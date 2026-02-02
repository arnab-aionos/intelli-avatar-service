"""
Pipecat WebRTC server for real-time avatar streaming.

This server handles WebRTC connections and orchestrates the AI pipeline:
Audio Input → ChatGPT → EdgeTTS → MuseTalk → Video Output

Uses LiveKit for WebRTC infrastructure.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator
)

# Import custom processors
import sys
sys.path.append(str(Path(__file__).parent.parent))
from processors.musetalk_processor import MuseTalkProcessor
from config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AvatarPipeline:
    """
    Main pipeline for avatar streaming.
    
    Orchestrates:
    1. Audio input from WebRTC (via LiveKit)
    2. Speech-to-text
    3. LLM response generation (ChatGPT)
    4. Text-to-speech
    5. Avatar video generation (MuseTalk)
    6. Video output to WebRTC
    """
    
    def __init__(
        self,
        room_name: str,
        token: str,
        livekit_url: str,
        avatar_image: str = None,
        system_prompt: str = None
    ):
        """
        Initialize avatar pipeline.
        
        Args:
            room_name: LiveKit room name
            token: LiveKit access token
            livekit_url: LiveKit WebSocket URL
            avatar_image: Path to avatar image
            system_prompt: System prompt for LLM
        """
        self.room_name = room_name
        self.token = token
        self.livekit_url = livekit_url
        self.avatar_image = avatar_image
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Pipeline components
        self.transport: Optional[LiveKitTransport] = None
        self.llm_service: Optional[OpenAILLMService] = None
        self.avatar_processor: Optional[MuseTalkProcessor] = None
        self.pipeline: Optional[Pipeline] = None
        self.runner: Optional[PipelineRunner] = None
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt for avatar."""
        return (
            "You are a helpful AI assistant appearing as a lifelike avatar. "
            "You provide clear, concise, and friendly responses. "
            "Keep your answers brief and conversational."
        )
    
    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("🔧 Initializing avatar pipeline...")
        
        # Initialize LiveKit transport
        logger.info("📡 Step 1/4: Initializing LiveKit transport...")
        try:
            self.transport = LiveKitTransport(
                url=self.livekit_url,
                token=self.token,
                room_name=self.room_name,
                params=LiveKitParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    video_out_enabled=True,
                    vad_enabled=True,
                    vad_audio_passthrough=True
                )
            )
            logger.info("✅ LiveKit transport initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LiveKit transport: {e}", exc_info=True)
            raise
        
        # Initialize Azure OpenAI LLM
        logger.info("🤖 Step 2/4: Initializing Azure OpenAI LLM...")
        try:
            from openai import AzureOpenAI
            
            # Create Azure OpenAI client
            azure_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version
            )
            
            self.llm_service = OpenAILLMService(
                api_key=settings.azure_openai_key,
                model=settings.azure_openai_deployment,
                system_prompt=self.system_prompt
            )
            
            logger.info(f"✅ Azure OpenAI initialized with deployment: {settings.azure_openai_deployment}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI: {e}")
            # Fallback to standard OpenAI if Azure fails
            if settings.openai_api_key:
                logger.warning("Falling back to standard OpenAI")
                self.llm_service = OpenAILLMService(
                    api_key=settings.openai_api_key,
                    model="gpt-3.5-turbo",
                    system_prompt=self.system_prompt
                )
            else:
                raise RuntimeError("No valid OpenAI configuration found")
        
        # Initialize MuseTalk processor
        logger.info("🎭 Step 3/4: Initializing MuseTalk processor...")
        try:
            self.avatar_processor = MuseTalkProcessor(
                avatar_image_path=self.avatar_image,
                fps=settings.musetalk_fps,
                batch_size=settings.musetalk_batch_size
            )
            logger.info("✅ MuseTalk processor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MuseTalk: {e}", exc_info=True)
            raise
        
        # Build pipeline
        logger.info("⚙️ Step 4/4: Building Pipecat pipeline...")
        try:
            self.pipeline = Pipeline([
                self.transport.input(),
                LLMUserResponseAggregator(),
                self.llm_service,
                LLMAssistantResponseAggregator(),
                self.avatar_processor,
                self.transport.output()
            ])
            logger.info("✅ Pipeline built")
            
            # Create pipeline task
            task = PipelineTask(
                self.pipeline,
                params=LiveKitParams(
                    audio_out_enabled=True,
                    video_out_enabled=True
                )
            )
            logger.info("✅ Pipeline task created")
        except Exception as e:
            logger.error(f"❌ Failed to build pipeline: {e}", exc_info=True)
            raise
        
        # Create runner
        self.runner = PipelineRunner()
        
        logger.info("✅ Avatar pipeline initialized successfully")
        
        return task
    
    async def run(self):
        """Run the pipeline."""
        logger.info(f"🎬 Starting avatar pipeline for room: {self.room_name}")
        
        try:
            # Initialize pipeline
            task = await self.initialize()
            
            logger.info("🚀 Running Pipecat pipeline - bot should join room soon...")
            
            # Run pipeline
            await self.runner.run(task)
            
        except asyncio.CancelledError:
            logger.info("🛑 Pipeline cancelled")
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
            raise
        finally:
            logger.info("🏁 Pipeline execution ended")
    
    async def cleanup(self):
        """Clean up pipeline resources."""
        logger.info("Cleaning up avatar pipeline...")
        
        if self.avatar_processor:
            await self.avatar_processor.cleanup()
        
        if self.transport:
            await self.transport.cleanup()
        
        logger.info("Avatar pipeline cleaned up")


async def main():
    """
    Main entry point for standalone server.
    
    For production, use the REST API to create sessions dynamically.
    This is for testing purposes.
    """
    logger.info("Starting IntelliAvatar Service (WebRTC Server)")
    logger.info(f"Server configured for {settings.musetalk_device} mode")
    logger.info(f"LiveKit URL: {settings.livekit_url}")
    
    if settings.musetalk_device == "cpu":
        logger.warning(
            "Running in CPU mode. Performance will be limited. "
            "GPU is required for real-time avatar generation."
        )
    
    logger.info(
        "WebRTC server ready. "
        "Use the REST API to create sessions and rooms."
    )
    
    # Keep server running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
