"""
Pipecat WebRTC server for real-time avatar streaming.

This server handles WebRTC connections and orchestrates the AI pipeline:
Audio Input → ChatGPT → EdgeTTS → MuseTalk → Video Output
"""
import asyncio
import logging
from pathlib import Path

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.services.openai import OpenAILLMService
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
    1. Audio input from WebRTC
    2. Speech-to-text (via Daily)
    3. LLM response generation (ChatGPT)
    4. Text-to-speech (via Daily)
    5. Avatar video generation (MuseTalk)
    6. Video output to WebRTC
    """
    
    def __init__(
        self,
        room_url: str,
        token: str,
        avatar_image: str = None,
        system_prompt: str = None
    ):
        """
        Initialize avatar pipeline.
        
        Args:
            room_url: Daily.co room URL for WebRTC
            token: Daily.co authentication token
            avatar_image: Path to avatar image
            system_prompt: System prompt for LLM
        """
        self.room_url = room_url
        self.token = token
        self.avatar_image = avatar_image
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Pipeline components
        self.transport: Optional[DailyTransport] = None
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
        
        # Initialize Daily transport (handles WebRTC + STT + TTS)
        logger.info("📡 Step 1/4: Initializing Daily.co transport...")
        try:
            self.transport = DailyTransport(
                self.room_url,
                self.token,
                "Avatar Bot",
                DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    video_out_enabled=True,
                    transcription_enabled=True,
                    vad_enabled=True,
                    vad_audio_passthrough=True
                )
            )
            logger.info("✅ Daily transport initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Daily transport: {e}", exc_info=True)
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
            
            # Note: Pipecat's OpenAILLMService needs to be configured for Azure
            # This might need adjustment based on Pipecat version
            self.llm_service = OpenAILLMService(
                api_key=settings.azure_openai_key,
                model=settings.azure_openai_deployment,  # Use deployment name for Azure
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
                params=DailyParams(
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
        logger.info(f"🎬 Starting avatar pipeline for room: {self.room_url}")
        
        try:
            # Initialize pipeline
            task = await self.initialize()
            
            logger.info("🚀 Running Pipecat pipeline - bot should join room soon...")
            
            # Run with timeout to prevent hanging
            await asyncio.wait_for(
                self.runner.run(task),
                timeout=None  # No timeout for now - session can be long
            )
            
        except asyncio.TimeoutError:
            logger.error("⏱️ Pipeline execution timed out")
            raise
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
    # For testing, you'd need to create a Daily.co room first
    # This is just a placeholder
    
    logger.info("Starting IntelliAvatar Service (WebRTC Server)")
    logger.info(f"Server configured for {settings.musetalk_device} mode")
    
    if settings.musetalk_device == "cpu":
        logger.warning(
            "Running in CPU mode. Performance will be limited. "
            "GPU is required for real-time avatar generation."
        )
    
    # In production, rooms are created via REST API
    # This would be called from the API layer
    logger.info(
        "WebRTC server ready. "
        "Use the REST API to create sessions and rooms."
    )
    
    # Keep server running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
