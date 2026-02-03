"""
WebSocket server for real-time conversational avatar.

Handles:
- Text/audio input from browser
- LLM responses via self-hosted H200 or Azure OpenAI
- TTS (Text-to-Speech) via Edge TTS
- Avatar video frame streaming via MuseTalk (GPU) or static image (CPU)
"""
import asyncio
import base64
import json
import logging
import os
import uuid
import tempfile
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
import httpx
import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Default avatar image path
DEFAULT_AVATAR_PATH = Path(__file__).parent.parent / "assets" / "default_avatar.png"


@dataclass
class ConversationSession:
    """Represents an active conversation session."""
    session_id: str
    websocket: WebSocket
    system_prompt: str = "You are a helpful AI assistant. Be concise and friendly."
    conversation_history: list = field(default_factory=list)
    is_processing: bool = False
    avatar_image_path: Optional[str] = None


class AvatarWebSocketHandler:
    """
    Handles WebSocket connections for real-time avatar conversations.
    
    Flow:
    1. Client connects and optionally sends config (avatar image, system prompt)
    2. Client performs browser-side speech recognition and sends TEXT
    3. Server: LLM → TTS → MuseTalk (if GPU) or Static Avatar (if CPU) → video frames
    4. Server streams video/audio back to client
    5. Repeat from step 2
    """
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self._llm_client = None
        self._tts_voice = "en-US-AriaNeural"
        self._musetalk = None
        self._musetalk_available = False
        self._http_client = None
        self._default_avatar_frame = None
        
    async def _init_services(self):
        """Lazy initialization of AI services."""
        if self._http_client is None:
            logger.info("🔧 Initializing AI services...")
            
            # Initialize HTTP client for self-hosted LLM
            self._http_client = httpx.AsyncClient(timeout=60.0)
            
            # Choose LLM backend
            if settings.use_self_hosted_llm:
                logger.info(f"✅ Using self-hosted LLM at: {settings.self_hosted_llm_url}")
                logger.info(f"   Model: {settings.self_hosted_llm_model}")
                self._use_self_hosted = True
            else:
                # Initialize Azure OpenAI LLM
                try:
                    from openai import AsyncAzureOpenAI
                    self._llm_client = AsyncAzureOpenAI(
                        api_key=settings.azure_openai_key,
                        azure_endpoint=settings.azure_openai_endpoint,
                        api_version=settings.azure_openai_api_version
                    )
                    logger.info(f"✅ Azure OpenAI initialized: {settings.azure_openai_deployment}")
                    self._use_self_hosted = False
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Azure OpenAI: {e}")
                    raise
            
            # Initialize TTS (Edge TTS - free)
            try:
                import edge_tts
                logger.info(f"✅ Edge TTS initialized with voice: {self._tts_voice}")
            except ImportError:
                logger.warning("⚠️ edge-tts not installed. Run: pip install edge-tts")
            
            # Try to initialize MuseTalk (requires GPU)
            await self._init_musetalk()
            
            # Load default avatar frame for CPU fallback
            await self._load_default_avatar()
    
    async def _init_musetalk(self):
        """Initialize MuseTalk if GPU is available."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("⚠️ GPU not available. Avatar will use static image mode.")
                logger.warning("   For lip-sync animation, deploy to GPU server (H200).")
                return
            
            # Import MuseTalk
            import sys
            sys.path.append(str(Path(__file__).parent.parent / "Musetalk"))
            from musetalk.MuseTalk import MuseTalk_RealTime
            
            logger.info("🎭 Initializing MuseTalk (GPU mode)...")
            loop = asyncio.get_event_loop()
            self._musetalk = await loop.run_in_executor(None, MuseTalk_RealTime)
            await loop.run_in_executor(None, self._musetalk.init_model)
            self._musetalk_available = True
            logger.info("✅ MuseTalk initialized successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ MuseTalk not installed: {e}")
        except Exception as e:
            logger.warning(f"⚠️ MuseTalk failed to initialize: {e}")
    
    async def _load_default_avatar(self):
        """Load default avatar image for static display."""
        try:
            # Check for default avatar in assets
            if DEFAULT_AVATAR_PATH.exists():
                img = cv2.imread(str(DEFAULT_AVATAR_PATH))
                if img is not None:
                    # Resize to standard size
                    img = cv2.resize(img, (512, 512))
                    _, buffer = cv2.imencode('.jpg', img)
                    self._default_avatar_frame = base64.b64encode(buffer).decode('utf-8')
                    logger.info("✅ Default avatar loaded")
                    return
            
            # Create a placeholder avatar if no default exists
            logger.info("📷 Creating placeholder avatar (no default image found)")
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            img[:] = (40, 40, 50)  # Dark background
            
            # Draw a simple avatar placeholder
            cv2.circle(img, (256, 200), 80, (100, 100, 120), -1)  # Head
            cv2.ellipse(img, (256, 420), (120, 100), 0, 180, 360, (100, 100, 120), -1)  # Body
            
            # Add text
            cv2.putText(img, "Avatar", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
            
            _, buffer = cv2.imencode('.jpg', img)
            self._default_avatar_frame = base64.b64encode(buffer).decode('utf-8')
            logger.info("✅ Placeholder avatar created")
            
        except Exception as e:
            logger.error(f"Failed to load avatar: {e}")
    
    async def handle_connection(self, websocket: WebSocket):
        """Handle a new WebSocket connection."""
        await websocket.accept()
        
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            websocket=websocket
        )
        self.sessions[session_id] = session
        
        logger.info(f"🔌 New connection: {session_id}")
        
        # Send welcome message with mode info
        mode = "gpu" if self._musetalk_available else "static"
        await self._send_message(websocket, {
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to Avatar Service",
            "avatar_mode": mode
        })
        
        await self._init_services()
        
        # Send initial avatar frame
        if self._default_avatar_frame:
            await self._send_message(websocket, {
                "type": "video_frame",
                "data": self._default_avatar_frame
            })
        
        try:
            await self._handle_messages(session)
        except WebSocketDisconnect:
            logger.info(f"🔌 Disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ Error in session {session_id}: {e}", exc_info=True)
        finally:
            self._cleanup_session(session_id)
    
    async def _handle_messages(self, session: ConversationSession):
        """Process incoming messages from the client."""
        while True:
            data = await session.websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "config":
                # Update session configuration
                session.system_prompt = message.get("system_prompt", session.system_prompt)
                session.avatar_image_path = message.get("avatar_image")
                logger.info(f"📋 Config updated for {session.session_id}")
                
                # If avatar image provided, prepare MuseTalk material
                if session.avatar_image_path and self._musetalk_available:
                    await self._prepare_avatar(session)
                    
            elif msg_type == "text_input":
                # Direct text input (from browser speech recognition)
                user_text = message.get("text", "").strip()
                if user_text and not session.is_processing:
                    session.is_processing = True
                    await self._process_conversation(session, user_text)
                    session.is_processing = False
                    
            elif msg_type == "interrupt":
                # User interrupted - stop current processing
                session.is_processing = False
                await self._send_message(session.websocket, {
                    "type": "interrupted",
                    "message": "Processing interrupted"
                })
                
            elif msg_type == "ping":
                await self._send_message(session.websocket, {"type": "pong"})
    
    async def _prepare_avatar(self, session: ConversationSession):
        """Prepare MuseTalk with avatar image (GPU only)."""
        if not self._musetalk_available or not session.avatar_image_path:
            return
        
        try:
            logger.info(f"🎭 Preparing avatar material: {session.avatar_image_path}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._musetalk.prepare_material(session.avatar_image_path, bbox_shift=5)
            )
            logger.info("✅ Avatar material prepared")
        except Exception as e:
            logger.error(f"Failed to prepare avatar: {e}")
    
    async def _process_conversation(self, session: ConversationSession, user_text: str):
        """
        Full conversation pipeline:
        Text → LLM → TTS → Avatar Video → Stream to client
        """
        try:
            logger.info(f"🎤 User said: {user_text}")
            await self._send_message(session.websocket, {
                "type": "user_text",
                "text": user_text
            })
            
            # Step 1: LLM Response
            await self._send_message(session.websocket, {
                "type": "processing",
                "status": "thinking"
            })
            
            response_text = await self._get_llm_response(session, user_text)
            logger.info(f"🤖 AI response: {response_text[:100]}...")
            
            await self._send_message(session.websocket, {
                "type": "response_text",
                "text": response_text
            })
            
            # Step 2: Text-to-Speech
            await self._send_message(session.websocket, {
                "type": "processing", 
                "status": "generating_speech"
            })
            
            audio_data, audio_path = await self._text_to_speech(response_text)
            
            # Step 3: Generate Avatar Video or Static Frame
            if self._musetalk_available and session.avatar_image_path and audio_path:
                await self._send_message(session.websocket, {
                    "type": "processing",
                    "status": "generating_avatar"
                })
                await self._generate_lip_sync_video(session, audio_path, audio_data)
            else:
                # Static avatar mode - send audio with static frame
                await self._send_static_avatar_with_audio(session, audio_data)
            
            await self._send_message(session.websocket, {
                "type": "response_complete"
            })
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
            await self._send_message(session.websocket, {
                "type": "error",
                "message": str(e)
            })
    
    async def _get_llm_response(self, session: ConversationSession, user_text: str) -> str:
        """Get response from LLM (self-hosted or Azure)."""
        # Build conversation history
        messages = [
            {"role": "system", "content": session.system_prompt}
        ]
        messages.extend(session.conversation_history)
        messages.append({"role": "user", "content": user_text})
        
        try:
            if settings.use_self_hosted_llm:
                # Use self-hosted LLM via HTTP
                response = await self._http_client.post(
                    f"{settings.self_hosted_llm_url}/chat/completions",
                    json={
                        "model": settings.self_hosted_llm_model,
                        "messages": messages,
                        "max_tokens": 500,
                        "temperature": 0.7
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
            else:
                # Use Azure OpenAI
                response = await self._llm_client.chat.completions.create(
                    model=settings.azure_openai_deployment,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                assistant_message = response.choices[0].message.content
            
            # Update conversation history
            session.conversation_history.append({"role": "user", "content": user_text})
            session.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep history manageable (last 10 exchanges)
            if len(session.conversation_history) > 20:
                session.conversation_history = session.conversation_history[-20:]
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm sorry, I had trouble processing that. Could you try again?"
    
    async def _text_to_speech(self, text: str) -> tuple[bytes, Optional[str]]:
        """Convert text to speech using Edge TTS. Returns (audio_bytes, temp_file_path)."""
        try:
            import edge_tts
            
            logger.info(f"🔊 Starting TTS for text: {text[:50]}...")
            
            communicate = edge_tts.Communicate(text, self._tts_voice)
            
            # Save to temp file (needed for MuseTalk)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
            
            await communicate.save(temp_path)
            
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"🔊 TTS generated {len(audio_data)} bytes of audio")
            
            return audio_data, temp_path
            
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return b"", None
    
    async def _generate_lip_sync_video(self, session: ConversationSession, audio_path: str, audio_data: bytes):
        """Generate lip-synced avatar video using MuseTalk (GPU only)."""
        try:
            logger.info("🎬 Generating lip-sync video with MuseTalk...")
            
            loop = asyncio.get_event_loop()
            video_path = await loop.run_in_executor(
                None,
                lambda: self._musetalk.inference_noprepare(
                    audio_path,
                    session.avatar_image_path,
                    bbox_shift=5,
                    batch_size=4,
                    fps=25
                )
            )
            
            logger.info(f"🎬 Video generated: {video_path}")
            
            # Read video and stream frames
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_delay = 1.0 / fps
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame
                await self._send_message(session.websocket, {
                    "type": "video_frame",
                    "data": frame_b64,
                    "frame_number": frame_count
                })
                
                # Pace the frames
                await asyncio.sleep(frame_delay)
                frame_count += 1
            
            cap.release()
            logger.info(f"🎬 Streamed {frame_count} video frames")
            
            # Also send audio
            await self._send_message(session.websocket, {
                "type": "audio_response",
                "data": base64.b64encode(audio_data).decode("utf-8")
            })
            
            # Cleanup temp files
            try:
                os.unlink(audio_path)
                os.unlink(video_path)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Lip-sync generation error: {e}", exc_info=True)
            # Fallback to static avatar
            await self._send_static_avatar_with_audio(session, audio_data)
    
    async def _send_static_avatar_with_audio(self, session: ConversationSession, audio_data: bytes):
        """Send static avatar frame with audio (CPU fallback)."""
        try:
            # Send static avatar frame (if not already sent)
            if self._default_avatar_frame:
                await self._send_message(session.websocket, {
                    "type": "video_frame",
                    "data": self._default_avatar_frame
                })
            
            # Send audio
            if audio_data:
                await self._send_message(session.websocket, {
                    "type": "audio_response",
                    "data": base64.b64encode(audio_data).decode("utf-8")
                })
            
        except Exception as e:
            logger.error(f"Static avatar error: {e}")
    
    async def _send_message(self, websocket: WebSocket, message: dict):
        """Send JSON message to client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    def _cleanup_session(self, session_id: str):
        """Clean up session resources."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🧹 Session {session_id} cleaned up")


# Global handler instance
avatar_handler = AvatarWebSocketHandler()
