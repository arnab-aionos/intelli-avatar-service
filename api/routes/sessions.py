"""
Session management endpoints for WebRTC connections.
"""
import uuid
import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.routes.auth import verify_token

router = APIRouter()
logger = logging.getLogger(__name__)

# Track active bot tasks
_bot_tasks: Dict[str, asyncio.Task] = {}


class SessionCreateRequest(BaseModel):
    """Request model for creating a new session."""
    avatar_id: Optional[str] = Field(default="default", description="Avatar configuration ID")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt for LLM")


class SessionResponse(BaseModel):
    """Response model for session."""
    session_id: str
    avatar_id: str
    webrtc_url: str
    webrtc_token: str
    status: str
    created_at: datetime
    expires_at: datetime


# In-memory session storage (use Redis in production)
_sessions = {}


async def spawn_pipecat_bot(session_id: str, room_url: str, bot_token: str, system_prompt: Optional[str] = None):
    """
    Spawn Pipecat bot to join Daily.co room and process speech.
    
    This runs as a background task and handles the full pipeline:
    - Audio input from user
    - Speech-to-text (Daily transcription)
    - LLM processing (Azure OpenAI)
    - Text-to-speech (Daily TTS)
    - Avatar generation (MuseTalk)
    - Video output to user
    """
    try:
        logger.info(f"🤖 Spawning Pipecat bot for session {session_id}")
        logger.info(f"Room URL: {room_url}")
        
        # Import here to avoid circular dependency
        from server.main import AvatarPipeline
        
        # Create and run pipeline
        pipeline = AvatarPipeline(
            room_url=room_url,
            token=bot_token,
            avatar_image=None,  # Use default avatar
            system_prompt=system_prompt
        )
        
        logger.info(f"✅ Bot joined room for session {session_id}")
        
        # Run pipeline (blocks until session ends)
        await pipeline.run()
        
        logger.info(f"🛑 Bot left room for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Bot error for session {session_id}: {e}", exc_info=True)
        # Update session status
        if session_id in _sessions:
            _sessions[session_id]["status"] = "error"
            _sessions[session_id]["error"] = str(e)
    finally:
        # Cleanup
        if session_id in _bot_tasks:
            del _bot_tasks[session_id]
        logger.info(f"🧹 Cleaned up bot task for session {session_id}")


@router.post("/", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest
    # Removed auth for dev mode - add back later: api_key: str = Depends(verify_token)
):
    """
    Create a new WebRTC session for real-time avatar streaming.
    
    Creates a Daily.co room and returns connection details.
    
    Returns:
        Session details including WebRTC URL and token
    """
    from server.daily_manager import daily_manager
    
    session_id = str(uuid.uuid4())
    
    try:
        # 1. Create Daily.co room
        room_name = f"avatar-{session_id}"
        room = await daily_manager.create_room(
            name=room_name,
            privacy="private",
            max_participants=2,
            enable_chat=False,
            enable_screenshare=False,
            start_video_off=True,  # User doesn't need camera
            start_audio_off=False,  # Need microphone
            exp=3600  # 1 hour expiration
        )
        
        webrtc_url = room.get("url")
        
        # 2. Create token for USER
        user_token_data = await daily_manager.create_token(
            room_name=room_name,
            is_owner=True,
            user_name="User",
            exp=3600
        )
        
        user_token = user_token_data.get("token")
        
        # 3. Create token for BOT
        bot_token_data = await daily_manager.create_token(
            room_name=room_name,
            is_owner=False,  # Bot is not owner
            user_name="Avatar Bot",
            exp=3600
        )
        
        bot_token = bot_token_data.get("token")
        
        logger.info(f"✅ Created room and tokens for session {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to create Daily.co room: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create Daily.co room: {str(e)}"
        )
    
    created_at = datetime.utcnow()
    expires_at = datetime.utcnow().replace(hour=created_at.hour + 1)
    
    # Store session
    session_data = {
        "session_id": session_id,
        "avatar_id": request.avatar_id or "default",
        "webrtc_url": webrtc_url,
        "webrtc_token": user_token,  # User token (not bot token)
        "status": "active",
        "created_at": created_at,
        "expires_at": expires_at,
        "api_key": "dev-mode",  # Dummy value for dev
        "system_prompt": request.system_prompt,
        "room_name": room_name  # Store for cleanup
    }
    
    _sessions[session_id] = session_data
    
    # 4. Spawn Pipecat bot as background task
    bot_task = asyncio.create_task(
        spawn_pipecat_bot(
            session_id=session_id,
            room_url=webrtc_url,
            bot_token=bot_token,
            system_prompt=request.system_prompt
        )
    )
    
    _bot_tasks[session_id] = bot_task
    logger.info(f"🚀 Pipecat bot task started for session {session_id}")
    
    return SessionResponse(**session_data)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details by ID."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session = _sessions[session_id]
    return SessionResponse(**session)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete/terminate a session and cleanup Daily.co room."""
    from server.daily_manager import daily_manager
    
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session = _sessions[session_id]
    
    # Cancel bot task if running
    if session_id in _bot_tasks:
        bot_task = _bot_tasks[session_id]
        if not bot_task.done():
            logger.info(f"Cancelling bot task for session {session_id}")
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass
        del _bot_tasks[session_id]
    
    # Delete Daily.co room
    room_name = session.get("room_name")
    if room_name:
        await daily_manager.delete_room(room_name)
    
    # Mark as terminated
    session["status"] = "terminated"
    _sessions[session_id] = session
    
    logger.info(f"🛑 Session {session_id} terminated and cleaned up")
    
    return {"status": "terminated", "session_id": session_id}


@router.get("/")
async def list_sessions():
    """List all sessions."""
    user_sessions = [
        SessionResponse(**session)
        for session in _sessions.values()
    ]
    
    return {"sessions": user_sessions, "count": len(user_sessions)}
