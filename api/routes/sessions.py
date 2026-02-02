"""
Session management endpoints for WebRTC connections.
Uses LiveKit for WebRTC infrastructure.
"""
import uuid
import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.routes.auth import verify_token
from config import settings

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
    room_name: str
    status: str
    created_at: datetime
    expires_at: datetime


# In-memory session storage (use Redis in production)
_sessions = {}


async def spawn_pipecat_bot(
    session_id: str,
    room_name: str,
    bot_token: str,
    livekit_url: str,
    system_prompt: Optional[str] = None
):
    """
    Spawn Pipecat bot to join LiveKit room and process speech.
    
    This runs as a background task and handles the full pipeline:
    - Audio input from user
    - Speech-to-text
    - LLM processing (Azure OpenAI)
    - Text-to-speech
    - Avatar generation (MuseTalk)
    - Video output to user
    """
    try:
        logger.info(f"🤖 Spawning Pipecat bot for session {session_id}")
        logger.info(f"Room: {room_name} @ {livekit_url}")
        
        # Import here to avoid circular dependency
        from server.main import AvatarPipeline
        
        # Create and run pipeline
        pipeline = AvatarPipeline(
            room_name=room_name,
            token=bot_token,
            livekit_url=livekit_url,
            avatar_image=None,  # Use default avatar
            system_prompt=system_prompt
        )
        
        logger.info(f"✅ Bot joining room for session {session_id}")
        
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
    
    Creates a LiveKit room and returns connection details.
    
    Returns:
        Session details including WebRTC URL and token
    """
    from server.livekit_manager import livekit_manager
    
    session_id = str(uuid.uuid4())
    room_name = f"avatar-{session_id}"
    
    try:
        # 1. Create token for USER
        user_token = livekit_manager.create_token(
            room_name=room_name,
            participant_name="User",
            is_publisher=True,
            ttl_seconds=3600
        )
        
        # 2. Create token for BOT
        bot_token = livekit_manager.create_token(
            room_name=room_name,
            participant_name="Avatar Bot",
            is_publisher=True,
            ttl_seconds=3600
        )
        
        # Get LiveKit URL
        livekit_url = livekit_manager.get_room_url(room_name)
        
        logger.info(f"✅ Created LiveKit tokens for session {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to create LiveKit tokens: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create LiveKit session: {str(e)}"
        )
    
    created_at = datetime.utcnow()
    expires_at = created_at + timedelta(hours=1)
    
    # Store session
    session_data = {
        "session_id": session_id,
        "avatar_id": request.avatar_id or "default",
        "webrtc_url": livekit_url,
        "webrtc_token": user_token,  # User token (not bot token)
        "room_name": room_name,
        "status": "active",
        "created_at": created_at,
        "expires_at": expires_at,
        "api_key": "dev-mode",  # Dummy value for dev
        "system_prompt": request.system_prompt
    }
    
    _sessions[session_id] = session_data
    
    # 3. Spawn Pipecat bot as background task
    bot_task = asyncio.create_task(
        spawn_pipecat_bot(
            session_id=session_id,
            room_name=room_name,
            bot_token=bot_token,
            livekit_url=livekit_url,
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
    """Delete/terminate a session."""
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
    
    # Mark as terminated
    session["status"] = "terminated"
    _sessions[session_id] = session
    
    logger.info(f"🛑 Session {session_id} terminated")
    
    return {"status": "terminated", "session_id": session_id}


@router.get("/")
async def list_sessions():
    """List all sessions."""
    user_sessions = [
        SessionResponse(**session)
        for session in _sessions.values()
    ]
    
    return {"sessions": user_sessions, "count": len(user_sessions)}
