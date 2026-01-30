"""
Asynchronous video generation endpoints (non-WebRTC).
"""
import uuid
from typing import Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.routes.auth import verify_token

router = APIRouter()


class GenerationStatus(str, Enum):
    """Status of video generation job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationRequest(BaseModel):
    """Request model for async video generation."""
    text: str = Field(..., description="Text for avatar to speak")
    avatar_id: str = Field(default="default", description="Avatar to use")
    voice: str = Field(default="en-US-JennyNeural", description="TTS voice")


class GenerationResponse(BaseModel):
    """Response model for generation job."""
    job_id: str
    status: GenerationStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    video_url: Optional[str] = None
    error: Optional[str] = None


# In-memory job storage (use database in production)
_jobs = {}


async def process_generation_job(job_id: str, request: GenerationRequest):
    """Background task to process video generation."""
    try:
        # Mark as processing
        _jobs[job_id]["status"] = GenerationStatus.PROCESSING
        
        # In production: Call MuseTalk to generate video
        # For now, simulate processing
        import asyncio
        await asyncio.sleep(5)  # Simulate processing time
        
        # Mock video URL
        video_url = f"https://storage.example.com/videos/{job_id}.mp4"
        
        # Update job
        _jobs[job_id].update({
            "status": GenerationStatus.COMPLETED,
           "completed_at": datetime.utcnow(),
            "video_url": video_url
        })
        
    except Exception as e:
        _jobs[job_id].update({
            "status": GenerationStatus.FAILED,
            "completed_at": datetime.utcnow(),
            "error": str(e)
        })


@router.post("/", response_model=GenerationResponse)
async def create_generation_job(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_token)
):
    """
    Create asynchronous video generation job.
    
    Use this for non-real-time video generation.
    For real-time streaming, use WebRTC sessions instead.
    """
    job_id = str(uuid.uuid4())
    
    job_data = {
        "job_id": job_id,
        "status": GenerationStatus.PENDING,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "video_url": None,
        "error": None,
        "api_key": api_key,
        "request": request.dict()
    }
    
    _jobs[job_id] = job_data
    
    # Schedule background processing
    background_tasks.add_task(process_generation_job, job_id, request)
    
    return GenerationResponse(**job_data)


@router.get("/{job_id}", response_model=GenerationResponse)
async def get_generation_job(
    job_id: str,
    api_key: str = Depends(verify_token)
):
    """Get status of video generation job."""
    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    job = _jobs[job_id]
    
    # Verify ownership
    if job["api_key"] != api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return GenerationResponse(**job)


@router.get("/")
async def list_generation_jobs(api_key: str = Depends(verify_token)):
    """List all generation jobs for the authenticated user."""
    user_jobs = [
        GenerationResponse(**job)
        for job in _jobs.values()
        if job["api_key"] == api_key
    ]
    
    return {"jobs": user_jobs, "count": len(user_jobs)}
