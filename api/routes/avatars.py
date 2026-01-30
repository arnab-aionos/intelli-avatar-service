"""
Avatar configuration endpoints.
"""
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.routes.auth import verify_token

router = APIRouter()


class AvatarConfig(BaseModel):
    """Avatar configuration model."""
    avatar_id: str
    name: str
    image_path: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AvatarCreateRequest(BaseModel):
    """Request model for creating avatar configuration."""
    name: str = Field(..., description="Avatar name")
    description: Optional[str] = Field(None, description="Avatar description")


# In-memory avatar storage (use database in production)
_avatars = {
    "default": {
        "avatar_id": "default",
        "name": "Default Avatar",
        "image_path": "../talkingavatar/inputs/girl.png",
        "description": "Default female avatar",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
}


@router.post("/", response_model=AvatarConfig)
async def create_avatar(
    request: AvatarCreateRequest,
    api_key: str = Depends(verify_token)
):
    """
    Create a new avatar configuration.
    
    Note: Image must be uploaded separately via /avatars/{avatar_id}/image
    """
    avatar_id = f"avatar_{len(_avatars)}"
    
    avatar_data = {
        "avatar_id": avatar_id,
        "name": request.name,
        "image_path": "",  # Set via upload endpoint
        "description": request.description,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    _avatars[avatar_id] = avatar_data
    
    return AvatarConfig(**avatar_data)


@router.get("/", response_model=List[AvatarConfig])
async def list_avatars(api_key: str = Depends(verify_token)):
    """List all available avatar configurations."""
    return [AvatarConfig(**avatar) for avatar in _avatars.values()]


@router.get("/{avatar_id}", response_model=AvatarConfig)
async def get_avatar(
    avatar_id: str,
    api_key: str = Depends(verify_token)
):
    """Get avatar configuration by ID."""
    if avatar_id not in _avatars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar not found"
        )
    
    return AvatarConfig(**_avatars[avatar_id])


@router.post("/{avatar_id}/image")
async def upload_avatar_image(
    avatar_id: str,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_token)
):
    """Upload avatar image."""
    if avatar_id not in _avatars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar not found"
        )
    
    # Save uploaded file
    upload_dir = Path("uploads/avatars")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{avatar_id}.png"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Update avatar config
    _avatars[avatar_id]["image_path"] = str(file_path)
    _avatars[avatar_id]["updated_at"] = datetime.utcnow()
    
    return {
        "avatar_id": avatar_id,
        "image_path": str(file_path),
        "message": "Image uploaded successfully"
    }


@router.delete("/{avatar_id}")
async def delete_avatar(
    avatar_id: str,
    api_key: str = Depends(verify_token)
):
    """Delete avatar configuration."""
    if avatar_id == "default":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default avatar"
        )
    
    if avatar_id not in _avatars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar not found"
        )
    
    del _avatars[avatar_id]
    
    return {"status": "deleted", "avatar_id": avatar_id}
