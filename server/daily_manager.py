"""
Daily.co API Manager

Handles creation, management, and cleanup of Daily.co rooms for WebRTC sessions.
"""
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import aiohttp

from config import settings

logger = logging.getLogger(__name__)


class DailyManager:
    """
    Manager for Daily.co room creation and lifecycle.
    
    Daily.co provides WebRTC infrastructure for real-time audio/video communication.
    This manager handles:
    - Room creation
    - Meeting token generation
    - Room cleanup
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Daily manager.
        
        Args:
            api_key: Daily.co API key. If None, uses settings.daily_api_key
        """
        self.api_key = api_key or settings.daily_api_key
        self.base_url = "https://api.daily.co/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def create_room(
        self,
        name: Optional[str] = None,
        privacy: str = "private",
        max_participants: int = 2,
        enable_chat: bool = False,
        enable_screenshare: bool = False,
        start_video_off: bool = True,
        start_audio_off: bool = False,
        exp: int = None
    ) -> Dict:
        """
        Create a Daily.co room.
        
        Args:
            name: Room name. If None, Daily.co generates random name
            privacy: "private" or "public"
            max_participants: Maximum number of participants
            enable_chat: Enable chat feature
            enable_screenshare: Enable screenshare
            start_video_off: Start with video off (user doesn't need camera)
            start_audio_off: Start with audio off (we need microphone on)
            exp: Expiration time in seconds from now. If None, room lasts forever.
        
        Returns:
            Dict with room details including:
                - id: Room ID
                - name: Room name
                - url: Room URL
                - config: Room configuration
                - created_at: Creation timestamp
        """
        try:
            # Minimal payload - just room name
            # No properties needed, Daily.co will use defaults
            payload = {}
            
            # Add custom name if provided
            if name:
                payload["name"] = name
            
            logger.info(f"Creating Daily.co room with payload: {payload}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/rooms",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Daily.co API error: {response.status} - {error_text}")
                        raise aiohttp.ClientError(f"{response.status}, message='{response.reason}', url='{response.url}'")
                    
                    room_data = await response.json()
                    
                    logger.info(f"✅ Created Daily.co room: {room_data.get('url')}")
                    return room_data
                    
        except aiohttp.ClientError as e:
            logger.error(f"Failed to create Daily.co room: {e}")
            raise
    
    async def create_token(
        self,
        room_name: str,
        is_owner: bool = True,
        user_name: Optional[str] = None,
        exp: int = 3600
    ) -> Dict:
        """
        Create a meeting token for a room.
        
        Args:
            room_name: Name of the room
            is_owner: Whether user has owner privileges
            user_name: Optional username to display
            exp: Token expiration in seconds (ignored for now - using Daily.co defaults)
        
        Returns:
            Dict with:
                - token: Meeting token string
        """
        try:
            # Minimal valid payload - just room name
            properties = {
                "room_name": room_name,
            }
            
            # Add is_owner if True
            if is_owner:
                properties["is_owner"] = True
            
            # Add username if provided
            if user_name:
                properties["user_name"] = user_name
            
            payload = {"properties": properties}
            
            logger.info(f"Creating token for room: {room_name}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/meeting-tokens",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Daily.co token API error: {response.status} - {error_text}")
                        raise aiohttp.ClientError(f"{response.status}, message='{response.reason}', url='{response.url}'")
                    
                    token_data = await response.json()
                    
                    logger.info(f"✅ Created token for room: {room_name}")
                    return token_data
                    
        except aiohttp.ClientError as e:
            logger.error(f"Failed to create token for room {room_name}: {e}")
            raise
    
    async def delete_room(self, room_name: str) -> bool:
        """
        Delete a Daily.co room.
        
        Args:
            room_name: Name of the room to delete
        
        Returns:
            bool: True if deleted successfully
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/rooms/{room_name}",
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    logger.info(f"Deleted Daily.co room: {room_name}")
                    return True
                    
        except aiohttp.ClientError as e:
            logger.error(f"Failed to delete room {room_name}: {e}")
            return False
    
    async def get_room(self, room_name: str) -> Optional[Dict]:
        """
        Get details of a room.
        
        Args:
            room_name: Name of the room
        
        Returns:
            Dict with room details or None if not found
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/rooms/{room_name}",
                    headers=self.headers
                ) as response:
                    if response.status == 404:
                        return None
                    response.raise_for_status()
                    room_data = await response.json()
                    return room_data
                    
        except aiohttp.ClientError as e:
            logger.error(f"Failed to get room {room_name}: {e}")
            return None
    
    async def list_rooms(self, limit: int = 100) -> list:
        """
        List all rooms.
        
        Args:
            limit: Maximum number of rooms to return
        
        Returns:
            List of room dicts
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/rooms",
                    headers=self.headers,
                    params={"limit": limit}
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("data", [])
                    
        except aiohttp.ClientError as e:
            logger.error(f"Failed to list rooms: {e}")
            return []


# Global instance
daily_manager = DailyManager()
