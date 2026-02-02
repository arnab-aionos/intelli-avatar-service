"""
LiveKit room and token management for IntelliAvatar Service.
Replaces Daily.co for WebRTC infrastructure.
"""
import logging
from datetime import timedelta
from livekit import api
from config import settings

logger = logging.getLogger(__name__)


class LiveKitManager:
    """Manages LiveKit rooms and access tokens."""
    
    def __init__(self):
        self.api_key = settings.livekit_api_key
        self.api_secret = settings.livekit_api_secret
        self.livekit_url = settings.livekit_url
    
    def create_token(
        self,
        room_name: str,
        participant_name: str,
        is_publisher: bool = True,
        ttl_seconds: int = 3600
    ) -> str:
        """
        Create an access token for a participant to join a room.
        
        Args:
            room_name: Name of the room to join
            participant_name: Display name for the participant
            is_publisher: Whether participant can publish audio/video
            ttl_seconds: Token validity in seconds
            
        Returns:
            JWT access token string
        """
        token = api.AccessToken(self.api_key, self.api_secret)
        token.with_identity(participant_name)
        token.with_name(participant_name)
        token.with_ttl(timedelta(seconds=ttl_seconds))
        
        # Grant room permissions
        grant = api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=is_publisher,
            can_subscribe=True,
            can_publish_data=True
        )
        token.with_grants(grant)
        
        jwt_token = token.to_jwt()
        logger.info(f"✅ Created LiveKit token for {participant_name} in room {room_name}")
        
        return jwt_token
    
    def get_room_url(self, room_name: str) -> str:
        """
        Get the WebSocket URL for a room.
        
        Args:
            room_name: Name of the room
            
        Returns:
            Full WebSocket URL for the room
        """
        return self.livekit_url


# Global manager instance
livekit_manager = LiveKitManager()
