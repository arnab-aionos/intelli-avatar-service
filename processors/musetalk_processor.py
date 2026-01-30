"""
Custom Pipecat processor for MuseTalk avatar generation.

This processor integrates MuseTalk for real-time frame-by-frame avatar video generation.
"""
import sys
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Add Musetalk to path to import MuseTalk
sys.path.append(str(Path(__file__).parent.parent / "Musetalk"))

try:
    from musetalk.MuseTalk import MuseTalk_RealTime
except ImportError:
    logging.warning("MuseTalk not found. Using mock processor for development.")
    MuseTalk_RealTime = None


from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    VideoFrame,
    StartFrame,
    EndFrame,
    SystemFrame
)

from config import settings

logger = logging.getLogger(__name__)


class MuseTalkProcessor(FrameProcessor):
    """
    Pipecat processor that converts audio frames to avatar video frames using MuseTalk.
    """
    
    def __init__(
        self,
        avatar_image_path: Optional[str] = None,
        bbox_shift: int = 5,
        fps: int = None,
        batch_size: int = None,
        **kwargs
    ):
        """
        Initialize MuseTalk processor.
        
        Args:
            avatar_image_path: Path to avatar image. If None, uses default.
            bbox_shift: Bounding box shift for face detection
            fps: Frames per second for generation
            batch_size: Batch size for processing
        """
        super().__init__(**kwargs)
        
        self.avatar_image_path = avatar_image_path
        self.bbox_shift = bbox_shift
        self.fps = fps or settings.musetalk_fps
        self.batch_size = batch_size or settings.musetalk_batch_size
        
        # Initialize MuseTalk
        self.musetalk: Optional[MuseTalk_RealTime] = None
        self._initialized = False
        self._audio_buffer = []
        self._buffer_duration = 0.0
        self._target_buffer_duration = 2.0  # Process 2-second chunks
        
        logger.info(
            f"MuseTalkProcessor initialized with fps={self.fps}, "
            f"batch_size={self.batch_size}"
        )
    
    async def initialize(self):
        """Initialize MuseTalk model (lazy loading)."""
        if self._initialized:
            return
        
        try:
            if MuseTalk_RealTime is None:
                logger.warning(
                    "MuseTalk not available. Running in mock mode for development."
                )
                self._initialized = True
                return
            
            logger.info("Initializing MuseTalk model...")
            
            # Initialize MuseTalk in separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.musetalk = await loop.run_in_executor(
                None,
                lambda: MuseTalk_RealTime()
            )
            
            # Prepare avatar material if image path provided
            if self.avatar_image_path:
                await loop.run_in_executor(
                    None,
                    lambda: self.musetalk.prepare_material(
                        self.avatar_image_path,
                        self.bbox_shift
                    )
                )
            
            self._initialized = True
            logger.info("MuseTalk model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MuseTalk: {e}")
            raise
    
    async def process_frame(self, frame: Frame, direction: str):
        """
        Process incoming frames.
        
        Args:
            frame: Input frame (audio or control)
            direction: Processing direction
        """
        await super().process_frame(frame, direction)
        
        # Initialize on first frame
        if isinstance(frame, StartFrame):
            await self.initialize()
            await self.push_frame(frame, direction)
            return
        
        # Pass through system frames
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return
        
        # Process audio frames
        if isinstance(frame, AudioRawFrame):
            await self._process_audio_frame(frame, direction)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
    
    async def _process_audio_frame(self, frame: AudioRawFrame, direction: str):
        """
        Process audio frame and generate corresponding video frames.
        
        Args:
            frame: Audio frame to process
            direction: Processing direction
        """
        if not self._initialized:
            logger.warning("MuseTalk not initialized, skipping frame")
            return
        
        # For mock mode (CPU development), just pass through
        if self.musetalk is None:
            logger.debug("Mock mode: generating placeholder video frame")
            await self._generate_mock_video_frame(frame, direction)
            return
        
        # Buffer audio until we have enough for processing
        self._audio_buffer.append(frame.audio)
        self._buffer_duration += len(frame.audio) / frame.sample_rate
        
        # Process when buffer reaches target duration
        if self._buffer_duration >= self._target_buffer_duration:
            await self._process_buffered_audio(frame.sample_rate, direction)
    
    async def _process_buffered_audio(self, sample_rate: int, direction: str):
        """Process buffered audio and generate video frames."""
        if not self._audio_buffer:
            return
        
        try:
            # Concatenate audio buffer
            audio_data = np.concatenate(self._audio_buffer)
            
            # Save to temporary file (MuseTalk requires file input)
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            ) as tmp_audio:
                sf.write(tmp_audio.name, audio_data, sample_rate)
                audio_path = tmp_audio.name
            
            # Generate video frames
            loop = asyncio.get_event_loop()
            video_path = await loop.run_in_executor(
                None,
                lambda: self.musetalk.inference_noprepare(
                    audio_path,
                    self.avatar_image_path,
                    self.bbox_shift,
                    self.batch_size,
                    self.fps
                )
            )
            
            # Read generated video and push frames
            await self._push_video_frames(video_path, direction)
            
            # Cleanup
            import os
            os.unlink(audio_path)
            
            # Clear buffer
            self._audio_buffer = []
            self._buffer_duration = 0.0
            
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
            self._audio_buffer = []
            self._buffer_duration = 0.0
    
    async def _push_video_frames(self, video_path: str, direction: str):
        """Read video file and push individual frames."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create video frame
                video_frame = VideoFrame(
                    image=frame_rgb.tobytes(),
                    size=(frame.shape[1], frame.shape[0]),
                    format="RGB"
                )
                
                await self.push_frame(video_frame, direction)
                
        finally:
            cap.release()
    
    async def _generate_mock_video_frame(self, audio_frame: AudioRawFrame, direction: str):
        """Generate placeholder video frame for CPU development mode."""
        # Create a simple colored frame (green screen for development)
        height, width = 512, 512
        frame_data = np.zeros((height, width, 3), dtype=np.uint8)
        frame_data[:, :] = [0, 255, 0]  # Green
        
        # Add text indicator
        import cv2
        cv2.putText(
            frame_data,
            "CPU Mode - GPU Required",
            (50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        video_frame = VideoFrame(
            image=frame_data.tobytes(),
            size=(width, height),
            format="RGB"
        )
        
        await self.push_frame(video_frame, direction)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.musetalk:
            # MuseTalk cleanup if needed
            pass
        
        self._audio_buffer = []
        self._buffer_duration = 0.0
        logger.info("MuseTalkProcessor cleaned up")
