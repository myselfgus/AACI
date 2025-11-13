"""
Utilities for AACI.
"""
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def validate_audio_duration(
    audio: np.ndarray,
    sample_rate: int,
    min_duration: float = 0.5,
    max_duration: float = 30.0
) -> bool:
    """
    Validate audio duration.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        
    Returns:
        True if valid, False otherwise
    """
    duration = len(audio) / sample_rate
    return min_duration <= duration <= max_duration


def prepare_audio_for_training(
    audio_path: str,
    max_duration: float = 30.0,
    target_sr: int = 16000
) -> Optional[np.ndarray]:
    """
    Prepare audio file for training.
    
    Args:
        audio_path: Path to audio file
        max_duration: Maximum duration in seconds
        target_sr: Target sample rate
        
    Returns:
        Processed audio array or None if invalid
    """
    try:
        # Load audio
        audio, sr = load_audio(audio_path, target_sr)
        
        # Validate duration
        if not validate_audio_duration(audio, sr, max_duration=max_duration):
            logger.warning(f"Audio {audio_path} duration out of bounds")
            return None
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
        
    except Exception as e:
        logger.error(f"Error preparing audio {audio_path}: {e}")
        return None


def format_time(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
