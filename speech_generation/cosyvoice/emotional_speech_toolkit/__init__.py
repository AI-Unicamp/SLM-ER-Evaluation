# File: emotional_speech_toolkit/__init__.py
"""
Emotional Speech Dataset Generation Toolkit
A comprehensive toolkit for generating emotional speech datasets using CosyVoice2 and ESD references
"""

__version__ = "1.0.0"
__author__ = "Emotional Speech Toolkit"

from .core.esd_manager import ESDManager
from .core.csv_processor import CSVProcessor
from .core.synthesis_engine import SynthesisEngine
from .utils.audio_utils import AudioUtils
from .utils.file_utils import FileUtils
from .config.emotion_config import EmotionConfig

__all__ = [
    'ESDManager',
    'CSVProcessor', 
    'SynthesisEngine',
    'AudioUtils',
    'FileUtils',
    'EmotionConfig'
]