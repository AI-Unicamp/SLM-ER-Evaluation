"""
Core modules for emotional speech synthesis
"""

from .esd_manager import ESDManager
from .csv_processor import CSVProcessor
from .synthesis_engine import SynthesisEngine

__all__ = ['ESDManager', 'CSVProcessor', 'SynthesisEngine']