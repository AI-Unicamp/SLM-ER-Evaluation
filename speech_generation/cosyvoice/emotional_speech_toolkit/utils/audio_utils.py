"""
Audio Utilities Module
Audio processing and analysis utilities
"""

import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

class AudioUtils:
    """Audio processing utilities"""
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample if needed
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            sr = target_sr
        
        return audio, sr
    
    @staticmethod
    def analyze_audio_quality(file_path: str) -> dict:
        """
        Analyze audio quality metrics
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            audio, sr = librosa.load(file_path, sr=None)
            
            # Basic metrics
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))
            
            # Spectral metrics
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Signal-to-noise ratio estimation
            # Simple approach: compare energy in speech vs quiet regions
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            frames = librosa.util.frame(audio, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            frame_energies = np.mean(frames**2, axis=0)
            
            # Estimate SNR (very rough)
            energy_threshold = np.percentile(frame_energies, 30)
            speech_energy = np.mean(frame_energies[frame_energies > energy_threshold])
            noise_energy = np.mean(frame_energies[frame_energies <= energy_threshold])
            snr_estimate = 10 * np.log10(speech_energy / (noise_energy + 1e-10))
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'rms_energy': float(rms_energy),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'snr_estimate': float(snr_estimate),
                'quality_score': min(100, max(0, (snr_estimate + 10) * 5))  # Rough quality score
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def trim_silence(audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """
        Trim silence from beginning and end of audio
        
        Args:
            audio: Audio tensor
            threshold: Energy threshold for silence detection
            
        Returns:
            Trimmed audio tensor
        """
        # Convert to numpy for processing
        audio_np = audio.squeeze().numpy()
        
        # Find non-silent regions
        energy = audio_np ** 2
        non_silent = energy > threshold
        
        if np.any(non_silent):
            start_idx = np.argmax(non_silent)
            end_idx = len(audio_np) - np.argmax(non_silent[::-1])
            audio_np = audio_np[start_idx:end_idx]
        
        return torch.tensor(audio_np).unsqueeze(0)
    
    @staticmethod
    def normalize_audio(audio: torch.Tensor, target_level: float = 0.8) -> torch.Tensor:
        """
        Normalize audio to target level
        
        Args:
            audio: Audio tensor
            target_level: Target peak level (0-1)
            
        Returns:
            Normalized audio tensor
        """
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio * (target_level / max_val)
        return audio

# File: emotional_speech_toolkit/utils/file_utils.py
"""
File Utilities Module
File system operations and data management utilities
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class FileUtils:
    """File system utilities"""
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if not
        
        Args:
            directory: Directory path
            
        Returns:
            Path object
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_filename(text: str, max_length: int = 50) -> str:
        """
        Create safe filename from text
        
        Args:
            text: Input text
            max_length: Maximum filename length
            
        Returns:
            Safe filename string
        """
        # Remove or replace unsafe characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        filename = ""
        
        for char in text:
            if char in safe_chars:
                filename += char
            elif char in " \t":
                filename += "_"
            # Skip other characters
        
        # Limit length
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        # Ensure not empty
        if not filename:
            filename = "untitled"
        
        return filename
    
    @staticmethod
    def save_json(data: Dict, file_path: Union[str, Path], indent: int = 2):
        """
        Save dictionary as JSON file
        
        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Saved JSON: {file_path}")
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict:
        """
        Load JSON file
        
        Args:
            file_path: JSON file path
            
        Returns:
            Loaded data dictionary
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON: {file_path}")
        return data
    
    @staticmethod
    def copy_file(source: Union[str, Path], 
                  destination: Union[str, Path], 
                  create_dirs: bool = True) -> Path:
        """
        Copy file to destination
        
        Args:
            source: Source file path
            destination: Destination file path
            create_dirs: Create destination directories if needed
            
        Returns:
            Destination path
        """
        source_path = Path(source)
        dest_path = Path(destination)
        
        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied: {source_path} → {dest_path}")
        
        return dest_path
    
    @staticmethod
    def find_files(directory: Union[str, Path], 
                   pattern: str = "*", 
                   recursive: bool = True) -> List[Path]:
        """
        Find files matching pattern
        
        Args:
            directory: Directory to search
            pattern: File pattern (e.g., "*.wav")
            recursive: Search recursively
            
        Returns:
            List of matching file paths
        """
        directory = Path(directory)
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        return files
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """Get file size in bytes"""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def get_directory_size(directory: Union[str, Path]) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    @staticmethod
    def cleanup_empty_dirs(directory: Union[str, Path]):
        """Remove empty directories recursively"""
        directory = Path(directory)
        
        for subdir in directory.rglob('*'):
            if subdir.is_dir():
                try:
                    subdir.rmdir()  # Only removes if empty
                    logger.info(f"Removed empty directory: {subdir}")
                except OSError:
                    pass  # Directory not empty
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

# File: emotional_speech_toolkit/config/emotion_config.py
"""
Emotion Configuration Module
Centralized emotion and synthesis configuration
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class EmotionInstruction:
    """Emotion instruction configuration"""
    emotion: str
    emotion_type: str
    instruction: str
    intensity: str = "medium"
    description: str = ""

class EmotionConfig:
    """Centralized emotion configuration"""
    
    def __init__(self):
        # Emotion mappings for CSV columns
        self.csv_emotion_mapping = {
            'happy_explicit': {'emotion': 'happy', 'type': 'explicit'},
            'happy_implicit': {'emotion': 'happy', 'type': 'implicit'},
            'sad_explicit': {'emotion': 'sad', 'type': 'explicit'},
            'sad_implicit': {'emotion': 'sad', 'type': 'implicit'},
            'angry_explicit': {'emotion': 'angry', 'type': 'explicit'},
            'angry_implicit': {'emotion': 'angry', 'type': 'implicit'},
            'neutral': {'emotion': 'neutral', 'type': 'neutral'}
        }
        
        # ESD emotion mappings
        self.esd_emotion_mapping = {
            'Happy': 'happy',
            'Sad': 'sad',
            'Angry': 'angry',
            'Neutral': 'neutral',
            'Surprise': 'surprised'
        }
        
        # Detailed emotion instructions
        self.emotion_instructions = {
            'happy_explicit': EmotionInstruction(
                emotion='happy',
                emotion_type='explicit',
                instruction='Speak with clear joy and excitement, use a bright and enthusiastic tone',
                intensity='high',
                description='Direct expressions of happiness with strong positive emotion'
            ),
            'happy_implicit': EmotionInstruction(
                emotion='happy',
                emotion_type='implicit',
                instruction='Speak with warmth and contentment, use a pleased and gentle tone',
                intensity='medium',
                description='Subtle happiness conveyed through content and context'
            ),
            'sad_explicit': EmotionInstruction(
                emotion='sad',
                emotion_type='explicit',
                instruction='Speak with clear sadness and sorrow, use a slow and melancholic tone',
                intensity='high',
                description='Direct expressions of sadness with clear emotional indicators'
            ),
            'sad_implicit': EmotionInstruction(
                emotion='sad',
                emotion_type='implicit',
                instruction='Speak with subtle sadness and disappointment, use a quiet and subdued tone',
                intensity='medium',
                description='Implied sadness through situational context and tone'
            ),
            'angry_explicit': EmotionInstruction(
                emotion='angry',
                emotion_type='explicit',
                instruction='Speak with clear anger and frustration, use a sharp and forceful tone',
                intensity='high',
                description='Direct expressions of anger with strong negative emotion'
            ),
            'angry_implicit': EmotionInstruction(
                emotion='angry',
                emotion_type='implicit',
                instruction='Speak with irritation and tension, use a controlled but strained tone',
                intensity='medium',
                description='Contained anger or frustration implied through context'
            ),
            'neutral': EmotionInstruction(
                emotion='neutral',
                emotion_type='neutral',
                instruction='Speak in a calm and natural tone without emotional emphasis',
                intensity='neutral',
                description='Emotionally neutral statements with balanced delivery'
            )
        }
        
        # Synthesis method preferences by emotion type
        self.method_preferences = {
            'explicit': ['audio_plus_tag', 'tag_only', 'audio_only'],
            'implicit': ['audio_plus_tag', 'audio_only', 'tag_only'],
            'neutral': ['tag_only', 'audio_plus_tag', 'audio_only']
        }
    
    def get_instruction(self, csv_column: str) -> Optional[EmotionInstruction]:
        """Get emotion instruction for CSV column"""
        return self.emotion_instructions.get(csv_column)
    
    def get_emotion_info(self, csv_column: str) -> Optional[Dict]:
        """Get emotion information for CSV column"""
        return self.csv_emotion_mapping.get(csv_column)
    
    def get_esd_emotion(self, esd_emotion: str) -> Optional[str]:
        """Map ESD emotion to standard emotion"""
        return self.esd_emotion_mapping.get(esd_emotion)
    
    def get_preferred_methods(self, emotion_type: str) -> List[str]:
        """Get preferred synthesis methods for emotion type"""
        return self.method_preferences.get(emotion_type, ['audio_plus_tag', 'tag_only', 'audio_only'])
    
    def list_emotions(self) -> List[str]:
        """List all unique emotions"""
        return list(set(info['emotion'] for info in self.csv_emotion_mapping.values()))
    
    def list_emotion_types(self) -> List[str]:
        """List all emotion types"""
        return list(set(info['type'] for info in self.csv_emotion_mapping.values()))