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