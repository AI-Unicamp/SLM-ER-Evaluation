"""
Synthesis Engine Module
Core synthesis engine using CosyVoice2 for emotional speech generation
"""

# OFFLINE MODE FIX: Mock wetext before any imports
import sys
from unittest.mock import MagicMock

class MockNormalizer:
    def __init__(self, *args, **kwargs):
        pass
    def normalize(self, text):
        return text  # Just return text as-is

wetext_mock = MagicMock()
wetext_mock.Normalizer = MockNormalizer
sys.modules['wetext'] = wetext_mock

import sys
import os
import torch
import torchaudio
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Add CosyVoice path
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logger = logging.getLogger(__name__)

@dataclass
class SynthesisJob:
    """Represents a synthesis job"""
    text: str
    emotion: str
    emotion_type: str
    reference_audio: Optional[str] = None
    instruction: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class SynthesisResult:
    """Represents synthesis result"""
    success: bool
    audio_path: Optional[str] = None
    audio_tensor: Optional[torch.Tensor] = None
    duration: Optional[float] = None
    method_used: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

class SynthesisEngine:
    """Core synthesis engine for emotional speech generation"""
    
    def __init__(self, 
             model_path: str = "pretrained_models/CosyVoice2-0.5B",
             output_dir: str = "synthesis_output",
             device: str = "auto"):
        """Initialize synthesis engine - MODIFIED for offline usage"""
        
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        '''# Initialize model
        logger.info(f"Loading CosyVoice2 model from {model_path}")
        self.model = CosyVoice2(
            model_path, 
            load_jit=False, 
            load_trt=False, 
            fp16=False)'''
        
        # Initialize model with offline settings
        logger.info(f"Loading CosyVoice2 model from {model_path} (offline mode)")
        
        # Set offline environment
        import os
        os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            self.model = CosyVoice2(
                model_path, 
                load_jit=False, 
                load_trt=False, 
                fp16=False
            )
        except Exception as e:
            logger.error(f"Failed to load model in offline mode: {e}")
            logger.info("Trying without offline restrictions...")
            # Clear offline env vars and retry
            for var in ['HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE']:
                os.environ.pop(var, None)
            
            self.model = CosyVoice2(
                model_path, 
                load_jit=False, 
                load_trt=False, 
                fp16=False
            )
        
        # Emotion instruction templates optimized for English
        self.emotion_instructions = {
            'happy': {
                'explicit': 'Speak with clear joy and excitement, use a bright and enthusiastic tone',
                'implicit': 'Speak with warmth and contentment, use a pleased and gentle tone'
            },
            'sad': {
                'explicit': 'Speak with clear sadness and sorrow, use a slow and melancholic tone',
                'implicit': 'Speak with subtle sadness and disappointment, use a quiet and subdued tone'
            },
            'angry': {
                'explicit': 'Speak with clear anger and frustration, use a sharp and forceful tone',
                'implicit': 'Speak with irritation and tension, use a controlled but strained tone'
            },
            'neutral': {
                'neutral': 'Speak in a calm and natural tone without emotional emphasis'
            },
            'surprised': {
                'explicit': 'Speak with clear surprise and amazement, use varied pitch and excitement',
                'implicit': 'Speak with subtle surprise and interest, use a curious tone'
            }
        }
        
        # Available synthesis methods
        self.synthesis_methods = ['audio_plus_tag', 'audio_only', 'tag_only']
        
        # Statistics tracking
        self.stats = {
            'total_jobs': 0,
            'successful': 0,
            'failed': 0,
            'by_emotion': {},
            'by_method': {},
            'by_emotion_type': {}
        }
    
    def get_instruction(self, emotion: str, emotion_type: str) -> str:
        """Get synthesis instruction for emotion and type"""
        if emotion in self.emotion_instructions:
            emotion_dict = self.emotion_instructions[emotion]
            if emotion_type in emotion_dict:
                return emotion_dict[emotion_type]
            else:
                # Fallback to first available instruction
                return list(emotion_dict.values())[0]
        
        # Generic fallback
        return f"Speak with {emotion} emotion using a {emotion_type} delivery"
    
    def _select_synthesis_method(self, job: SynthesisJob) -> str:
        """Select optimal synthesis method based on available resources"""
        if job.reference_audio and os.path.exists(job.reference_audio):
            return 'audio_plus_tag'  # Best quality
        else:
            return 'tag_only'  # Fallback
    
    def _synthesize_audio_plus_tag(self, job: SynthesisJob) -> SynthesisResult:
        """Synthesize using audio reference + emotion instruction"""
        try:
            # Load reference audio
            ref_speech = load_wav(job.reference_audio, 16000)
            
            # Limit reference duration
            max_samples = 30 * 16000
            if ref_speech.shape[1] > max_samples:
                ref_speech = ref_speech[:, :max_samples]
            
            # Get instruction
            instruction = job.instruction or self.get_instruction(job.emotion, job.emotion_type)
            
            speech_list = []
            
            # Try inference_instruct2 first (best method)
            if hasattr(self.model, 'inference_instruct2'):
                for result in self.model.inference_instruct2(
                    job.text, instruction, ref_speech, stream=False
                ):
                    speech_list.append(result['tts_speech'])
            else:
                # Fallback to zero-shot with emotional prompt
                prompt_text = f"This is a {job.emotion} voice speaking with {job.emotion_type} emotion"
                for result in self.model.inference_zero_shot(
                    job.text, prompt_text, ref_speech, stream=False
                ):
                    speech_list.append(result['tts_speech'])
            
            if speech_list:
                combined_speech = torch.concat(speech_list, dim=1)
                return SynthesisResult(
                    success=True,
                    audio_tensor=combined_speech,
                    method_used='audio_plus_tag',
                    metadata={'instruction': instruction, 'reference_audio': job.reference_audio}
                )
            else:
                return SynthesisResult(
                    success=False,
                    error_message="No speech generated",
                    method_used='audio_plus_tag'
                )
                
        except Exception as e:
            return SynthesisResult(
                success=False,
                error_message=str(e),
                method_used='audio_plus_tag'
            )
    
    def _synthesize_audio_only(self, job: SynthesisJob) -> SynthesisResult:
        """Synthesize using only audio reference"""
        try:
            ref_speech = load_wav(job.reference_audio, 16000)
            
            # Limit reference duration
            max_samples = 30 * 16000
            if ref_speech.shape[1] > max_samples:
                ref_speech = ref_speech[:, :max_samples]
            
            prompt_text = f"Reference {job.emotion} speech"
            speech_list = []
            
            for result in self.model.inference_zero_shot(
                job.text, prompt_text, ref_speech, stream=False
            ):
                speech_list.append(result['tts_speech'])
            
            if speech_list:
                combined_speech = torch.concat(speech_list, dim=1)
                return SynthesisResult(
                    success=True,
                    audio_tensor=combined_speech,
                    method_used='audio_only',
                    metadata={'prompt_text': prompt_text, 'reference_audio': job.reference_audio}
                )
            else:
                return SynthesisResult(
                    success=False,
                    error_message="No speech generated",
                    method_used='audio_only'
                )
                
        except Exception as e:
            return SynthesisResult(
                success=False,
                error_message=str(e),
                method_used='audio_only'
            )
    
    def _synthesize_tag_only(self, job: SynthesisJob) -> SynthesisResult:
        """Synthesize using only emotion instruction"""
        try:
            # Get available speakers
            available_speakers = self.model.list_available_spks()
            
            # Look for English speakers
            english_speakers = [spk for spk in available_speakers 
                              if any(keyword in spk.lower() 
                                   for keyword in ['english', 'en', 'male', 'female', '英'])]
            
            if not english_speakers:
                return SynthesisResult(
                    success=False,
                    error_message="No suitable English speakers found",
                    method_used='tag_only'
                )
            
            speaker = english_speakers[0]
            instruction = job.instruction or self.get_instruction(job.emotion, job.emotion_type)
            speech_list = []
            
            if hasattr(self.model, 'inference_instruct'):
                for result in self.model.inference_instruct(
                    job.text, speaker, instruction, stream=False
                ):
                    speech_list.append(result['tts_speech'])
            else:
                return SynthesisResult(
                    success=False,
                    error_message="Instruct mode not available",
                    method_used='tag_only'
                )
            
            if speech_list:
                combined_speech = torch.concat(speech_list, dim=1)
                return SynthesisResult(
                    success=True,
                    audio_tensor=combined_speech,
                    method_used='tag_only',
                    metadata={'instruction': instruction, 'speaker': speaker}
                )
            else:
                return SynthesisResult(
                    success=False,
                    error_message="No speech generated",
                    method_used='tag_only'
                )
                
        except Exception as e:
            return SynthesisResult(
                success=False,
                error_message=str(e),
                method_used='tag_only'
            )
    
    def synthesize_single(self, job: SynthesisJob, method: str = "auto") -> SynthesisResult:
        """
        Synthesize a single text sample
        
        Args:
            job: Synthesis job configuration
            method: Synthesis method (auto, audio_plus_tag, audio_only, tag_only)
            
        Returns:
            Synthesis result
        """
        self.stats['total_jobs'] += 1
        
        # Auto-select method if needed
        if method == "auto":
            method = self._select_synthesis_method(job)
        
        # Execute synthesis based on method
        if method == "audio_plus_tag":
            result = self._synthesize_audio_plus_tag(job)
        elif method == "audio_only":
            result = self._synthesize_audio_only(job)
        elif method == "tag_only":
            result = self._synthesize_tag_only(job)
        else:
            result = SynthesisResult(
                success=False,
                error_message=f"Unknown synthesis method: {method}"
            )
        
        # Update statistics
        if result.success:
            self.stats['successful'] += 1
            self.stats['by_method'][result.method_used] = self.stats['by_method'].get(result.method_used, 0) + 1
            self.stats['by_emotion'][job.emotion] = self.stats['by_emotion'].get(job.emotion, 0) + 1
            self.stats['by_emotion_type'][job.emotion_type] = self.stats['by_emotion_type'].get(job.emotion_type, 0) + 1
            
            # Calculate duration
            if result.audio_tensor is not None:
                result.duration = result.audio_tensor.shape[1] / self.model.sample_rate
        else:
            self.stats['failed'] += 1
            logger.warning(f"Synthesis failed: {result.error_message}")
        
        return result
    
    def save_result(self, result: SynthesisResult, filename: str = None) -> str:
        """
        Save synthesis result to file
        
        Args:
            result: Synthesis result to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if not result.success or result.audio_tensor is None:
            raise ValueError("Cannot save failed synthesis result")
        
        if filename is None:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthesis_{timestamp}.wav"
        
        # Ensure .wav extension
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        audio_path = self.output_dir / "audio" / filename
        torchaudio.save(str(audio_path), result.audio_tensor, self.model.sample_rate)
        
        result.audio_path = str(audio_path)
        logger.info(f"Saved audio: {audio_path}")
        
        return str(audio_path)
    
    def batch_synthesize(self, 
                        jobs: List[SynthesisJob], 
                        method: str = "auto",
                        save_files: bool = True) -> List[SynthesisResult]:
        """
        Synthesize multiple jobs in batch
        
        Args:
            jobs: List of synthesis jobs
            method: Synthesis method to use
            save_files: Whether to save audio files
            
        Returns:
            List of synthesis results
        """
        logger.info(f"Starting batch synthesis of {len(jobs)} jobs")
        
        results = []
        for i, job in enumerate(jobs):
            logger.info(f"Processing job {i+1}/{len(jobs)}: {job.emotion}_{job.emotion_type}")
            
            result = self.synthesize_single(job, method)
            
            if result.success and save_files:
                # Generate filename based on job
                text_hash = hashlib.md5(job.text.encode()).hexdigest()[:8]
                filename = f"{job.emotion}_{job.emotion_type}_{text_hash}.wav"
                self.save_result(result, filename)
            
            results.append(result)
        
        # Print batch summary
        successful = len([r for r in results if r.success])
        success_rate = (successful / len(results)) * 100 if results else 0
        
        logger.info(f"Batch synthesis complete: {successful}/{len(results)} ({success_rate:.1f}%)")
        
        return results
    
    def create_dataset_manifest(self, 
                               results: List[SynthesisResult], 
                               jobs: List[SynthesisJob]) -> Dict:
        """Create comprehensive dataset manifest"""
        
        successful_results = [(r, j) for r, j in zip(results, jobs) if r.success]
        
        manifest = {
            'dataset_info': {
                'name': 'Emotional Speech Dataset',
                'created_at': datetime.now().isoformat(),
                'total_samples': len(successful_results),
                'failed_samples': len(results) - len(successful_results),
                'sample_rate': self.model.sample_rate,
                'model_used': self.model_path
            },
            'samples': [],
            'statistics': self.stats.copy(),
            'failed_jobs': []
        }
        
        # Add successful samples
        for i, (result, job) in enumerate(successful_results):
            # Safely merge metadata
            combined_metadata = {}
            if job.metadata:
                combined_metadata.update(job.metadata)
            if result.metadata:
                combined_metadata.update(result.metadata)
            
            sample_info = {
                'id': f"sample_{i:06d}",
                'audio_path': result.audio_path,
                'text': job.text,
                'emotion': job.emotion,
                'emotion_type': job.emotion_type,
                'duration_seconds': result.duration,
                'synthesis_method': result.method_used,
                'metadata': combined_metadata
            }
            manifest['samples'].append(sample_info)
        
        # Add failed jobs
        for result, job in zip(results, jobs):
            if not result.success:
                manifest['failed_jobs'].append({
                    'text': job.text,
                    'emotion': job.emotion,
                    'emotion_type': job.emotion_type,
                    'error': result.error_message,
                    'method_attempted': result.method_used
                })
        
        return manifest
    
    def save_manifest(self, manifest: Dict, filename: str = "dataset_manifest.json"):
        """Save dataset manifest to file"""
        manifest_path = self.output_dir / filename
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved manifest: {manifest_path}")
        return str(manifest_path)
    
    def get_statistics(self) -> Dict:
        """Get synthesis statistics"""
        total = self.stats['total_jobs']
        success_rate = (self.stats['successful'] / total * 100) if total > 0 else 0
        
        return {
            **self.stats,
            'success_rate': success_rate
        }
    
    def print_statistics(self):
        """Print synthesis statistics"""
        stats = self.get_statistics()
        
        print("📊 Synthesis Statistics")
        print("=" * 30)
        print(f"Total jobs: {stats['total_jobs']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
        if stats['by_emotion']:
            print(f"\n🎭 By emotion:")
            for emotion, count in stats['by_emotion'].items():
                print(f"   {emotion:10}: {count:3d}")
        
        if stats['by_method']:
            print(f"\n⚙️ By method:")
            for method, count in stats['by_method'].items():
                print(f"   {method:15}: {count:3d}")
        
        if stats['by_emotion_type']:
            print(f"\n📝 By emotion type:")
            for etype, count in stats['by_emotion_type'].items():
                print(f"   {etype:10}: {count:3d}")

    def synthesize_single_with_tensor_reference(self, job: SynthesisJob, method: str = "auto") -> SynthesisResult:
        """
        Synthesize with reference audio as tensor (for concatenated references)
        MODIFIED version that accepts tensor input instead of file path
        """
        self.stats['total_jobs'] += 1
        
        # Auto-select method if needed
        if method == "auto":
            method = 'audio_plus_tag'  # Force audio+tag since we have reference
        
        # Execute synthesis with tensor reference
        if method == "audio_plus_tag":
            result = self._synthesize_audio_plus_tag_with_tensor(job)
        elif method == "audio_only":
            result = self._synthesize_audio_only_with_tensor(job)
        else:
            result = SynthesisResult(
                success=False,
                error_message=f"Method {method} not supported with tensor reference"
            )
        
        # Update statistics
        if result.success:
            self.stats['successful'] += 1
            self.stats['by_method'][result.method_used] = self.stats['by_method'].get(result.method_used, 0) + 1
            self.stats['by_emotion'][job.emotion] = self.stats['by_emotion'].get(job.emotion, 0) + 1
            self.stats['by_emotion_type'][job.emotion_type] = self.stats['by_emotion_type'].get(job.emotion_type, 0) + 1
            
            # Calculate duration
            if result.audio_tensor is not None:
                result.duration = result.audio_tensor.shape[1] / self.model.sample_rate
        else:
            self.stats['failed'] += 1
            logger.warning(f"Synthesis failed: {result.error_message}")
        
        return result
    
    def _synthesize_audio_plus_tag_with_tensor(self, job: SynthesisJob) -> SynthesisResult:
        """
        Synthesize using tensor reference audio + emotion instruction
        MODIFIED version for tensor input
        """
        try:
            # Reference audio is already a tensor (concatenated from multiple files)
            ref_speech = job.reference_audio
            
            if not isinstance(ref_speech, torch.Tensor):
                return SynthesisResult(
                    success=False,
                    error_message="Expected tensor reference audio",
                    method_used='audio_plus_tag'
                )
            
            # Ensure tensor is properly shaped (1, samples)
            if ref_speech.dim() == 1:
                ref_speech = ref_speech.unsqueeze(0)
            
            # Limit reference duration (already done in concatenation, but double-check)
            max_samples = 30 * 16000
            if ref_speech.shape[1] > max_samples:
                ref_speech = ref_speech[:, :max_samples]
            
            # Get instruction
            instruction = job.instruction or self.get_instruction(job.emotion, job.emotion_type)
            
            speech_list = []
            
            # Try inference_instruct2 first (best method)
            if hasattr(self.model, 'inference_instruct2'):
                logger.info(f"    Using inference_instruct2 with concatenated reference ({ref_speech.shape[1]/16000:.2f}s)")
                for result in self.model.inference_instruct2(
                    job.text, instruction, ref_speech, stream=False
                ):
                    speech_list.append(result['tts_speech'])
            else:
                # Fallback to zero-shot with emotional prompt
                prompt_text = f"This is a {job.emotion} voice speaking with {job.emotion_type} emotion"
                logger.info(f"    Using inference_zero_shot fallback")
                for result in self.model.inference_zero_shot(
                    job.text, prompt_text, ref_speech, stream=False
                ):
                    speech_list.append(result['tts_speech'])
            
            if speech_list:
                combined_speech = torch.concat(speech_list, dim=1)
                
                # Get reference metadata from job
                ref_metadata = job.metadata.get('reference_metadata', {})
                
                return SynthesisResult(
                    success=True,
                    audio_tensor=combined_speech,
                    method_used='audio_plus_tag_concatenated',
                    metadata={
                        'instruction': instruction,
                        'reference_type': 'concatenated_tensor',
                        'reference_duration': ref_speech.shape[1] / 16000,
                        **ref_metadata
                    }
                )
            else:
                return SynthesisResult(
                    success=False,
                    error_message="No speech generated",
                    method_used='audio_plus_tag_concatenated'
                )
                
        except Exception as e:
            return SynthesisResult(
                success=False,
                error_message=str(e),
                method_used='audio_plus_tag_concatenated'
            )

    def _synthesize_audio_only_with_tensor(self, job: SynthesisJob) -> SynthesisResult:
        """
        Synthesize using only tensor reference audio
        MODIFIED version for tensor input
        """
        try:
            ref_speech = job.reference_audio
            
            if not isinstance(ref_speech, torch.Tensor):
                return SynthesisResult(
                    success=False,
                    error_message="Expected tensor reference audio",
                    method_used='audio_only'
                )
            
            # Ensure tensor is properly shaped
            if ref_speech.dim() == 1:
                ref_speech = ref_speech.unsqueeze(0)
            
            # Limit reference duration
            max_samples = 30 * 16000
            if ref_speech.shape[1] > max_samples:
                ref_speech = ref_speech[:, :max_samples]
            
            prompt_text = f"Reference {job.emotion} speech"
            speech_list = []
            
            for result in self.model.inference_zero_shot(
                job.text, prompt_text, ref_speech, stream=False
            ):
                speech_list.append(result['tts_speech'])
            
            if speech_list:
                combined_speech = torch.concat(speech_list, dim=1)
                return SynthesisResult(
                    success=True,
                    audio_tensor=combined_speech,
                    method_used='audio_only_concatenated',
                    metadata={
                        'prompt_text': prompt_text,
                        'reference_type': 'concatenated_tensor',
                        'reference_duration': ref_speech.shape[1] / 16000
                    }
                )
            else:
                return SynthesisResult(
                    success=False,
                    error_message="No speech generated",
                    method_used='audio_only_concatenated'
                )
                
        except Exception as e:
            return SynthesisResult(
                success=False,
                error_message=str(e),
                method_used='audio_only_concatenated'
            )