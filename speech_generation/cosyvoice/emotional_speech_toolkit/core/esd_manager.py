"""
ESD Dataset Manager - MODIFIED for CSV-based reference selection
Handles ESD dataset operations, reference extraction, and speaker management
"""

import os
import sys
import shutil
import random
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torchaudio
import torch

# Add CosyVoice path
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.utils.file_utils import load_wav

logger = logging.getLogger(__name__)

class ESDManager:
    """Manages ESD dataset operations and reference extraction"""
    
    def __init__(self, esd_path: str = "ESD_English", output_ref_dir: str = "references", 
                 reference_csv: str = None):
        """
        Initialize ESD Manager
        
        Args:
            esd_path: Path to ESD_English directory containing speakers 0011-0020
            output_ref_dir: Output directory for extracted references
            reference_csv: Path to CSV file containing reference file selections (NEW)
        """
        self.esd_path = Path(esd_path)
        self.output_ref_dir = Path(output_ref_dir)
        self.reference_csv = reference_csv
        self.english_speakers = [f"00{i}" for i in range(11, 21)]  # 0011-0020
        
        # ESD emotion mapping
        self.esd_emotions = {
            'Happy': 'happy',
            'Sad': 'sad',
            'Angry': 'angry',
            'Neutral': 'neutral',
            'Surprise': 'surprised'
        }
        
        # Load reference CSV if provided
        self.reference_df = None
        if reference_csv:
            self._load_reference_csv()
        
        self._validate_setup()
    
    def _load_reference_csv(self):
        """Load the reference CSV file"""
        try:
            logger.info(f"Loading reference CSV: {self.reference_csv}")
            self.reference_df = pd.read_csv(self.reference_csv)
            logger.info(f"Loaded {len(self.reference_df)} reference entries")
            
            # Display CSV structure
            logger.info(f"CSV columns: {list(self.reference_df.columns)}")
            
            # Check for required columns - use wav_name for actual filenames
            required_cols = ['speaker', 'emotion', 'wav_name']
            missing_cols = [col for col in required_cols if col not in self.reference_df.columns]
            if missing_cols:
                logger.warning(f"Missing expected columns: {missing_cols}")
            
            # Show sample of data
            logger.info("Sample reference entries:")
            for i, row in self.reference_df.head(3).iterrows():
                logger.info(f"  {row.get('speaker', 'N/A')} - {row.get('emotion', 'N/A')} - {row.get('wav_name', 'N/A')}")
                
        except Exception as e:
            logger.error(f"Failed to load reference CSV: {e}")
            self.reference_df = None
    
    def _validate_setup(self):
        """Validate ESD setup and directory structure"""
        if not self.esd_path.exists():
            raise FileNotFoundError(f"ESD_English directory not found: {self.esd_path}")
        
        # Check for English speakers
        found_speakers = []
        for speaker in self.english_speakers:
            speaker_path = self.esd_path / speaker
            if speaker_path.exists():
                found_speakers.append(speaker)
        
        if not found_speakers:
            raise FileNotFoundError("No English speakers (0011-0020) found in ESD_English directory")
        
        logger.info(f"Found {len(found_speakers)} English speakers: {found_speakers}")
        self.available_speakers = found_speakers
    
    def get_csv_files_for_emotion(self, emotion: str, max_files: int = 7) -> List[Path]:
        """
        Get audio files for emotion from CSV selection (MODIFIED for 7 files)
        
        Args:
            emotion: Target emotion (Happy, Sad, Angry, Neutral)
            max_files: Maximum number of files to return (now 7)
            
        Returns:
            List of file paths from CSV selection
        """
        if self.reference_df is None:
            logger.warning("No reference CSV loaded, falling back to directory search")
            return self.get_longest_files_fallback(emotion, max_files)
        
        # Filter CSV by emotion
        emotion_rows = self.reference_df[self.reference_df['emotion'] == emotion]
        
        if emotion_rows.empty:
            logger.warning(f"No files found for emotion {emotion} in CSV, trying fallback")
            return self.get_longest_files_fallback(emotion, max_files)
        
        # Get file paths
        selected_files = []
        for _, row in emotion_rows.head(max_files).iterrows():
            speaker = str(row['speaker']).zfill(4)  # FIXED: Ensure 4 digits (17 → 0017)
            wav_name = row['wav_name']     # Actual filename (e.g., "0017_001364.wav")
            
            # Construct full path using wav_name (the actual filename in ESD)
            file_path = self.esd_path / speaker / emotion / wav_name
            
            if file_path.exists():
                selected_files.append(file_path)
                logger.info(f"Selected from CSV: {file_path}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        logger.info(f"Selected {len(selected_files)} files for {emotion} from CSV")
        return selected_files
    
    def get_longest_files_fallback(self, emotion: str, num_files: int = 7) -> List[Path]:
        """
        Fallback method: Get longest files when CSV is not available (MODIFIED for 7 files)
        """
        logger.info(f"Using fallback method for {emotion}")
        
        all_files_with_duration = []
        
        # Collect files from all available speakers
        for speaker_id in self.available_speakers:
            files_with_duration = self.get_audio_files_with_duration(speaker_id, emotion)
            all_files_with_duration.extend(files_with_duration)
        
        if not all_files_with_duration:
            logger.warning(f"No audio files found for emotion {emotion}")
            return []
        
        # Sort by duration (longest first) and take top N
        sorted_files = sorted(all_files_with_duration, key=lambda x: x[1], reverse=True)
        longest_files = [file_path for file_path, duration in sorted_files[:num_files]]
        
        logger.info(f"Selected {len(longest_files)} longest files for {emotion}")
        return longest_files
    
    def get_audio_files_with_duration(self, speaker_id: str, emotion: str) -> List[Tuple[Path, float]]:
        """Get audio files with their durations for specific speaker and emotion"""
        emotion_path = self.esd_path / speaker_id / emotion
        if not emotion_path.exists():
            return []
        
        audio_files_with_duration = []
        for audio_file in emotion_path.glob("*.wav"):
            try:
                info = torchaudio.info(str(audio_file))
                duration = info.num_frames / info.sample_rate
                audio_files_with_duration.append((audio_file, duration))
            except Exception as e:
                logger.warning(f"Failed to get duration for {audio_file}: {e}")
                continue
        
        return audio_files_with_duration
    
    def concatenate_reference_audio(self, audio_files: List[Path], max_duration: float = 100.0) -> torch.Tensor:
        """
        Concatenate multiple audio files into single reference (MODIFIED for longer duration)
        
        Args:
            audio_files: List of audio file paths to concatenate
            max_duration: Maximum total duration in seconds (increased to 100s for 7 files)
            
        Returns:
            Concatenated audio tensor
        """
        if not audio_files:
            raise ValueError("No audio files provided for concatenation")
        
        concatenated_audio = []
        total_duration = 0.0
        target_sr = 16000
        
        for audio_file in audio_files:
            try:
                # Load audio
                audio, sr = torchaudio.load(str(audio_file))
                
                # Convert to mono if needed
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                
                # Resample if needed
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    audio = resampler(audio)
                
                # Check if adding this would exceed max duration
                audio_duration = audio.shape[1] / target_sr
                if total_duration + audio_duration > max_duration:
                    # Truncate this audio to fit within max_duration
                    remaining_duration = max_duration - total_duration
                    remaining_samples = int(remaining_duration * target_sr)
                    if remaining_samples > 0:
                        audio = audio[:, :remaining_samples]
                        concatenated_audio.append(audio)
                    break
                else:
                    concatenated_audio.append(audio)
                    total_duration += audio_duration
                
            except Exception as e:
                logger.warning(f"Failed to load {audio_file}: {e}")
                continue
        
        if not concatenated_audio:
            raise ValueError("No audio files could be loaded for concatenation")
        
        # Concatenate all audio
        final_audio = torch.cat(concatenated_audio, dim=1)
        final_duration = final_audio.shape[1] / target_sr
        
        logger.info(f"Concatenated {len(concatenated_audio)} files, total duration: {final_duration:.2f}s")
        
        return final_audio
    
    def create_concatenated_reference(self, emotion: str, num_files: int = 7, 
                                    speaker_id: str = None) -> Tuple[torch.Tensor, str, List[str]]:
        """
        Create concatenated reference audio for synthesis (MODIFIED for 7 files and CSV selection)
        
        Args:
            emotion: Target emotion (Happy, Sad, Angry, Neutral)
            num_files: Number of files to concatenate (now 7)
            speaker_id: Specific speaker ID (ignored when using CSV)
            
        Returns:
            Tuple of (concatenated_audio, selected_speaker_info, file_list)
        """
        logger.info(f"Creating concatenated reference: {emotion}, Target files: {num_files}")
        
        # Use CSV-based selection if available
        if self.reference_csv and self.reference_df is not None:
            selected_files = self.get_csv_files_for_emotion(emotion, num_files)
            speaker_info = "csv_mixed_speakers"  # Multiple speakers from CSV
        else:
            # Fallback to longest files method
            selected_files = self.get_longest_files_fallback(emotion, num_files)
            speaker_info = "fallback_mixed_speakers"
        
        if not selected_files:
            raise ValueError(f"No audio files found for emotion {emotion}")
        
        # Concatenate the files
        concatenated_audio = self.concatenate_reference_audio(selected_files, max_duration=100.0)
        
        # Return file names for metadata
        file_names = [f.name for f in selected_files]
        
        logger.info(f"Successfully created concatenated reference with {len(file_names)} files")
        
        return concatenated_audio, speaker_info, file_names
    
    def get_concatenated_reference_for_synthesis(self, emotion: str) -> Tuple[torch.Tensor, Dict]:
        """
        Get concatenated reference audio ready for synthesis with metadata (MODIFIED)
        
        Args:
            emotion: Target emotion (happy, sad, angry, neutral)
            
        Returns:
            Tuple of (audio_tensor, metadata_dict)
        """
        # Map standard emotions to ESD emotions
        esd_emotion_map = {
            'happy': 'Happy',
            'sad': 'Sad', 
            'angry': 'Angry',
            'neutral': 'Neutral'
        }
        
        esd_emotion = esd_emotion_map.get(emotion.lower())
        if not esd_emotion:
            raise ValueError(f"Unsupported emotion: {emotion}")
        
        # Create concatenated reference with 7 files
        concatenated_audio, speaker_used, file_names = self.create_concatenated_reference(
            esd_emotion, num_files=7
        )
        
        # Prepare metadata
        metadata = {
            'reference_type': 'concatenated_multi_file_csv',
            'emotion': emotion,
            'esd_emotion': esd_emotion,
            'speaker_used': speaker_used,
            'num_files_concatenated': len(file_names),
            'concatenated_files': file_names,
            'total_duration': concatenated_audio.shape[1] / 16000,
            'reference_source': 'csv_selection' if self.reference_csv else 'fallback_longest',
            'reference_csv': self.reference_csv
        }
        
        return concatenated_audio, metadata
    
    def analyze_dataset(self) -> Dict:
        """Analyze ESD dataset structure and coverage"""
        analysis = {
            'total_speakers': len(self.available_speakers),
            'speakers': {},
            'emotion_coverage': {},
            'total_files': 0
        }
        
        for speaker_id in self.available_speakers:
            speaker_analysis = {
                'emotions': {},
                'total_files': 0
            }
            
            for esd_emotion in self.esd_emotions.keys():
                audio_files = self.get_audio_files(speaker_id, esd_emotion)
                if audio_files:
                    # Analyze durations
                    durations = []
                    for audio_file in audio_files[:5]:  # Sample first 5
                        try:
                            info = torchaudio.info(str(audio_file))
                            duration = info.num_frames / info.sample_rate
                            durations.append(duration)
                        except:
                            continue
                    
                    speaker_analysis['emotions'][esd_emotion] = {
                        'count': len(audio_files),
                        'avg_duration': sum(durations) / len(durations) if durations else 0
                    }
                    speaker_analysis['total_files'] += len(audio_files)
            
            analysis['speakers'][speaker_id] = speaker_analysis
            analysis['total_files'] += speaker_analysis['total_files']
        
        # Calculate emotion coverage
        for emotion in self.esd_emotions.keys():
            speakers_with_emotion = sum(1 for s in analysis['speakers'].values() 
                                      if emotion in s['emotions'])
            analysis['emotion_coverage'][emotion] = speakers_with_emotion
        
        return analysis
    
    def print_dataset_summary(self):
        """Print a summary of the ESD dataset"""
        analysis = self.analyze_dataset()
        
        print("📊 ESD English Dataset Summary")
        print("=" * 40)
        print(f"Total English speakers: {analysis['total_speakers']}")
        print(f"Total audio files: {analysis['total_files']}")
        
        print(f"\n🎭 Emotion coverage:")
        for emotion, speaker_count in analysis['emotion_coverage'].items():
            print(f"   {emotion:10}: {speaker_count}/{analysis['total_speakers']} speakers")
        
        print(f"\n👥 Speaker details:")
        for speaker_id, data in analysis['speakers'].items():
            emotions = list(data['emotions'].keys())
            print(f"   {speaker_id}: {len(emotions)} emotions, {data['total_files']} files")
        
        # If using CSV, show CSV info
        if self.reference_csv and self.reference_df is not None:
            print(f"\n📋 CSV Reference Information:")
            print(f"   CSV file: {self.reference_csv}")
            print(f"   Total entries: {len(self.reference_df)}")
            
            # Show emotion distribution in CSV
            csv_emotions = self.reference_df['emotion'].value_counts()
            print(f"   Emotion distribution in CSV:")
            for emotion, count in csv_emotions.items():
                print(f"     {emotion:10}: {count:3d} files")
    
    def get_random_speaker(self) -> str:
        """Get a random English speaker ID"""
        return random.choice(self.available_speakers)
    
    def get_reference_audio(self, emotion: str, speaker_preference: str = None) -> Optional[str]:
        """
        Get a reference audio file for synthesis
        
        Args:
            emotion: Target emotion (happy, sad, angry, neutral)
            speaker_preference: Preferred speaker ID (optional)
            
        Returns:
            Path to reference audio file
        """
        emotion_dir = self.output_ref_dir / emotion
        if not emotion_dir.exists():
            return None
        
        audio_files = list(emotion_dir.glob("*.wav"))
        if not audio_files:
            return None
        
        # If speaker preference specified, try to find it
        if speaker_preference:
            preferred_files = [f for f in audio_files if speaker_preference in f.name]
            if preferred_files:
                return str(preferred_files[0])
        
        # Otherwise return random file
        return str(random.choice(audio_files))
    def get_speaker_emotions(self, speaker_id: str) -> List[str]:
        """Get available emotions for a specific speaker"""
        speaker_path = self.esd_path / speaker_id
        if not speaker_path.exists():
            return []
        
        emotions = []
        for emotion_dir in speaker_path.iterdir():
            if emotion_dir.is_dir() and emotion_dir.name in self.esd_emotions:
                emotions.append(emotion_dir.name)
        
        return emotions
    
    def get_audio_files(self, speaker_id: str, emotion: str) -> List[Path]:
        """Get audio files for specific speaker and emotion"""
        emotion_path = self.esd_path / speaker_id / emotion
        if not emotion_path.exists():
            return []
        
        audio_files = list(emotion_path.glob("*.wav"))
        return audio_files
    
    def extract_references(self, samples_per_emotion: int = 5) -> Dict[str, List[str]]:
        """Extract high-quality reference samples for each emotion"""
        logger.info("🎵 Extracting ESD references from English speakers")
        
        # Create output directories
        self.output_ref_dir.mkdir(parents=True, exist_ok=True)
        for emotion in self.esd_emotions.values():
            (self.output_ref_dir / emotion).mkdir(exist_ok=True)
        
        extracted_files = {emotion: [] for emotion in self.esd_emotions.values()}
        
        for esd_emotion, ref_emotion in self.esd_emotions.items():
            logger.info(f"Processing {esd_emotion} → {ref_emotion}")
            
            samples_extracted = 0
            speakers_to_try = self.available_speakers.copy()
            random.shuffle(speakers_to_try)
            
            for speaker_id in speakers_to_try:
                if samples_extracted >= samples_per_emotion:
                    break
                
                if esd_emotion not in self.get_speaker_emotions(speaker_id):
                    continue
                
                best_ref = self.select_best_reference(speaker_id, esd_emotion)
                if best_ref is None:
                    continue
                
                try:
                    output_filename = f"esd_{speaker_id}_{ref_emotion}_{samples_extracted+1}.wav"
                    output_path = self.output_ref_dir / ref_emotion / output_filename
                    
                    shutil.copy2(best_ref, output_path)
                    extracted_files[ref_emotion].append(str(output_path))
                    
                    logger.info(f"   ✅ {output_filename} (from speaker {speaker_id})")
                    samples_extracted += 1
                    
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to copy {best_ref}: {e}")
                    continue
            
            logger.info(f"   📊 Extracted {samples_extracted} samples for {ref_emotion}")
        
        return extracted_files
    
    def select_best_reference(self, speaker_id: str, emotion: str, 
                            max_duration: float = 10.0) -> Optional[Path]:
        """Select best reference audio file based on quality criteria"""
        audio_files = self.get_audio_files(speaker_id, emotion)
        if not audio_files:
            return None
        
        candidates = []
        for audio_file in audio_files:
            try:
                info = torchaudio.info(str(audio_file))
                duration = info.num_frames / info.sample_rate
                
                if 3.0 <= duration <= max_duration:
                    quality_score = 10 - abs(6.5 - duration)
                    candidates.append((audio_file, quality_score, duration))
            except Exception as e:
                logger.warning(f"Failed to analyze {audio_file}: {e}")
                continue
        
        if not candidates:
            return audio_files[0] if audio_files else None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]