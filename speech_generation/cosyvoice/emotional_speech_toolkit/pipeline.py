"""
Main Pipeline Script - COMPLETE FIXED VERSION
Orchestrates the complete emotional speech dataset generation pipeline
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Add CosyVoice path
sys.path.append('third_party/Matcha-TTS')

from .core.esd_manager import ESDManager
from .core.csv_processor import CSVProcessor, TextSample
from .core.synthesis_engine import SynthesisEngine, SynthesisJob, SynthesisResult
from .utils.file_utils import FileUtils
from .config.emotion_config import EmotionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionalSpeechPipeline:
    """Main pipeline for emotional speech dataset generation"""
    
    def __init__(self, 
                 csv_file: str = "emotion_sentences_dataset.csv",
                 esd_path: str = "ESD_English", 
                 model_path: str = "pretrained_models/CosyVoice2-0.5B",
                 output_dir: str = "emotional_dataset_output",
                 reference_csv: str = None):
        """
        Initialize the pipeline
        
        Args:
            csv_file: Path to emotion sentences CSV
            esd_path: Path to ESD_English directory
            model_path: Path to CosyVoice2 model
            output_dir: Output directory for generated dataset
            reference_csv: Path to CSV file with reference file selections
        """
        self.csv_file = csv_file
        self.esd_path = esd_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.reference_csv = reference_csv
        
        # Initialize components
        self.emotion_config = EmotionConfig()
        self.esd_manager = None
        self.csv_processor = None
        self.synthesis_engine = None
        
        # Pipeline state
        self.reference_files = {}
        self.text_samples = []
        self.synthesis_results = []
        
        # Create output structure
        self._setup_output_directories()
    
    def _setup_output_directories(self):
        """Create output directory structure"""
        directories = [
            self.output_dir,
            self.output_dir / "audio",
            self.output_dir / "metadata", 
            self.output_dir / "references",
            self.output_dir / "logs"
        ]
        
        for directory in directories:
            FileUtils.ensure_directory(directory)
        
        logger.info(f"Output directory structure created: {self.output_dir}")
    
    def setup_esd_references(self, samples_per_emotion: int = 5) -> bool:
        """
        Setup ESD reference system - MODIFIED for per-synthesis randomization
        
        Args:
            samples_per_emotion: Not used in new approach
            
        Returns:
            Success status
        """
        try:
            logger.info("🎵 Setting up ESD reference system (per-synthesis randomization)")
            
            # Initialize ESD manager with CSV support
            reference_dir = self.output_dir / "references"
            self.esd_manager = ESDManager(
                self.esd_path, 
                str(reference_dir),
                reference_csv=self.reference_csv
            )
            
            # Print ESD analysis
            self.esd_manager.print_dataset_summary()
            
            # If using CSV, report CSV info
            if self.reference_csv:
                logger.info(f"📋 Using reference CSV: {self.reference_csv}")
                if self.esd_manager.reference_df is not None:
                    logger.info(f"📊 CSV contains {len(self.esd_manager.reference_df)} reference entries")
                    logger.info("🎲 Will randomize speaker and select 7 files per synthesis")
                else:
                    logger.warning("⚠️  CSV loading failed, will use fallback method")
            else:
                logger.info("🎲 Will randomize speaker and select 7 longest files per synthesis")
            
            # No need to extract references upfront - will be done per synthesis
            logger.info("✅ ESD reference system ready for per-synthesis selection")
            return True
            
        except Exception as e:
            logger.error(f"ESD setup failed: {e}")
            return False
    
    def load_csv_data(self) -> bool:
        """
        Load and process CSV data
        
        Returns:
            Success status
        """
        try:
            logger.info("📊 Loading CSV data")
            
            # Initialize CSV processor
            self.csv_processor = CSVProcessor(self.csv_file)
            
            # Print analysis
            self.csv_processor.print_summary()
            
            # Validate dataset
            is_valid, issues = self.csv_processor.validate_for_synthesis()
            if not is_valid:
                logger.warning("CSV validation issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            
            # Get text samples
            self.text_samples = self.csv_processor.text_samples
            logger.info(f"✅ Loaded {len(self.text_samples)} text samples")
            
            return True
            
        except Exception as e:
            logger.error(f"CSV loading failed: {e}")
            return False
    
    def initialize_synthesis(self) -> bool:
        """
        Initialize synthesis engine
        
        Returns:
            Success status
        """
        try:
            logger.info("🎙️ Initializing synthesis engine")
            
            # Initialize synthesis engine
            self.synthesis_engine = SynthesisEngine(
                model_path=self.model_path,
                output_dir=str(self.output_dir)
            )
            
            logger.info("✅ Synthesis engine ready")
            return True
            
        except Exception as e:
            logger.error(f"Synthesis initialization failed: {e}")
            return False
    
    def create_synthesis_jobs(self) -> List[SynthesisJob]:
        """
        Create synthesis jobs for multi-emotion with concatenated references
        Each unique text will be synthesized with all 4 emotions using concatenated references
        """
        logger.info("📋 Creating multi-emotion synthesis jobs with concatenated references")
        
        # Get unique texts from CSV (avoid duplicates) but preserve line numbers
        unique_texts = {}
        for sample in self.text_samples:
            text_key = sample.text.lower().strip()
            if text_key not in unique_texts:
                unique_texts[text_key] = sample
        
        logger.info(f"📝 Found {len(unique_texts)} unique texts from CSV")
        
        # Target emotions for synthesis
        target_emotions = ['happy', 'sad', 'angry', 'neutral']
        
        jobs = []
        total_jobs = len(unique_texts) * len(target_emotions)
        
        logger.info(f"🎭 Creating {len(unique_texts)} texts × {len(target_emotions)} emotions = {total_jobs} synthesis jobs")
        
        job_counter = 0
        for original_sample in unique_texts.values():
            for target_emotion in target_emotions:
                job_counter += 1
                
                logger.info(f"Creating job {job_counter}/{total_jobs}: {target_emotion} for line {original_sample.line_number}")
                
                # Get emotion instruction for target emotion
                emotion_type = 'explicit'  # Use explicit for stronger emotional expression
                csv_column = f"{target_emotion}_{emotion_type}"
                emotion_instruction = self.emotion_config.get_instruction(csv_column)
                instruction = emotion_instruction.instruction if emotion_instruction else f"Speak with {target_emotion} emotion"
                
                # Create job with concatenated reference strategy
                job = SynthesisJob(
                    text=original_sample.text,
                    emotion=target_emotion,
                    emotion_type=emotion_type,
                    reference_audio=None,  # Will be generated dynamically
                    instruction=instruction,
                    metadata={
                        'original_emotion': original_sample.emotion,
                        'original_csv_column': original_sample.csv_column,
                        'target_emotion': target_emotion,
                        'synthesis_type': 'multi_emotion_concatenated',
                        'text_length': len(original_sample.text),
                        'job_number': job_counter,
                        'total_jobs': total_jobs,
                        'csv_line_number': original_sample.line_number,
                        'csv_row_index': original_sample.row_index if hasattr(original_sample, 'row_index') else None
                    }
                )
                jobs.append(job)
        
        logger.info(f"✅ Created {len(jobs)} multi-emotion synthesis jobs")
        return jobs
    
    def run_synthesis_with_concatenated_references(self, 
                                             jobs: List[SynthesisJob], 
                                             method: str = "auto",
                                             batch_size: int = 7,
                                             num_files: int = 7) -> List:  # ADD parameter
        """
        Run synthesis with dynamically generated concatenated references
        UPDATED: Per-synthesis speaker randomization with configurable file count
        """
        import random  # Add import for randomization
        
        logger.info(f"🚀 Starting synthesis with randomized speaker selection: {len(jobs)} jobs")
        logger.info(f"🔗 Using {num_files} files per emotion reference")
        
        all_results = []
        
        # Process in batches to manage memory
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(jobs) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} jobs)")
            
            # Process each job in batch with concatenated reference
            batch_results = []
            for job_idx, job in enumerate(batch):
                job_num = i + job_idx + 1
                csv_line = job.metadata.get('csv_line_number', 'unknown')
                logger.info(f"Job {job_num}/{len(jobs)}: Line {csv_line} - Synthesizing '{job.emotion}' emotion")
                
                try:
                    # Map emotion to CSV format (CSV uses title case: Happy, Sad, Angry, Neutral)
                    csv_emotion = job.emotion.title()  # happy → Happy, neutral → Neutral
                    
                    # STEP 1: Randomize speaker selection for this emotion
                    logger.info(f"  🎲 Step 1: Random speaker selection for {csv_emotion}")
                    
                    # If using CSV, get random speaker that has this emotion in CSV
                    if self.esd_manager.reference_csv and self.esd_manager.reference_df is not None:
                        # Get available speakers for this emotion from CSV
                        emotion_speakers = self.esd_manager.reference_df[
                            self.esd_manager.reference_df['emotion'] == csv_emotion
                        ]['speaker'].unique()
                        
                        logger.info(f"  📋 Step 2: Check CSV for {csv_emotion} - found {len(emotion_speakers)} speakers")
                        
                        if len(emotion_speakers) > 0:
                            # Randomly select one speaker
                            selected_speaker = random.choice(emotion_speakers)
                            selected_speaker = str(selected_speaker).zfill(4)  # Ensure 4 digits
                            logger.info(f"  👤 Step 3: Selected speaker {selected_speaker}")
                            
                            # Get files for this specific speaker+emotion from CSV
                            speaker_emotion_files = self.esd_manager.reference_df[
                                (self.esd_manager.reference_df['emotion'] == csv_emotion) &
                                (self.esd_manager.reference_df['speaker'] == int(selected_speaker))
                            ]
                            
                            logger.info(f"  📁 Step 4: Found {len(speaker_emotion_files)} available files in CSV")
                            logger.info(f"  ⏱️ Step 5: Finding {num_files} longest files and concatenating...")
                            
                            # Get up to N LONGEST files for this speaker+emotion from CSV
                            ref_audio_tensor, ref_metadata = self._get_concatenated_reference_from_csv_speaker(
                                speaker_emotion_files, csv_emotion, selected_speaker, num_files
                            )
                        else:
                            logger.warning(f"  ⚠️ No speakers found for emotion '{csv_emotion}' in CSV")
                            logger.info(f"  📋 Available emotions in CSV: {self.esd_manager.reference_df['emotion'].unique()}")
                            logger.warning(f"  🔄 Using fallback method (longest files)")
                            ref_audio_tensor, ref_metadata = self.esd_manager.get_concatenated_reference_for_synthesis(job.emotion)
                    else:
                        # Fallback: random speaker selection from available speakers
                        logger.info(f"  🔄 No CSV available, using fallback method for {csv_emotion}")
                        ref_audio_tensor, ref_metadata = self.esd_manager.get_concatenated_reference_for_synthesis(job.emotion)
                    
                    # Create a temporary SynthesisJob with the reference audio tensor
                    job_with_ref = SynthesisJob(
                        text=job.text,
                        emotion=job.emotion,
                        emotion_type=job.emotion_type,
                        reference_audio=ref_audio_tensor,  # Pass tensor directly
                        instruction=job.instruction,
                        metadata={
                            **job.metadata,
                            'reference_metadata': ref_metadata
                        }
                    )
                    
                    # Step 6: Synthesize with CosyVoice
                    logger.info(f"  🎙️ Step 6: Synthesizing with CosyVoice using {ref_metadata.get('num_files_concatenated', 0)} concatenated references")
                    result = self.synthesis_engine.synthesize_single_with_tensor_reference(job_with_ref, method)
                    
                    # Save result if successful
                    if result.success:
                        # Generate filename with CSV line number
                        csv_line_number = job.metadata.get('csv_line_number', 0)
                        original_csv_column = job.metadata.get('original_csv_column', 'unknown')
                        synthesized_emotion = job.emotion
                        speaker_id = ref_metadata.get('speaker_used', 'unknown')

                        # Filename format: line_{csv_line}_{original_emotion}_{synthesized_emotion}_{speaker}_COSY.wav
                        if csv_line_number and csv_line_number != 0:
                            filename = f"{csv_line_number:02d}_{original_csv_column}_{synthesized_emotion}_{speaker_id}_COSY.wav"
                        else:
                            # Fallback if no line number available
                            import hashlib
                            text_hash = hashlib.md5(job.text.encode()).hexdigest()[:8]
                            filename = f"{original_csv_column}_{synthesized_emotion}_{speaker_id}_{text_hash}_COSY.wav"
                        
                        self.synthesis_engine.save_result(result, filename)
                        
                        logger.info(f"  ✅ Success: {filename}")
                    else:
                        logger.warning(f"  ❌ Failed: {result.error_message}")
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"  ❌ Job failed: {e}")
                    # Create failed result
                    failed_result = SynthesisResult(
                        success=False,
                        error_message=str(e),
                        method_used=method
                    )
                    batch_results.append(failed_result)
            
            all_results.extend(batch_results)
            
            # Print batch summary
            successful = len([r for r in batch_results if r.success])
            logger.info(f"Batch {batch_num} complete: {successful}/{len(batch)} successful")
        
        # Final summary
        total_successful = len([r for r in all_results if r.success])
        success_rate = (total_successful / len(all_results)) * 100 if all_results else 0
        
        logger.info(f"🎉 Synthesis complete: {total_successful}/{len(all_results)} ({success_rate:.1f}%)")
        
        return all_results
    
    def create_final_dataset(self) -> Dict:
        """
        Create final dataset with manifest and organization
        
        Returns:
            Dataset creation summary
        """
        logger.info("📦 Creating final dataset")
        
        # Create manifest
        manifest = self.synthesis_engine.create_dataset_manifest(
            self.synthesis_results, 
            self.create_synthesis_jobs()
        )
        
        # Save manifest
        manifest_path = self.synthesis_engine.save_manifest(manifest)
        
        # Save additional metadata
        metadata = {
            'pipeline_config': {
                'csv_file': str(self.csv_file),
                'esd_path': str(self.esd_path),
                'model_path': str(self.model_path),
                'output_dir': str(self.output_dir),
                'reference_csv': str(self.reference_csv) if self.reference_csv else None
            },
            'reference_files': self.reference_files,
            'emotion_config': {
                'csv_mapping': self.emotion_config.csv_emotion_mapping,
                'esd_mapping': self.emotion_config.esd_emotion_mapping
            }
        }
        
        metadata_path = self.output_dir / "pipeline_metadata.json"
        FileUtils.save_json(metadata, metadata_path)
        
        # Create README
        self._create_readme(manifest)
        
        # Calculate final statistics
        successful_results = [r for r in self.synthesis_results if r.success]
        total_duration = sum(r.duration for r in successful_results if r.duration)
        total_size = FileUtils.get_directory_size(self.output_dir / "audio")
        
        summary = {
            'total_samples': len(successful_results),
            'total_duration_minutes': total_duration / 60,
            'total_size_mb': total_size / (1024 * 1024),
            'success_rate': len(successful_results) / len(self.synthesis_results) * 100,
            'output_directory': str(self.output_dir),
            'manifest_path': manifest_path
        }
        
        logger.info("✅ Final dataset created")
        return summary
    
    def _create_readme(self, manifest: Dict):
        """Create README file for the dataset"""
        readme_content = f"""# Emotional Speech Dataset

Generated using CosyVoice2 with ESD references and emotion_sentences_dataset.csv

## Dataset Information

- **Total Samples**: {manifest['dataset_info']['total_samples']}
- **Failed Samples**: {manifest['dataset_info']['failed_samples']}
- **Sample Rate**: {manifest['dataset_info']['sample_rate']} Hz
- **Created**: {manifest['dataset_info']['created_at']}
- **Reference Selection**: {'CSV-based' if self.reference_csv else 'Automatic'}
- **Concatenated Files per Reference**: 7 files (increased from 5)

"""
        if self.reference_csv:
            readme_content += f"""## Reference CSV Information

- **Reference CSV**: {self.reference_csv}
- **Selection Method**: CSV-based file selection
- **Files per Emotion**: Up to 7 files concatenated per emotion reference

"""
        
        readme_content += """## Emotions Covered

"""
        
        # Add emotion statistics
        emotion_counts = {}
        for sample in manifest['samples']:
            emotion_key = f"{sample['emotion']}_{sample['emotion_type']}"
            emotion_counts[emotion_key] = emotion_counts.get(emotion_key, 0) + 1
        
        for emotion, count in emotion_counts.items():
            readme_content += f"- **{emotion}**: {count} samples\n"
        
        readme_content += f"""
## Filename Format

Files are named using the pattern:
`{{csv_line_number}}_{{original_csv_column}}_{{synthesized_emotion}}_{{speaker_info}}_COSY.wav`

Example: `42_happy_explicit_sad_csv_mixed_speakers_COSY.wav`
- `42` = CSV line number 42
- `happy_explicit` = Original emotion from CSV
- `sad` = Synthesized target emotion  
- `csv_mixed_speakers` = Mixed speakers from CSV selection
- `COSY` = CosyVoice synthesis marker

## Directory Structure

```
{self.output_dir.name}/
├── audio/                    # Generated audio files (.wav)
├── metadata/                 # Metadata and logs
├── references/               # ESD reference audio files
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   └── neutral/
├── dataset_manifest.json     # Complete dataset information
├── pipeline_metadata.json   # Pipeline configuration
└── README.md                # This file
```

## Usage

Use `dataset_manifest.json` to programmatically access all sample information including:
- Audio file paths
- Original text
- Emotion labels and types
- CSV line numbers
- Reference file information (7 files per emotion)
- Synthesis metadata
- Duration information

## Synthesis Methods Used

"""
        
        method_counts = {}
        for sample in manifest['samples']:
            method = sample['synthesis_method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        for method, count in method_counts.items():
            readme_content += f"- **{method}**: {count} samples\n"
        
        readme_content += f"""
## Quality Information

- **Success Rate**: {len(manifest['samples']) / (len(manifest['samples']) + len(manifest['failed_jobs'])) * 100:.1f}%
- **Reference Source**: {'CSV-selected files' if self.reference_csv else 'ESD English speakers (0011-0020)'}
- **Text Source**: emotion_sentences_dataset.csv
- **Reference Audio**: 7 concatenated files per emotion (increased for better quality)

Generated with the Emotional Speech Toolkit
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def run_complete_pipeline(self, 
                         concatenated_files_per_emotion: int = 7,  # RENAMED for clarity
                         synthesis_method: str = "auto",
                         batch_size: int = 7) -> Dict:
        """
        Run the complete pipeline
        
        Args:
            concatenated_files_per_emotion: Number of files to concatenate per emotion reference (7)
            synthesis_method: Synthesis method to use
            batch_size: Number of jobs per batch
        """
        logger.info("🎯 Starting Complete Multi-Emotion Pipeline with Concatenated References")
        logger.info(f"🔗 Using {concatenated_files_per_emotion} files per emotion reference")
        logger.info("=" * 80)
        
        # Store the concatenation count for use in synthesis
        self.concatenated_files_count = concatenated_files_per_emotion
        
        pipeline_steps = [
            ("ESD Reference Setup", lambda: self.setup_esd_references()),  # No parameter needed
            ("CSV Data Loading", self.load_csv_data),
            ("Synthesis Initialization", self.initialize_synthesis),
        ]
        
        # Execute setup steps
        for step_name, step_func in pipeline_steps:
            logger.info(f"Step: {step_name}")
            success = step_func()
            if not success:
                logger.error(f"Pipeline failed at step: {step_name}")
                return {'success': False, 'failed_step': step_name}
        
        # Create multi-emotion jobs
        jobs = self.create_synthesis_jobs()
        
        # Run synthesis with concatenated references
        results = self.run_synthesis_with_concatenated_references(
            jobs, synthesis_method, batch_size, concatenated_files_per_emotion
        )
        
        # Store results for manifest creation
        self.synthesis_results = results
        
        # Create final dataset
        summary = self.create_final_dataset()
        
        # Print final statistics
        self.synthesis_engine.print_statistics()
        
        logger.info("🎉 Multi-Emotion Pipeline Complete!")
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info(f"🎵 Total jobs processed: {len(results)}")
        logger.info(f"✅ Success Rate: {summary['success_rate']:.1f}%")
        
        return {
            'success': True,
            'summary': summary,
            'output_directory': str(self.output_dir),
            'synthesis_mode': 'multi_emotion_concatenated_references'
        }
    
    def _get_concatenated_reference_from_csv_speaker(self, speaker_emotion_files, emotion, speaker_id, num_files=7):
        """
        Get concatenated reference from specific speaker using CSV selection
        FIXED: Pick 7 longest files from CSV for this speaker+emotion
        
        Args:
            speaker_emotion_files: DataFrame rows for this speaker+emotion
            emotion: Target emotion
            speaker_id: Selected speaker ID
            num_files: Number of files to concatenate (7)
            
        Returns:
            Tuple of (audio_tensor, metadata)
        """
        import torchaudio
        
        logger.info(f"    🔍 Finding longest files for speaker {speaker_id} + {emotion}")
        logger.info(f"    📋 Available files in CSV: {len(speaker_emotion_files)}")
        
        # Get all available files from CSV with their durations
        files_with_duration = []
        
        for idx, row in speaker_emotion_files.iterrows():
            wav_name = row['wav_name']
            file_path = self.esd_manager.esd_path / speaker_id / emotion / wav_name
            
            if file_path.exists():
                try:
                    # Get duration of this file
                    info = torchaudio.info(str(file_path))
                    duration = info.num_frames / info.sample_rate
                    files_with_duration.append((file_path, duration, wav_name))
                    logger.info(f"    📁 {wav_name}: {duration:.2f}s")
                except Exception as e:
                    logger.warning(f"    ⚠️ Failed to get duration for {wav_name}: {e}")
            else:
                logger.warning(f"    ❌ File not found: {file_path}")
        
        if not files_with_duration:
            raise ValueError(f"No valid files found for speaker {speaker_id}, emotion {emotion}")
        
        # Sort by duration (longest first) and take top N
        files_with_duration.sort(key=lambda x: x[1], reverse=True)
        longest_files = files_with_duration[:num_files]
        
        logger.info(f"    🏆 Selected {len(longest_files)} longest files:")
        for i, (file_path, duration, wav_name) in enumerate(longest_files):
            logger.info(f"      {i+1:2d}. {wav_name} ({duration:.2f}s)")
        
        # Extract just the file paths for concatenation
        selected_files = [file_path for file_path, duration, wav_name in longest_files]
        
        # Concatenate the selected files
        concatenated_audio = self.esd_manager.concatenate_reference_audio(selected_files, max_duration=100.0)
        
        # Create metadata
        total_duration = sum(duration for _, duration, _ in longest_files)
        metadata = {
            'reference_type': 'csv_longest_files',
            'emotion': emotion,
            'speaker_used': speaker_id,
            'num_files_concatenated': len(selected_files),
            'concatenated_files': [wav_name for _, _, wav_name in longest_files],
            'individual_durations': [duration for _, duration, _ in longest_files],
            'total_source_duration': total_duration,
            'concatenated_duration': concatenated_audio.shape[1] / 16000,
            'selection_method': 'csv_longest_files'
        }
        
        logger.info(f"    ✅ Concatenated {len(selected_files)} files, total: {total_duration:.2f}s → {metadata['concatenated_duration']:.2f}s")
        return concatenated_audio, metadata