"""
Main execution script for the emotional speech pipeline with CSV support
Place this in the root directory alongside your CSV files
"""

import sys
import os
from pathlib import Path

# Add the toolkit to path
sys.path.append('emotional_speech_toolkit')

from cosyvoice.emotional_speech_toolkit.pipeline import EmotionalSpeechPipeline

def main():
    """Main execution function"""
    
    print("🎙️ Emotional Speech Dataset Generation Pipeline")
    print("=" * 60)
    
    # Configuration - EDIT THESE PATHS AS NEEDED
    config = {
        'csv_file': 'emotion_sentences_dataset.csv',
        'esd_path': 'ESD_English',  # Your ESD English speakers directory
        'model_path': 'pretrained_models/CosyVoice2-0.5B',
        'output_dir': 'final_emotional_dataset',
        'reference_csv': 'esd_test_v2.csv',  # CSV with reference file selections
        'samples_per_emotion': 7,  # Number of ESD references per emotion (for traditional method)
        'synthesis_method': 'auto',  # auto, audio_plus_tag, tag_only
        'batch_size': 25  # Adjust based on your GPU memory
    }
    
    # Validate required files exist
    required_files = [config['csv_file'], config['esd_path'], config['model_path']]
    if config['reference_csv']:
        required_files.append(config['reference_csv'])
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file/directory not found: {file_path}")
            return
    
    print("✅ All required files found")
    
    if config['reference_csv']:
        print(f"📋 Using reference CSV: {config['reference_csv']}")
        print("🎵 Will concatenate 7 files per emotion reference")
    
    # Initialize and run pipeline
    pipeline = EmotionalSpeechPipeline(
        csv_file=config['csv_file'],
        esd_path=config['esd_path'],
        model_path=config['model_path'],
        output_dir=config['output_dir'],
        reference_csv=config['reference_csv']
    )
    
    # Run complete pipeline
    result = pipeline.run_complete_pipeline(
        concatenated_files_per_emotion=7,
        synthesis_method=config['synthesis_method'],
        batch_size=config['batch_size']
    )
    
    if result['success']:
        print("\n🎊 SUCCESS! Your emotional speech dataset is ready!")
        print(f"📁 Location: {result['output_directory']}")
        print(f"📋 Check the README.md and dataset_manifest.json for details")
        if config['reference_csv']:
            print(f"🎵 Used CSV-based reference selection with 7-file concatenation")
    else:
        print(f"\n❌ Pipeline failed at: {result.get('failed_step', 'unknown step')}")
        print("Check the logs for more details")

if __name__ == "__main__":
    main()