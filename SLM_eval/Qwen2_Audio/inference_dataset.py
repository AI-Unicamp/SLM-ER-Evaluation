from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioEmotionProcessor:
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct"):
        """Initialize the audio emotion processor"""
        # Set offline mode
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        logger.info("Loading model and processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name, device_map="auto", local_files_only=True
            )
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
            self.device = next(self.model.parameters()).device
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def process_single_audio(self, audio_path, custom_prompt=None):
        """Process a single audio file and return emotion prediction"""
        try:
            # Default prompt
            if custom_prompt is None:
                custom_prompt = "Using tone of voice only (prosody: pitch, rhythm, loudness, timbre). Ignore word meaning; do not transcribe. Reply with exactly one: angry | happy | sad | neutral."
            
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": custom_prompt},
                ]}
            ]

            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []

            # Load audio file
            if Path(audio_path).exists():
                audio_data = librosa.load(audio_path, sr=self.sampling_rate)[0]
                audios.append(audio_data)
            else:
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Process inputs
            inputs = self.processor(
                text=text, 
                audios=audios, 
                return_tensors="pt", 
                sampling_rate=self.sampling_rate, 
                padding=True
            )

            # Move tensors to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_length=256)
                generate_ids = generate_ids[:, inputs['input_ids'].size(1):]

            response = self.processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            return response.strip()

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return f"ERROR: {str(e)}"

    def process_dataset(self, dataset_path, output_file=None, custom_prompt=None, file_pattern="*.wav"):
        """Process entire dataset of audio files"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Find all audio files
        audio_files = list(dataset_path.glob(file_pattern))
        if not audio_files:
            # Try recursive search
            audio_files = list(dataset_path.rglob(file_pattern))
        
        if not audio_files:
            raise FileNotFoundError(f"No audio files found with pattern {file_pattern} in {dataset_path}")

        logger.info(f"Found {len(audio_files)} audio files to process")

        # Process files
        results = []
        failed_files = []

        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                emotion = self.process_single_audio(audio_file, custom_prompt)
                
                result = {
                    'file_path': str(audio_file),
                    'filename': audio_file.name,
                    'emotion': emotion,
                    'processed_at': datetime.now().isoformat()
                }
                results.append(result)
                
                logger.info(f"Processed {audio_file.name}: {emotion}")
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")
                failed_files.append({'file': str(audio_file), 'error': str(e)})

        # Save results
        if output_file is None:
            output_file = f"emotion_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save as CSV
        df = pd.DataFrame(results)
        csv_file = f"{output_file}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")

        # Save as JSON for more detailed info
        json_file = f"{output_file}.json"
        output_data = {
            'metadata': {
                'total_files': len(audio_files),
                'successful': len(results),
                'failed': len(failed_files),
                'dataset_path': str(dataset_path),
                'processed_at': datetime.now().isoformat(),
                'prompt_used': custom_prompt or "Using tone of voice only (prosody: pitch, rhythm, loudness, timbre). Ignore word meaning; do not transcribe. Reply with exactly one: angry | happy | sad | neutral."
            },
            'results': results,
            'failed_files': failed_files
        }
        
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Detailed results saved to {json_file}")

        return results, failed_files

    def get_emotion_summary(self, results):
        """Get summary statistics of emotions detected"""
        if not results:
            return {}
        
        emotions = [result['emotion'] for result in results if not result['emotion'].startswith('ERROR')]
        emotion_counts = {}
        
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Sort by frequency
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_processed': len(emotions),
            'unique_emotions': len(emotion_counts),
            'emotion_distribution': dict(sorted_emotions),
            'most_common': sorted_emotions[0] if sorted_emotions else None
        }

def main():
    """Main function to run the batch processing"""
    # Configuration
    DATASET_PATH = "F5TTS/audio/7_ref_SERtestSet_F5"  # Change this to your dataset path
    OUTPUT_FILE = "F5TTS/f5tts_predictions_qwen2audio_7ref_v4"
    CUSTOM_PROMPT = None  # Use None for default prompt or specify your own
    FILE_PATTERN = "*.wav"  # Change to "*.mp3" or other formats as needed
    
    try:
        # Initialize processor
        processor = AudioEmotionProcessor()
        
        # Process dataset
        results, failed_files = processor.process_dataset(
            dataset_path=DATASET_PATH,
            output_file=OUTPUT_FILE,
            custom_prompt=CUSTOM_PROMPT,
            file_pattern=FILE_PATTERN
        )
        
        # Print summary
        summary = processor.get_emotion_summary(results)
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total files processed: {summary.get('total_processed', 0)}")
        print(f"Failed files: {len(failed_files)}")
        print(f"Unique emotions detected: {summary.get('unique_emotions', 0)}")
        
        if summary.get('emotion_distribution'):
            print("\nEmotion Distribution:")
            for emotion, count in summary['emotion_distribution'].items():
                print(f"  {emotion}: {count}")
        
        if summary.get('most_common'):
            emotion, count = summary['most_common']
            print(f"\nMost common emotion: {emotion} ({count} files)")
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()