import os
import pandas as pd
import random
import librosa
import tempfile
import logging
import soundfile as sf
import numpy as np
import json
import torch
import torchaudio
from pathlib import Path
from f5_tts.infer.utils_infer import load_vocoder

# F5-TTS specific imports
from f5_tts.infer.utils_infer import (
    load_model, 
    load_vocoder,
    preprocess_ref_audio_text, 
    infer_process
)
from f5_tts.model import DiT

BASE_REFERENCE_PATH = '/home/joao.lima/data/ESD'

FULL_CSV_FILE_PATH = '/home/joao.lima/experiments/StyleTTS2/emotion_sentences_dataset.csv'

TEST_SET_CSV_FILE_PATH = '/home/joao.lima/experiments/StyleTTS2/esd_test.csv'

OUTPUT_DIR = '/home/victor.moreno/paper_pedro_2025/audio_generator/ref'
OUTPUT_METADATA_DIR = os.path.join(OUTPUT_DIR, 'metadata') 

EMOTIONS_TO_USE = ['Angry', 'Happy', 'Neutral', 'Sad']
SPEAKERS_TO_USE = [f"00{i}" for i in range(11, 21)]
NUM_REF_SAMPLES = 7
SAMPLE_RATE = 24000

# F5-TTS specific parameters
F5TTS_MODEL_CFG = dict(
    dim=1024, 
    depth=22, 
    heads=16, 
    ff_mult=2, 
    text_dim=512, 
    conv_layers=4
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_random_reference_audio(emotion: str) -> str:
    """
    Selects a random reference audio file for a given emotion and speaker.

    Args:
        emotion (str): The target emotion for the reference style.

    Returns:
        tuple: (used_refs, temp_file_path, speaker_id, ref_dur) or None if failed
    """
    test_set_df = pd.read_csv(TEST_SET_CSV_FILE_PATH)

    try:
        # 1. Randomly select a speaker
        speaker_id = random.choice(SPEAKERS_TO_USE)
        
        # 2. Construct the path to the speaker's emotion directory
        emotion_dir = os.path.join(BASE_REFERENCE_PATH, speaker_id, emotion)
            
        # 3. Get WAV files (only select those present in SER test set)
        wav_files = [f for f in os.listdir(emotion_dir) if f in test_set_df['wav_name'].values]
        
        # 4. Get duration for each WAV file
        durations_paths = []
        for wav_filename in wav_files:
            full_path = os.path.join(emotion_dir, wav_filename)
            try:
                audio_data, sr = librosa.load(full_path, sr=SAMPLE_RATE)
                duration = librosa.get_duration(y=audio_data, sr=sr)

                durations_paths.append((duration, full_path))
            except Exception as e:
                logging.warning(f"Could not process {full_path}: {e}")

        # 5. Sort by duration and get the longest N files
        durations_paths.sort(key=lambda x: x[0], reverse=True)
        longest_paths = [path for _, path in durations_paths[:NUM_REF_SAMPLES]]
        longest_paths_basenames = [os.path.basename(path) for path in longest_paths]

        ref_dur = sum(duration for duration, _ in durations_paths[:NUM_REF_SAMPLES])

        if not longest_paths:
            logging.error(f"Could not find any valid audio files to process for {emotion_dir}")
            return None, None

        # 6. Load and concatenate the audio data
        full_ref = []
        for filepath in longest_paths:
            audio_data, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            full_ref.append(audio_data)

        concatenated_audio = np.concatenate(full_ref)

        # 7. Create a temporary file and write the concatenated audio to it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, concatenated_audio, SAMPLE_RATE)
        
        return longest_paths_basenames, temp_file.name, speaker_id, ref_dur
        
    except Exception as e:
        logging.error(f"Error selecting reference for emotion '{emotion}': {e}")
        return None


def transcribe_reference_audio(audio_path: str) -> str:
    """
    Simple transcription placeholder. In practice, you might want to use
    an ASR model to automatically transcribe the reference audio.
    For now, return a generic text.
    
    You can either:
    1. Use empty string "" (F5-TTS will auto-transcribe but uses more GPU memory)
    2. Implement ASR here
    3. Use predefined text based on emotion/speaker
    """
    # Option 1: Return empty string for auto-transcription
    return ""
    
    # Option 2: Return a generic placeholder
    # return "This is a reference audio sample."
    
    # Option 3: You could return emotion-specific text
    # emotion_texts = {
    #     'Angry': "I am feeling very angry right now!",
    #     'Happy': "I am so happy and excited today!",
    #     'Neutral': "This is a neutral statement.",
    #     'Sad': "I feel so sad and disappointed."
    # }
    # return emotion_texts.get(emotion, "This is a reference audio sample.")


def main():
    """
    Main function to run the speech synthesis process using F5-TTS.
    """
    total_reference_durations = []
    df_rows = []

    if not os.path.isdir(BASE_REFERENCE_PATH):
        logging.error("Invalid BASE_REFERENCE_PATH. Please set the correct path to your audio dataset.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)
    logging.info(f"Output directory '{OUTPUT_DIR}' is ready.")

    try:
        df = pd.read_csv(FULL_CSV_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"CSV file not found at '{FULL_CSV_FILE_PATH}'. Please check the path.")
        return
    except Exception as e:
        logging.error(f"Error reading or processing CSV file: {e}")
        return

    # Initialize F5-TTS model - load it once at the beginning for efficiency
    logging.info("Loading F5-TTS model...")
    try:
        # Load the model (will download from HuggingFace if not cached)
        ema_model = load_model(
            DiT, 
            F5TTS_MODEL_CFG, 
            "/home/victor.moreno/paper_pedro_2025/F5-TTS/ckpts/model_1250000.safetensors"
        )
        logging.info("F5-TTS model loaded successfully.")
        vocoder = load_vocoder()
    except Exception as e:
        logging.error(f"Failed to load F5-TTS model: {e}")
        return

    # Generate Audio by Iterating Through Columns and Rows
    # Iterate through each column in the CSV
    for text_emotion in df.columns:
        sentences = df[text_emotion].dropna().reset_index(drop=True)
        for sentence_index, text_content in sentences.items():
            
            logging.info(f"\nProcessing {text_emotion} sentence {sentence_index} | Text: '{text_content}...'")

            # For each sentence, generate audio using every possible reference emotion
            for ref_emotion in EMOTIONS_TO_USE:
                
                # --- Get Reference Audio and Speaker ID ---
                ref_result = get_random_reference_audio(ref_emotion)
                if not ref_result or len(ref_result) < 4:
                    logging.warning(f"Skipping generation for ref_emotion '{ref_emotion}' due to missing reference.")
                    continue
                
                used_refs, ref_style_path, speaker_id, ref_dur = ref_result
                if not ref_style_path:
                    logging.warning(f"Skipping generation for ref_emotion '{ref_emotion}' due to missing reference path.")
                    continue

                # --- Determine Condition (Explicit vs. Implicit) ---
                logging.info(f"  -> Reference Emotion: '{ref_emotion}' | Speaker: {speaker_id}")

                # --- Prepare Reference Text ---
                ref_text = transcribe_reference_audio(ref_style_path)

                # --- Synthesize Speech ---
                try:
                    # Preprocess reference audio and text
                    processed_ref_audio, processed_ref_text = preprocess_ref_audio_text(
                        ref_style_path, ref_text
                    )
                    
                    # Perform F5-TTS inference
                    final_wave, final_sample_rate, combined_spectrogram = infer_process(
                        processed_ref_audio,
                        processed_ref_text, 
                        text_content,  # gen_text
                        ema_model,
                        vocoder,
                        cross_fade_duration=0.15,
                        speed=1.0
                    )

                    total_reference_durations.append(ref_dur)
                    row = {
                        'sentence_index': sentence_index + 1,
                        'text_emotion': text_emotion,
                        'ref_emotion': ref_emotion,
                        'speaker_id': speaker_id,
                        'used_refs': used_refs,
                        'ref_dur': round(ref_dur, 2)
                    }
                    df_rows.append(row)
                    
                    # --- Save the Output Audio with the New Filename Format ---
                    output_filename = f"{sentence_index + 1:02}_{text_emotion}_{ref_emotion.lower()}_{speaker_id}_F5TTS.wav"
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    
                    # Convert to numpy array if it's a tensor
                    if torch.is_tensor(final_wave):
                        final_wave = final_wave.cpu().numpy()
                    
                    # Ensure the audio is in the right format for saving
                    if final_wave.ndim > 1:
                        final_wave = final_wave.squeeze()
                    
                    sf.write(output_filepath, final_wave, final_sample_rate)
                    logging.info(f"    Generated and saved: {output_filename}")

                except Exception as e:
                    logging.error(f"     Failed to generate audio. Error: {e}")
                finally:
                    # Clean up temporary reference file
                    try:
                        os.unlink(ref_style_path)
                    except:
                        pass

    # Save metadata
    output_json_path = os.path.join(OUTPUT_METADATA_DIR, 'metadata.json')
    with open(output_json_path, 'w') as f:
        json.dump(df_rows, f, indent=4)

    # Calculate and log statistics
    if total_reference_durations:
        mean_ref_dur = np.mean(total_reference_durations)
        std_ref_dur = np.std(total_reference_durations)
        logging.info(f"\n--- Batch generation complete! ---\nMean reference duration: {mean_ref_dur:.2f} seconds ---\nStandard deviation: {std_ref_dur:.2f} seconds")
    else:
        logging.info("\n--- Batch generation complete! ---\nNo reference durations recorded.")


if __name__ == '__main__':
    main()