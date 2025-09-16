import os
import pandas as pd
import random
from scipy.io.wavfile import write
import msinference
import librosa
import tempfile
import logging
import soundfile as sf
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--esd', type=str, help='path to original ESD dataset')
parser.add_argument('--txts', type=str, help='path to generated emotional sentences')
parser.add_argument('--esd_ser_test', type=str, help='path to SER test set')
parser.add_argument('--out', type=str, help='output directory')
args = parser.parse_args()

BASE_REFERENCE_PATH = args.esd
FULL_CSV_FILE_PATH = args.txts
TEST_SET_CSV_FILE_PATH = args.esd_ser_test
OUTPUT_DIR = args.out
OUTPUT_METADATA_DIR = os.path.join(OUTPUT_DIR, 'metadata')

EMOTIONS_TO_USE = ['Angry', 'Happy', 'Neutral', 'Sad']
SPEAKERS_TO_USE = [f"00{i}" for i in range(11, 21)]
NUM_REF_SAMPLES = 7
SAMPLE_RATE = 24000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_random_reference_audio(emotion: str) -> str:
    """
    Selects a random reference audio file for a given emotion and speaker.

    Args:
        emotion (str): The target emotion for the reference style.

    Returns:
        str: The full path to a randomly selected WAV file.
             Returns None if a valid file cannot be found.
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
                duration = librosa.get_duration(path=full_path)
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


def main():
    """
    Main function to run the speech synthesis process.
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

    # Generate Audio by Iterating Through Columns and Rows
    # Iterate through each column in the CSV
    for text_emotion in df.columns:
        sentences = df[text_emotion].dropna().reset_index(drop=True)
        for sentence_index, text_content in sentences.items():
            
            logging.info(f"\nProcessing {text_emotion} sentence {sentence_index} | Text: '{text_content}...'")

            # For each sentence, generate audio using every possible reference emotion
            for ref_emotion in EMOTIONS_TO_USE:
                
                # --- Get Reference Audio and Speaker ID ---
                used_refs, ref_style_path, speaker_id, ref_dur = get_random_reference_audio(ref_emotion)
                if not ref_style_path:
                    logging.warning(f"Skipping generation for ref_emotion '{ref_emotion}' due to missing reference.")
                    continue

                # --- Determine Condition (Explicit vs. Implicit) ---
                logging.info(f"  -> Reference Emotion: '{ref_emotion}' | Speaker: {speaker_id}")

                # --- Synthesize Speech ---
                try:
                    voice = msinference.compute_style(ref_style_path)
                    wav_data = msinference.inference(
                        text_content, 
                        voice, 
                        alpha=0.2, 
                        beta=0.1, 
                        diffusion_steps=15, 
                        embedding_scale=1
                    )

                    total_reference_durations.append(ref_dur)
                    row = {'sentence_index': sentence_index + 1,
                           'text_emotion': text_emotion,
                           'ref_emotion': ref_emotion,
                           'speaker_id': speaker_id,
                           'used_refs': used_refs,
                           'ref_dur': round(ref_dur, 2)}
                    df_rows.append(row)
                    
                    # --- Save the Output Audio with the New Filename Format ---
                    output_filename = f"{sentence_index + 1:02}_{text_emotion}_{ref_emotion.lower()}_{speaker_id}_STYLE.wav"
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    write(output_filepath, SAMPLE_RATE, wav_data)

                except Exception as e:
                    logging.error(f"     Failed to generate audio. Error: {e}")

    output_json_path = os.path.join(OUTPUT_METADATA_DIR, 'metadata.json')
    with open(output_json_path, 'w') as f:
        json.dump(df_rows, f, indent=4)

    mean_ref_dur = np.mean(total_reference_durations)
    std_red_dur = np.std(total_reference_durations)
    logging.info(f"\n--- Batch generation complete! ---\nMean reference duration: {mean_ref_dur:.2f} seconds ---\nStandard deviation: {std_red_dur:.2f} seconds")


if __name__ == '__main__':
    main()