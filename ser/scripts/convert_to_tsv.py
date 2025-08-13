import os
import pandas as pd
import soundfile as sf
import sys

# Define paths
# This should be the root directory where your WAV files are located.
ESD_ROOT_PATH = "/home/lucas.ueda/expressive_datasets/esd/files"
# This is the directory where your esd_train.csv, esd_val.csv, esd_test.csv are saved.
# Assuming you run this script from SLM-ER-Evaluation/ser/ and CSVs are also there.
OUTPUT_CSV_DIR = "../"
# This is where the generated TSV, LENGTHS, and EMO files will be saved.
OUTPUT_MANIFEST_DIR = "./esd_manifests"
# This directory will hold the extracted features (.npy and .lengths files).
# It's created here to ensure it exists before feature extraction.
OUTPUT_FEATURES_DIR = "./esd_features" 

def create_manifest(csv_file_path, manifest_output_prefix, root_dir):
    """
    Reads a CSV, creates a Fairseq-compatible TSV manifest,
    and writes lengths file and emotion labels file.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}. Skipping manifest generation.")
        return

    os.makedirs(os.path.dirname(manifest_output_prefix), exist_ok=True)

    tsv_file_path = f"{manifest_output_prefix}.tsv"
    lengths_file_path = f"{manifest_output_prefix}.lengths"
    emo_file_path = f"{manifest_output_prefix}.emo" # New: for emotion labels

    with open(tsv_file_path, 'w') as tsv_f, \
         open(lengths_file_path, 'w') as len_f, \
         open(emo_file_path, 'w') as emo_f: # New: open emo file
        
        # Write the root directory as the first line of the TSV
        tsv_f.write(f"{root_dir}\n")
        
        for index, row in df.iterrows():
            full_wav_path = row['wav_file']
            wav_name = row['wav_name'] # Get wav_name from CSV
            emotion_label = row['emotion'] # Get emotion from CSV
            
            try:
                # Get audio length in frames
                info = sf.info(full_wav_path)
                length_frames = info.frames

                # Get path relative to the root_dir
                relative_wav_path = os.path.relpath(full_wav_path, root_dir)
                
                # Write to TSV and lengths file
                tsv_f.write(f"{relative_wav_path}\t{length_frames}\n")
                len_f.write(f"{length_frames}\n")
                
                # Write to EMO file: <wav_name>\t<emotion_label>
                emo_f.write(f"{wav_name}\t{emotion_label}\n") # New: write emotion label
                
            except Exception as e:
                print(f"Warning: Could not process {full_wav_path}: {e}. Skipping.")
                continue

    print(f"Manifest created: {tsv_file_path}")
    print(f"Lengths file created: {lengths_file_path}")
    print(f"Emotion labels file created: {emo_file_path}") # New print

if __name__ == "__main__":
    os.makedirs(OUTPUT_MANIFEST_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True) # Ensure features dir is created

    splits = ['train', 'val', 'test']
    for split in splits:
        csv_path = os.path.join(OUTPUT_CSV_DIR, f"esd_{split}.csv")
        # e.g., ./esd_manifests/train
        manifest_path_prefix = os.path.join(OUTPUT_MANIFEST_DIR, split) 

        print(f"Attempting to process {csv_path}...")
        create_manifest(csv_path, manifest_path_prefix, ESD_ROOT_PATH)