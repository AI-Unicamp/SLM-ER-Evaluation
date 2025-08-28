import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="Aggregate SLM prediction results into one big .CSV file")
parser.add_argument('--in_dir', type=str)
parser.add_argument('--exp_ver', type=str)
parser.add_argument('--out_file', type=str)
args = parser.parse_args()

BASE_DIR = args.in_dir #hadatasets/<user>

out_df = pd.DataFrame()

def parse_filename(filename):
    filename = Path(filename).stem
    parts = filename.split('_')
    
    txt_emo = parts[1]
    if txt_emo == 'neutral':
        txt_cond = 'neutral'
        wav_emo = parts[2]
        tts = parts[4]
    else:
        txt_emo = parts[1]
        txt_cond = parts[2]
        wav_emo = parts[3]
        tts = parts[5]

    return txt_emo, txt_cond, wav_emo, tts

def apply_parsing(df, slm_name):
    if slm_name == 'salmonn':
        filename = df['file_path']
        new_columns = df['file_path'].apply(parse_filename)
    elif slm_name == 'flamingo3':
        filename = df['file_name']
        new_columns = df['file_name'].apply(parse_filename)
    else:
        filename = df['filename']
        new_columns = df['filename'].apply(parse_filename)
    
    df = pd.DataFrame(new_columns.tolist(), columns=['txt_emo', 'txt_cond', 'wav_emo', 'tts_model'])
    df['filename'] = filename
    return df

def get_predictions(df, slm_name):
    if slm_name == 'salmonn':
        preds = df['output'].str.replace(".", "", regex=False).str.lower()
    elif slm_name == 'flamingo3':
        preds = df['response'].str.replace(".", "", regex=False).str.lower()
    elif slm_name == 'desta2':
        preds = df['predicted_emotion'].str.replace(".", "", regex=False).str.lower()
    elif slm_name == 'qwen':
        preds = df['emotion'].str.replace(".", "", regex=False).str.lower()
    
    return preds

dfs_to_concat = []

for dir in os.listdir(BASE_DIR):                            # There are SAMPLE and RESULT folders
    if dir.split('_')[0] == 'results':                      # Look only for RESULT folders
        full_path_results_dir = os.path.join(BASE_DIR, dir)
        slm_name = dir.split('_')[-1]

        for file in os.listdir(full_path_results_dir):      #Look for .CSV files
            if Path(file).stem.split('_')[-1] == args.exp_ver:      #Specify experiment version (prompt)
                full_path_csv = os.path.join(full_path_results_dir, file)
                df = pd.read_csv(full_path_csv)
                new_df = apply_parsing(df, slm_name)
                preds = get_predictions(df, slm_name)
                new_df['pred_emo'] = preds
                new_df['slm'] = slm_name

                dfs_to_concat.append(new_df)

final_df = pd.concat(dfs_to_concat, ignore_index=True)
final_df.to_csv(args.out_file, index=False)
