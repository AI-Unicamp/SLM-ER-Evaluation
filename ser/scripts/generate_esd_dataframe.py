import os
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa

ESD_PATH = "/home/lucas.ueda/expressive_datasets/esd/files"
test_fixed_samples_path = "/home/lucas.ueda/github/SLM-ER-Evaluation/ser/emotional_speech_analysis/test_set_reserved_files.csv"

fixed_test_df = pd.read_csv(test_fixed_samples_path)

speakers = []
emotions = []
wav_files = []
wav_names = []
durations = []

for folder in os.listdir(ESD_PATH):
    speaker = folder.split('_')[0]
    emotion = folder.split('_')[1]
    
    # Skip Surprise emotion
    if emotion.lower() == 'surprise':
        continue

    for wav_file in os.listdir(os.path.join(ESD_PATH, folder)):
        if wav_file.endswith('.wav'):
            wav_path = os.path.join(ESD_PATH, folder, wav_file)
            
            # Get audio duration in seconds
            duration = librosa.get_duration(path=wav_path)
            
            speakers.append(speaker)
            emotions.append(emotion)
            wav_files.append(wav_path)
            wav_names.append(wav_file)
            durations.append(duration)

full_df = pd.DataFrame({
    'speaker': speakers,
    'emotion': emotions,
    'wav_file': wav_files,
    'wav_name': wav_names,
    'duration': durations
})

# Select 30 longest audios for each emotion+speaker combination as test set
test_dfs = []
for (emotion, speaker), group in full_df.groupby(['emotion', 'speaker']):
    # Sort by duration descending and take top 30
    longest_30 = group.nlargest(30, 'duration')
    test_dfs.append(longest_30)

test_df = pd.concat(test_dfs, ignore_index=True)

# Remove test samples from full_df to create train/val sets
train_val_df = full_df[~full_df.index.isin(test_df.index)].reset_index(drop=True)

# Split remaining data into train and validation with stratification
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=0.1, 
    random_state=42, 
    stratify=train_val_df[['emotion', 'speaker']]
)

# Save the splits
train_df.to_csv(os.path.join("../", 'esd_train.csv'), index=False)
val_df.to_csv(os.path.join("../", 'esd_val.csv'), index=False)
test_df.to_csv(os.path.join("../", 'esd_test.csv'), index=False)