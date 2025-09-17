# F5-TTS Batch Audio Generation

Generate synthetic speech with emotional style transfer using F5-TTS.

## Setup

```bash
# Install dependencies
pip install pandas librosa soundfile numpy torch torchaudio

# Install F5-TTS (included in this repo)
cd F5-TTS && pip install -e . && cd ..
```

## Data Requirements

1. **ESD Dataset**: Download from [here](https://github.com/HLTSingapore/Emotional-Speech-Data)
   - Structure: `data/ESD/{speaker_id}/{emotion}/{audio_files}.wav`
   - Speakers: `0011` to `0020`
   - Emotions: `Angry`, `Happy`, `Neutral`, `Sad`

2. **Text CSV**: Create `data/emotion_sentences_dataset.csv`
   ```csv
   Angry,Happy,Neutral,Sad
   "I can't believe this!","What a great day!","The weather is nice.","I feel sad."
   ```

## Configure & Run

1. Edit paths in `batch_inference_f5tts.py`:
   ```python
   BASE_REFERENCE_PATH = 'data/ESD'
   FULL_CSV_FILE_PATH = 'data/emotion_sentences_dataset.csv'
   OUTPUT_DIR = 'output/generated_audio'
   ```

2. Run:
   ```bash
   python batch_inference_f5tts.py
   ```

## Output

Files named: `{sentence}_{text_emotion}_{ref_emotion}_{speaker}_F5TTS.wav`

Example: `01_angry_explicit_happy_0011_F5TTS.wav` = Angry text with happy voice style