#!/bin/bash
#SBATCH --job-name=flamingo3_run
#SBATCH --output=/home/pedro.correa/slurm/flamingo3_run_%j.out
#SBATCH --error=/home/pedro.correa/slurm/flamingo3_run_%j.err
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mail-user=p243236@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate af3

cd /home/pedro.correa/audio-flamingo/

python llava/cli/infer_audio_batch.py \
    --model-base nvidia/audio-flamingo-3 \
    --conv-mode auto \
    --text "Using tone of voice only (prosody: pitch, rhythm, loudness, timbre). Ignore word meaning; do not transcribe. Reply with exactly one: angry | happy | sad | neutral." \
    --folder F5TTS/7_ref_SERtestSet_F5 \
    --output F5TTS/f5tts_predictions_flamingo3_7ref_v4.csv