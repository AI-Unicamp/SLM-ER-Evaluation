import argparse
import os
import os.path as osp
import sys
import numpy as np
import soundfile as sf
import pandas as pd
from dataclasses import dataclass
import tqdm
import torch
import torch.nn.functional as F
import fairseq

import torchaudio 
import torchaudio.transforms as T 

# Import Namespace for checkpoint patching
from argparse import Namespace

# --- Import your local modules ---
# Assuming data.py, model.py, utils.py are in the same directory as this script
from model import BaseModel # Your downstream classification model
# No need to import data or utils directly for inference, but keeping for reference if needed


# --- Emotion2vec Feature Reader (Adapted for Inference) ---
@dataclass
class UserDirModule:
    user_dir: str

class Emotion2vecFeatureReader(object):
    def __init__(self, model_file, checkpoint, layer):
        model_path = UserDirModule(model_file)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """
        Load an audio file, resample to 16kHz if necessary, and return PCM along with the sample rate.
        """
        wav, sr = sf.read(fname)
        channel = sf.info(fname).channels
        
        assert channel == 1, f"Channel should be 1 (mono), but got {channel} in file {fname}"

        # Convert to PyTorch tensor for resampling
        wav_tensor = torch.from_numpy(wav).float()

        if sr != 16000:
            print(f"Warning: Sample rate of {fname} is {sr}kHz. Resampling to 16kHz.")
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            wav_tensor = resampler(wav_tensor)
            sr = 16000 # Update sample rate after resampling
        
        return wav_tensor.numpy()


    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            res = self.model.extract_features(source, padding_mask=None, remove_extra_tokens=True)
            return res['x'].squeeze(0).cpu()


# --- Main Inference Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Perform inference using emotion2vec and a downstream model."
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to a single WAV file or a directory containing WAV files.'
    )
    parser.add_argument(
        '--emotion2vec_checkpoint',
        type=str,
        required=True,
        help='Path to the pre-trained emotion2vec base model checkpoint (e.g., emotion2vec_base.pt).'
    )
    parser.add_argument(
        '--downstream_checkpoint',
        type=str,
        required=True,
        help='Path to the best downstream model checkpoint (e.g., best_model.pth).'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the Fairseq "upstream" directory (e.g., /path/to/emotion2vec/upstream).'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Optional: Path to save the output CSV file if processing a directory of WAVs.'
    )
    args = parser.parse_args()

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Initialize Emotion2vec Feature Reader ---
    emotion2vec_reader = Emotion2vecFeatureReader(
        model_file=args.model_dir,
        checkpoint=args.emotion2vec_checkpoint,
        layer=11
    )

    # --- 2. Load Downstream Model ---
    # Define label mapping (must match your training)
    label_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    idx_to_label = {v: k for k, v in label_dict.items()} # Reverse mapping for output
    num_classes = len(label_dict)

    downstream_model = BaseModel(input_dim=768, output_dim=num_classes)
    
    # Load downstream model weights
    try:
        downstream_ckpt = torch.load(args.downstream_checkpoint, map_location=device)
        downstream_model.load_state_dict(downstream_ckpt, strict=True)
        downstream_model.eval() # Set to evaluation mode
        downstream_model.to(device) # Move to device
        print(f"Downstream model loaded from {args.downstream_checkpoint} and moved to {device}")
    except Exception as e:
        print(f"Error loading downstream model from {args.downstream_checkpoint}: {e}")
        sys.exit(1)

    # --- 3. Process Input ---
    results = []

    if os.path.isfile(args.input_path):
        # Process a single WAV file
        print(f"\nProcessing single WAV file: {args.input_path}")
        try:
            # Extract frame-level features
            feats = emotion2vec_reader.get_feats(args.input_path)
            
            # Convert to batch for downstream model (1, T, D)
            feats_batch = feats.unsqueeze(0).to(device)
            
            # Create a dummy padding mask for a single sample (all False as it's a single utterance)
            padding_mask = torch.zeros(feats_batch.shape[0], feats_batch.shape[1], dtype=torch.bool).to(device)

            with torch.no_grad():
                outputs = downstream_model(feats_batch, padding_mask)
                _, predicted_idx = torch.max(outputs.data, 1)
                predicted_emotion = idx_to_label[predicted_idx.item()]
            
            print(f"Predicted Emotion: {predicted_emotion}")
            results.append({
                'filename': os.path.basename(args.input_path),
                'predicted_emotion': predicted_emotion
            })

        except Exception as e:
            print(f"Error processing {args.input_path}: {e}")

    elif os.path.isdir(args.input_path):
        # Process a directory of WAV files
        print(f"\nProcessing WAV files in directory: {args.input_path}")
        wav_files = [f for f in os.listdir(args.input_path) if f.endswith('.wav')]
        
        if not wav_files:
            print(f"No WAV files found in {args.input_path}")
            return

        for wav_file in tqdm.tqdm(wav_files, desc="Processing WAVs"):
            full_wav_path = os.path.join(args.input_path, wav_file)
            try:
                feats = emotion2vec_reader.get_feats(full_wav_path)
                feats_batch = feats.unsqueeze(0).to(device)
                padding_mask = torch.zeros(feats_batch.shape[0], feats_batch.shape[1], dtype=torch.bool).to(device)

                with torch.no_grad():
                    outputs = downstream_model(feats_batch, padding_mask)
                    _, predicted_idx = torch.max(outputs.data, 1)
                    predicted_emotion = idx_to_label[predicted_idx.item()]
                
                results.append({
                    'filename': wav_file,
                    'predicted_emotion': predicted_emotion
                })
            except Exception as e:
                print(f"Warning: Could not process {full_wav_path}: {e}. Skipping.")
        
        # Save results to CSV if output_csv path is provided
        if args.output_csv:
            output_df = pd.DataFrame(results)
            output_df.to_csv(args.output_csv, index=False)
            print(f"\nInference results saved to: {args.output_csv}")
        else:
            print("\nInference complete. No output CSV saved (use --output_csv to save).")
            print("Results:")
            for res in results:
                print(f"  {res['filename']}: {res['predicted_emotion']}")

    else:
        print(f"Error: Input path '{args.input_path}' is neither a file nor a directory.")
        sys.exit(1)

if __name__ == '__main__':
    main()