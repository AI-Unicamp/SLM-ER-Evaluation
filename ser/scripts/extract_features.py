import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import os
import sys

import torch
import torch.nn.functional as F
import fairseq

def get_parser():
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks"
    )
    parser.add_argument('--source_file', help='location of source wav files', required=True)
    parser.add_argument('--target_file', help='location of target npy files', required=True)
    parser.add_argument('--model_dir', type=str, help='pretrained model', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', required=True)
    parser.add_argument('--granularity', type=str, help='which granularity to use, frame or utterance', required=True)

    return parser

@dataclass
class UserDirModule:
    user_dir: str

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    source_file = args.source_file
    target_file = args.target_file
    model_dir = args.model_dir
    checkpoint_dir = args.checkpoint_dir
    granularity = args.granularity

    # Determine the device to use (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Add model_dir to sys.path to enable fairseq to find custom modules
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    # Use fairseq's utility to import user-defined modules.
    # This will automatically find and register models and tasks
    # from the 'models' and 'tasks' subdirectories within 'model_dir'.
    model_path_obj = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path_obj)
    print(f"Fairseq's user modules imported from {model_dir}")

    print(f"Loading checkpoint from: {checkpoint_dir}")

    # Load the model ensemble and task
    # fairseq.checkpoint_utils.load_model_ensemble_and_task automatically handles
    # loading to CPU first, so we can then explicitly move it to the desired device.
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    
    # Access the first model in the ensemble and move it to the determined device
    model = model[0]
    model.eval() # Set model to evaluation mode
    model.to(device) # Move the model to the selected device
    print(f"Model moved to {device}")

    if source_file.endswith('.wav'):
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert sr == 16e3, f"Sample rate should be 16kHz, but got {sr} in file {source_file}"
        assert channel == 1, f"Channel should be 1, but got {channel} in file {source_file}"
        
    with torch.no_grad():
        # Move the input waveform to the determined device
        source = torch.from_numpy(wav).float().to(device)
        
        if task.cfg.normalize:
            # Layer normalization typically operates on the last dimension,
            # so the device handling for 'source' is sufficient.
            source = F.layer_norm(source, source.shape)
        
        source = source.view(1, -1) # Reshape for batch processing
        
        try:
            # extract_features will use the model already on the correct device
            feats = model.extract_features(source, padding_mask=None)
            
            # Move features back to CPU before converting to NumPy
            feats = feats['x'].squeeze(0).cpu().numpy()

            print(f"Extracted features shape: {feats.shape}")
            if granularity == 'frame':
                # feats is already frame-level
                pass
            elif granularity == 'utterance':
                feats = np.mean(feats, axis=0)
            else:
                raise ValueError(f"Unknown granularity: {args.granularity}. Choose 'frame' or 'utterance'.")
            
            np.save(target_file, feats)
            print(f"Features saved to: {target_file}")
        except Exception as e:
            print(f"Error in extracting features from {source_file}: {e}")
            sys.exit(1)


if __name__ == '__main__':
    # Import Namespace here, as it's needed for state["args"] patching
    # (though not directly used in this version, it's good practice if patching is ever re-introduced)
    from argparse import Namespace 
    main()