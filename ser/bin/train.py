import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

import torch
from torch import nn, optim
from torch.utils.data import DataLoader # <--- ADD THIS IMPORT

# Import necessary modules directly from the current directory
from data import load_dataset, SpeechDataset
from model import BaseModel
from utils import train_one_epoch, validate_and_test

import logging

logger = logging.getLogger('ESD_Downstream_Training')

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"\nTotal number of parameters: {total_params}")

@hydra.main(config_path='config', config_name='default.yaml')
def train_esd(cfg: DictConfig):
    torch.manual_seed(cfg.common.seed)

    # Define label dictionary for ESD dataset.
    # label_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'surprise': 4}
    label_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}

    num_classes = len(label_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # Load features for each split from the 'esd_features' directory
    train_feat_path_prefix = os.path.join(cfg.dataset.feat_path, "train")
    val_feat_path_prefix = os.path.join(cfg.dataset.feat_path, "val")
    test_feat_path_prefix = os.path.join(cfg.dataset.feat_path, "test")

    # Load data for each split using load_dataset and SpeechDataset
    train_data_npy, train_sizes, train_offsets, train_labels_raw = load_dataset(train_feat_path_prefix, labels='emo', min_length=1)
    val_data_npy, val_sizes, val_offsets, val_labels_raw = load_dataset(val_feat_path_prefix, labels='emo', min_length=1)
    test_data_npy, test_sizes, test_offsets, test_labels_raw = load_dataset(test_feat_path_prefix, labels='emo', min_length=1)

    # Mapping from full-word, capitalized labels (from .emo.tsv) to abbreviated lowercase labels (for label_dict)
    # raw_label_conversion_map = {
    #     'Neutral': 'neu',
    #     'Happy': 'hap',
    #     'Sad': 'sad',
    #     'Angry': 'ang',
    #     'Surprise': 'surprise',
    # }
    raw_label_conversion_map = {
        'Neutral': 'neu',
        'Happy': 'hap',
        'Sad': 'sad',
        'Angry': 'ang'
    }

    # Convert raw labels (e.g., 'Neutral') to the abbreviated format (e.g., 'neu')
    train_labels = [label_dict[raw_label_conversion_map[elem]] for elem in train_labels_raw]
    val_labels = [label_dict[raw_label_conversion_map[elem]] for elem in val_labels_raw]
    test_labels = [label_dict[raw_label_conversion_map[elem]] for elem in test_labels_raw]

    # Create SpeechDataset instances for each split
    train_dataset = SpeechDataset(feats=train_data_npy, sizes=train_sizes, offsets=train_offsets, labels=train_labels)
    val_dataset = SpeechDataset(feats=val_data_npy, sizes=val_sizes, offsets=val_offsets, labels=val_labels)
    test_dataset = SpeechDataset(feats=test_data_npy, sizes=test_sizes, offsets=test_offsets, labels=test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, collate_fn=train_dataset.collator, 
                              num_workers=0, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, collate_fn=val_dataset.collator, 
                            num_workers=0, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, collate_fn=test_dataset.collator, 
                             num_workers=0, pin_memory=True, shuffle=False)

    model = BaseModel(input_dim=768, output_dim=num_classes)
    model = model.to(device)

    # count_parameters(model) # Uncomment to see model parameters
    optimizer = optim.RMSprop(model.parameters(), lr=cfg.optimization.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.optimization.lr, max_lr=1e-3, step_size_up=10)
    criterion = nn.CrossEntropyLoss()

    best_val_wa = 0
    best_val_wa_epoch = 0
    
    # Define save directory for the trained model
    save_dir_base = os.path.join(str(Path.cwd()), "esd_downstream_model")
    os.makedirs(save_dir_base, exist_ok=True)
    model_save_path = os.path.join(save_dir_base, "best_model.pth")


    for epoch in range(cfg.optimization.epoch):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        scheduler.step()
        
        val_wa, val_ua, val_f1 = validate_and_test(model, val_loader, device, num_classes=num_classes)

        if val_wa > best_val_wa:
            best_val_wa = val_wa
            best_val_wa_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saving best model with WA: {best_val_wa:.2f}% at epoch {best_val_wa_epoch + 1}")

        logger.info(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.6f}, Validation WA: {val_wa:.2f}%; UA: {val_ua:.2f}%; F1: {val_f1:.2f}%")

    # Load best model for final evaluation
    if os.path.exists(model_save_path):
        ckpt = torch.load(model_save_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        logger.info(f"Loaded best model from {model_save_path} (epoch {best_val_wa_epoch + 1}) for final evaluation.")
    else:
        logger.warning("No best model checkpoint found to load for final evaluation.")

    test_wa, test_ua, test_f1 = validate_and_test(model, test_loader, device, num_classes=num_classes)
    logger.info(f"Final Test Results: WA {test_wa:.2f}%; UA {test_ua:.2f}%; F1 {test_f1:.2f}%")
    

if __name__ == '__main__':
    # Add a basic logging configuration if not handled by Hydra default
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    train_esd()