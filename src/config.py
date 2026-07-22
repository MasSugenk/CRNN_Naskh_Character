"""
config.py - Konfigurasi untuk CRNN Arabic Naskh Recognition
"""

import os
import torch


class Config:
    # ==========================================
    # PATH DATASET
    # ==========================================
    BASE_PATH = "E:/SUGENG/dataset/Full_Dataset"
    DATA_SETS = ["Set1", "Set2", "Set3"]

    # ==========================================
    # PATH MODEL & OUTPUT
    # ==========================================
    CHECKPOINT_DIR = "checkpoints_naskh"
    OUTPUT_DIR = "paper_results"

    # ==========================================
    # MODEL PARAMETERS
    # ==========================================
    IMG_HEIGHT = 64
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 8
    EPOCHS = 50
    LR = 0.0005
    WEIGHT_DECAY = 1e-4

    # ==========================================
    # DEVICE
    # ==========================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def create_dirs(cls):
        """Buat direktori yang diperlukan"""
        for d in [cls.CHECKPOINT_DIR, cls.OUTPUT_DIR]:
            os.makedirs(d, exist_ok=True)