"""
train.py - Training dan Validation CRNN Arabic Naskh Recognition
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import cv2
import editdistance
from config import Config


# ==========================================
# 1. CHARMAP
# ==========================================
class CharMap:
    def __init__(self, df=None):
        if df is not None:
            chars = set()
            for label in df['label'].astype(str):
                chars.update(list(label))
            self.char_to_id = {c: i + 1 for i, c in enumerate(sorted(chars))}
            self.char_to_id['[blank]'] = 0
            self.id_to_char = {i: c for c, i in self.char_to_id.items()}
            self.num_classes = len(self.char_to_id)
        else:
            self.char_to_id = {}
            self.id_to_char = {}
            self.num_classes = 0

    def encode(self, text):
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        tokens = []
        for i in range(len(ids)):
            if ids[i] != 0 and (i == 0 or ids[i] != ids[i - 1]):
                if ids[i] in self.id_to_char:
                    tokens.append(self.id_to_char[ids[i]])
        return "".join(tokens)


# ==========================================
# 2. DATASET
# ==========================================
class NaskhDataset(Dataset):
    def __init__(self, df, char_map):
        self.df = df.reset_index(drop=True)
        self.char_map = char_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['full_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Flip untuk RTL
        img = cv2.flip(img, 1)

        # Resize
        h, w = img.shape
        new_w = int((Config.IMG_HEIGHT / h) * w)
        new_w = max(1000, (new_w // 4) * 4)
        img = cv2.resize(img, (new_w, Config.IMG_HEIGHT))
        img = (img.astype(np.float32) / 127.5) - 1.0

        label = str(row['label'])
        return torch.FloatTensor(img).unsqueeze(0), self.char_map.encode(label), label


def collate_fn(batch):
    batch = [b for b in batch if b is not None and len(b[1]) > 0]
    if not batch:
        return None

    imgs, labels, raws = zip(*batch)
    max_w = max([img.shape[2] for img in imgs])
    padded = [nn.functional.pad(img, (0, max_w - img.shape[2], 0, 0), value=1.0) for img in imgs]

    return (torch.stack(padded),
            torch.IntTensor([i for l in labels for i in l]),
            torch.IntTensor([len(l) for l in labels]),
            raws)


def load_data():
    """Load data from all sets"""
    all_dfs = []
    for s in Config.DATA_SETS:
        csv_path = os.path.join(Config.BASE_PATH, s, "labels.csv")
        img_dir = os.path.join(Config.BASE_PATH, s, "images")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, header=None, names=['id', 'filename', 'class', 'label'])
        df = df[df['class'] == 'Naskh'].dropna()
        df['full_path'] = df['filename'].apply(lambda x: os.path.join(img_dir, x))
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ==========================================
# 3. MODEL CRNN
# ==========================================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, (0, 1)), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True, num_layers=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        if h > 1:
            x = nn.functional.adaptive_avg_pool2d(x, (1, w))
        x = x.squeeze(2).permute(0, 2, 1)
        seq_len = x.size(1)
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x.permute(1, 0, 2), seq_len


# ==========================================
# 4. METRIK
# ==========================================
def calc_metrics(gt, pred):
    cer = editdistance.eval(gt, pred) / max(1, len(gt))
    wer = editdistance.eval(gt.split(), pred.split()) / max(1, len(gt.split()))
    return cer, wer


# ==========================================
# 5. TRAINING
# ==========================================
def train():
    Config.create_dirs()

    print("=" * 60)
    print("TRAINING CRNN - ARABIC NASKH")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")

    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    if df.empty:
        print("Error: No data found!")
        return

    char_map = CharMap(df)
    print(f"Total samples: {len(df)}")
    print(f"Characters: {char_map.num_classes}")

    # Split data
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Dataloaders
    train_loader = DataLoader(
        NaskhDataset(train_df, char_map),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        NaskhDataset(val_df, char_map),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )

    # Model
    model = CRNN(char_map.num_classes).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = GradScaler()

    # History
    history = {'train_loss': [], 'val_loss': [], 'val_cer': [], 'val_wer': []}
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(Config.EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        scaled_loss_accumulated = False

        for i, batch in enumerate(train_loader):
            if batch is None:
                continue

            imgs, targets, tg_lens, _ = batch
            imgs, targets = imgs.to(Config.DEVICE), targets.to(Config.DEVICE)

            with autocast():
                logits, seq_len = model(imgs)
                input_lengths = torch.full((imgs.size(0),), seq_len, dtype=torch.long).to(Config.DEVICE)
                loss = criterion(logits.log_softmax(2), targets, input_lengths, tg_lens)

            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss / Config.ACCUMULATION_STEPS).backward()
                train_loss += loss.item()
                num_batches += 1
                scaled_loss_accumulated = True

            if (i + 1) % Config.ACCUMULATION_STEPS == 0 and scaled_loss_accumulated:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scaled_loss_accumulated = False

        avg_train_loss = train_loss / max(1, num_batches)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        num_val_batches = 0
        total_cer, total_wer, total_samples = 0, 0, 0
        val_gt, val_pred = "", ""

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                v_imgs, v_targets, v_tg_lens, v_raws = batch
                v_imgs = v_imgs.to(Config.DEVICE)

                v_logits, v_seq_len = model(v_imgs)
                v_input_lengths = torch.full((v_imgs.size(0),), v_seq_len, dtype=torch.long).to(Config.DEVICE)

                loss = criterion(v_logits.log_softmax(2), v_targets.to(Config.DEVICE),
                                 v_input_lengths, v_tg_lens)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    num_val_batches += 1

                _, p_idx = v_logits.max(2)
                for b in range(v_imgs.size(0)):
                    pred_text = char_map.decode(p_idx[:, b].cpu().tolist())
                    gt_text = v_raws[b]

                    if total_samples == 0:
                        val_gt, val_pred = gt_text, pred_text

                    cer, wer = calc_metrics(gt_text, pred_text)
                    total_cer += cer
                    total_wer += wer
                    total_samples += 1

        avg_val_loss = val_loss / max(1, num_val_batches)
        avg_cer = total_cer / max(1, total_samples)
        avg_wer = total_wer / max(1, total_samples)

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_cer'].append(avg_cer)
        history['val_wer'].append(avg_wer)

        # Print
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Val CER:    {avg_cer * 100:.2f}%")
        print(f"  Val WER:    {avg_wer * 100:.2f}%")
        print(f"  Sample GT : {val_gt}")
        print(f"  Sample Pred: {val_pred}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state': model.state_dict(),
                'char_to_id': char_map.char_to_id,
                'id_to_char': char_map.id_to_char,
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_cer': avg_cer,
                'val_wer': avg_wer
            }
            torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ✅ Best model saved!")

    # Save final
    torch.save({
        'model_state': model.state_dict(),
        'char_to_id': char_map.char_to_id,
        'id_to_char': char_map.id_to_char,
        'history': history
    }, os.path.join(Config.CHECKPOINT_DIR, "final_model.pth"))

    # Save history
    pd.DataFrame(history).to_csv(os.path.join(Config.OUTPUT_DIR, "training_history.csv"), index=False)
    print(f"\n✅ Training completed! History saved to {Config.OUTPUT_DIR}/training_history.csv")


if __name__ == "__main__":
    train()
