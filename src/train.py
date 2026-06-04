import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import editdistance  # Wajib untuk CER dan WER

# ==========================================
# 1. KONFIGURASI
# ==========================================
class Config:
    BASE_PATH = "E:/SUGENG/dataset/Full_Dataset"
    DATA_SETS = ["Set1", "Set2", "Set3"]
    CHECKPOINT_DIR = "checkpoints_naskh"
    OUTPUT_DIR = "paper_results"
    
    IMG_HEIGHT = 64  
    BATCH_SIZE = 4          
    ACCUMULATION_STEPS = 8  
    EPOCHS = 50
    LR = 0.0005 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. KAMUS KARAKTER & METRIKS EVALUASI
# ==========================================
class CharMap:
    def __init__(self, df):
        chars = set()
        for label in df['label'].astype(str):
            chars.update(list(label))
        self.char_to_id = {char: idx + 1 for idx, char in enumerate(sorted(list(chars)))}
        self.char_to_id['[blank]'] = 0
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.num_classes = len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        tokens = []
        for i in range(len(ids)):
            if ids[i] != 0 and (i == 0 or ids[i] != ids[i-1]):
                if ids[i] in self.id_to_char:
                    tokens.append(self.id_to_char[ids[i]])
        return "".join(tokens)

def calculate_metrics(gt_text, pred_text):
    """Menghitung CER dan WER berbasis Levenshtein Distance"""
    # CER
    cer_dist = editdistance.eval(gt_text, pred_text)
    cer = cer_dist / max(1, len(gt_text))
    
    # WER
    gt_words = gt_text.split()
    pred_words = pred_text.split()
    wer_dist = editdistance.eval(gt_words, pred_words)
    wer = wer_dist / max(1, len(gt_words))
    
    return cer, wer

# ==========================================
# 3. DATASET & LOADING
# ==========================================
def load_naskh_data():
    all_dfs = []
    for s in Config.DATA_SETS:
        csv_path = os.path.join(Config.BASE_PATH, s, "labels.csv")
        img_dir = os.path.join(Config.BASE_PATH, s, "images")
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path, header=None, names=['id', 'filename', 'class', 'label'])
        df = df[df['class'] == 'Naskh'].dropna()
        df['full_path'] = df['filename'].apply(lambda x: os.path.join(img_dir, x))
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

class NaskhDataset(Dataset):
    def __init__(self, df, char_map):
        self.df = df.reset_index(drop=True)
        self.char_map = char_map

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['full_path'], cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        img = cv2.flip(img, 1)  # Kunci kesuksesan teks Arab (RTL ke LTR)
        
        h, w = img.shape
        new_w = int((Config.IMG_HEIGHT / h) * w)
        new_w = max(1000, (new_w // 4) * 4) 
        img = cv2.resize(img, (new_w, Config.IMG_HEIGHT))
        img = (img.astype(np.float32) / 127.5) - 1.0
        return torch.FloatTensor(img).unsqueeze(0), self.char_map.encode(str(row['label'])), str(row['label'])

def collate_fn(batch):
    batch = [b for b in batch if b is not None and len(b[1]) > 0]
    if not batch: return None
    imgs, labels, raws = zip(*batch)
    max_w = max([img.shape[2] for img in imgs])
    padded_imgs = [nn.functional.pad(img, (0, max_w - img.shape[2], 0, 0), value=1.0) for img in imgs]
    return torch.stack(padded_imgs), torch.IntTensor([i for l in labels for i in l]), torch.IntTensor([len(l) for l in labels]), raws

# ==========================================
# 4. ARSITEKTUR CRNN
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
        if h > 1: x = nn.functional.adaptive_avg_pool2d(x, (1, w))
        x = x.squeeze(2).permute(0, 2, 1) 
        seq_len = x.size(1)
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x.permute(1, 0, 2), seq_len

# ==========================================
# 5. UTAMA: PROSES TRAINING
# ==========================================
def train():
    df = load_naskh_data()
    char_map = CharMap(df)
    
    # SPLIT DATA: 90% Train, 10% Validation (Krusial untuk Jurnal Sinta 1)
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_loader = DataLoader(NaskhDataset(train_df, char_map), Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(NaskhDataset(val_df, char_map), Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CRNN(char_map.num_classes).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = GradScaler()

    # List Penyimpan Log untuk Grafik
    history = {'train_loss': [], 'val_loss': [], 'val_cer': [], 'val_wer': []}
    best_val_loss = float('inf')

    print(f"Dataset Split: Train={len(train_df)} | Val={len(val_df)}")

    for epoch in range(Config.EPOCHS):
        # --- PHASE 1: TRAINING ---
        model.train()
        t_loss, t_count = 0, 0
        optimizer.zero_grad()
        scaled_loss_accumulated = False 

        for i, batch in enumerate(train_loader):
            if batch is None: continue
            imgs, targets, tg_lens, _ = batch
            imgs, targets = imgs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            with autocast():
                logits, seq_len = model(imgs)
                input_lengths = torch.full((imgs.size(0),), seq_len, dtype=torch.int32).to(Config.DEVICE)
                loss = criterion(logits.log_softmax(2), targets, input_lengths, tg_lens)

            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss / Config.ACCUMULATION_STEPS).backward()
                t_loss += loss.item()
                t_count += 1
                scaled_loss_accumulated = True

            if (i + 1) % Config.ACCUMULATION_STEPS == 0 and scaled_loss_accumulated:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scaled_loss_accumulated = False 

        avg_train_loss = t_loss / max(1, t_count)

        # --- PHASE 2: VALIDATION & METRIC EVALUATION ---
        model.eval()
        v_loss, v_count = 0, 0
        total_cer, total_wer, total_samples = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                imgs, targets, tg_lens, raws = batch
                imgs = imgs.to(Config.DEVICE)
                logits, seq_len = model(imgs)
                input_lengths = torch.full((imgs.size(0),), seq_len, dtype=torch.int32).to(Config.DEVICE)
                
                loss = criterion(logits.log_softmax(2), targets.to(Config.DEVICE), input_lengths, tg_lens)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    v_loss += loss.item()
                    v_count += 1
                
                # Hitung CER/WER untuk Batch Validasi ini
                _, p_idx = logits.max(2)
                p_idx = p_idx.transpose(1, 0)
                for b in range(p_idx.size(0)):
                    pred_text = char_map.decode(p_idx[b].cpu().tolist())
                    gt_text = raws[b]
                    cer, wer = calculate_metrics(gt_text, pred_text)
                    total_cer += cer
                    total_wer += wer
                    total_samples += 1

        avg_val_loss = v_loss / max(1, v_count)
        avg_cer = total_cer / max(1, total_samples)
        avg_wer = total_wer / max(1, total_samples)

        # Simpan History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_cer'].append(avg_cer)
        history['val_wer'].append(avg_wer)

        # Cetak Progress Struktural
        print(f"\n[Epoch {epoch+1}/{Config.EPOCHS}]")
        print(f"Loss Latih: {avg_train_loss:.4f} | Loss Validasi: {avg_val_loss:.4f}")
        print(f"Metrik Val: CER = {avg_cer*100:.2f}% | WER = {avg_wer*100:.2f}%")
        print(f"Sampel GT  : {raws[0]}")
        print(f"Sampel Pred: {char_map.decode(p_idx[0].cpu().tolist())}")
        print("-" * 50)

        # SAVE CHECKPOINT TERBAIK
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Simpan Weights beserta metadata kamus karakter agar portabel saat inferensi
            checkpoint = {
                'model_state': model.state_dict(),
                'char_to_id': char_map.char_to_id,
                'id_to_char': char_map.id_to_char
            }
            torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            print("=> Model Terbaik Disimpan Berdasarkan Loss Validasi Terendah.")

    # Simpan data history ke CSV untuk backup data kuantitatif paper
    pd.DataFrame(history).to_csv(os.path.join(Config.OUTPUT_DIR, "training_history.csv"), index=False)
    print("\nTraining Selesai. Seluruh Log Data Telah Diekspor.")

if __name__ == "__main__":
    train()
