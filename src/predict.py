"""
predict.py - Testing / Inference CRNN Arabic Naskh Recognition
"""

import os
import torch
import cv2
import pandas as pd
import numpy as np
import editdistance
from config import Config


# ==========================================
# 1. CHARMAP
# ==========================================
class CharMap:
    def __init__(self, char_to_id=None, id_to_char=None):
        if char_to_id is not None and id_to_char is not None:
            self.char_to_id = char_to_id
            self.id_to_char = id_to_char
            self.num_classes = len(self.char_to_id)
        else:
            self.char_to_id = {}
            self.id_to_char = {}
            self.num_classes = 0

    def decode(self, ids):
        tokens = []
        for i in range(len(ids)):
            if ids[i] != 0 and (i == 0 or ids[i] != ids[i - 1]):
                if ids[i] in self.id_to_char:
                    tokens.append(self.id_to_char[ids[i]])
        return "".join(tokens)


# ==========================================
# 2. MODEL CRNN
# ==========================================
import torch.nn as nn


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
# 3. FUNGSI METRIK
# ==========================================
def calc_metrics(gt, pred):
    cer = editdistance.eval(gt, pred) / max(1, len(gt))
    wer = editdistance.eval(gt.split(), pred.split()) / max(1, len(gt.split()))
    return cer, wer


# ==========================================
# 4. PREPROCESS & PREDICT
# ==========================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.flip(img, 1)  # RTL

    h, w = img.shape
    new_w = int((Config.IMG_HEIGHT / h) * w)
    new_w = max(1000, (new_w // 4) * 4)
    img = cv2.resize(img, (new_w, Config.IMG_HEIGHT))
    img = (img.astype(np.float32) / 127.5) - 1.0

    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def predict_single(model, char_map, image_path):
    """Predict one image"""
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return ""

    with torch.no_grad():
        img_tensor = img_tensor.to(Config.DEVICE)
        logits, _ = model(img_tensor)
        _, pred_ids = logits.max(2)
        return char_map.decode(pred_ids[:, 0].cpu().tolist())


# ==========================================
# 5. LOAD MODEL
# ==========================================
def load_model(model_path):
    """Load model from checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=Config.DEVICE)

    char_map = CharMap(checkpoint['char_to_id'], checkpoint['id_to_char'])

    model = CRNN(char_map.num_classes).to(Config.DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"✅ Model loaded: {os.path.basename(model_path)}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}" if 'val_loss' in checkpoint else "")
    print(f"   Characters: {char_map.num_classes}")

    return model, char_map


# ==========================================
# 6. LOAD DATA UNTUK TESTING
# ==========================================
def load_test_data():
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
# 7. TESTING FUNCTIONS
# ==========================================
def test_batch(model_path, num_samples=10):
    """Test random samples from dataset"""
    print("\n" + "=" * 60)
    print("BATCH TEST")
    print("=" * 60)

    model, char_map = load_model(model_path)

    df = load_test_data()
    if df.empty:
        print("No data found!")
        return

    test_df = df.sample(min(num_samples, len(df)), random_state=42)
    print(f"Testing {len(test_df)} samples...")
    print("-" * 60)

    results = []
    correct = 0

    for idx, (_, row) in enumerate(test_df.iterrows(), 1):
        img_path = row['full_path']
        gt = str(row['label'])
        pred = predict_single(model, char_map, img_path)

        if pred == "":
            continue

        cer, wer = calc_metrics(gt, pred)
        is_correct = (gt == pred)
        if is_correct:
            correct += 1

        results.append({
            'image': os.path.basename(img_path),
            'ground_truth': gt,
            'prediction': pred,
            'cer': cer,
            'wer': wer,
            'exact_match': is_correct
        })

        print(f"[{idx}/{len(test_df)}] {os.path.basename(img_path)}")
        print(f"  GT  : {gt}")
        print(f"  Pred: {pred}")
        print(f"  CER : {cer * 100:.2f}% | {'✅' if is_correct else '❌'}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_cer = np.mean([r['cer'] for r in results])
    avg_wer = np.mean([r['wer'] for r in results])
    print(f"Total: {len(results)} samples")
    print(f"Exact Match: {correct}/{len(results)} ({correct / len(results) * 100:.1f}%)")
    print(f"Avg CER: {avg_cer * 100:.2f}%")
    print(f"Avg WER: {avg_wer * 100:.2f}%")

    # Save results
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(Config.OUTPUT_DIR, "test_results.csv"), index=False)
    print(f"\n✅ Results saved to {Config.OUTPUT_DIR}/test_results.csv")

    return results


def test_single(model_path, image_path, ground_truth=None):
    """Test one image"""
    print("\n" + "=" * 60)
    print("SINGLE IMAGE TEST")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    model, char_map = load_model(model_path)
    pred = predict_single(model, char_map, image_path)

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Pred: {pred}")

    if ground_truth:
        cer, wer = calc_metrics(ground_truth, pred)
        print(f"GT  : {ground_truth}")
        print(f"CER : {cer * 100:.2f}%")
        print(f"WER : {wer * 100:.2f}%")
        print(f"Match: {'✅' if ground_truth == pred else '❌'}")

    return pred


def test_folder(model_path, folder_path):
    """Test all images in a folder"""
    print("\n" + "=" * 60)
    print("FOLDER TEST")
    print("=" * 60)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    model, char_map = load_model(model_path)

    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    print(f"Found {len(images)} images")
    print("-" * 60)

    results = []
    for idx, img_file in enumerate(images, 1):
        img_path = os.path.join(folder_path, img_file)
        pred = predict_single(model, char_map, img_path)
        results.append({'image': img_file, 'prediction': pred})
        print(f"[{idx}/{len(images)}] {img_file}: {pred}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(Config.OUTPUT_DIR, "folder_predictions.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")

    return results


# ==========================================
# 8. MAIN
# ==========================================
def main():
    Config.create_dirs()

    # Cek model
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("\nAvailable models:")
        if os.path.exists(Config.CHECKPOINT_DIR):
            for f in os.listdir(Config.CHECKPOINT_DIR):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        return

    print("=" * 60)
    print("ARABIC NASKH - TESTING TOOL")
    print("=" * 60)
    print(f"Model: {os.path.basename(model_path)}")
    print("\nModes:")
    print("  1. Batch test (random from dataset)")
    print("  2. Single image test")
    print("  3. Folder test")
    print("  4. Interactive test")
    print("  0. Exit")

    choice = input("\nSelect mode (0-4): ").strip()

    if choice == '0':
        return
    elif choice == '1':
        num = input("Number of samples (default 10): ").strip()
        num = int(num) if num else 10
        test_batch(model_path, num)
    elif choice == '2':
        img = input("Image path: ").strip()
        gt = input("Ground truth (optional): ").strip()
        test_single(model_path, img, gt if gt else None)
    elif choice == '3':
        folder = input("Folder path: ").strip()
        test_folder(model_path, folder)
    elif choice == '4':
        print("\nInteractive mode - type 'exit' to quit")
        print("-" * 40)
        model, char_map = load_model(model_path)
        while True:
            img = input("\nImage path: ").strip()
            if img.lower() == 'exit':
                break
            if not os.path.exists(img):
                print("File not found")
                continue
            pred = predict_single(model, char_map, img)
            print(f"Pred: {pred}")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()