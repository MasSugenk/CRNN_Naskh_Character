# CRNN_Naskh_Character
Implementation of Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) Loss for Arabic Naskh from HICMA dataset.

Implementation of Convolutional Recurrent Neural Network (CRNN) combined with Connectionist Temporal Classification (CTC) Loss for Arabic Naskh handwritten text recognition. This repository focuses on analyzing the impact of horizontal spatial resolution and gradient accumulation steps on stabilizing CTC Loss and preventing numerical instability (`NaN` issues) during training on long Arabic sequences.

## 🚀 Key Features
- **Spatial Resolution Optimization:** Specifically tailored for handling long cursive Arabic Naskh scripts by preventing aggressive horizontal pooling.
- **Gradient Accumulation Integration:** Simulates large batch sizes to stabilize gradient estimation under resource-constrained environments (VRAM limitations).
- **Automated Metric Evaluation:** Real-time computation of Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein Distance during validation phases.

## 📁 Repository Structure
```text
├── src/
│   ├── train.py          # Main training loop with dataset splitting and metric evaluation
│   ├── plot_metrics.py   # High-resolution (300 DPI) visualization generator for paper publication
│   └── inference.py      # Independent inference script for single-image testing
└── paper_results/        # Output directory for loss curves and error rate plots
