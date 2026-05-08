# Audio Deepfake Detection

Binary classification of audio samples as **bonafide** or **deepfake** using a 2D CNN on LFCC spectral features.

## Model

The input features are pre-processed LFCC (Linear Frequency Cepstral Coefficients) including delta and delta-delta coefficients, represented as `[180, 321]` tensors (180 time frames × 321 feature dimensions). These are treated as single-channel 2D images.

**Architecture (`AudioCNN`):**
- Conv2d(1→16, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)
- Conv2d(16→32, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)
- AdaptiveAvgPool2d(4×4) → Linear(512→1)

**Training:**
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam (lr=0.001)
- Epochs: 12
- Training loss: 0.3111 → 0.0602

**Evaluation metric:** Equal Error Rate (EER) — the point where false acceptance rate equals false rejection rate. Lower is better (0% = perfect, 50% = random).

## Data Format

Features and labels are provided as `.pkl` files containing Pandas DataFrames:

| File | Columns | Notes |
|---|---|---|
| `features.pkl` | `uttid`, `features` | `features` is a `torch.Tensor` of shape `[180, 321]` |
| `labels.pkl` | `uttid`, `label` | `label` is `1` (bonafide) or `0` (deepfake) |

Expected directory structure:

```
data/
├── train/
│   ├── features.pkl
│   └── labels.pkl
└── test/
    └── features.pkl
```

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train and run inference:** Open and run `deepfake_detection.ipynb` cell by cell. Update the paths in Cell 3 to point to your data directory.

**Evaluate EER on a labeled set:**
```bash
python evaluation.py prediction.pkl data/dev/labels.pkl
```

## Requirements

See `requirements.txt`. A GPU is not required — the notebook was developed and tested on CPU only.
