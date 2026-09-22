# CaptionLens 🔭 (Very-bad-IC)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-Supported-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning Image Captioning pipeline and interactive web application combining an **EfficientNet-B3** convolutional feature extractor with a **Pre-LN Transformer Decoder** and **Beam Search** decoding.

---

## 📌 Overview

**CaptionLens** (repository `Very-bad-IC`) addresses the limitations of classical CNN-LSTM image captioning systems (e.g., vanishing gradients, repetitive text loops, and lack of spatial grounding). 

By pairing a convolutional backbone with a multi-head self-attention sequence model, the system learns spatial patch correlations and generates fluent, descriptive English captions from raw images.

---

## 🏗 Architecture

```
   Raw Input Image [3 x 256 x 256]
                 │
                 ▼
    ┌──────────────────────────┐
    │     EfficientNet-B3      │  Pretrained ImageNet weights; top 4 blocks
    │   Convolutional Stages   │  unfrozen during progressive fine-tuning
    └────────────┬─────────────┘
                 │ Feature map: [B, 1536, 8, 8]
                 ▼
    ┌──────────────────────────┐
    │ 1x1 Conv + GroupNorm(8)  │  Channel projection down to
    │         + GELU           │  embedding dimension (d_model = 512)
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │ Learnable Spatial Attn   │  Learns 2D spatial importance mask [B, 1, 8, 8]
    │  Conv(512->1) + Sigmoid  │  via element-wise feature scaling
    └────────────┬─────────────┘
                 │
                 ▼
    Visual Memory Keys/Values: [64, Batch, 512]
                 │
                 │  Cross-Attention
                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 Transformer Decoder (4 Layers)              │
    │  - Target Self-Attention with Causal Upper-Triangular Mask   │
    │  - Multi-Head Cross-Attention over Visual Patches (8 Heads) │
    │  - Position-wise Feed-Forward Network (d_ff = 2048)         │
    │  - Pre-Layer Normalization (Pre-LN)                         │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Tied Output Linear Projection Head (d_model -> Vocab:10049)│
    └────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                     Next Token Logits / Probabilities
```

### Key Highlights
- **Encoder**: EfficientNet-B3 backbone (~12M params) extracting 64 spatial feature patches ($8 \times 8 \times 1536$), projected down to $d_{\text{model}} = 512$ with `GroupNorm` and a learnable spatial gating mechanism.
- **Decoder**: 4-layer, 8-head Transformer Decoder using **Pre-Layer Normalization (`norm_first=True`)** for gradient stability and **Weight Tying** between the token embedding matrix and classification head.
- **Beam Search Decoding**: Implements $K$-beam expansion with dynamic **Repetition Penalty** ($\gamma = 1.3$) and polynomial **Length Normalization** ($\alpha = 0.7$) to eliminate repetitive and truncated phrases.
- **Interactive UI**: Streamlit web dashboard featuring dark-mode styling, real-time hyperparameter sliders, confidence bar proxies, and ranked alternative beam candidates.

---

## ⚙️ Model Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| **Embedding Size (`EMBED_SIZE`)** | `512` | Latent representation dimension |
| **Attention Heads (`NHEAD`)** | `8` | Multi-head attention heads |
| **Decoder Layers (`DEC_LAYERS`)** | `4` | Transformer decoder blocks |
| **Feed-Forward Dimension (`FFN_DIM`)** | `2048` | Intermediate dimension in FFN |
| **Vocabulary Size** | `10,049` | Minimum word frequency threshold $\ge 3$ |
| **Batch Size & Accumulation** | `24 × 2 = 48` | Gradient accumulation factor of 2 |
| **Optimization** | AdamW | Weight decay $10^{-4}$, $\beta = (0.9, 0.98)$ |
| **Label Smoothing** | `0.08` | Prevents overconfident Dirichlet peaks |
| **Best Val Loss** | `3.9197` | Checkpointed at epoch 3 |
| **Corpus BLEU-4** | `13.96%` | Validation metric with Chen & Cherry smoothing |

---

## 📂 Repository Structure

```
.
├── app.py             # Streamlit web interface with interactive beam search
├── train.py           # Model definitions, dataset loader, and AMP training loop
├── vocab.pkl          # Serialized Vocabulary object (10,049 tokens)
├── captions.txt       # Dataset annotations mapping image filenames to text captions
├── images/            # Directory of sample images
├── .gitignore         # Git ignore rules (includes best_model.pth)
└── README.md          # Project documentation
```

> **Note:** The trained checkpoint `best_model.pth` (~135 MB) is gitignored to keep the repository lightweight. You can train your own or download it as described below.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Dev-coder21/Very-bad-IC.git
cd Very-bad-IC
```

Create a virtual environment (Python 3.10 or 3.11 recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install torch torchvision streamlit nltk pillow tqdm numpy
```

Download the required NLTK tokenizer resources:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

### 2. Model Checkpoint

Ensure `best_model.pth` is placed in the project root directory alongside `vocab.pkl`. 

If you are training from scratch, running the training script will automatically create `vocab.pkl` and save `best_model.pth` upon achieving improved validation loss.

---

### 3. Running the Web Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Or run headlessly on a custom port:

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Open your browser at **`http://localhost:8501`**.

#### Features in the UI:
- **Image Upload**: Supports JPG, PNG, and WebP.
- **Beam Width Slider**: Adjust search breadth ($1$ to $10$).
- **Max Length Slider**: Control maximum caption length ($10$ to $80$ tokens).
- **Length Normalization ($\alpha$)**: Adjust preference for longer vs shorter sentences ($0.0$ to $1.0$).
- **Repetition Penalty**: Penalize duplicate token generation ($1.0$ to $3.0$).
- **Diagnostics**: Real-time confidence score and ranked alternative beam candidates.

---

### 4. Training the Model

To train the model on the full caption dataset:

```bash
python train.py
```

The training pipeline includes:
- **Automatic Mixed Precision (AMP)** via `torch.cuda.amp.autocast`.
- **Progressive Unfreezing**: EfficientNet-B3 backbone is frozen for the first 3 epochs and unfreezes its top 4 blocks starting at Epoch 4 with differential learning rates ($\eta_{\text{enc}} = 5 \times 10^{-6}$, $\eta_{\text{dec}} = 5 \times 10^{-5}$).
- **Cosine Annealing Scheduler** with 2 epochs of initial linear warmup.
- **Corpus BLEU-4 Evaluation** on validation sets every 3 epochs.

---

## 📊 Evaluation & Metrics

The model is evaluated using **Corpus BLEU-4** (Bilingual Evaluation Understudy) with Chen & Cherry smoothing method 1 against multiple ground-truth reference sentences per image.

$$\text{BLEU-4} = \text{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
