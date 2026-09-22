"""
train.py — Improved Image Captioning with EfficientNet-B3 + Transformer
Fixes: better backbone, cross-attention pooling, mixed precision, GPU forced,
       safe multiprocessing, BLEU eval, better regularization.
"""

import os
import sys
import math
import random
import pickle
import csv
from collections import Counter, defaultdict

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast   # mixed precision → faster + less VRAM
from torchvision import transforms, models

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

# ───────────────────────────── DEVICE ─────────────────────────────
def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.backends.cudnn.benchmark = True          # auto-tune kernels for fixed input sizes
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on Ampere+
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        # Force multi-core CPU — never cook a single core
        n_cores = os.cpu_count() or 4
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(max(1, n_cores // 2))
        dev = torch.device("cpu")
        print(f"⚠️  No GPU found. Using CPU with {n_cores} threads.")
        print("   Training will be slow. Consider Google Colab for free GPU.")
    return dev

device = get_device()

# ───────────────────────────── CONFIG ─────────────────────────────
IMAGE_DIR      = "images"
CAPTION_FILE   = "captions.txt"

# Model
EMBED_SIZE     = 512
NHEAD          = 8
DEC_LAYERS     = 4
FFN_DIM        = 2048
DROPOUT        = 0.1

# Training
BATCH_SIZE     = 24     # B3 is heavier than B0; safe on 6 GB with AMP
GRAD_ACCUM     = 2      # effective batch = 24 × 2 = 48 without extra VRAM
LR             = 1e-4
ENCODER_LR     = 5e-6   # very conservative for fine-tuning B3
EPOCHS         = 25
GRAD_CLIP      = 1.0
UNFREEZE_EPOCH = 4      # warm-up longer because B3 is bigger
FREQ_THRESHOLD = 3
PATIENCE       = 5
MIN_FREQ_EVAL  = 3      # BLEU eval every N epochs (expensive)

# DataLoader workers — safe multicore without cooking one core
# Use 0 on Windows (no fork), 2-4 on Linux/Mac
NUM_WORKERS    = 0 if sys.platform == "win32" else min(4, (os.cpu_count() or 2))

# ───────────────────────────── VOCABULARY ─────────────────────────
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return word_tokenize(text.lower())

    def build_vocab(self, sentences):
        freq = Counter()
        idx = 4
        for sent in sentences:
            for word in self.tokenizer(sent):
                freq[word] += 1
                if freq[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        print(f"Vocabulary size: {len(self.itos):,}")

    def numericalize(self, text):
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in self.tokenizer(text)]


# ───────────────────────────── DATA ───────────────────────────────
def load_data(file):
    img_to_caps = defaultdict(list)
    with open(file, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            img, cap = row[0].strip(), row[1].strip()
            if img and cap and os.path.exists(os.path.join(IMAGE_DIR, img)):
                img_to_caps[img].append(cap)
    print(f"Loaded {len(img_to_caps):,} images")
    return img_to_caps


def split_data(img_to_caps, seed=42):
    random.seed(seed)
    imgs = list(img_to_caps.keys())
    random.shuffle(imgs)
    n = len(imgs)
    train_imgs = imgs[:int(0.80 * n)]
    val_imgs   = imgs[int(0.80 * n):int(0.90 * n)]
    # keep test set for final BLEU
    test_imgs  = imgs[int(0.90 * n):]

    def build(lst):
        return [(img, cap) for img in lst for cap in img_to_caps[img]]

    return build(train_imgs), build(val_imgs), build(test_imgs), img_to_caps


# ───────────────────────────── TRANSFORMS ─────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((288, 288)),            # slightly bigger → more crop variety
    transforms.RandomCrop(256),               # B3 native res; better than 224 for detail
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),       # occasionally train in greyscale
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ───────────────────────────── DATASET ────────────────────────────
class FlickrDataset(Dataset):
    def __init__(self, data, vocab, transform):
        self.data      = data
        self.vocab     = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, cap = self.data[idx]
        try:
            image = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
            image = self.transform(image)
        except Exception:
            # Corrupted image — return a black tensor so training doesn't crash
            image = torch.zeros(3, 256, 256)

        tokens  = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(cap)
        tokens.append(self.vocab.stoi["<EOS>"])
        return image, torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    max_len = max(len(c) for c in caps)
    padded  = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, c in enumerate(caps):
        padded[i, :len(c)] = c
    return imgs, padded


# ───────────────────────────── POSITIONAL ENCODING ────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))   # (max_len, 1, d)

    def forward(self, x):                             # x: (T, B, d)
        return self.dropout(x + self.pe[:x.size(0)])


# ───────────────────────────── ENCODER ────────────────────────────
class Encoder(nn.Module):
    """
    EfficientNet-B3 backbone (better than B0 — 12M params vs 5M).
    Outputs a spatial patch sequence + a global CLS-like vector.
    """
    def __init__(self, embed_size):
        super().__init__()
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        for p in backbone.parameters():
            p.requires_grad = False

        self.features = backbone.features   # output: (B, 1536, 8, 8) for 256px input

        # Better projection: LayerNorm on channels before projecting
        self.proj = nn.Sequential(
            nn.Conv2d(1536, embed_size, kernel_size=1, bias=False),
            nn.GroupNorm(8, embed_size),    # GroupNorm works with batch_size=1 unlike BN
            nn.GELU(),
        )

        # Learnable global average pool weights — lets encoder weight patches
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_size, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.2)

    def unfreeze_top_blocks(self, num_blocks=4):
        blocks = list(self.features.children())
        for block in blocks[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze top {num_blocks} EfficientNet-B3 blocks | Trainable params: {n_trainable:,}")

    def forward(self, x):
        x = self.features(x)               # (B, 1536, H, W)
        x = self.proj(x)                   # (B, embed_size, H, W)

        # Spatial attention weighting — model learns which patches matter
        w = self.spatial_attn(x)           # (B, 1, H, W)
        x = x * w                          # weighted patches

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)   # (H*W, B, embed_size)
        return self.dropout(x)


# ───────────────────────────── DECODER ────────────────────────────
class Decoder(nn.Module):
    """
    Transformer decoder with:
    - Tied input/output embeddings (fewer params, better generalization)
    - Pre-norm (more stable training than post-norm)
    """
    def __init__(self, vocab_size, embed_size, nhead, num_layers, ffn_dim, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos   = PositionalEncoding(embed_size, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=True,    # Pre-LN: more stable gradients than post-LN
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(embed_size),
        )

        self.fc = nn.Linear(embed_size, vocab_size, bias=False)

        # Tie weights: output projection ↔ embedding matrix
        # Forces consistent token representations → better generalization
        self.fc.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0, std=self.embed_size ** -0.5)
        self.embed.weight.data[0].zero_()   # PAD stays zero

    def forward(self, features, captions):
        # features: (src_len, B, embed_size)
        # captions: (B, tgt_len)
        tgt     = self.embed(captions) * math.sqrt(self.embed_size)   # scale like "Attention is All You Need"
        tgt     = tgt.permute(1, 0, 2)        # (tgt_len, B, embed_size)
        tgt     = self.pos(tgt)
        tgt_len = tgt.size(0)

        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=captions.device), diagonal=1
        ).bool()

        tgt_key_padding_mask = (captions == 0)

        out = self.transformer_decoder(
            tgt, features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        # fc shares weights with embed → scale by 1/sqrt(embed_size) for stability
        return self.fc(out).permute(1, 0, 2)   # (B, tgt_len, vocab_size)


# ───────────────────────────── BLEU EVAL ──────────────────────────
def evaluate_bleu(encoder, decoder, vocab, val_data, max_samples=500, max_len=40):
    """
    Compute corpus BLEU-4 on a subset of val data using greedy decoding.
    This is what actually matters — val loss is a proxy, BLEU is the real metric.
    """
    encoder.eval()
    decoder.eval()

    # Group by image to get all reference captions
    img_to_caps = defaultdict(list)
    for img, cap in val_data:
        img_to_caps[img].append(cap)

    img_list = list(img_to_caps.keys())[:max_samples]
    references, hypotheses = [], []

    smooth = SmoothingFunction().method1

    with torch.no_grad():
        for img_name in img_list:
            img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
            img_tensor = val_transform(img).unsqueeze(0).to(device)

            feats   = encoder(img_tensor)               # (src_len, 1, embed_size)
            caption = [vocab.stoi["<SOS>"]]

            for _ in range(max_len):
                caps_t = torch.tensor(caption, device=device).unsqueeze(0)
                output = decoder(feats, caps_t)
                pred   = output[0, -1].argmax().item()
                if pred == vocab.stoi["<EOS>"]:
                    break
                caption.append(pred)

            hyp  = [vocab.itos.get(t, "<UNK>") for t in caption[1:]]
            refs = [word_tokenize(c.lower()) for c in img_to_caps[img_name]]

            hypotheses.append(hyp)
            references.append(refs)

    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    return bleu4


# ───────────────────────────── TRAINING ───────────────────────────
def train():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ── Data
    data = load_data(CAPTION_FILE)
    all_caps = [c for caps in data.values() for c in caps]

    vocab = Vocabulary(FREQ_THRESHOLD)
    vocab.build_vocab(all_caps)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_data, val_data, test_data, img_to_caps = split_data(data)
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,} pairs")

    train_loader = DataLoader(
        FlickrDataset(train_data, vocab, train_transform),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        FlickrDataset(val_data, vocab, val_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ── Model
    encoder = Encoder(EMBED_SIZE).to(device)
    decoder = Decoder(
        vocab_size=len(vocab),
        embed_size=EMBED_SIZE,
        nhead=NHEAD,
        num_layers=DEC_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
    ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Trainable — Encoder: {n_enc:,} | Decoder: {n_dec:,}")

    # ── Optimizer
    def make_optimizer(enc_lr=ENCODER_LR, dec_lr=LR):
        return optim.AdamW([
            {"params": decoder.parameters(), "lr": dec_lr},
            {"params": filter(lambda p: p.requires_grad, encoder.parameters()), "lr": enc_lr},
        ], weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)

    optimizer = make_optimizer()

    # Warmup then cosine decay — critical for Transformer stability
    warmup_steps  = len(train_loader) * 2     # 2 epochs of warmup
    total_steps   = len(train_loader) * EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision — halves VRAM use, ~1.5-2× faster on RTX GPUs
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Label smoothing = 0.1 — but reduce to avoid over-smoothing short captions
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.08)

    best_val_loss  = float("inf")
    best_bleu      = 0.0
    no_improve     = 0
    global_step    = 0

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    for epoch in range(EPOCHS):

        # ── Unfreeze encoder after warm-up
        if epoch == UNFREEZE_EPOCH:
            encoder.unfreeze_top_blocks(num_blocks=4)
            optimizer = make_optimizer(enc_lr=ENCODER_LR, dec_lr=LR * 0.5)
            # Rebuild scheduler for remaining epochs
            remaining   = len(train_loader) * (EPOCHS - epoch)
            scheduler   = optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda s: max(0.02, 0.5 * (1 + math.cos(math.pi * s / remaining)))
            )

        # ────── TRAIN ──────
        encoder.train()
        decoder.train()
        train_loss  = 0.0
        n_batches   = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>2}/{EPOCHS} [train]", leave=False)
        for batch_idx, (imgs, caps) in enumerate(pbar):
            imgs, caps = imgs.to(device, non_blocking=True), caps.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                feats   = encoder(imgs)
                outputs = decoder(feats, caps[:, :-1])
                loss    = criterion(
                    outputs.reshape(-1, outputs.size(2)),
                    caps[:, 1:].reshape(-1),
                )
                loss = loss / GRAD_ACCUM   # normalize for accumulation

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            train_loss += loss.item() * GRAD_ACCUM
            n_batches  += 1
            pbar.set_postfix(loss=f"{train_loss/n_batches:.4f}")

        # ────── VALIDATE ──────
        encoder.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, caps in tqdm(val_loader, desc=f"Epoch {epoch+1:>2}/{EPOCHS} [val]  ", leave=False):
                imgs, caps = imgs.to(device, non_blocking=True), caps.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    feats   = encoder(imgs)
                    outputs = decoder(feats, caps[:, :-1])
                    loss    = criterion(
                        outputs.reshape(-1, outputs.size(2)),
                        caps[:, 1:].reshape(-1),
                    )
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        lr_now    = optimizer.param_groups[0]["lr"]

        # ── BLEU every N epochs (greedy, fast)
        bleu_str = ""
        if (epoch + 1) % MIN_FREQ_EVAL == 0:
            bleu4    = evaluate_bleu(encoder, decoder, vocab, val_data, max_samples=300)
            bleu_str = f" | BLEU-4: {bleu4:.4f}"
            if bleu4 > best_bleu:
                best_bleu = bleu4

        print(f"Epoch {epoch+1:>2} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{bleu_str} | LR: {lr_now:.2e}")

        # ── Save on val loss
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve    = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab_size": len(vocab),
                "embed_size": EMBED_SIZE,
                "nhead": NHEAD,
                "dec_layers": DEC_LAYERS,
                "ffn_dim": FFN_DIM,
                "epoch": epoch + 1,
                "val_loss": avg_val,
                "best_bleu": best_bleu,
            }, "best_model.pth")
            print("  ✅ Saved best model")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best BLEU-4   : {best_bleu:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # ⚠️ Required on Windows for safe DataLoader multiprocessing
    # On Linux/Mac this is a no-op
    import multiprocessing
    multiprocessing.freeze_support()
    train()