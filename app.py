"""
app.py — CaptionLens Streamlit UI
Compatible with the improved train.py (EfficientNet-B3 + Transformer).
"""

import streamlit as st
import torch
import pickle
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
from nltk.tokenize import word_tokenize


from train import (
    Encoder, Decoder,
    EMBED_SIZE, NHEAD, DEC_LAYERS, FFN_DIM,
    IMAGENET_MEAN, IMAGENET_STD,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="CaptionLens", page_icon="🔭", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:ital,wght@0,400;1,400&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0d0d0d; color: #f0ece4; }
.stApp {
    background: radial-gradient(ellipse at 20% 50%, rgba(255,100,30,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(255,210,80,0.06) 0%, transparent 50%), #0d0d0d;
}
h1 { font-family:'Syne',sans-serif!important; font-weight:800!important; font-size:3rem!important;
     background:linear-gradient(90deg,#ff6e1f,#ffd250); -webkit-background-clip:text;
     -webkit-text-fill-color:transparent; background-clip:text; }
.subtitle { font-family:'DM Mono',monospace; font-size:.78rem; color:#555; letter-spacing:3px; text-transform:uppercase; margin-bottom:2rem; }
[data-testid="stFileUploader"] { border:1.5px dashed #333!important; border-radius:12px!important; background:rgba(255,255,255,.02)!important; }
[data-testid="stImage"] img { border-radius:10px; border:1px solid #222; }
.caption-box { background:rgba(255,110,31,.07); border:1px solid rgba(255,110,31,.25); border-left:3px solid #ff6e1f;
               border-radius:8px; padding:1.2rem 1.5rem; margin-top:.5rem; font-size:1.15rem; font-weight:700; color:#f0ece4; line-height:1.6; }
.caption-label { font-family:'DM Mono',monospace; font-size:.7rem; color:#ff6e1f; letter-spacing:2px;
                 text-transform:uppercase; margin-bottom:.3rem; margin-top:1rem; }
.alt-box { background:rgba(255,255,255,.02); border:1px solid #1e1e1e; border-radius:8px; padding:.7rem 1.1rem;
           margin-top:.4rem; font-family:'DM Mono',monospace; font-size:.8rem; color:#999;
           display:flex; justify-content:space-between; }
.score-pill { background:#151515; border:1px solid #2a2a2a; border-radius:20px; padding:2px 10px; font-size:.7rem; color:#444; }
.device-badge { display:inline-block; font-family:'DM Mono',monospace; font-size:.7rem; background:#151515;
                border:1px solid #222; border-radius:20px; padding:3px 12px; color:#555; letter-spacing:1px; }
.conf-bar-track { background:#1a1a1a; border-radius:4px; height:4px; width:100%; margin-top:.8rem; overflow:hidden; }
.conf-bar-fill { height:4px; border-radius:4px; background:linear-gradient(90deg,#ff6e1f,#ffd250); }
hr { border-color:#1e1e1e!important; }
</style>
""", unsafe_allow_html=True)

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


# ── Load model
@st.cache_resource(show_spinner=False)
def load_model():
    ckpt = torch.load("best_model.pth", map_location=device)
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    embed  = ckpt.get("embed_size",  EMBED_SIZE)
    nhead  = ckpt.get("nhead",       NHEAD)
    layers = ckpt.get("dec_layers",  DEC_LAYERS)
    ffn    = ckpt.get("ffn_dim",     FFN_DIM)
    enc = Encoder(embed).to(device)
    dec = Decoder(len(vocab), embed, nhead, layers, ffn).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    return enc, dec, vocab, ckpt

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@torch.no_grad()
def beam_search(enc, dec, vocab, image, beam_size=5, max_len=50, alpha=0.7, rep_penalty=1.3):
    img_t    = val_transform(image).unsqueeze(0).to(device)
    features = enc(img_t)
    sos, eos = vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]
    beams, completed = [([sos], 0.0)], []

    for _ in range(max_len):
        if not beams:
            break
        n      = len(beams)
        max_t  = max(len(b[0]) for b in beams)
        padded = torch.zeros(n, max_t, dtype=torch.long, device=device)
        for i, (seq, _) in enumerate(beams):
            padded[i, :len(seq)] = torch.tensor(seq, device=device)

        feat_b = features.expand(-1, n, -1)
        out    = dec(feat_b, padded)
        logits = out[:, -1, :].clone()

        candidates = []
        for i, (seq, score) in enumerate(beams):
            lg = logits[i].clone()
            for tok in set(seq):
                lg[tok] /= rep_penalty
            lp = F.log_softmax(lg, dim=-1)
            topk_v, topk_i = torch.topk(lp, beam_size)
            for k in range(beam_size):
                tok, ns = topk_i[k].item(), score + topk_v[k].item()
                new_seq  = seq + [tok]
                if tok == eos:
                    completed.append((new_seq, ns / (len(new_seq) ** alpha)))
                else:
                    candidates.append((new_seq, ns))

        beams = sorted(candidates, key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)[:beam_size]

    if not completed:
        completed = [(s, sc / (len(s) ** alpha)) for s, sc in beams]
    completed = sorted(completed, key=lambda x: x[1], reverse=True)

    def decode(seq):
        return " ".join(vocab.itos[t] for t in seq if t not in (sos, eos) and t in vocab.itos)

    return [(decode(s), sc) for s, sc in completed[:beam_size]]

# ── UI
st.markdown("<h1>CaptionLens 🔭</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">EfficientNet-B3 · Transformer · Beam Search</p>', unsafe_allow_html=True)

dev_label = f"⚡ {torch.cuda.get_device_name(0)}" if device.type == "cuda" else "💻 CPU"
st.markdown(f'<span class="device-badge">{dev_label}</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    try:
        encoder, decoder_model, vocab, ckpt = load_model()
        ep   = ckpt.get("epoch", "?")
        vloss= ckpt.get("val_loss", float("nan"))
        bleu = ckpt.get("best_bleu", None)
        info = f"Ready — vocab: {len(vocab):,} · epoch: {ep} · val loss: {vloss:.4f}"
        if bleu:
            info += f" · BLEU-4: {bleu:.4f}"
        st.success(info, icon="✅")
    except FileNotFoundError:
        st.error("⚠️  `best_model.pth` or `vocab.pkl` not found. Train the model first.")
        st.stop()

st.divider()

c1, c2 = st.columns(2)
with c1: beam_size = st.slider("Beam width", 1, 10, 5)
with c2: max_len   = st.slider("Max length", 10, 80, 45)

c3, c4 = st.columns(2)
with c3: alpha   = st.slider("Length norm α", 0.0, 1.0, 0.7, 0.05, help="Higher → prefer longer captions")
with c4: rep_pen = st.slider("Repetition penalty", 1.0, 3.0, 1.3, 0.1, help="Higher → fewer repeated words")

st.divider()

uploaded = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Running beam search..."):
        results = beam_search(encoder, decoder_model, vocab, image,
                              beam_size=beam_size, max_len=max_len,
                              alpha=alpha, rep_penalty=rep_pen)

    best_cap, best_score = results[0]
    confidence = min(100, max(0, int((1 + best_score / max_len) * 100)))

    st.markdown('<p class="caption-label">Best Caption</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="caption-box">"{best_cap}"</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{confidence}%"></div></div>
    <p style="font-family:'DM Mono',monospace;font-size:.68rem;color:#444;margin-top:4px;">
      CONFIDENCE PROXY — {confidence}%</p>""", unsafe_allow_html=True)

    if len(results) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="caption-label">Beam Alternatives</p>', unsafe_allow_html=True)
        for i, (cap, score) in enumerate(results[1:], 2):
            st.markdown(f"""
            <div class="alt-box"><span>#{i} &nbsp; {cap}</span>
            <span class="score-pill">{score:.3f}</span></div>""", unsafe_allow_html=True)