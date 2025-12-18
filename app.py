import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="BART Simplifier Demo", page_icon="📖", layout="centered")
st.title("📖 BART Sentence Simplifiero for Readability")
st.caption("Fine-tuned BART demo — paste a complex sentence and get a simplified version.")

# ----------------------------
# Config
# ----------------------------
DEFAULT_CKPT = "jiwonbae1124/bart-simplifier-readability"
DEFAULT_MAX_SOURCE_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model/tokenizer (cached)
# ----------------------------
@st.cache_resource
def load_model(ckpt_path: str):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
    model.eval()
    return tokenizer, model

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Settings")

ckpt_path = st.sidebar.text_input("Checkpoint path", value=DEFAULT_CKPT)

max_source_length = st.sidebar.number_input(
    "Max source length (tokenization)",
    min_value=32,
    max_value=512,
    value=DEFAULT_MAX_SOURCE_LENGTH,
    step=8,
)

num_beams = st.sidebar.slider("Beam size", min_value=1, max_value=10, value=4)
max_new_tokens = st.sidebar.slider("Max new tokens", min_value=16, max_value=256, value=64, step=8)
no_repeat_ngram_size = st.sidebar.slider("No repeat n-gram size", min_value=0, max_value=6, value=3)
length_penalty = st.sidebar.slider("Length penalty", min_value=0.5, max_value=2.0, value=1.1, step=0.1)

use_cache = st.sidebar.checkbox("Use KV cache", value=True)
early_stopping = st.sidebar.checkbox("Early stopping (beam)", value=True)

# ----------------------------
# Main input
# ----------------------------
default_text = (
    "Adjacent counties are Marin (to the south), Mendocino (to the north), "
    "Lake (northeast), Napa (to the east), and Solano and Contra Costa (to the southeast)."
)

text = st.text_area("Enter a complex sentence:", value=default_text, height=130)

col1, col2 = st.columns([1, 1])
run_btn = col1.button("Simplify ✨", type="primary")
clear_btn = col2.button("Clear")

if clear_btn:
    st.experimental_rerun()

# ----------------------------
# Inference
# ----------------------------
def simplify(text: str, tokenizer, model) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_source_length),
    ).to(device)

    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.num_beams = int(num_beams)
    gen_config.max_new_tokens = int(max_new_tokens)
    gen_config.no_repeat_ngram_size = int(no_repeat_ngram_size)
    gen_config.length_penalty = float(length_penalty)
    gen_config.early_stopping = bool(early_stopping)
    gen_config.use_cache = bool(use_cache)

    with torch.no_grad():
        out_ids = model.generate(**enc, generation_config=gen_config)

    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

if run_btn:
    try:
        with st.spinner("Loading model + simplifying..."):
            tokenizer, model = load_model(ckpt_path)
            simplified = simplify(text, tokenizer, model)

        st.subheader("✅ Simplified output")
        st.write(simplified if simplified else "⚠️ (empty output)")

    except Exception as e:
        st.error("Oops — something broke while loading or generating.")
        st.exception(e)

st.markdown("---")
st.caption("Tip: if outputs are too long/short, tweak length penalty + max_new_tokens.")
