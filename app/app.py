"""
SupportAI — Intelligent Ticket Routing & Response System (Phase 6 demo)

Gradio + ZeroGPU app. For a customer-support ticket it returns:
  1. Category classification + confidence (TF-IDF + Logistic Regression — the
     project's recommended production classifier; self-fits from the dataset
     if the committed pkls aren't present, so the Space is self-contained).
  2. A retrieval reply (CPU, instant): the resolved reply of the most similar
     past ticket — a zero-cost baseline (Phase-5 finding: retrieve-and-return
     is ROUGE-competitive on this dataset).
  3. On demand, a reply drafted by the Phase-4 QLoRA Mistral-7B, grounded in
     the retrieved tickets (RAG), run on ZeroGPU.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import spaces                      # MUST precede torch / any CUDA import
import torch
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BASE_LLM = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR = os.environ.get("MISTRAL_ADAPTER", "models/mistral_qlora_gen")

# ---------------------------------------------------------------------------
# 1. Historical ticket corpus (also the retrieval knowledge base)
# ---------------------------------------------------------------------------
def _shape(df):
    keep = [c for c in ["instruction", "response", "category", "intent"] if c in df.columns]
    return df[keep].dropna(subset=["instruction", "response"]).reset_index(drop=True)

def load_corpus():
    """Prefer a BUNDLED local CSV — zero runtime Hub/token dependency, so the app
    boots even if the dataset is removed/gated or the HF_TOKEN is invalid. Only if
    the file is absent do we fall back to the public dataset (anonymously — a bad
    HF_TOKEN would otherwise 401 a public read as 'cannot be accessed')."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ["customer_support_clean.csv", "data/customer_support_clean.csv",
              os.path.join(here, "data", "customer_support_clean.csv")]:
        if os.path.exists(p):
            print(f"Corpus from bundled file: {p}")
            return _shape(pd.read_csv(p))
    print("No bundled corpus; loading public dataset anonymously...")
    from datasets import load_dataset
    return _shape(pd.DataFrame(load_dataset(DATASET_ID, split="train", token=False)))

print("Loading corpus...")
CORPUS = load_corpus()
CATS_FROM_DATA = sorted(CORPUS["category"].unique().tolist())
print(f"Corpus: {len(CORPUS)} tickets, {len(CATS_FROM_DATA)} categories")

# ---------------------------------------------------------------------------
# 2. Classifier — committed Phase-1 artifacts if present, else fit fresh
# ---------------------------------------------------------------------------
def _find_models_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in [os.environ.get("MODELS_DIR"), os.path.join(here, "..", "models"),
              os.path.join(here, "models"), "models", here]:
        if d and os.path.exists(os.path.join(d, "lr_tfidf.pkl")):
            return d
    return None

def get_classifier():
    md = _find_models_dir()
    if md:
        try:
            import joblib
            tfidf = joblib.load(os.path.join(md, "tfidf_vectorizer.pkl"))
            lr = joblib.load(os.path.join(md, "lr_tfidf.pkl"))
            le = joblib.load(os.path.join(md, "label_encoder.pkl"))
            if not hasattr(lr, "multi_class"):      # sklearn version shim
                lr.multi_class = "multinomial"
            print(f"Loaded committed Phase-1 classifier from {md}")
            return tfidf, lr, list(le.classes_), "committed Phase-1 model"
        except Exception as e:
            print(f"Committed model load failed ({e}); fitting fresh.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    print("Fitting TF-IDF + LogReg from the dataset...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    X = tfidf.fit_transform(CORPUS["instruction"])
    lr = LogisticRegression(max_iter=1000).fit(X, CORPUS["category"])
    return tfidf, lr, CATS_FROM_DATA, "fitted from dataset"

TFIDF, LR, CATEGORIES, CLF_SOURCE = get_classifier()
print("Vectorizing corpus for retrieval...")
CORPUS_VEC = TFIDF.transform(CORPUS["instruction"])

def classify(query):
    proba = LR.predict_proba(TFIDF.transform([query]))[0]
    order = proba.argsort()[::-1][:5]
    return {CATEGORIES[i]: float(proba[i]) for i in order}

def retrieve(query, k=3):
    sims = cosine_similarity(TFIDF.transform([query]), CORPUS_VEC).ravel()
    top = sims.argsort()[::-1][:k]
    return [(CORPUS.iloc[i]["instruction"], CORPUS.iloc[i]["response"],
             CORPUS.iloc[i]["category"], float(sims[i])) for i in top]

# ---------------------------------------------------------------------------
# 3. LLM — loaded at module scope (bnb 4-bit + LoRA adapter). ZeroGPU-aware:
#    bitsandbytes' loader accepts device_map="cuda" at module scope.
# ---------------------------------------------------------------------------
# The 4-bit base loads at module scope — bitsandbytes' loader is ZeroGPU-aware,
# so device_map="cuda" is intercepted and the weights are packed/streamed. The
# LoRA adapter is attached LATER, inside the GPU worker: PEFT's load touches real
# CUDA, which doesn't exist at module scope (that raised "No CUDA GPUs available").
HAS_ADAPTER = bool(ADAPTER_DIR and os.path.isdir(ADAPTER_DIR))
BASE, TOK, LLM_STATUS = None, None, "not loaded"
def load_base():
    global BASE, TOK, LLM_STATUS
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        TOK = AutoTokenizer.from_pretrained(ADAPTER_DIR if HAS_ADAPTER else BASE_LLM)
        if TOK.pad_token is None:
            TOK.pad_token = TOK.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        BASE = AutoModelForCausalLM.from_pretrained(
            BASE_LLM, quantization_config=bnb, device_map="cuda",
            dtype=torch.bfloat16, attn_implementation="sdpa")
        LLM_STATUS = "Mistral-7B loaded" + (" + fine-tuned adapter" if HAS_ADAPTER else "")
        print(LLM_STATUS)
    except Exception as e:
        LLM_STATUS = f"unavailable ({e})"
        print("base load failed:", e)

load_base()
_MODEL = None      # PEFT-wrapped model, built once on the first GPU call

@spaces.GPU(duration=120)
def llm_reply(query):
    """Draft a support reply with the fine-tuned Mistral-7B, grounded in the
    most similar past resolved tickets (RAG). Runs on ZeroGPU."""
    global _MODEL
    query = (query or "").strip()
    if not query:
        return "_Enter a ticket first._"
    neigh = retrieve(query, 3)
    if BASE is None:
        return (f"⚠️ LLM {LLM_STATUS}. Retrieval reply instead:\n\n{neigh[0][1]}")
    if _MODEL is None:                       # attach adapter here — real CUDA exists
        _MODEL = BASE
        if HAS_ADAPTER:
            try:
                from peft import PeftModel
                _MODEL = PeftModel.from_pretrained(BASE, ADAPTER_DIR)
            except Exception as e:
                print("adapter attach failed:", e)
        _MODEL.eval()
    model, tok = _MODEL, TOK
    ctx = "\n\n".join(f"Example ticket: {q}\nResolved reply: {r}" for q, r, _, _ in neigh)
    user = ("You are a customer-support agent. Use the example resolved tickets below "
            "as guidance for tone and content, then write a reply to the new customer "
            f"message.\n\n{ctx}\n\nNew customer message: {query}")
    prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# 4. CPU handler: classify + retrieval reply + provenance
# ---------------------------------------------------------------------------
def analyze(query):
    """Classify the ticket and suggest a reply from the most similar past ticket."""
    query = (query or "").strip()
    if not query:
        return {}, "_Enter a ticket above._", ""
    cats = classify(query)
    neigh = retrieve(query, 3)
    nbr = "\n".join(f"{i+1}. **{c}** · similarity {s:.2f} — {q}"
                    for i, (q, _, c, s) in enumerate(neigh))
    return cats, neigh[0][1], nbr

# ---------------------------------------------------------------------------
# 5. UI
# ---------------------------------------------------------------------------
EXAMPLES = [
    "I want to cancel order {{Order Number}}, how do I do that?",
    "there is a problem setting up my new account",
    "can you help me get a refund for my last purchase?",
    "where is my package, it hasn't arrived yet",
    "how do I change the shipping address on an existing order?",
]
THESIS = (
    "For **classification/routing**, TF-IDF + Logistic Regression hits **0.995 macro-F1** — "
    "deeper models (BERT, LLMs) add <0.5% for 100–1000× the cost. The LLM earns its place "
    "only for **reply drafting**, as a human-in-the-loop assist. This demo reflects that: "
    "classification + a retrieval reply run instantly on CPU; the fine-tuned LLM drafts a "
    "grounded reply on demand (ZeroGPU)."
)

with gr.Blocks(title="SupportAI Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎫 SupportAI — Ticket Routing & Response")
    gr.Markdown(f"_Classifier: {CLF_SOURCE}. LLM: {LLM_STATUS}._")
    with gr.Row():
        with gr.Column(scale=3):
            ticket = gr.Textbox(label="Customer ticket", lines=3,
                                placeholder="e.g. I need to cancel my order...")
            with gr.Row():
                analyze_btn = gr.Button("Analyze (classify + retrieve)", variant="primary")
                llm_btn = gr.Button("✨ Draft reply with fine-tuned LLM (ZeroGPU)")
            gr.Examples(EXAMPLES, inputs=ticket)
        with gr.Column(scale=2):
            cats_out = gr.Label(label="Predicted category (top-5 confidence)", num_top_classes=5)
    reply_out = gr.Textbox(label="Suggested reply", lines=6)
    with gr.Accordion("Retrieved similar tickets (provenance)", open=False):
        neigh_out = gr.Markdown()
    gr.Markdown("---\n### Why it's built this way\n" + THESIS)

    analyze_btn.click(analyze, [ticket], [cats_out, reply_out, neigh_out])
    ticket.submit(analyze, [ticket], [cats_out, reply_out, neigh_out])
    # LLM button: classify + retrieve first (so the category & neighbours reflect
    # THIS ticket), then overwrite the reply with the grounded LLM draft.
    llm_btn.click(analyze, [ticket], [cats_out, reply_out, neigh_out]).then(
        llm_reply, [ticket], [reply_out])

if __name__ == "__main__":
    demo.launch()
