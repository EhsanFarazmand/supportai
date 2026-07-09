---
title: SupportAI Demo
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
short_description: Classify support tickets and draft replies
startup_duration_timeout: 1h
pinned: false
license: mit
---

# SupportAI — Ticket Routing & Response (Phase 6 demo)

**🔴 Live: https://huggingface.co/spaces/EFarazmand/supportai-demo** (Gradio + ZeroGPU)

Gradio demo for the SupportAI project. Paste a customer-support ticket and get:

1. **Category classification + confidence** — TF-IDF + Logistic Regression, the project's recommended production classifier (0.995 macro-F1 at ~zero cost).
2. **A suggested reply** via a **hybrid** strategy:
   - **Retrieval (default, CPU)** — the reply from the most similar past *resolved* ticket. Always works; ties to the Phase-5 finding that retrieve-and-return is ROUGE-competitive on this dataset.
   - **Fine-tuned LLM (optional, GPU)** — the Phase-4 QLoRA Mistral-7B, grounded in the retrieved tickets (RAG).
3. **Retrieved neighbours** — for transparency / auditability.

## Run locally

```bash
cd app
pip install -r requirements.txt
python app.py            # opens http://127.0.0.1:7860
```

The classifier loads the committed Phase-1 artifacts from `../models/` if present; otherwise it fits an equivalent TF-IDF + LogReg from the dataset at startup (a few seconds), so the app is always self-contained. The retrieval corpus is downloaded from the HuggingFace `datasets` hub on first run.

## Deploy to HuggingFace Spaces

1. Create a **Gradio** Space.
2. Push `app.py` and `requirements.txt` to it. (The classifier self-fits from the dataset, so no model files are strictly required; to use the *exact* committed model, also copy `models/{label_encoder,tfidf_vectorizer,lr_tfidf}.pkl` into the Space and they'll be picked up.)
3. The **CPU free tier** runs classification + retrieval replies.

### Enabling the LLM reply (GPU Space)

The 7B model needs a GPU:

1. Upgrade the Space hardware to a GPU tier.
2. Uncomment the LLM dependencies in `requirements.txt`.
3. Upload the Phase-4 LoRA adapter (`models/mistral_qlora_gen/`) to a HuggingFace model repo, then set the Space secret/variable:
   ```
   MISTRAL_ADAPTER = <your-username>/mistral_qlora_gen
   ```
   (A local path also works when running locally on a GPU box.)

The app auto-detects the GPU + adapter; without them it silently stays on retrieval — no code change needed.

## Environment variables

| Var | Purpose | Default |
|-----|---------|---------|
| `MISTRAL_ADAPTER` | HF repo id or local path of the Phase-4 LoRA adapter | *(unset → retrieval only)* |
| `MODELS_DIR` | Directory holding the committed `*.pkl` classifier artifacts | auto-detected (`../models`) |
