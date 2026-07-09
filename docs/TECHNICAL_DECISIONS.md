# Technical Decisions — SupportAI

A record of the key choices, the evidence behind them, and the trade-offs accepted. Numbers link back to `EXPERIMENTS.md` and the per-phase `notebooks/*_FINDINGS.md`.

## TD-1 · Ship TF-IDF + Logistic Regression for classification
**Decision**: Use the classical model in production for routing.
**Why**: It hits **0.9955 macro-F1** — above the 0.85 target — training in ~2 s on CPU. BERT (0.9992) and DistilBERT (0.9999) add **<0.5%** for 100–1000× the compute/cost; the LSTM (0.994) needs 1.1M params and ~1500 s to train for no gain.
**Trade-off accepted**: We forgo the last ~0.4% F1. Justified: the dataset is clean and keyword-driven, so a linear model over sparse TF-IDF is already near-ceiling. Revisit only if real traffic proves more ambiguous than this dataset.

## TD-2 · Multi-task encoder for category + intent
**Decision**: If both `category` and `intent` are needed, serve them from one shared DistilBERT encoder with two heads.
**Why**: Multi-task matched single-task models (category 0.9993 vs 0.9986; intent 0.9966 vs 0.9967) at **~half the parameters/serving cost** (Phase 5).
**Trade-off**: A shared encoder can't emit free-text generation (needs a decoder), so this covers classification only.

## TD-3 · LLMs are for reply generation, not classification
**Decision**: Do **not** use an LLM to classify.
**Why**: Off-the-shelf Mistral-7B reached only **0.69 zero-shot / 0.80 few-shot** macro-F1 (Phase 4) — ~0.20 below fine-tuned BERT — while being far slower and costlier. Even chain-of-thought on the hardest tickets (0.85, Phase 5) trails BERT. The LLM's unique value is **generation**.
**Trade-off**: None meaningful — this is strictly cheaper and more accurate for classification.

## TD-4 · QLoRA (4-bit) for fine-tuning the generator
**Decision**: Fine-tune Mistral-7B with QLoRA rather than fp16 LoRA or full fine-tuning.
**Why**: QLoRA used **58% of the VRAM** of fp16 LoRA for a negligible −0.01 ROUGE difference (Phase 4), which is what makes a 7B model trainable on a **16 GB T4** at all (peaked ~6.4 GB).
**Trade-off**: QLoRA was **~27% slower** to train (4-bit dequant overhead). Accepted: memory is the binding constraint on commodity GPUs, not training wall-clock.

## TD-5 · Reply generation is a human-in-the-loop assist, not automation
**Decision**: The generator drafts; a human sends.
**Why**: Fine-tuning lifted ROUGE-L to 0.36 (on-brand replies), but ROUGE is a **weak proxy** on this near-duplicate dataset (Phase 5 showed RAG *lowered* ROUGE while improving grounding). Quality has not been human-validated at scale.
**Trade-off**: No full automation yet. Gated on the human-eval rubric (`METRICS.md`).

## TD-6 · RAG kept for grounding, not for the metric
**Decision**: Offer retrieval-grounded replies, but don't claim a ROUGE win from RAG.
**Why**: Retrieval was perfect (category-match@3 = 1.00) but FT+RAG **0.354 < FT-only 0.372** on ROUGE — the dataset is too near-duplicate for RAG to add lexical signal (Phase 5). RAG's value is **grounding/auditability** (replies traceable to real resolutions) and it should help on more diverse real traffic.
**Trade-off**: Extra retrieval latency for benefit that this benchmark under-measures. Kept because production traffic is more varied than this dataset.

## TD-7 · Hybrid reply strategy in the demo
**Decision**: The demo defaults to **CPU retrieve-and-return** and uses the fine-tuned LLM only when a GPU + adapter are present.
**Why**: Retrieve-and-return scored ROUGE-L 0.361 — competitive with the LLM (0.372) at **zero GPU cost** — so the demo is fully functional on a free CPU Space, degrading gracefully.
**Trade-off**: On CPU the reply is a past reply, not a freshly generated one. Acceptable for a demo; the LLM path is one env var away.

## TD-8 · Active learning for the labeling loop
**Decision**: Prioritize human labeling by model uncertainty (entropy), not randomly.
**Why**: Uncertainty sampling reached 0.95 F1 with **1000 labels vs 1200 random — 16.7% fewer** (Phase 5).
**Trade-off**: Requires a retraining loop; worth it wherever labeling costs real money.

## Cross-cutting engineering notes
- **Reproducible splits**: every notebook uses the same 70/15/15 stratified split (seed 42) so metrics compare across phases.
- **Colab realities**: bitsandbytes must be ≥0.44 on current Colab (new triton removed `triton.ops`); the Phase-1 pkls were saved with a newer sklearn than Colab ships, so `LogisticRegression.predict_proba` needs a `multi_class` shim. Both are handled in code.
- **Artifacts**: small models/encoders are committed; large weights (transformer/LLM) are gitignored and regenerated or loaded from the Hub.
