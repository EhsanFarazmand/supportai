# Deployment Considerations — SupportAI

How the pieces would actually run in production, and the cost/latency/accuracy trade-offs behind each choice.

## Component map
| Component | Model | Hardware | Role |
|-----------|-------|----------|------|
| Router | TF-IDF + LogReg (or multi-task DistilBERT) | CPU | Category (+ intent) per ticket |
| Reply drafter | QLoRA Mistral-7B (+ RAG) | GPU (or API) | Draft reply for agent review |
| Retriever | TF-IDF / embedding NN over resolved tickets | CPU | Grounding + zero-cost fallback reply |
| Labeling loop | Uncertainty sampling on the router | CPU (offline) | Prioritize what humans label next |

## Cost / latency / accuracy trade-offs

### Routing (the high-volume path)
- **TF-IDF + LogReg**: sub-millisecond CPU inference, effectively **$0 per ticket**, 0.995 macro-F1. This runs on *every* ticket.
- **DistilBERT / multi-task**: ~ms on GPU, higher infra cost, ≤0.5% better. Choose it **only** if intent is needed too (multi-task serves both from one encoder) or if real traffic proves harder than the benchmark.
- **Rule**: never put an LLM on the routing path — 0.80 F1 at LLM cost/latency is strictly worse than 0.995 at ~zero cost.

### Reply drafting (the low-volume, high-value path)
- **QLoRA Mistral-7B**: needs a GPU. Peak ~6.4 GB in 4-bit — fits a 16 GB T4 (well under the <8 GB inference budget). Latency is seconds per reply → **async assist**, not inline.
- **Retrieve-and-return**: CPU, near-zero cost, ROUGE-competitive on this data. The sensible **default/fallback**; upgrade to the LLM where reply quality justifies GPU spend.
- **Deployment shapes for the LLM**: (a) self-hosted GPU (control, fixed cost), (b) serverless GPU / inference endpoint (scales to zero, pay-per-use), (c) hosted LLM API (no infra, per-token cost, data leaves your boundary). Pick by volume and data-sensitivity.

## Meeting the stated constraints (`METRICS.md`)
| Constraint | Target | Reality |
|------------|--------|---------|
| Latency (routing) | < 500 ms p95 | ✅ classical is sub-ms; DistilBERT ~ms on GPU. *(Not yet formally load-tested.)* |
| Memory (inference) | < 8 GB GPU | ✅ Mistral-7B 4-bit ≈ 6.4 GB; classifiers CPU-only. |
| Cost | < $0.01 / ticket | ✅ routing ≈ free; LLM drafting tracked separately, justified as an assist. |

## Scaling & operations
- **Throughput**: routing is CPU-bound and trivially horizontally scalable. The LLM is the bottleneck — batch requests, cache by near-duplicate ticket (retrieval already surfaces these), and route only low-confidence or high-value tickets to the LLM.
- **Confidence-gated routing**: auto-apply high-confidence categories; send low-confidence ones (the active-learning frontier) to human review — this both protects quality and generates the next labeling batch.
- **Monitoring**: track macro-F1 drift per category, reply accept/edit/reject rates (the real generation-quality signal), and per-ticket cost. A rising edit-rate or category drift triggers retraining.
- **Retraining loop**: uncertainty sampling → human labels → periodic router refit (cheap) and adapter refresh (scheduled GPU job). ~17% fewer labels for target F1 (Phase 5).

## Risks & mitigations
- **Generation not human-validated** → keep human-in-the-loop; run the 100-response eval rubric before any automation.
- **Benchmark ≠ real traffic** (this dataset is clean/near-duplicate) → monitor drift; RAG and the LLM are expected to matter more on messier real tickets than on this benchmark.
- **LLM cost creep** → confidence/volume gating + retrieval caching; the CPU fallback guarantees the product still works if the GPU budget is cut.
- **PII / data residency** (if using a hosted API) → prefer self-hosted or endpoint deployment for the LLM; scrub placeholders (the dataset already uses `{{...}}` tokens).

## Demo deployment (Phase 6) — LIVE
Deployed to **HuggingFace Spaces (Gradio + ZeroGPU)**: **https://huggingface.co/spaces/EFarazmand/supportai-demo**

- Classification + retrieval reply run on CPU (instant); the fine-tuned **Mistral-7B (QLoRA)** drafts a RAG-grounded reply on demand via **ZeroGPU**.
- ZeroGPU note: the 4-bit base loads at module scope (bitsandbytes is ZeroGPU-aware), but the **LoRA adapter is attached inside the `@spaces.GPU` worker** — attaching it at module scope raises "No CUDA GPUs available".
- Gated base model (`mistralai/Mistral-7B-Instruct-v0.3`) is pulled via an `HF_TOKEN` Space secret. See `app/README.md`.
