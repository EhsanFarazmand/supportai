# Product Brief — SupportAI: Intelligent Ticket Routing & Response

## Problem
Customer-support teams triage every incoming ticket by hand: read it, decide which queue/category it belongs to, then write a reply. This is slow, repetitive, and scales linearly with ticket volume. Two costs dominate: **routing time** (deciding where a ticket goes) and **drafting time** (composing a correct, on-brand reply).

## Solution
An assistant that, for each incoming ticket, instantly:
1. **Classifies** it into one of 11 categories (ACCOUNT, ORDER, REFUND, INVOICE, CONTACT, PAYMENT, FEEDBACK, DELIVERY, SHIPPING, SUBSCRIPTION, CANCEL) — and a finer 27-way intent — with a confidence score.
2. **Drafts a suggested reply** grounded in how similar tickets were resolved historically.

The human agent stays in the loop: they accept, edit, or reject the draft. The system removes the blank-page and routing overhead, not the human judgement.

## Users & user stories
- **Support agent**: "As an agent, I want an incoming ticket pre-categorised with a suggested reply so I can respond in seconds instead of minutes."
- **Support lead**: "As a lead, I want confident tickets auto-routed and low-confidence ones flagged for review so my team spends attention where it matters."
- **ML/Ops owner**: "As the system owner, I want to know which model to run and what it costs per ticket, so routing stays cheap and only reply-drafting pays for an LLM."

## Scope (what this project delivers)
- A benchmarked classifier (production-ready: TF-IDF + Logistic Regression, **0.995 macro-F1**).
- A multi-task encoder that serves `category` + `intent` from one model at ~half the serving cost.
- A fine-tuned reply generator (QLoRA Mistral-7B) plus a zero-cost retrieval fallback.
- A working demo (`app/`) and the decision docs in `docs/`.

## Out of scope (explicitly)
- **Full automation** — replies are drafts for human review, not auto-sent. Generation quality is not yet human-validated at scale (see the pending human-eval rubric in `METRICS.md`).
- Multi-lingual support, live ticketing-system integration, and PII handling — future work.

## Success metrics
| Dimension | Metric | Target | Status |
|-----------|--------|--------|--------|
| Routing quality | Macro-F1 | > 0.85 | ✅ **0.995** (classical) |
| Reply quality | ROUGE-L (proxy) | — | 0.36 (LLM); ROUGE is a weak proxy here → human eval pending |
| Reply quality | Human eval (relevance/accuracy/tone/completeness) | ≥ 4/5 | ⬜ pending (Phase 6) |
| Cost | $ per ticket (routing) | < $0.01 | ✅ classical routing ≈ free |
| Latency | p95 per ticket (routing) | < 500 ms | ✅ sub-ms classifier (not yet formally benchmarked) |

## The core product bet
Most of the value (routing) is captured by a **cheap** model; the expensive model (LLM) is reserved for the one task that genuinely needs generation (reply drafting), and even there it runs as an assist. This keeps unit economics viable at scale — the central thesis validated across Phases 1–5 (see `EXPERIMENTS.md`).

## Roadmap position
Phases 0–5 (data → classical → deep learning → transformers → LLM fine-tuning → advanced experiments) are complete. Phase 6 (this brief + the demo + the decision docs) packages the findings into a product story. Next: human-eval study, live integration, and a retraining/active-learning loop.
