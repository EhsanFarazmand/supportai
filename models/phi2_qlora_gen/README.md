# Model card — phi2_qlora_gen

- **Base model**: `microsoft/phi-2`
- **Task**: response generation
- **Method**: QLoRA 4-bit
- **Dataset**: Bitext Customer Support (2000 train / 200 eval subset)
- **LoRA**: r=16, alpha=32, dropout=0.05, target=all-linear
- **Train config**: epochs=1, lr=0.0002, max_seq_len=512

## Metrics
```json
{
  "method": "QLoRA 4-bit",
  "rougeL_base": 0.1966,
  "rougeL_ft": 0.3031,
  "peak_vram_gb": 3.89,
  "train_time_s": 994.8
}
```

## How to load
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model = AutoPeftModelForCausalLM.from_pretrained("models/phi2_qlora_gen", device_map="auto")
tok = AutoTokenizer.from_pretrained("models/phi2_qlora_gen")
```

## Intended use
Drafting customer-support replies for agent review (assist, not full automation).
Not validated for autonomous deployment — see ../METRICS.md for the human-eval rubric.
