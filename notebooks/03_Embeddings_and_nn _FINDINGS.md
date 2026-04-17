# Phase 2: Embeddings & Neural Networks - Key Learnings

## What I Learned

### 1. **Pre-trained Embeddings (GloVe)**
- **What**: Word vectors trained on Wikipedia
- **How**: Average word vectors to get document embedding
- **Performance**: 91%
- **When to Use**:
  ✅ Small datasets (borrow knowledge)
  ✅ Semantic similarity important
  ❌ Keyword-driven tasks (TF-IDF better)

### 2. **Neural Networks**
- **Architecture**: 3-layer feedforward network
- **Performance**: 94%
- **Learning**: Can capture non-linear patterns
- **Limitation**: Still uses averaged embeddings (loses word order)

### 3. **LSTM (Recurrent Neural Network)**
- **Architecture**: Bidirectional LSTM with 2 layers
- **Performance**: 99.4%
- **Advantage**: Processes sequences, remembers context
- **Cost**: 100x more parameters, longer training

## Key Insight for AI PMs

**"More complex ≠ Better performance"**

For this dataset:
- TF-IDF baseline: 99.5% F1, 5s training, ~53K params
- LSTM: 99.4% F1, 1461s training, ~1.13M params

**Decision**: Stick with TF-IDF for production because:
1. Equal or better performance
2. 100x faster inference
3. 1000x cheaper to deploy
4. Easier to debug and explain

## When Would Deep Learning Help?

✅ **Use embeddings/neural nets when**:
- Context matters: "This is not bad" vs "This is bad"
- Small training data (transfer learning)
- Semantic similarity: "broken" should match "defective"
- Multi-lingual support

❌ **Don't use when**:
- Clear keyword signals (like our case)
- Large clean dataset available
- Speed/cost critical
- Interpretability needed

## Next Phase Preview

Phase 3 will explore **Transformers** - these handle:
- Long-range dependencies
- Bidirectional context
- Pre-trained on massive datasets