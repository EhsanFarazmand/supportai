# SupportAI: Success Metrics

## Business Metrics
- **Response Time Reduction**: Target 40% reduction in average ticket response time
- **Agent Productivity**: Target 30% more tickets handled per agent per day
- **Customer Satisfaction**: Maintain or improve CSAT score (>4.0/5.0)

## Technical Metrics

### Classification Task
- **Primary Metric**: Macro F1-Score (treats all categories equally)
  - Target: >0.85 for production
  - Baseline: TBD after Phase 1
  
- **Secondary Metrics**:
  - Accuracy: Overall correct predictions
  - Per-category F1: Identify weak categories
  - Confusion Matrix: Understand misclassification patterns

### Response Generation Task
- **Automated Metrics**:
  - ROUGE-L: Overlap with reference responses
  - BLEU: N-gram precision
  - Perplexity: Model confidence
  
- **Human Evaluation** (sample 100 responses):
  - Relevance: Does it address the query? (1-5 scale)
  - Accuracy: Is information correct? (1-5 scale)
  - Tone: Appropriate customer service tone? (1-5 scale)
  - Completeness: Answers all parts? (1-5 scale)

## Experiment Comparison Framework

| Experiment | Macro F1 | Accuracy | Inference Time | Memory | ROUGE-L | Notes |
|------------|----------|----------|----------------|---------|---------|-------|
| Baseline   | -        | -        | -              | -       | -       | -     |

## Deployment Constraints
- **Latency**: <500ms per ticket (95th percentile)
- **Memory**: <8GB GPU for inference
- **Cost**: <$0.01 per ticket processed