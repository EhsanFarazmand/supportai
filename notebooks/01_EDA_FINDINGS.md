# Phase 0: Data Exploration Findings
*Date: April, 2026*

## Dataset Overview
- **Source**: Bitext Customer Support Dataset
- **Total Samples**: 26,872
- **Categories**: 11 total (ACCOUNT, ORDER, REFUND, INVOICE, CONTACT, PAYMENT, FEEDBACK, DELIVERY, SHIPPING, SUBSCRIPTION, CANCEL)
- **Avg Query Length**: 8.69 Words


## Key Insights

### 1. Category Distribution
- Most common categories: ACCOUNT, ORDER, REFUND
- Class balance: Balanced
- Imbalance ratio: 6.30x

### 2. Text Characteristics
- Average instruction length: 8.69 words
- Average response length: 104.79 words

### Word Count Analysis

| Metric | Instruction (Query) | Response (Reply) |
|--------|-------------------|-----------------|
| Mean | 8.7 words | 104.8 words |
| Median | 9 words | 90 words |
| Std Dev | 2.6 | 53.0 |
| Range | 1 – 16 words | 9 – 402 words |
| IQR | 7 – 11 words | 72 – 124 words |

### Character Length Analysis
| Metric | Instruction | Response |
|--------|-------------|----------|
| Mean | 46.9 chars | 634.1 chars |
| Median | 48 chars | 540 chars |
| Std Dev | 10.9 | 331.6 |
| Range | 6 – 92 chars | 57 – 2,472 chars |
| IQR | 40 – 55 chars | 427 – 753 chars |
**Key Insights:**
- Instructions are consistently **brief** (IQR: 40–55 chars).
- Responses show **extreme length variation** (max: 2,472 chars); consider:
  - Truncating long responses for model input limits
  - Using hierarchical models for very long replies
  - Analyzing if length correlates with category or resolution quality

### 3. Potential Challenges
- Class imbalance in [specific categories]
- Very short/long queries that might need special handling

### 4. Next Steps for Phase 1
[ ] Build TF-IDF + Logistic Regression baseline
[ ] Create train/validation/test splits