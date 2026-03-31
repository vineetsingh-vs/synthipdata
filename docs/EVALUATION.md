# Evaluation Framework

## Overview

SynthIPData evaluates synthetic data quality using **8 metrics across 4 buckets**. This multi-dimensional evaluation ensures the generated data is not only realistic but genuinely useful for downstream tasks.

## Bucket 1: Utility (Most Important)

> "Does the synthetic data actually help AI models perform better?"

### Metric 1: Downstream Task Accuracy
- **What**: Train PatentBERT on real-only data vs real+synthetic data, compare accuracy on rare-case classification
- **How**: Fine-tune `anferico/bert-for-patents` on rejection type classification task
- **Target**: 10%+ accuracy improvement on rare categories

### Metric 2: Few-Shot Improvement
- **What**: How much does synthetic data help when real data is extremely scarce?
- **How**: Train with only 10, 25, 50 real examples + synthetic augmentation
- **Target**: Significant improvement at all data scarcity levels

## Bucket 2: Coverage

> "Does the synthetic data fill gaps across all rare categories?"

### Metric 3: Distribution Coverage
- **What**: Are synthetic embeddings spread across the vector space or clustered?
- **How**: BGE-M3 embeddings in Qdrant, measure density across regions
- **Target**: Low KL divergence between synthetic and seed distributions

### Metric 4: Category Balance
- **What**: Is synthetic data evenly distributed across all 8 categories?
- **How**: Chi-square test of category counts
- **Target**: No category has <5% of total synthetic data

## Bucket 3: Validity

> "Does the synthetic data look like real patent documents?"

### Metric 5: Linguistic Quality
- **What**: Is the generated text fluent and natural?
- **How**: Perplexity score from a held-out language model
- **Target**: Perplexity within 1.5x of real document perplexity

### Metric 6: Structural Correctness
- **What**: Does the text follow patent document conventions?
- **How**: Rule-based checks (claim numbering, section headers, legal citation format)
- **Target**: 85%+ format compliance rate

## Bucket 4: Risk

> "Is the synthetic data safe from memorization and privacy issues?"

### Metric 7: Memorization Rate
- **What**: How many synthetic documents are near-copies of real ones?
- **How**: Cosine similarity between synthetic and seed embeddings (threshold: 0.95)
- **Target**: <2% memorization rate

### Metric 8: Deduplication
- **What**: Are there near-duplicates within the synthetic dataset?
- **How**: Pairwise cosine similarity within synthetic embeddings (threshold: 0.98)
- **Target**: <1% near-duplicate rate

## Baseline Comparisons

| Method | Description |
|--------|------------|
| No augmentation | Real data only (floor) |
| Simple paraphrasing | LLM rephrasing without fine-tuning |
| Generic LLM | Base Mistral-7B with prompting only |
| **SynthIPData (ours)** | **QLoRA fine-tuned Mistral-7B** |

All baselines use the same evaluation pipeline and test set for fair comparison.
