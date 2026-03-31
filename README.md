# SynthIPData: Synthetic Data Augmentation for Rare-Case IP Lifecycle Scenarios

[![Dataset on HuggingFace](https://img.shields.io/badge/_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/vineetsingh-vs/synthipdata)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

## Overview

**SynthIPData** addresses a critical gap in legal AI: the scarcity of training data for rare patent examination scenarios. While common rejection types (§102, §103) have abundant training examples, rare but important cases — such as §101 rejections in biotech or double patenting in AI/ML patents — lack sufficient data to train reliable AI models.

This project uses **QLoRA-finetuned Mistral-7B** to generate high-quality synthetic patent examination documents for 8 rare-case categories, validated through an 8-metric evaluation framework across 4 measurement buckets (Utility, Coverage, Validity, Risk).

<p align="center">
  <img src="docs/architecture.png" alt="SynthIPData Pipeline Architecture" width="700">
</p>

## Key Results

| Metric | Baseline (Real Only) | + SynthIPData | Improvement |
|--------|---------------------|---------------|-------------|
| Rare-case classification (PatentBERT) | TBD | TBD | TBD |
| Coverage across rare categories | TBD | TBD | TBD |
| Memorization rate | — | TBD | — |
| Validity (linguistic quality) | — | TBD | — |

> *Results will be updated as experiments complete.*

## Rare-Case Categories

We target 8 rare rejection-type × technology-area combinations identified through frequency analysis of USPTO examination data (2015–2024):

| # | Category | Rejection Type | Technology Area | Est. Seed Size |
|---|----------|---------------|-----------------|----------------|
| 1 | 101 in Biotech | §101 Subject Matter | Biotech/Pharma | ~150–300 |
| 2 | 101 in AI/ML | §101 Subject Matter | AI/Machine Learning | ~200–400 |
| 3 | Double Patenting in AI/ML | Double Patenting | AI/Machine Learning | ~100–200 |
| 4 | 112(f) in Biotech | §112(f) Means-Plus-Function | Biotech/Pharma | ~80–150 |
| 5 | Restriction in AI/ML | Restriction Requirement | AI/Machine Learning | ~100–200 |
| 6 | 112(a) in AI/ML | §112(a) Written Description | AI/Machine Learning | ~150–300 |
| 7 | 101 in Nanotech | §101 Subject Matter | Materials/Nanotech | ~60–120 |
| 8 | Double Patenting in Semiconductors | Double Patenting | Semiconductors | ~150–250 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                             │
│  USPTO Office Action Dataset  ←→  HUPD (HuggingFace)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  SEED COLLECTION                             │
│  Filter rare categories → Extract full text → Clean → S3    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 EMBEDDING + INDEXING                          │
│  BGE-M3 embeddings → Qdrant vector store                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               SYNTHETIC GENERATION                           │
│  Mistral-7B + QLoRA fine-tuning → Generate variations       │
│  Qdrant similarity check → Filter memorized/garbage         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ UTILITY  │ │ COVERAGE │ │ VALIDITY │ │   RISK   │      │
│  │PatentBERT│ │  Vector  │ │Linguistic│ │Memorize  │      │
│  │ before/  │ │  space   │ │ quality  │ │ check +  │      │
│  │  after   │ │ analysis │ │  scores  │ │ dedup    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PUBLICATION                               │
│  Validated dataset → HuggingFace Hub (Parquet + Dataset Card)│
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Synthetic Generation | Mistral-7B + QLoRA | Generate rare-case patent documents |
| Embeddings | BGE-M3 (1024-dim) | Semantic similarity and coverage analysis |
| Vector Database | Qdrant | Memorization detection + coverage mapping |
| Downstream Evaluation | PatentBERT | Utility testing (before/after comparison) |
| Data Sources | USPTO Office Action Dataset, HUPD | Real patent examination data |
| Storage | AWS S3 | Shared team storage for seeds, models, outputs |
| Dataset Hosting | HuggingFace Hub | Public dataset distribution |
| Compute | Google Colab / RunPod | GPU for training and generation |

## Project Structure

```
synthipdata/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
│
├── notebooks/                    # Colab notebooks (run these in order)
│   ├── 01_discovery.ipynb        # Day 1: Find rare categories
│   ├── 02_seed_collection.ipynb  # Day 1: Collect seed documents
│   ├── 03_embedding.ipynb        # Day 2: Embed seeds into Qdrant
│   ├── 04_finetuning.ipynb       # Day 3: Fine-tune Mistral with QLoRA
│   ├── 05_generation.ipynb       # Day 4: Generate synthetic data
│   ├── 06_evaluation.ipynb       # Day 5: Run all 8 metrics
│   └── 07_publish.ipynb          # Day 6: Push to HuggingFace
│
├── scripts/                      # Standalone Python scripts
│   ├── download_data.py          # Download USPTO datasets
│   ├── discover_rare_cases.py    # Frequency analysis
│   ├── collect_seeds.py          # Extract seed documents
│   ├── embed_seeds.py            # BGE-M3 embedding pipeline
│   ├── finetune_mistral.py       # QLoRA training script
│   ├── generate_synthetic.py     # Synthetic data generation
│   ├── evaluate.py               # Full evaluation pipeline
│   ├── upload_s3.py              # S3 upload helper
│   └── publish_hf.py             # HuggingFace publishing
│
├── configs/                      # Configuration files
│   ├── categories.yaml           # Rare-case category definitions
│   ├── qlora_config.yaml         # QLoRA hyperparameters
│   ├── generation_config.yaml    # Generation prompts and settings
│   └── evaluation_config.yaml    # Evaluation metric thresholds
│
├── data/                         # Local data (gitignored)
│   ├── raw/                      # Downloaded USPTO files
│   ├── processed/                # Cleaned intermediate files
│   └── seeds/                    # Final seed corpus
│
├── results/                      # Experiment outputs
│   ├── figures/                  # Charts, heatmaps
│   └── metrics/                  # Evaluation scores (JSON/CSV)
│
├── docs/                         # Documentation
│   ├── architecture.png          # Pipeline diagram
│   ├── DATA_SOURCES.md           # Data source documentation
│   ├── EVALUATION.md             # Evaluation framework details
│   └── REPRODUCTION.md           # Steps to reproduce results
│
└── tests/                        # Unit tests
    ├── test_discovery.py
    ├── test_embedding.py
    └── test_evaluation.py
```

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/vineetsingh-vs/synthipdata.git
cd synthipdata
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

The easiest way is to run the notebooks in order on Google Colab:

```
01_discovery.ipynb       → Find rare patent categories
02_seed_collection.ipynb → Collect real patent documents
03_embedding.ipynb       → Build vector representations
04_finetuning.ipynb      → Train Mistral on rare cases
05_generation.ipynb      → Generate synthetic documents
06_evaluation.ipynb      → Evaluate quality
07_publish.ipynb         → Publish to HuggingFace
```

### 4. Use the dataset

```python
from datasets import load_dataset

ds = load_dataset("vineetsingh-vs/synthipdata")
print(ds)
```

## Evaluation Framework

We evaluate synthetic data quality across 4 buckets with 8 metrics:

### Utility (Does it help?)
- **Downstream task accuracy**: PatentBERT classification performance on rare cases before/after augmentation
- **Few-shot improvement**: Performance gain in low-resource scenarios

### Coverage (Does it fill the gaps?)
- **Distribution coverage**: Vector space analysis across all 8 rare categories
- **Category balance**: Uniformity of synthetic data across categories

### Validity (Does it look real?)
- **Linguistic quality**: Perplexity and fluency scores
- **Structural correctness**: Patent document format adherence

### Risk (Is it safe?)
- **Memorization rate**: Cosine similarity check against seed corpus (threshold: 0.95)
- **Deduplication**: Near-duplicate detection within synthetic dataset

## Baseline Comparisons

We compare our QLoRA fine-tuned approach against:

| Method | Description |
|--------|------------|
| No augmentation | Train on real data only (floor baseline) |
| Simple paraphrasing | LLM rephrasing without domain fine-tuning |
| Generic LLM generation | Base Mistral-7B with prompting (no fine-tuning) |
| **SynthIPData (ours)** | **QLoRA fine-tuned Mistral-7B** |

## Data Sources

| Source | What It Provides | Access |
|--------|-----------------|--------|
| [HUPD](https://huggingface.co/datasets/HUPD/hupd) | Full patent text, claims, metadata, decisions | HuggingFace streaming |
| [USPTO Office Action Dataset](https://www.uspto.gov/ip-policy/economic-research/research-datasets) | Rejection types, examiner details, structured labels | CSV download |

## Reproducibility

All experiments are fully reproducible:
- Random seeds are fixed across all scripts
- Model checkpoints are versioned on S3
- Dataset versions are tracked via HuggingFace commit SHAs
- Exact hyperparameters documented in `configs/`

See [REPRODUCTION.md](docs/REPRODUCTION.md) for detailed instructions.

## Citation

If you use SynthIPData in your research, please cite:

```bibtex
@article{singh2026synthipdata,
  title={SynthIPData: Synthetic Data Augmentation for Rare-Case Scenarios 
         Across the IP Examination Lifecycle},
  author={Singh, Vineet},
  year={2026},
  url={https://github.com/vineetsingh-vs/synthipdata}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Vineet Singh**  
Senior Software Engineer — AI Infrastructure for IP Litigation Support  
[GitHub](https://github.com/vineetsingh-vs) | [HuggingFace](https://huggingface.co/vineetsingh-vs)
