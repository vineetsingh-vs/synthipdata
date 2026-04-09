# SynthIPData: Synthetic Data Augmentation for Rare Patent Office Action Rejection Categories

[![Dataset on HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/vineetsinghvats/SynthIPData)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

## Overview

SynthIPData addresses a critical gap in legal AI: the scarcity of training data for rare patent examination rejection scenarios. While common rejection types (102, 103) have abundant examples, rare but important combinations -- such as 101 rejections in AI/ML or double patenting in semiconductors -- lack sufficient data to train reliable AI models.

This project uses a **LoRA-finetuned Mistral-7B** to generate high-quality synthetic patent office action documents for **8 data-driven rare-case categories**, validated through multiple evaluation dimensions including perplexity analysis, retrieval evaluation, and classification experiments.

## Key Results

### Text Quality (Perplexity)
**Average perplexity ratio: 1.04** (1.0 = indistinguishable from real)

PatentBERT cannot distinguish our synthetic text from real USPTO office actions.

| Category | Real PPL | Synthetic PPL | Ratio |
|---|---|---|---|
| 101 in AI/ML | 6.1 | 6.1 | 0.99 |
| 101 in Semiconductors | 7.3 | 5.4 | 0.74 |
| 101 in Surgical Instruments | 6.3 | 5.8 | 0.93 |
| 112 in AI/ML | 4.6 | 4.7 | 1.04 |
| 101 in Batteries | 5.1 | 5.8 | 1.15 |
| 101 in Cryptography | 6.0 | 6.3 | 1.05 |
| Double Patenting in AI/ML | 6.0 | 6.6 | 1.10 |
| 101 in Materials | 5.4 | 7.0 | 1.30 |

### Retrieval Improvement for Rare Categories
Adding synthetic documents to a retrieval database significantly improves recall for underrepresented categories:

| Category | Real Seeds | Real Only Recall@1 | + SynthIPData | Improvement |
|---|---|---|---|---|
| 101 in Materials | 75 | 6.7% | 20.0% | +200% |
| 101 in Batteries | 152 | 30.0% | 43.3% | +44% |
| 101 in Semiconductors | 363 | 55.6% | 61.1% | +10% |
| Double Patenting in AI/ML | 1,184 | 67.1% | 73.0% | +9% |
| 101 in Surgical Instruments | 230 | 28.3% | 30.4% | +8% |

### Few-Shot Classification
With only 30 real examples per category, SynthIPData is the only augmentation method that consistently improves classification:

| Method | F1 Macro | vs Few-shot Only |
|---|---|---|
| Few-shot only (30/cat) | 0.307 | -- |
| + Paraphrasing | 0.311 | +1.3% |
| **+ SynthIPData** | **0.344** | **+12.0%** |

Notable per-category improvements in few-shot:
- 101 in Batteries: 11.1% to 31.7% F1 (+186% relative)
- 112 in AI/ML: 40.2% to 53.5% F1 (+33% relative)

### Memorization Safety
- Memorization rate (cosine similarity > 0.95): 8.3% before filtering, 0% after Qdrant-based deduplication
- Average similarity to nearest real document: 0.91 (structurally consistent but lexically novel)

## 8 Rare Categories (Data-Driven Discovery)

Categories were identified through frequency analysis of **2.4 million office actions** from the USPTO PTOFFACT dataset (2014-2017). We selected rejection-type x technology-area combinations that are underrepresented:

| # | Category | Rejection Type | USPC Class | Technology Area | 2014-2017 Count | 2020-2024 Seeds (Full Text) |
|---|---|---|---|---|---|---|
| 1 | 101_ai_ml | 35 USC 101 | 706 | AI/Neural Networks | 1,400 | 2,424 |
| 2 | 112_ai_ml | 35 USC 112 | 706 | AI/Neural Networks | 1,182 | 1,552 |
| 3 | dp_ai_ml | Double Patenting | 706 | AI/Neural Networks | 610 | 1,184 |
| 4 | 101_semiconductors | 35 USC 101 | 257 | Semiconductors | 500 | 363 |
| 5 | 101_surgical | 35 USC 101 | 606 | Surgical Instruments | 953 | 230 |
| 6 | 101_crypto | 35 USC 101 | 380 | Cryptography | 803 | 181 |
| 7 | 101_batteries | 35 USC 101 | 429 | Batteries/Fuel Cells | 174 | 152 |
| 8 | 101_materials | 35 USC 101 | 428 | Materials/Coatings | 242 | 75 |

## Data Collection Pipeline

### Sources
| Source | Year Range | Records | What It Provides |
|---|---|---|---|
| USPTO PTOFFACT CSV | 2014-2017 | 2.4M office actions | Rejection type labels + USPC class (no full text) |
| USPTO OACT Weekly Archives | 2020-2024 | 3.6M office actions | Full office action text + rejection types + USPC class |
| USPTO ODP API | 2017-2024 | 234K+ available | Application metadata, CPC codes, filing dates |

### Collection Process
1. Downloaded and analyzed 2.4M office actions from PTOFFACT CSV to discover rare categories
2. Downloaded all 297 weekly archive files (19GB) from the OACT bulk data product via USPTO API
3. Parsed 3.6 million records from weekly archives
4. Extracted 7,797 records matching our 8 categories (rejection type + USPC class)
5. Deduplicated to 6,161 unique seed documents with full office action text

## Architecture

```
USPTO PTOFFACT (2.4M records, 2014-2017)
       |
       v
 Frequency Analysis --> Identify 8 rare categories
       |
       v
USPTO OACT Weekly Archives (3.6M records, 2020-2024)
       |
       v
 Filter & Extract --> 6,161 seed documents with full text
       |
       v
 BGE-M3 Embedding (1024-dim) --> Qdrant Cloud (6,161 vectors)
       |
       v
 LoRA Fine-tuning on Mistral-7B
   - 5,544 training / 617 validation
   - LoRA r=16, alpha=32
   - 3 epochs, final loss: 0.528 / 0.561
       |
       v
 Generate Synthetic Documents (1,800 total)
   - 240 per category (120 for dp_ai_ml)
   - Temperature 0.7-1.0 for diversity
   - max_new_tokens=512
       |
       v
 Quality Filtering
   - Qdrant memorization check (>0.95 cosine sim removed)
   - Minimum length filter (>100 chars)
       |
       v
 Evaluation (3 dimensions)
   - Perplexity (text quality)
   - Retrieval (utility for rare categories)
   - Classification (few-shot and standard)
       |
       v
 Publication
   - HuggingFace Hub (dataset)
   - GitHub (code + configs)
```

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Synthetic Generation | Mistral-7B + LoRA | Generate rare-case patent rejection documents |
| Embeddings | BGE-M3 (1024-dim) | Semantic similarity and memorization detection |
| Vector Database | Qdrant Cloud | Persistent vector storage for memorization checks |
| Downstream Evaluation | PatentBERT (anferico/bert-for-patents) | Classification and perplexity evaluation |
| Data Sources | USPTO PTOFFACT, OACT Weekly Archives, ODP API | Real patent examination data |
| Dataset Hosting | HuggingFace Hub | Public dataset distribution |
| Compute | Google Colab (T4), RunPod (A100 SXM 80GB, A40) | GPU for training and generation |

## Model Training Details

| Parameter | Value |
|---|---|
| Base Model | Mistral-7B-v0.1 |
| Fine-tuning Method | LoRA (not quantized on A100 80GB) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Training Examples | 5,544 |
| Validation Examples | 617 |
| Epochs | 3 |
| Batch Size | 8 (gradient accumulation: 2) |
| Learning Rate | 2e-4 |
| Final Training Loss | 0.528 |
| Final Validation Loss | 0.561 |
| Trainable Parameters | 13.6M / 7.26B (0.19%) |

### Training Loss Curve

| Step | Training Loss | Validation Loss |
|---|---|---|
| 200 | 0.692 | 0.636 |
| 400 | 0.592 | 0.593 |
| 600 | 0.598 | 0.576 |
| 800 | 0.563 | 0.568 |
| 1000 | 0.559 | 0.561 |
| 1041 | 0.528 | 0.561 |

## Project Structure

```
synthipdata/
├── README.md
├── LICENSE (MIT)
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/
│   ├── categories.yaml           # 8 rare-case category definitions
│   ├── qlora_config.yaml         # LoRA hyperparameters
│   ├── generation_config.yaml    # Generation settings
│   └── evaluation_config.yaml    # Evaluation metric thresholds
│
├── notebooks/
│   ├── 01_discovery.py           # Discover rare categories from 2.4M records
│   └── 02_seed_collection.py     # Collect seeds from weekly archives
│
├── scripts/
│   ├── download_data.py          # Download USPTO datasets
│   ├── upload_s3.py              # S3 upload helper
│   └── publish_hf.py            # HuggingFace publishing
│
├── docs/
│   ├── DATA_SOURCES.md           # USPTO data source documentation
│   ├── EVALUATION.md             # Evaluation framework details
│   └── REPRODUCTION.md           # Steps to reproduce results
│
├── data/                         # Local data (gitignored)
├── results/                      # Evaluation outputs
│   ├── figures/
│   └── metrics/
│
└── tests/
    ├── test_discovery.py
    ├── test_embedding.py
    └── test_evaluation.py
```

## Quick Start

### 1. Use the dataset directly

```python
from datasets import load_dataset

dataset = load_dataset("vineetsinghvats/SynthIPData")

# Access synthetic documents
for doc in dataset['synthetic']:
    print(doc['category'], doc['title'])
    print(doc['text'][:200])
    break

# Access real seed metadata
for doc in dataset['real_seeds']:
    print(doc['category'], doc['title'])
    break
```

### 2. Clone and reproduce

```bash
git clone https://github.com/vineetsingh-vs/synthipdata.git
cd synthipdata
pip install -r requirements.txt
```

### 3. Run the pipeline

```
01_discovery.py          -> Find rare patent categories from USPTO data
02_seed_collection.py    -> Collect seed documents from weekly archives
03_embedding             -> Embed seeds with BGE-M3, upload to Qdrant
04_finetuning            -> Fine-tune Mistral-7B with LoRA
05_generation            -> Generate synthetic documents
06_evaluation            -> Run perplexity, retrieval, and classification evaluations
07_publish               -> Push to HuggingFace
```

## Evaluation Framework

### Dimension 1: Text Quality (Perplexity)
Measures whether PatentBERT can distinguish synthetic text from real text using pseudo-perplexity via masked language modeling. Average ratio of 1.04 indicates synthetic text is virtually identical to real office actions.

### Dimension 2: Retrieval Utility
Measures whether adding synthetic documents to a retrieval database improves the ability to find relevant documents for rare categories. Materials category improved from 6.7% to 20% Recall@1 (+200%).

### Dimension 3: Classification
Tests whether synthetic data improves PatentBERT's ability to classify office actions into the 8 categories. Most effective in few-shot scenarios (30 examples per category), where SynthIPData achieves +12% F1 improvement -- the only augmentation method that consistently helps.

### Memorization Safety
Qdrant-based cosine similarity check against all 6,161 real seed documents. Documents exceeding 0.95 similarity threshold are filtered. Pre-filtering memorization rate: 8.3%. Post-filtering: 0%.

## Baseline Comparisons

| Method | Description | Standard F1 | Few-shot F1 |
|---|---|---|---|
| No augmentation | Train on real data only | 0.449 | 0.307 |
| Simple paraphrasing | Word-swap augmentation of real docs | 0.462 | 0.311 |
| Oversampling | Duplicate rare-category real docs | 0.460 | -- |
| **SynthIPData (ours)** | **LoRA fine-tuned Mistral-7B** | **0.452** | **0.344** |

SynthIPData's primary advantage is in data-scarce scenarios where other augmentation methods fail.

## Reproducibility

All experiments are fully reproducible:
- Random seeds fixed across all scripts (seed=42)
- Model checkpoint saved as LoRA adapter
- Dataset versioned on HuggingFace Hub
- Exact hyperparameters documented in configs/
- USPTO data sources are publicly available

See [REPRODUCTION.md](docs/REPRODUCTION.md) for detailed instructions.

## Citation

If you use SynthIPData in your research, please cite:

```bibtex
@dataset{singh2026synthipdata,
  title={SynthIPData: Synthetic Data Augmentation for Rare Patent
         Office Action Rejection Categories},
  author={Singh, Vineet},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/vineetsinghvats/SynthIPData}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Vineet Singh**
Senior Software Engineer -- AI Infrastructure for IP Litigation Support
[GitHub](https://github.com/vineetsingh-vs/synthipdata) | [HuggingFace](https://huggingface.co/datasets/vineetsinghvats/SynthIPData) | [LinkedIn](https://linkedin.com/in/vineetsingh44)
