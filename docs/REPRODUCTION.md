# Reproduction Guide

## Prerequisites

- Google account (for Colab)
- AWS account with S3 access
- HuggingFace account (free)
- Python 3.10+

## Step-by-Step Reproduction

### Step 1: Setup
```bash
git clone https://github.com/vineetsingh-vs/synthipdata.git
cd synthipdata
pip install -r requirements.txt
```

### Step 2: Run Notebooks in Order

Open each notebook in Google Colab and run all cells:

| Notebook | Purpose | Time | GPU Needed? |
|----------|---------|------|-------------|
| `01_discovery.py` | Find rare categories in USPTO data | 30 min | No |
| `02_seed_collection.py` | Collect seed documents from HUPD | 1-1.5 hrs | No |
| `03_embedding.py` | Embed seeds with BGE-M3 | 30-45 min | Recommended |
| `04_finetuning.py` | Fine-tune Mistral-7B with QLoRA | 2-3 hrs | **Yes** |
| `05_generation.py` | Generate synthetic documents | 2-3 hrs | **Yes** |
| `06_evaluation.py` | Run all 8 evaluation metrics | 1-2 hrs | Recommended |
| `07_publish.py` | Push dataset to HuggingFace | 15 min | No |

### Step 3: Configuration

All hyperparameters and settings are in `configs/`:
- `categories.yaml` — Rare-case category definitions
- `qlora_config.yaml` — QLoRA training hyperparameters
- `generation_config.yaml` — Generation prompts and settings
- `evaluation_config.yaml` — Evaluation metric thresholds

### Step 4: Verify Results

After running `06_evaluation.py`, compare your results against the reported metrics in the paper. Results should be within ±2% due to random seed variations.

## Random Seeds

All random seeds are set to `42` across the pipeline for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Python: `random.seed(42)`
- Transformers: `TrainingArguments(seed=42)`

## Hardware Requirements

| Phase | Minimum | Recommended |
|-------|---------|-------------|
| Discovery + Collection | CPU only, 8GB RAM | Any |
| Embedding | CPU (slow) or GPU | T4 GPU (Colab free) |
| Fine-tuning | T4 16GB GPU | A100 40GB (RunPod) |
| Generation | T4 16GB GPU | A100 40GB (RunPod) |
| Evaluation | CPU + 16GB RAM | T4 GPU |

## Estimated Costs

| Resource | Cost |
|----------|------|
| Google Colab Pro (1 month) | $10 |
| RunPod A100 (~6 hours) | $10-15 |
| AWS S3 storage | $1-3/month |
| HuggingFace | Free |
| **Total** | **~$25-30** |
