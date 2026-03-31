"""
SynthIPData — Publish Dataset to HuggingFace Hub

Usage:
    python scripts/publish_hf.py --data-dir seeds/ --repo vineetsingh-vs/synthipdata
"""

import os
import argparse
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def load_jsonl(filepath):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Publish dataset to HuggingFace")
    parser.add_argument("--data-dir", required=True, help="Directory with JSONL files")
    parser.add_argument("--repo", default="vineetsingh-vs/synthipdata", help="HuggingFace repo")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HuggingFace token not found.")
        print("   Set it with: export HF_TOKEN='your_token'")
        print("   Get your token at: https://huggingface.co/settings/tokens")
        return
    
    # Load all JSONL files
    print(f"📂 Loading data from {args.data_dir}")
    all_records = []
    
    for filename in sorted(os.listdir(args.data_dir)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(args.data_dir, filename)
            records = load_jsonl(filepath)
            all_records.extend(records)
            print(f"   Loaded {len(records):>5} records from {filename}")
    
    print(f"\n   Total records: {len(all_records):,}")
    
    if not all_records:
        print("❌ No records found. Check your data directory.")
        return
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(all_records)
    
    # Split into train/test
    split = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "test": split["test"]
    })
    
    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(dataset_dict['train']):,} records")
    print(f"   Test:  {len(dataset_dict['test']):,} records")
    
    # Push to HuggingFace
    print(f"\n⬆️  Pushing to HuggingFace: {args.repo}")
    dataset_dict.push_to_hub(
        args.repo,
        token=hf_token,
        private=args.private
    )
    
    print(f"\n✅ Published!")
    print(f"   URL: https://huggingface.co/datasets/{args.repo}")
    print(f"\n   Anyone can now use it with:")
    print(f"   ds = load_dataset('{args.repo}')")


if __name__ == "__main__":
    main()
