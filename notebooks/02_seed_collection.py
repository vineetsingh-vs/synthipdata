# ============================================================
# SynthIPData — Notebook 02: Seed Collection
# ============================================================
# PURPOSE:  Pull full patent text from HUPD for our rare cases
# DEPENDS:  Run 01_discovery.py first (need application_numbers_by_category.json)
# RUN ON:   Google Colab (free tier)
# TIME:     ~1-1.5 hours
# ============================================================


# ==========================
# CELL 1: Install & Setup
# ==========================

!pip install datasets pandas boto3 tqdm -q

import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import boto3
import os

print("✅ Libraries ready!")


# ==========================================
# CELL 2: Load Application Numbers from Day 1
# ==========================================

# Load the application numbers we found in Notebook 01
with open("application_numbers_by_category.json", "r") as f:
    app_numbers = json.load(f)

print("📋 Application numbers per category:")
total = 0
for category, numbers in app_numbers.items():
    print(f"   {category:35s} → {len(numbers):>5} applications")
    total += len(numbers)

print(f"\n   Total applications to fetch: {total:,}")

# Create a flat set of all application numbers for quick lookup
all_app_nums = set()
for nums in app_numbers.values():
    all_app_nums.update(str(n) for n in nums)

print(f"   Unique application numbers: {len(all_app_nums):,}")


# ==========================================
# CELL 3: Stream HUPD and Extract Matching Patents
# ==========================================
# HUPD is ~360GB total, but we STREAM it — meaning we read
# one record at a time and only keep the ones we need.
# We never download the full 360GB.

print("⏳ Streaming HUPD dataset from HuggingFace...")
print("   This reads records one at a time (no 360GB download!)")
print("   It will take 30-60 minutes to scan through the dataset.")
print()

# Load HUPD in streaming mode
try:
    hupd = load_dataset(
        "HUPD/hupd",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
except Exception as e:
    print(f"⚠️  Error loading HUPD: {e}")
    print("   Try: load_dataset('HUPD/hupd', name='sample', split='train')")
    print("   This loads a smaller sample instead of the full dataset.")
    hupd = load_dataset("HUPD/hupd", name="sample", split="train", streaming=True)

# Collect matching records
matched_records = []
scanned = 0

for record in tqdm(hupd, desc="Scanning HUPD"):
    scanned += 1
    
    # Check if this application matches any of our target numbers
    app_num = str(record.get("application_number", ""))
    patent_num = str(record.get("patent_number", ""))
    
    if app_num in all_app_nums or patent_num in all_app_nums:
        matched_records.append({
            "application_number": app_num,
            "patent_number": patent_num,
            "title": record.get("title", ""),
            "abstract": record.get("abstract", ""),
            "claims": record.get("claims", ""),
            "description": record.get("description", ""),
            "cpc_codes": record.get("main_cpc_label", ""),
            "filing_date": record.get("filing_date", ""),
            "decision": record.get("decision", ""),
        })
        
        # Progress update every 50 matches
        if len(matched_records) % 50 == 0:
            print(f"\n   Found {len(matched_records)} matches so far (scanned {scanned:,})")
    
    # Safety: if we've scanned a lot and found most matches, we can stop
    if len(matched_records) >= len(all_app_nums) * 0.9:
        print(f"\n   Found 90%+ of target applications. Stopping early.")
        break
    
    # Progress checkpoint every 100K records
    if scanned % 100000 == 0:
        print(f"\n   Scanned {scanned:,} records, found {len(matched_records)} matches")

print(f"\n✅ Scan complete!")
print(f"   Scanned: {scanned:,} records")
print(f"   Matched: {len(matched_records)} applications")
print(f"   Target:  {len(all_app_nums)} applications")
print(f"   Hit rate: {len(matched_records)/len(all_app_nums)*100:.1f}%")


# ==========================================
# CELL 4: Organize by Category
# ==========================================
# Assign each matched record to its rare-case category

# Create a lookup: application_number → category
app_to_category = {}
for category, numbers in app_numbers.items():
    for num in numbers:
        app_to_category[str(num)] = category

# Organize records
categorized = {cat: [] for cat in app_numbers.keys()}

for record in matched_records:
    app_num = record["application_number"]
    cat = app_to_category.get(app_num)
    if cat:
        record["category"] = cat
        categorized[cat].append(record)

print("📋 Records per category:")
for cat, records in categorized.items():
    status = "✅" if len(records) >= 50 else "⚠️" if len(records) >= 20 else "❌"
    print(f"   {status} {cat:35s} → {len(records):>5} seed documents")


# ==========================================
# CELL 5: Clean and Structure Seed Documents
# ==========================================
# Clean up the text and create the final seed format

def clean_text(text):
    """Basic text cleaning for patent documents"""
    if not text or text == "None":
        return ""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove very short texts (likely errors)
    if len(text) < 50:
        return ""
    return text

all_seeds = []

for cat, records in categorized.items():
    for record in records:
        cleaned = {
            "application_number": record["application_number"],
            "category": cat,
            "title": clean_text(record.get("title", "")),
            "abstract": clean_text(record.get("abstract", "")),
            "claims": clean_text(record.get("claims", "")),
            "description": clean_text(record.get("description", "")),
            "cpc_codes": record.get("cpc_codes", ""),
            "filing_date": record.get("filing_date", ""),
            "decision": record.get("decision", ""),
            "full_text": ""  # Combined text for embedding later
        }
        
        # Create combined full_text field
        parts = []
        if cleaned["title"]:
            parts.append(f"Title: {cleaned['title']}")
        if cleaned["abstract"]:
            parts.append(f"Abstract: {cleaned['abstract']}")
        if cleaned["claims"]:
            parts.append(f"Claims: {cleaned['claims']}")
        if cleaned["description"]:
            parts.append(f"Description: {cleaned['description'][:2000]}")  # Truncate long descriptions
        
        cleaned["full_text"] = "\n\n".join(parts)
        
        # Only keep records with meaningful text
        if len(cleaned["full_text"]) > 200:
            all_seeds.append(cleaned)

print(f"✅ Cleaned {len(all_seeds)} seed documents")
print(f"   (Dropped {sum(len(r) for r in categorized.values()) - len(all_seeds)} empty/short records)")

# Show sample
if all_seeds:
    sample = all_seeds[0]
    print(f"\n📄 Sample seed document:")
    print(f"   App Number: {sample['application_number']}")
    print(f"   Category:   {sample['category']}")
    print(f"   Title:      {sample['title'][:100]}...")
    print(f"   Text length: {len(sample['full_text'])} characters")


# ==========================================
# CELL 6: Save Locally
# ==========================================
# Save seed documents as JSONL files (one per category + one combined)

import json

# Create output directory
os.makedirs("seeds", exist_ok=True)

# Save per category
for cat in app_numbers.keys():
    cat_seeds = [s for s in all_seeds if s["category"] == cat]
    filepath = f"seeds/{cat}.jsonl"
    with open(filepath, "w") as f:
        for seed in cat_seeds:
            f.write(json.dumps(seed) + "\n")
    print(f"💾 Saved {len(cat_seeds):>4} seeds → {filepath}")

# Save combined
with open("seeds/all_seeds.jsonl", "w") as f:
    for seed in all_seeds:
        f.write(json.dumps(seed) + "\n")

print(f"\n💾 Combined file: seeds/all_seeds.jsonl ({len(all_seeds)} records)")


# ==========================================
# CELL 7: Upload to S3
# ==========================================
# Upload seed documents to your S3 bucket
#
# ⚠️ REPLACE these with your actual AWS credentials
#    (or skip this cell and upload manually later)

AWS_ACCESS_KEY = "YOUR_ACCESS_KEY_HERE"      # ← Replace this
AWS_SECRET_KEY = "YOUR_SECRET_KEY_HERE"       # ← Replace this
S3_BUCKET = "synthipdata"                     # ← Replace if different
AWS_REGION = "us-west-2"                      # ← Replace if different

if "YOUR_ACCESS_KEY" not in AWS_ACCESS_KEY:
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        # Create bucket if it doesn't exist
        try:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
            )
            print(f"✅ Created S3 bucket: {S3_BUCKET}")
        except s3.exceptions.BucketAlreadyOwnedByYou:
            print(f"✅ S3 bucket already exists: {S3_BUCKET}")
        except Exception as e:
            print(f"   Bucket may already exist: {e}")
        
        # Upload each file
        for filename in os.listdir("seeds"):
            filepath = f"seeds/{filename}"
            s3_key = f"seeds/{filename}"
            s3.upload_file(filepath, S3_BUCKET, s3_key)
            print(f"   ⬆️  Uploaded: s3://{S3_BUCKET}/{s3_key}")
        
        print(f"\n✅ All seeds uploaded to S3!")
        
    except Exception as e:
        print(f"⚠️  S3 upload failed: {e}")
        print("   You can upload manually later. Seeds are saved locally in /seeds/")
else:
    print("⚠️  AWS credentials not set. Skipping S3 upload.")
    print("   Replace AWS_ACCESS_KEY and AWS_SECRET_KEY above to enable upload.")
    print("   Seeds are saved locally in /seeds/ — you can upload manually later.")


# ==========================================
# CELL 8: Summary
# ==========================================

print("=" * 60)
print("📋 DAY 1 — SEED COLLECTION COMPLETE")
print("=" * 60)
print()
print("What you now have:")
print(f"  🌱 {len(all_seeds)} seed documents across 8 rare categories")
print(f"  💾 Saved locally in /seeds/ folder")
print(f"  ☁️  {'Uploaded to S3' if 'YOUR_ACCESS_KEY' not in AWS_ACCESS_KEY else 'Not yet uploaded to S3'}")
print()

print("Seeds per category:")
for cat in app_numbers.keys():
    count = len([s for s in all_seeds if s["category"] == cat])
    bar = "█" * (count // 10)
    print(f"  {cat:35s} → {count:>4} seeds {bar}")

print()
print("Each seed document contains:")
print("  - Full patent title")
print("  - Abstract")
print("  - Claims text")
print("  - Description (truncated to 2000 chars)")
print("  - CPC classification codes")
print("  - Filing date and decision")
print()
print("=" * 60)
print("NEXT: Run Notebook 03 (Embedding)")
print("  → Converts seed text into vectors using BGE-M3")
print("  → Stores vectors in Qdrant for similarity search")
print("  → Takes about 30-45 minutes")
print("=" * 60)
