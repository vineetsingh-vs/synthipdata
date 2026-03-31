# ============================================================
# SynthIPData — Notebook 01: Discovery
# ============================================================
# PURPOSE:  Download USPTO data, find rare patent categories
# RUN ON:   Google Colab (free tier)
# TIME:     ~30 minutes
# 
# INSTRUCTIONS:
# 1. Go to colab.research.google.com
# 2. Click "New Notebook"
# 3. Copy each CELL into a separate Colab cell
# 4. Run cells top to bottom (click play button)
# ============================================================


# ==========================
# CELL 1: Install Libraries
# ==========================

!pip install pandas matplotlib seaborn requests tqdm datasets -q

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import requests
import os

print("Libraries installed and imported!")


# =========================================
# CELL 2: Download Office Action Dataset
# =========================================
# This downloads the USPTO Office Action Research Dataset
# Source: https://www.uspto.gov/ip-policy/economic-research/research-datasets
#
# If the URL below doesn't work, go to the source page above
# and find the current download link for the Office Action dataset

import urllib.request

# USPTO Office Action dataset URL
# NOTE: Update this URL if it changes. Check the USPTO research datasets page.
OA_URL = "https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2024/office_action_research_dataset.csv.zip"

print("Downloading USPTO Office Action Dataset...")
print("   (This may take 5-10 minutes depending on your connection)")
print("   If the download fails, the URL may have changed.")
print("   Check: https://www.uspto.gov/ip-policy/economic-research/research-datasets")

try:
    urllib.request.urlretrieve(OA_URL, "oa_dataset.zip")
    print("Downloaded!")
    
    # Unzip
    import zipfile
    with zipfile.ZipFile("oa_dataset.zip", "r") as z:
        z.extractall("oa_data/")
    print("Unzipped!")
    
    # Find the CSV file
    csv_files = [f for f in os.listdir("oa_data/") if f.endswith(".csv")]
    print(f"Found files: {csv_files}")
    
except Exception as e:
    print(f"Download failed: {e}")
    print("")
    print("Don't worry! Here's what to do:")
    print("1. Go to: https://www.uspto.gov/ip-policy/economic-research/research-datasets")
    print("2. Find 'Office Action Research Dataset'")
    print("3. Download the CSV file manually")
    print("4. Upload it to Colab (click the folder icon on the left → upload)")
    print("5. Then continue to the next cell")


# ==========================================
# CELL 3: Load and Explore the Data
# ==========================================
# This loads the CSV and shows you what's inside

# Try to find the CSV file
import glob

csv_files = glob.glob("oa_data/*.csv") + glob.glob("*.csv")
if csv_files:
    csv_path = csv_files[0]
    print(f"Loading: {csv_path}")
else:
    # If download failed, user needs to upload manually
    csv_path = input("Enter the path to your CSV file: ")

# Load the data
print("Loading data (this may take a minute for large files)...")
df = pd.read_csv(csv_path, low_memory=False)

print(f"\nLoaded {len(df):,} records")
print(f"\nColumns in the dataset:")
for col in df.columns:
    print(f"   - {col}")

print(f"\nFirst 5 rows:")
df.head()


# ==========================================
# CELL 4: Understand the Data
# ==========================================
# Before finding rare cases, let's understand what we have

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"\nTotal records: {len(df):,}")
print(f"Date range: {df['filing_date'].min() if 'filing_date' in df.columns else 'N/A'} to {df['filing_date'].max() if 'filing_date' in df.columns else 'N/A'}")

# Show all column names and their types
print(f"\nColumn details:")
for col in df.columns:
    non_null = df[col].count()
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"   {col:40s} | {str(dtype):10s} | {non_null:>10,} values | {unique:>8,} unique")

# The key columns we need to find:
# - Rejection type (might be called: rejection_type, rejection_101, rejection_102, etc.)
# - Technology area (might be called: uspc_class, cpc_code, technology_center, etc.)
# - Date (might be called: filing_date, mail_date, etc.)

print("\nLooking for rejection-related columns:")
rej_cols = [c for c in df.columns if any(term in c.lower() for term in ['reject', '101', '102', '103', '112'])]
print(f"   Found: {rej_cols}")

print("\nLooking for technology/classification columns:")
tech_cols = [c for c in df.columns if any(term in c.lower() for term in ['cpc', 'uspc', 'class', 'tech', 'art_unit'])]
print(f"   Found: {tech_cols}")

print("\nLooking for date columns:")
date_cols = [c for c in df.columns if any(term in c.lower() for term in ['date', 'year'])]
print(f"   Found: {date_cols}")

print("\nIMPORTANT: Copy the column names above and share them with Claude.")
print("   Claude will help you write the exact filtering code for YOUR dataset.")


# ==========================================
# CELL 5: Filter to 2015-2024
# ==========================================
# We only want post-Alice (2014) data
# 
# You may need to adjust the column name below
# based on what you saw in Cell 4

# Try common date column names
date_col = None
for candidate in ['filing_date', 'mail_date', 'date', 'app_filing_date']:
    if candidate in df.columns:
        date_col = candidate
        break

if date_col:
    # Convert to datetime and extract year
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['year'] = df[date_col].dt.year
    
    # Filter to 2015-2024
    df_filtered = df[df['year'].between(2015, 2024)].copy()
    
    print(f"Date column used: {date_col}")
    print(f"Before filter: {len(df):,} records")
    print(f"After filter (2015-2024): {len(df_filtered):,} records")
    
    print(f"\nRecords per year:")
    print(df_filtered['year'].value_counts().sort_index())
else:
    print("Could not find a date column automatically.")
    print("   Check the column names from Cell 4 and set date_col manually:")
    print("   date_col = 'your_column_name_here'")
    df_filtered = df.copy()  # Use all data for now


# ==========================================
# CELL 6: Count Rejection Types
# ==========================================
# This is the KEY discovery step
# We count how often each rejection type appears
#
# The column names below may need adjustment
# based on your actual data from Cell 4

# Common patterns in USPTO Office Action datasets:
# Option A: Single column with rejection type
# Option B: Separate boolean columns (rejection_101=1, rejection_102=1, etc.)

# Try Option A first
rej_type_col = None
for candidate in ['rejection_type', 'rejection_ground', 'ifw_number']:
    if candidate in df_filtered.columns:
        rej_type_col = candidate
        break

if rej_type_col:
    print(f"Using rejection column: {rej_type_col}")
    print(f"\nRejection Type Counts:")
    print("=" * 50)
    counts = df_filtered[rej_type_col].value_counts()
    for rej_type, count in counts.items():
        pct = count / len(df_filtered) * 100
        bar = "█" * int(pct)
        print(f"  {str(rej_type):25s} → {count:>10,} ({pct:5.1f}%) {bar}")
else:
    # Try Option B: boolean columns
    print("Looking for individual rejection columns...")
    rej_bool_cols = [c for c in df_filtered.columns if any(
        term in c.lower() for term in ['101', '102', '103', '112', 'double_pat', 'restrict']
    )]
    
    if rej_bool_cols:
        print(f"\nFound boolean rejection columns: {rej_bool_cols}")
        print(f"\nRejection Type Counts:")
        print("=" * 50)
        for col in rej_bool_cols:
            count = df_filtered[col].sum() if df_filtered[col].dtype in ['int64', 'float64', 'bool'] else df_filtered[col].value_counts().get(1, 0)
            pct = count / len(df_filtered) * 100
            bar = "█" * int(pct)
            print(f"  {col:25s} → {count:>10,} ({pct:5.1f}%) {bar}")
    else:
        print("Could not find rejection type columns automatically.")
        print("   Share the column names from Cell 4 with Claude for help.")


# ==========================================
# CELL 7: Find Rare Combinations
# ==========================================
# Cross-reference rejection types with technology areas
# THIS is where rare cases reveal themselves

# Try to find technology area column
tech_col = None
for candidate in ['uspc_class', 'cpc_group', 'cpc_code', 'technology_center', 'art_unit', 'gau']:
    if candidate in df_filtered.columns:
        tech_col = candidate
        break

if tech_col and rej_type_col:
    print(f"Cross-referencing: {rej_type_col} × {tech_col}")
    print("=" * 60)
    
    # Count combinations
    combos = df_filtered.groupby([rej_type_col, tech_col]).size().reset_index(name='count')
    combos = combos.sort_values('count')
    
    # Find the rare ones (bottom 10%)
    threshold = combos['count'].quantile(0.10)
    rare = combos[combos['count'] <= threshold]
    
    print(f"\nTotal unique combinations: {len(combos):,}")
    print(f"Rarity threshold (bottom 10%): {threshold:.0f} or fewer cases")
    print(f"Rare combinations found: {len(rare):,}")
    
    print(f"\nTOP 30 RAREST COMBINATIONS:")
    print("-" * 60)
    for _, row in rare.head(30).iterrows():
        print(f"  {str(row[rej_type_col]):20s} + {str(row[tech_col]):20s} → {row['count']:>6,} cases")
    
    print(f"\nTOP 10 MOST COMMON COMBINATIONS:")
    print("-" * 60)
    for _, row in combos.tail(10).iterrows():
        print(f"  {str(row[rej_type_col]):20s} + {str(row[tech_col]):20s} → {row['count']:>6,} cases")

elif tech_col:
    # Just show technology area distribution
    print(f"Technology area distribution ({tech_col}):")
    print(df_filtered[tech_col].value_counts().head(20))
    print("\nNeed rejection type column to find rare combinations.")
    print("   Share column names with Claude for help.")
else:
    print("Could not find technology area column automatically.")
    print("   Share the column names from Cell 4 with Claude for help.")


# ==========================================
# CELL 8: Map to Our 8 Target Categories
# ==========================================
# Now let's check how many real examples exist for each of
# our pre-defined rare categories

# CPC code to tech area mapping
cpc_to_tech = {
    "A61": "Biotech/Pharma",
    "C07": "Biotech/Pharma",
    "C12": "Biotech/Pharma",
    "G06N": "AI/Machine Learning",
    "B82": "Materials/Nanotech",
    "C01B": "Materials/Nanotech",
    "H01L": "Semiconductors"
}

# Our 8 target categories
target_categories = [
    {"id": "101_biotech", "name": "§101 in Biotech", "rejection": "101", "cpc_prefixes": ["A61", "C07", "C12"]},
    {"id": "101_ai_ml", "name": "§101 in AI/ML", "rejection": "101", "cpc_prefixes": ["G06N"]},
    {"id": "dp_ai_ml", "name": "Double Patenting in AI/ML", "rejection": "double_patenting", "cpc_prefixes": ["G06N"]},
    {"id": "112f_biotech", "name": "§112(f) in Biotech", "rejection": "112f", "cpc_prefixes": ["A61", "C07", "C12"]},
    {"id": "restriction_ai_ml", "name": "Restriction in AI/ML", "rejection": "restriction", "cpc_prefixes": ["G06N"]},
    {"id": "112a_ai_ml", "name": "§112(a) in AI/ML", "rejection": "112a", "cpc_prefixes": ["G06N"]},
    {"id": "101_nanotech", "name": "§101 in Nanotech", "rejection": "101", "cpc_prefixes": ["B82", "C01B"]},
    {"id": "dp_semiconductors", "name": "Double Patenting in Semicon", "rejection": "double_patenting", "cpc_prefixes": ["H01L"]},
]

print("=" * 70)
print("SEED AVAILABILITY CHECK — Our 8 Target Categories")
print("=" * 70)
print()
print(f"NOTE: This cell requires the correct column names from YOUR dataset.")
print(f"   If the counts below show 0 for everything, the column names need fixing.")
print(f"   Share the Cell 4 output with Claude and he'll fix it for you.")
print()

# Try to count (this depends on actual column names)
if rej_type_col and tech_col:
    for cat in target_categories:
        # Filter by rejection type
        mask_rej = df_filtered[rej_type_col].astype(str).str.contains(cat["rejection"], case=False, na=False)
        
        # Filter by CPC code
        mask_tech = False
        for prefix in cat["cpc_prefixes"]:
            mask_tech = mask_tech | df_filtered[tech_col].astype(str).str.startswith(prefix)
        
        count = len(df_filtered[mask_rej & mask_tech])
        status = "" if count >= 50 else "" if count >= 20 else ""
        
        print(f"  {status} {cat['name']:35s} → {count:>6,} real cases found")
        
    print()
    print("= Enough seeds (50+)")
    print("= Low but usable (20-49)")
    print("= Very few, may need to broaden search (<20)")
else:
    print("Cannot count yet — need correct column names.")
    print("   Share Cell 4 output with Claude to proceed.")


# ==========================================
# CELL 9: Get Application Numbers
# ==========================================
# For each category, extract the list of application numbers
# These will be used to pull full text from HUPD

if rej_type_col and tech_col:
    app_num_col = None
    for candidate in ['application_number', 'app_id', 'patent_id', 'application_id']:
        if candidate in df_filtered.columns:
            app_num_col = candidate
            break
    
    if app_num_col:
        print("=" * 60)
        print("EXTRACTING APPLICATION NUMBERS PER CATEGORY")
        print("=" * 60)
        
        all_app_numbers = {}
        
        for cat in target_categories:
            mask_rej = df_filtered[rej_type_col].astype(str).str.contains(cat["rejection"], case=False, na=False)
            mask_tech = False
            for prefix in cat["cpc_prefixes"]:
                mask_tech = mask_tech | df_filtered[tech_col].astype(str).str.startswith(prefix)
            
            app_numbers = df_filtered[mask_rej & mask_tech][app_num_col].unique().tolist()
            all_app_numbers[cat["id"]] = app_numbers
            
            print(f"  {cat['name']:35s} → {len(app_numbers):>5} application numbers")
        
        # Save to file
        import json
        with open("application_numbers_by_category.json", "w") as f:
            json.dump(all_app_numbers, f, indent=2, default=str)
        
        print(f"\nSaved to: application_numbers_by_category.json")
        print(f"   Total unique applications: {sum(len(v) for v in all_app_numbers.values()):,}")
    else:
        print("Could not find application number column.")
        print("   Check Cell 4 output for the correct column name.")


# ==========================================
# CELL 10: Visualization
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Rejection type distribution
if rej_type_col:
    counts = df_filtered[rej_type_col].value_counts().head(10)
    counts.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Top 10 Rejection Types (2015-2024)', fontsize=13)
    axes[0].set_xlabel('Count')

# Plot 2: Our categories seed counts
if rej_type_col and tech_col:
    cat_names = []
    cat_counts = []
    for cat in target_categories:
        mask_rej = df_filtered[rej_type_col].astype(str).str.contains(cat["rejection"], case=False, na=False)
        mask_tech = False
        for prefix in cat["cpc_prefixes"]:
            mask_tech = mask_tech | df_filtered[tech_col].astype(str).str.startswith(prefix)
        count = len(df_filtered[mask_rej & mask_tech])
        cat_names.append(cat["name"])
        cat_counts.append(count)
    
    colors = ['#d32f2f' if c < 50 else '#ff9800' if c < 200 else '#4caf50' for c in cat_counts]
    axes[1].barh(cat_names, cat_counts, color=colors)
    axes[1].set_title('Seed Availability per Category', fontsize=13)
    axes[1].set_xlabel('Real Cases Found')

plt.tight_layout()
plt.savefig("discovery_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: discovery_results.png")


# ==========================================
# CELL 11: Summary & Next Steps
# ==========================================

print("=" * 60)
print("DAY 1 — DISCOVERY COMPLETE")
print("=" * 60)
print()
print("What you now have:")
print("  1. USPTO Office Action data loaded and filtered (2015-2024)")
print("  2. Rejection type frequency counts")
print("  3. Rare-case categories identified")
print("  4. Application numbers extracted per category")
print("  5. Visualization of data distribution")
print()
print("Files created:")
print("  application_numbers_by_category.json")
print("  discovery_results.png")
print()
print("=" * 60)
print("NEXT STEP: Run Notebook 02 (Seed Collection)")
print("  → Uses the application numbers to pull full text from HUPD")
print("  → Takes about 1-1.5 hours")
print("=" * 60)
print()
print("If any cells showed warnings about column names,")
print("   copy the Cell 4 output and share it with Claude.")
print("   Claude will give you the exact fix.")
