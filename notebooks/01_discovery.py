# ============================================================
# SynthIPData -- Notebook 01: Discovery
# ============================================================
# PURPOSE:  Download USPTO data, automatically discover rare
#           patent categories, and generate configs for the
#           rest of the pipeline.
#
# NOTHING IS HARDCODED. The data tells us what's rare.
#
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

!pip install pandas matplotlib seaborn requests tqdm datasets pyyaml -q

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import requests
import os
import json
import yaml

print("Libraries installed and imported!")


# =========================================
# CELL 2: Download Office Action Dataset
# =========================================
# This downloads the USPTO Office Action Research Dataset.
# Source: https://www.uspto.gov/ip-policy/economic-research/research-datasets
#
# If the URL doesn't work, go to the source page and find the current link.

import urllib.request
import zipfile

OA_URL = "https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2024/office_action_research_dataset.csv.zip"

print("Downloading USPTO Office Action Dataset...")
print("(This may take 5-10 minutes depending on your connection)")
print()

try:
    urllib.request.urlretrieve(OA_URL, "oa_dataset.zip")
    print("Download complete!")

    with zipfile.ZipFile("oa_dataset.zip", "r") as z:
        z.extractall("oa_data/")
    print("Unzipped!")

    csv_files = [f for f in os.listdir("oa_data/") if f.endswith(".csv")]
    print(f"Found files: {csv_files}")

except Exception as e:
    print(f"Download failed: {e}")
    print()
    print("Manual download steps:")
    print("1. Go to: https://www.uspto.gov/ip-policy/economic-research/research-datasets")
    print("2. Find 'Office Action Research Dataset'")
    print("3. Download the CSV file")
    print("4. Upload it to Colab (click the folder icon on the left, then upload)")
    print("5. Continue to the next cell")


# ==========================================
# CELL 3: Load the Data
# ==========================================

import glob

csv_files = glob.glob("oa_data/*.csv") + glob.glob("*.csv")
if csv_files:
    csv_path = csv_files[0]
    print(f"Loading: {csv_path}")
else:
    csv_path = input("Enter the path to your CSV file: ")

print("Loading data (this may take a minute for large files)...")
df = pd.read_csv(csv_path, low_memory=False)

print(f"Loaded {len(df):,} records")
print(f"\nColumns in the dataset:")
for col in df.columns:
    print(f"   - {col}")

print(f"\nFirst 3 rows:")
df.head(3)


# ==========================================
# CELL 4: Auto-Detect Column Types
# ==========================================
# The script automatically finds the right columns.
# If it guesses wrong, you can override manually at the bottom.

print("=" * 60)
print("AUTO-DETECTING COLUMN TYPES")
print("=" * 60)

# --- Find rejection type column(s) ---
rej_type_col = None
rej_bool_cols = []

# Option A: Single column with rejection type labels
for candidate in ['rejection_type', 'rejection_ground', 'rejection_basis',
                   'ground_type', 'statutory_basis', 'basis']:
    if candidate in df.columns:
        rej_type_col = candidate
        break

# Option B: Separate boolean columns (rejection_101=1, rejection_102=1)
if not rej_type_col:
    rej_bool_cols = [c for c in df.columns if any(
        term in c.lower() for term in ['101', '102', '103', '112', 'double_pat', 'restrict']
    )]

# --- Find technology/classification column ---
tech_col = None
for candidate in ['uspc_class', 'uspc_subclass', 'cpc_group', 'cpc_code',
                   'cpc_section', 'technology_center', 'art_unit', 'gau',
                   'tc', 'tech_center']:
    if candidate in df.columns:
        tech_col = candidate
        break

# --- Find date column ---
date_col = None
for candidate in ['filing_date', 'mail_date', 'mail_dt', 'date',
                   'app_filing_date', 'action_date']:
    if candidate in df.columns:
        date_col = candidate
        break

# --- Find application number column ---
app_num_col = None
for candidate in ['application_number', 'app_id', 'patent_id',
                   'application_id', 'appln_id', 'app_no']:
    if candidate in df.columns:
        app_num_col = candidate
        break

# --- Report findings ---
print(f"\nRejection type column:    {rej_type_col or 'NOT FOUND'}")
if rej_bool_cols:
    print(f"Rejection boolean cols:   {rej_bool_cols}")
print(f"Technology area column:   {tech_col or 'NOT FOUND'}")
print(f"Date column:              {date_col or 'NOT FOUND'}")
print(f"Application number col:   {app_num_col or 'NOT FOUND'}")

# --- Show sample values for detected columns ---
print("\n" + "-" * 60)
print("SAMPLE VALUES (verify these look correct)")
print("-" * 60)

if rej_type_col:
    print(f"\n{rej_type_col} (top 5 values):")
    print(df[rej_type_col].value_counts().head())

if tech_col:
    print(f"\n{tech_col} (top 5 values):")
    print(df[tech_col].value_counts().head())

if date_col:
    print(f"\n{date_col} (sample):")
    print(df[date_col].head())

# --- Manual override section ---
print("\n" + "=" * 60)
print("If any column above is wrong, uncomment and set manually:")
print("# rej_type_col = 'your_column_name'")
print("# tech_col = 'your_column_name'")
print("# date_col = 'your_column_name'")
print("# app_num_col = 'your_column_name'")
print("=" * 60)

# UNCOMMENT AND EDIT BELOW IF NEEDED:
# rej_type_col = 'your_column_name'
# tech_col = 'your_column_name'
# date_col = 'your_column_name'
# app_num_col = 'your_column_name'


# ==========================================
# CELL 5: Filter to 2015-2024
# ==========================================
# Post-Alice (2014) data only for consistent legal standards.

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['year'] = df[date_col].dt.year
    df_filtered = df[df['year'].between(2015, 2024)].copy()

    print(f"Date column: {date_col}")
    print(f"Before filter: {len(df):,} records")
    print(f"After filter (2015-2024): {len(df_filtered):,} records")
    print(f"\nRecords per year:")
    print(df_filtered['year'].value_counts().sort_index().to_string())
else:
    print("No date column found. Using all data.")
    df_filtered = df.copy()


# ==========================================
# CELL 6: Count Everything Automatically
# ==========================================
# This is where rare cases reveal themselves.
# We count EVERY rejection + technology combination
# and sort by frequency. No hardcoded categories.

print("=" * 60)
print("COUNTING ALL REJECTION x TECHNOLOGY COMBINATIONS")
print("=" * 60)

# --- Handle two data formats ---

# If we have a single rejection type column
if rej_type_col and tech_col:
    # Clean up values
    df_filtered[rej_type_col] = df_filtered[rej_type_col].astype(str).str.strip()
    df_filtered[tech_col] = df_filtered[tech_col].astype(str).str.strip()

    # Count all combinations
    combos = (df_filtered
              .groupby([rej_type_col, tech_col])
              .size()
              .reset_index(name='count'))
    combos = combos.sort_values('count', ascending=True)

    total_records = len(df_filtered)
    combos['percentage'] = (combos['count'] / total_records * 100).round(3)

    print(f"\nTotal records: {total_records:,}")
    print(f"Unique rejection types: {df_filtered[rej_type_col].nunique()}")
    print(f"Unique tech areas: {df_filtered[tech_col].nunique()}")
    print(f"Unique combinations: {len(combos):,}")

# If we have boolean rejection columns instead
elif rej_bool_cols and tech_col:
    df_filtered[tech_col] = df_filtered[tech_col].astype(str).str.strip()

    rows = []
    for col in rej_bool_cols:
        if df_filtered[col].dtype in ['int64', 'float64']:
            subset = df_filtered[df_filtered[col] == 1]
        else:
            subset = df_filtered[df_filtered[col] == True]
        for tech, group in subset.groupby(tech_col):
            rows.append({
                'rejection_type': col,
                tech_col: tech,
                'count': len(group)
            })

    combos = pd.DataFrame(rows).sort_values('count', ascending=True)
    total_records = len(df_filtered)
    combos['percentage'] = (combos['count'] / total_records * 100).round(3)
    rej_type_col = 'rejection_type'

    print(f"\nTotal records: {total_records:,}")
    print(f"Rejection columns: {len(rej_bool_cols)}")
    print(f"Unique tech areas: {df_filtered[tech_col].nunique()}")
    print(f"Unique combinations: {len(combos):,}")

else:
    print("ERROR: Could not find required columns.")
    print("Go back to Cell 4 and set the column names manually.")
    combos = pd.DataFrame()


# ==========================================
# CELL 7: Automatically Identify Rare Cases
# ==========================================
# We define "rare" based on percentiles calculated from YOUR data.
# Nothing is pre-defined. The script finds them automatically.

if len(combos) > 0:
    # Calculate rarity thresholds from the data
    p10 = combos['count'].quantile(0.10)
    p25 = combos['count'].quantile(0.25)
    median = combos['count'].quantile(0.50)

    # Classify each combination
    def classify_rarity(count):
        if count <= p10:
            return "VERY RARE"
        elif count <= p25:
            return "RARE"
        elif count <= median:
            return "UNCOMMON"
        else:
            return "COMMON"

    combos['rarity'] = combos['count'].apply(classify_rarity)

    print("=" * 60)
    print("RARITY ANALYSIS (auto-discovered from your data)")
    print("=" * 60)
    print(f"\nThresholds (calculated from the data):")
    print(f"  VERY RARE:  <= {p10:.0f} cases (bottom 10%)")
    print(f"  RARE:       <= {p25:.0f} cases (bottom 25%)")
    print(f"  UNCOMMON:   <= {median:.0f} cases (bottom 50%)")
    print(f"  COMMON:     >  {median:.0f} cases (top 50%)")

    print(f"\nDistribution:")
    print(combos['rarity'].value_counts().to_string())

    # Show all rare and very rare combinations
    rare_combos = combos[combos['rarity'].isin(['VERY RARE', 'RARE'])].copy()
    rare_combos = rare_combos.sort_values('count', ascending=True)

    print(f"\n" + "=" * 60)
    print(f"ALL RARE + VERY RARE COMBINATIONS ({len(rare_combos)} found)")
    print("=" * 60)
    print(f"{'#':>3}  {'Rejection Type':25s}  {'Tech Area':25s}  {'Count':>8}  {'%':>7}  {'Rarity'}")
    print("-" * 100)

    for i, (_, row) in enumerate(rare_combos.iterrows(), 1):
        print(f"{i:>3}  {str(row[rej_type_col]):25s}  {str(row[tech_col]):25s}  {row['count']:>8,}  {row['percentage']:>6.3f}%  {row['rarity']}")

    print(f"\nTotal rare combinations found: {len(rare_combos)}")
    print(f"These are your candidates for synthetic data augmentation.")
else:
    print("No data to analyze. Fix column detection in Cell 4.")


# ==========================================
# CELL 8: Select Categories
# ==========================================
# The data has shown you what's rare. Now pick which ones
# to target for your paper.
#
# TWO MODES:
#   "auto"   -- Script picks the top N rare categories that
#               have enough seed data (recommended for first run)
#   "manual" -- You pick specific row numbers from Cell 7 output
#
# Change the settings below:

SELECTION_MODE = "auto"   # "auto" or "manual"
AUTO_SELECT_COUNT = 8     # How many categories to auto-select
MIN_SEED_COUNT = 20       # Minimum cases needed (below this is too few to learn from)

# For manual mode, uncomment and list row numbers from Cell 7:
# MANUAL_SELECTIONS = [1, 3, 5, 7, 12, 15, 18, 22]

if len(combos) > 0:
    if SELECTION_MODE == "auto":
        # Pick the rarest combinations that still have enough seeds
        candidates = rare_combos[rare_combos['count'] >= MIN_SEED_COUNT].copy()

        if len(candidates) < AUTO_SELECT_COUNT:
            print(f"Only {len(candidates)} rare categories have >= {MIN_SEED_COUNT} cases.")
            print(f"Including UNCOMMON categories to reach {AUTO_SELECT_COUNT}.")
            uncommon = combos[combos['rarity'] == 'UNCOMMON']
            uncommon = uncommon[uncommon['count'] >= MIN_SEED_COUNT]
            candidates = pd.concat([candidates, uncommon])

        selected = candidates.head(AUTO_SELECT_COUNT).copy()

        print("=" * 60)
        print(f"AUTO-SELECTED {len(selected)} CATEGORIES")
        print("=" * 60)

    else:
        # Manual: pick by row numbers from Cell 7
        rare_indexed = rare_combos.reset_index(drop=True)
        indices = [i - 1 for i in MANUAL_SELECTIONS]
        selected = rare_indexed.iloc[indices].copy()

        print("=" * 60)
        print(f"MANUALLY SELECTED {len(selected)} CATEGORIES")
        print("=" * 60)

    # Display final selection
    print(f"\n{'#':>3}  {'Rejection Type':25s}  {'Tech Area':25s}  {'Seed Count':>10}  {'Rarity'}")
    print("-" * 80)
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"{i:>3}  {str(row[rej_type_col]):25s}  {str(row[tech_col]):25s}  {row['count']:>10,}  {row['rarity']}")

    print(f"\nTotal seed documents available: {selected['count'].sum():,}")
    print()
    print("Happy with these? Continue to the next cell.")
    print("Want different ones? Set SELECTION_MODE = 'manual' and pick row numbers from Cell 7.")


# ==========================================
# CELL 9: Generate Config Files Automatically
# ==========================================
# This creates:
#   categories.yaml                      -- Pipeline config
#   application_numbers_by_category.json -- App numbers per category
#   discovery_all_combinations.csv       -- Full analysis
#   discovery_rare_combinations.csv      -- Rare ones only
#
# Everything is generated FROM the data. Nothing hardcoded.

if len(selected) > 0:
    # --- Build categories.yaml ---
    categories_config = {
        'year_range': {
            'start': 2015,
            'end': 2024,
            'rationale': 'Post-Alice (2014) framework ensures consistent legal standards'
        },
        'discovery_stats': {
            'total_records': int(total_records),
            'rarity_threshold_p10': int(p10),
            'rarity_threshold_p25': int(p25),
            'median_count': int(median),
            'total_rare_combinations': int(len(rare_combos))
        },
        'categories': [],
        'evaluation': {
            'memorization_threshold': 0.95,
            'garbage_threshold': 0.30,
            'min_validity_score': 0.70
        }
    }

    all_app_numbers = {}

    print("=" * 60)
    print("GENERATING PIPELINE CONFIGS")
    print("=" * 60)
    print()

    for i, (_, row) in enumerate(selected.iterrows(), 1):
        rej_val = str(row[rej_type_col])
        tech_val = str(row[tech_col])
        count = int(row['count'])

        # Create a clean ID from the values
        clean_rej = rej_val.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        clean_tech = tech_val.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        cat_id = f"{clean_rej}_{clean_tech}"[:50]

        # Add to categories config
        categories_config['categories'].append({
            'id': cat_id,
            'name': f"{rej_val} in {tech_val}",
            'rejection_type': rej_val,
            'tech_area': tech_val,
            'seed_count': count,
            'rarity': row['rarity'],
            'synthetic_target': min(count * 10, 2000),
            'min_seeds_target': max(count, MIN_SEED_COUNT)
        })

        # Extract application numbers
        mask_rej = df_filtered[rej_type_col].astype(str).str.strip() == rej_val
        mask_tech = df_filtered[tech_col].astype(str).str.strip() == tech_val
        app_nums = df_filtered[mask_rej & mask_tech][app_num_col].unique().tolist()
        all_app_numbers[cat_id] = [str(x) for x in app_nums]

        print(f"  Category {i}: {rej_val} + {tech_val}")
        print(f"    ID: {cat_id}")
        print(f"    Seeds: {count} | Synthetic target: {min(count * 10, 2000)}")
        print(f"    Application numbers: {len(app_nums)}")
        print()

    # --- Save all files ---
    with open("categories.yaml", "w") as f:
        yaml.dump(categories_config, f, default_flow_style=False, sort_keys=False)
    print("Saved: categories.yaml")

    with open("application_numbers_by_category.json", "w") as f:
        json.dump(all_app_numbers, f, indent=2)
    print("Saved: application_numbers_by_category.json")

    combos.to_csv("discovery_all_combinations.csv", index=False)
    print("Saved: discovery_all_combinations.csv")

    rare_combos.to_csv("discovery_rare_combinations.csv", index=False)
    print("Saved: discovery_rare_combinations.csv")

    # --- Print the generated config ---
    print("\n" + "=" * 60)
    print("GENERATED categories.yaml:")
    print("=" * 60)
    print(yaml.dump(categories_config, default_flow_style=False, sort_keys=False))


# ==========================================
# CELL 10: Visualization
# ==========================================

if len(combos) > 0 and len(selected) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Overall rejection type distribution
    if rej_type_col:
        rej_counts = df_filtered[rej_type_col].value_counts().head(15)
        rej_counts.plot(kind='barh', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Top 15 Rejection Types (2015-2024)', fontsize=13)
        axes[0, 0].set_xlabel('Count')

    # Plot 2: Rarity distribution
    rarity_counts = combos['rarity'].value_counts()
    colors_map = {'VERY RARE': '#d32f2f', 'RARE': '#ff9800', 'UNCOMMON': '#ffeb3b', 'COMMON': '#4caf50'}
    rarity_colors = [colors_map.get(r, '#999999') for r in rarity_counts.index]
    rarity_counts.plot(kind='bar', ax=axes[0, 1], color=rarity_colors)
    axes[0, 1].set_title('Combination Rarity Distribution', fontsize=13)
    axes[0, 1].set_ylabel('Number of Combinations')
    axes[0, 1].tick_params(axis='x', rotation=0)

    # Plot 3: Selected categories seed counts
    cat_names = [f"{row[rej_type_col]}\n{row[tech_col]}" for _, row in selected.iterrows()]
    cat_counts = selected['count'].tolist()
    bar_colors = ['#d32f2f' if r == 'VERY RARE' else '#ff9800' for r in selected['rarity']]
    axes[1, 0].barh(cat_names, cat_counts, color=bar_colors)
    axes[1, 0].set_title('Selected Categories -- Seed Availability', fontsize=13)
    axes[1, 0].set_xlabel('Real Cases Available')

    # Plot 4: Log-scale frequency distribution
    combos['count'].plot(kind='hist', bins=50, ax=axes[1, 1],
                         color='steelblue', alpha=0.7, log=True)
    axes[1, 1].axvline(x=p10, color='red', linestyle='--',
                        label=f'Very Rare threshold ({p10:.0f})')
    axes[1, 1].axvline(x=p25, color='orange', linestyle='--',
                        label=f'Rare threshold ({p25:.0f})')
    axes[1, 1].set_title('Frequency Distribution (log scale)', fontsize=13)
    axes[1, 1].set_xlabel('Cases per Combination')
    axes[1, 1].set_ylabel('Number of Combinations (log)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("discovery_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Saved: discovery_results.png")


# ==========================================
# CELL 11: Summary
# ==========================================

print("=" * 60)
print("DAY 1 -- DISCOVERY COMPLETE")
print("=" * 60)
print()
print("What happened:")
print(f"  1. Loaded {total_records:,} USPTO office action records (2015-2024)")
print(f"  2. Found {len(combos):,} unique rejection x technology combinations")
print(f"  3. Identified {len(rare_combos):,} rare combinations (bottom 25%)")
print(f"  4. Selected {len(selected)} categories for synthetic augmentation")
print(f"  5. Extracted {sum(len(v) for v in all_app_numbers.values()):,} application numbers")
print()
print("Files created:")
print("  categories.yaml                      -- Pipeline config (auto-generated)")
print("  application_numbers_by_category.json -- App numbers per category")
print("  discovery_all_combinations.csv       -- Full frequency analysis")
print("  discovery_rare_combinations.csv      -- Rare combinations only")
print("  discovery_results.png                -- Visualization")
print()
print("Key difference from a hardcoded approach:")
print("  Categories were discovered FROM the data, not pre-defined.")
print("  Rarity thresholds were calculated FROM the data.")
print("  Re-run on different data and categories update automatically.")
print()
print("=" * 60)
print("NEXT: Run Notebook 02 (Seed Collection)")
print("  Uses application_numbers_by_category.json to pull full text from HUPD.")
print("  Takes about 1-1.5 hours.")
print("=" * 60)
print()
print("To change selected categories:")
print("  Set SELECTION_MODE = 'manual' in Cell 8")
print("  List row numbers from Cell 7 output")
print("  Re-run Cells 8-11")
