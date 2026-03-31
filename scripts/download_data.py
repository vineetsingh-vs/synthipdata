"""
SynthIPData — Download USPTO Datasets
Run this first to get the raw data needed for discovery.
"""

import os
import urllib.request
import zipfile
import argparse


def download_file(url, output_path):
    """Download a file with progress reporting."""
    print(f"⬇️  Downloading: {url}")
    print(f"   Saving to: {output_path}")
    
    def progress_hook(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        print(f"\r   Progress: {pct:.1f}%", end="")
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Downloaded ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download USPTO datasets for SynthIPData")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SynthIPData — Data Download")
    print("=" * 60)
    
    # USPTO Office Action Research Dataset
    print("\n[1/1] USPTO Office Action Research Dataset")
    print("   Source: USPTO Economic Research")
    print("   Note: If URL is outdated, visit:")
    print("   https://www.uspto.gov/ip-policy/economic-research/research-datasets")
    
    oa_url = "https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2024/office_action_research_dataset.csv.zip"
    oa_path = os.path.join(args.output_dir, "oa_dataset.zip")
    
    try:
        download_file(oa_url, oa_path)
        
        # Unzip
        print("   Extracting...")
        with zipfile.ZipFile(oa_path, "r") as z:
            z.extractall(args.output_dir)
        print("   ✅ Extracted!")
        
    except Exception as e:
        print(f"   ⚠️  Download failed: {e}")
        print("   Please download manually from the USPTO website.")
    
    print("\n" + "=" * 60)
    print("HUPD will be streamed directly in Notebook 02.")
    print("No separate download needed for HUPD.")
    print("=" * 60)


if __name__ == "__main__":
    main()
