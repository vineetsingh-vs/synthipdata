"""
SynthIPData — S3 Upload Helper
Upload seed documents and pipeline outputs to AWS S3.

Usage:
    python scripts/upload_s3.py --dir seeds/ --prefix seeds/
    python scripts/upload_s3.py --dir results/ --prefix eval/
"""

import os
import argparse
import boto3
from tqdm import tqdm


def upload_directory(s3_client, local_dir, bucket, s3_prefix):
    """Upload all files in a directory to S3."""
    files = []
    for root, dirs, filenames in os.walk(local_dir):
        for filename in filenames:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path)
            files.append((local_path, s3_key))
    
    print(f"📤 Uploading {len(files)} files to s3://{bucket}/{s3_prefix}")
    
    for local_path, s3_key in tqdm(files, desc="Uploading"):
        s3_client.upload_file(local_path, bucket, s3_key)
    
    print(f"✅ Upload complete!")


def main():
    parser = argparse.ArgumentParser(description="Upload files to S3")
    parser.add_argument("--dir", required=True, help="Local directory to upload")
    parser.add_argument("--prefix", required=True, help="S3 prefix (folder)")
    parser.add_argument("--bucket", default="synthipdata", help="S3 bucket name")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    args = parser.parse_args()
    
    # Check for AWS credentials
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if not access_key or not secret_key:
        print("⚠️  AWS credentials not found in environment variables.")
        print("   Set them with:")
        print('   export AWS_ACCESS_KEY_ID="your_key"')
        print('   export AWS_SECRET_ACCESS_KEY="your_secret"')
        return
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=args.region
    )
    
    upload_directory(s3, args.dir, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
