# Data Sources

## Overview

SynthIPData uses 2 primary data sources, both free and publicly accessible.

## 1. HUPD — Harvard USPTO Patent Dataset

- **What**: Full text of US patent applications (title, abstract, claims, description) with metadata and decisions
- **Size**: ~360GB total (we stream and extract only matching records)
- **Access**: `load_dataset('HUPD/hupd', streaming=True)`
- **URL**: https://huggingface.co/datasets/HUPD/hupd
- **License**: CC BY 4.0
- **Citation**: Suzgun et al., "The Harvard USPTO Patent Dataset", NeurIPS 2023

## 2. USPTO Office Action Research Dataset

- **What**: Structured data on patent office actions including rejection types, examiner details, and application metadata
- **Size**: ~2-3GB (CSV)
- **Access**: Direct download from USPTO
- **URL**: https://www.uspto.gov/ip-policy/economic-research/research-datasets
- **License**: Public domain (US government data)

## How They Work Together

1. **Office Action Dataset** identifies which applications received rare rejection types
2. **HUPD** provides the full patent text for those applications
3. Records are matched on `application_number`

## Data Scope

- **Year range**: 2015–2024 (post-Alice Corp. v. CLS Bank, 2014)
- **Focus**: Examination and prosecution stage of the IP lifecycle
- **Filter**: 8 rare rejection-type × technology-area combinations

## Potential Future Sources

These sources could enrich the dataset in future versions:

| Source | What It Adds | Why Deferred |
|--------|-------------|--------------|
| PatEx | Detailed prosecution timelines | HUPD covers basics |
| PatentsView | Enriched metadata | HUPD metadata sufficient for v1 |
| Google Patents BigQuery | Cross-validation | Not needed for initial release |
| PTAB Decisions | Appeal reasoning | Different lifecycle stage |
