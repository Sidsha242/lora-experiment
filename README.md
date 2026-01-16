# LoRA Experiment — Disease & Symptoms Dataset

This repository explores the **Disease and Symptoms Dataset 2023** for the UMass NLP Fall 2025 LoRA experiment.

## Dataset

**Source:** [Disease and Symptoms Dataset 2023](https://data.mendeley.com/datasets/2cxccsxydc/1)

| Property | Value |
|----------|-------|
| Diseases | 773 unique |
| Symptoms | 377 unique |
| Rows | ~246,000 |
| Format | CSV (binary symptom flags) |
| License | CC BY 4.0 |
| DOI | 10.17632/2cxccsxydc.1 |

### Download

1. Visit: https://data.mendeley.com/datasets/2cxccsxydc/1
2. Click **Download All** (or download individual files)
3. Place `Disease and symptoms dataset.csv` in the repository root

Or use the command line:

```bash
# Using curl (the dataset is ~180MB)
curl -L -o "Disease and symptoms dataset.csv" \
  "https://data.mendeley.com/public-files/datasets/2cxccsxydc/files/Disease%20and%20symptoms%20dataset.csv/download"
```

## Quick Start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. Run the exploration script:

```bash
# Full analysis with plots
python scripts/data_explore.py

# Quick console summary (no plots)
python scripts/data_explore.py --quick

# Custom input/output paths
python scripts/data_explore.py --input path/to/data.csv --output out/results
```

## What the Script Does

- **Dataset Overview**: Row/column counts, memory usage, data types
- **Disease Analysis**: Top diseases by frequency, distribution stats
- **Symptom Analysis**: Most/least common symptoms, correlation patterns
- **Visualizations**: Bar charts for top diseases and symptoms (saved as PNG)
- **JSON Summary**: Machine-readable statistics for downstream processing

## Files

| File | Description |
|------|-------------|
| `README.md` | This file |
| `scripts/data_explore.py` | Dataset exploration script |
| `requirements.txt` | Python dependencies |
| `Disease and symptoms dataset.csv` | The dataset (download separately) |

## Citation

If you use this dataset, please cite:

```
Stark, Bran (2025), "Disease and symptoms dataset 2023", Mendeley Data, V1, doi: 10.17632/2cxccsxydc.1
```
