#!/usr/bin/env python3
"""
Disease & Symptoms Dataset Explorer

Analyzes the Disease and Symptoms Dataset 2023 from Mendeley Data.

"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Default dataset path (relative to repo root)
DEFAULT_INPUT = "Disease and symptoms dataset.csv"
DEFAULT_OUTPUT = "out/explore"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV dataset into a DataFrame."""
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """Extract basic dataset information."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "disease_column": df.columns[0],
        "symptom_columns": list(df.columns[1:]),
        "num_symptoms": len(df.columns) - 1,
    }


def analyze_diseases(df: pd.DataFrame) -> dict:
    """Analyze disease distribution."""
    disease_col = df.columns[0]
    disease_counts = df[disease_col].value_counts()

    return {
        "unique_diseases": disease_counts.nunique(),
        "total_records": len(df),
        "top_10_diseases": disease_counts.head(10).to_dict(),
        "bottom_10_diseases": disease_counts.tail(10).to_dict(),
        "avg_records_per_disease": round(disease_counts.mean(), 2),
        "max_records": int(disease_counts.max()),
        "min_records": int(disease_counts.min()),
    }


def analyze_symptoms(df: pd.DataFrame) -> dict:
    """Analyze symptom patterns."""
    symptom_cols = df.columns[1:]
    symptom_sums = df[symptom_cols].sum().sort_values(ascending=False)

    # Symptoms per record
    symptoms_per_row = df[symptom_cols].sum(axis=1)

    return {
        "total_symptoms": len(symptom_cols),
        "most_common_symptoms": symptom_sums.head(15).to_dict(),
        "least_common_symptoms": symptom_sums.tail(10).to_dict(),
        "avg_symptoms_per_record": round(symptoms_per_row.mean(), 2),
        "max_symptoms_per_record": int(symptoms_per_row.max()),
        "min_symptoms_per_record": int(symptoms_per_row.min()),
    }


def print_summary(info: dict, diseases: dict, symptoms: dict):
    """Print a formatted summary to the console."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows:            {info['rows']:,}")
    print(f"  Columns:         {info['columns']:,}")
    print(f"  Memory:          {info['memory_mb']} MB")
    print(f"  Disease column:  {info['disease_column']}")
    print(f"  Symptom columns: {info['num_symptoms']}")

    print("\n" + "-" * 60)
    print("DISEASE ANALYSIS")
    print("-" * 60)
    print(f"  Unique diseases:         {diseases['unique_diseases']}")
    print(f"  Avg records per disease: {diseases['avg_records_per_disease']}")
    print(f"  Max records:             {diseases['max_records']}")
    print(f"  Min records:             {diseases['min_records']}")

    print("\n  Top 10 diseases:")
    for disease, count in diseases["top_10_diseases"].items():
        print(f"    - {disease}: {count:,}")

    print("\n" + "-" * 60)
    print("SYMPTOM ANALYSIS")
    print("-" * 60)
    print(f"  Total symptoms:           {symptoms['total_symptoms']}")
    print(f"  Avg symptoms per record:  {symptoms['avg_symptoms_per_record']}")
    print(f"  Max symptoms per record:  {symptoms['max_symptoms_per_record']}")
    print(f"  Min symptoms per record:  {symptoms['min_symptoms_per_record']}")

    print("\n  Top 15 most common symptoms:")
    for symptom, count in symptoms["most_common_symptoms"].items():
        print(f"    - {symptom}: {count:,}")

    print("\n" + "=" * 60)


def create_plots(df: pd.DataFrame, output_dir: Path):
    """Generate and save visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    disease_col = df.columns[0]
    symptom_cols = df.columns[1:]

    # Plot 1: Top 20 diseases by frequency
    plt.figure(figsize=(12, 8))
    disease_counts = df[disease_col].value_counts().head(20)
    disease_counts.plot(kind="barh", color="steelblue")
    plt.xlabel("Number of Records")
    plt.ylabel("Disease")
    plt.title("Top 20 Diseases by Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "top_diseases.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'top_diseases.png'}")

    # Plot 2: Top 20 symptoms by occurrence
    plt.figure(figsize=(12, 8))
    symptom_sums = df[symptom_cols].sum().sort_values(ascending=False).head(20)
    symptom_sums.plot(kind="barh", color="coral")
    plt.xlabel("Number of Occurrences")
    plt.ylabel("Symptom")
    plt.title("Top 20 Most Common Symptoms")
    plt.tight_layout()
    plt.savefig(output_dir / "top_symptoms.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'top_symptoms.png'}")

    # Plot 3: Distribution of symptoms per record
    plt.figure(figsize=(10, 6))
    symptoms_per_row = df[symptom_cols].sum(axis=1)
    symptoms_per_row.hist(bins=30, color="seagreen", edgecolor="black")
    plt.xlabel("Number of Symptoms")
    plt.ylabel("Number of Records")
    plt.title("Distribution of Symptoms per Record")
    plt.tight_layout()
    plt.savefig(output_dir / "symptoms_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'symptoms_distribution.png'}")


def save_summary(info: dict, diseases: dict, symptoms: dict, output_dir: Path):
    """Save summary as JSON."""
    summary = {
        "dataset_info": info,
        "disease_analysis": diseases,
        "symptom_analysis": symptoms,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore the Disease and Symptoms Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data_explore.py                    # Full analysis
  python scripts/data_explore.py --quick            # Quick summary only
  python scripts/data_explore.py -i data.csv -o out # Custom paths
        """,
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to dataset CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: console summary only, no plots or files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Check if input exists
    if not input_path.exists():
        print(f"Error: Dataset not found at '{input_path}'")
        print("\nTo download the dataset:")
        print("  1. Visit: https://data.mendeley.com/datasets/2cxccsxydc/1")
        print("  2. Click 'Download All'")
        print(f"  3. Place the CSV file at: {input_path}")
        sys.exit(1)

    # Load and analyze
    df = load_dataset(input_path)
    info = get_basic_info(df)
    diseases = analyze_diseases(df)
    symptoms = analyze_symptoms(df)

    # Print summary
    print_summary(info, diseases, symptoms)

    # Quick mode: stop here
    if args.quick:
        print("\n[Quick mode - skipping file output]")
        return

    # Create output directory and save results
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nSaving results...")
    save_summary(info, diseases, symptoms, output_dir)
    create_plots(df, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
