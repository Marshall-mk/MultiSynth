#!/usr/bin/env python3
"""
Merge two training log CSV files, handling overlapping epochs and different columns.
"""
import pandas as pd
import argparse
from pathlib import Path


def merge_training_logs(file1_path, file2_path, output_path=None):
    """
    Merge two training log CSV files.

    Args:
        file1_path: Path to first CSV file (earlier run)
        file2_path: Path to second CSV file (later run, takes priority)
        output_path: Path for merged output (optional)

    Returns:
        Path to the merged CSV file
    """
    # Read both CSV files
    print(f"Reading {file1_path}...")
    df1 = pd.read_csv(file1_path)
    print(f"  - Found {len(df1)} rows, epochs {df1['epoch'].min()}-{df1['epoch'].max()}")

    print(f"Reading {file2_path}...")
    df2 = pd.read_csv(file2_path)
    print(f"  - Found {len(df2)} rows, epochs {df2['epoch'].min()}-{df2['epoch'].max()}")

    # Identify the overlap
    min_epoch_2 = df2['epoch'].min()
    max_epoch_1 = df1['epoch'].max()

    if min_epoch_2 <= max_epoch_1:
        overlap_start = min_epoch_2
        overlap_end = max_epoch_1
        print(f"\nDetected overlap: epochs {overlap_start}-{overlap_end}")
        print(f"  - Keeping epochs {df1['epoch'].min()}-{min_epoch_2-1} from first file")
        print(f"  - Removing epochs {overlap_start}-{overlap_end} from first file")
        print(f"  - Keeping all epochs from second file")
    else:
        print(f"\nNo overlap detected")
        overlap_start = None

    # Filter first dataframe to remove overlapping epochs
    if overlap_start is not None:
        df1_filtered = df1[df1['epoch'] < min_epoch_2].copy()
        print(f"  - First file after filtering: {len(df1_filtered)} rows")
    else:
        df1_filtered = df1.copy()

    # Get all columns from both dataframes
    all_columns = list(df1.columns) + [col for col in df2.columns if col not in df1.columns]

    # Identify new columns
    new_columns = [col for col in df2.columns if col not in df1.columns]
    if new_columns:
        print(f"\nNew columns in second file: {new_columns}")
        print(f"  - Adding these columns to first file with 0.0 values")

    # Add missing columns to df1_filtered with 0.0 values
    for col in new_columns:
        df1_filtered[col] = 0.0

    # Reorder columns in df1_filtered to match df2
    df1_filtered = df1_filtered[df2.columns]

    # Concatenate the dataframes
    merged_df = pd.concat([df1_filtered, df2], ignore_index=True)

    # Sort by epoch to ensure correct order
    merged_df = merged_df.sort_values('epoch').reset_index(drop=True)

    print(f"\nMerged dataframe:")
    print(f"  - Total rows: {len(merged_df)}")
    print(f"  - Epoch range: {merged_df['epoch'].min()}-{merged_df['epoch'].max()}")
    print(f"  - Total columns: {len(merged_df.columns)}")

    # Determine output path
    if output_path is None:
        # Create output filename based on input files
        base_dir = Path(file1_path).parent
        output_path = base_dir / "training_log_merged.csv"

    # Save merged dataframe
    print(f"\nSaving merged file to: {output_path}")
    merged_df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge two training log CSV files, handling overlapping epochs and different columns.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two log files (output will be training_log_merged.csv in same directory)
  python merge_training_logs.py -i log1.csv log2.csv

  # Merge with custom output file
  python merge_training_logs.py -i log1.csv log2.csv -o merged_logs.csv
        """
    )

    parser.add_argument('-i', '--input', nargs=2, required=True, metavar=('FILE1', 'FILE2'),
                        help='Two CSV files to merge (earlier run first, later run second)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output CSV file path (default: training_log_merged.csv in same directory as first file)')

    args = parser.parse_args()

    result_path = merge_training_logs(args.input[0], args.input[1], args.output)
    print(f"\n✓ Successfully merged training logs to: {result_path}")
