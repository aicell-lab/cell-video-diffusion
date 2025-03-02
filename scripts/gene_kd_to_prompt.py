import pandas as pd
import argparse
from pathlib import Path

def filter_csv_by_plates(input_file, output_file, plates):
    """
    Filter a CSV file to only include rows from specified plates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        plates (list): List of plate IDs to keep
    """
    # Load the dataset
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Get original row count
    original_count = len(df)
    
    # Filter by plates
    df_filtered = df[df["Plate"].isin(plates)]
    
    # Get filtered row count
    filtered_count = len(df_filtered)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    
    print(f"Original dataset: {original_count} rows")
    print(f"Filtered dataset: {filtered_count} rows ({filtered_count/original_count*100:.2f}% of original)")
    print(f"Saved filtered dataset to {output_file}")
    
    # Return the filtered dataframe for further processing
    return df_filtered

def main():
    parser = argparse.ArgumentParser(description="Filter CSV by plates and generate gene knockdown prompts")
    parser.add_argument("--output", type=str, required=True, help="Path to output filtered CSV file")
    args = parser.parse_args()
    
    # Hardcoded input path
    input_file = "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/idr0013-screenA-annotation.csv"
    
    # Hardcoded plates to keep
    plates = [
        'LT0001_02', 'LT0001_09', 'LT0001_12', 
        'LT0002_02', 'LT0002_24', 'LT0002_51', 
        'LT0003_02', 'LT0003_15', 'LT0003_40', 
        'LT0004_06'
    ]
    
    # Filter the CSV
    filtered_df = filter_csv_by_plates(input_file, args.output, plates)
    
    # Print plate distribution in filtered dataset
    plate_counts = filtered_df["Plate"].value_counts()
    print("\nRows per plate in filtered dataset:")
    for plate, count in plate_counts.items():
        print(f"{plate}: {count} rows")
    
    # Print some statistics about the filtered dataset
    print("\nFiltered dataset statistics:")
    print(f"Number of unique gene symbols: {filtered_df['Gene Symbol'].nunique()}")
    print(f"Number of rows with missing gene symbols: {filtered_df['Gene Symbol'].isna().sum()}")
    
    # Print the top 10 most common genes in the filtered dataset
    top_genes = filtered_df['Gene Symbol'].value_counts().head(10)
    print("\nTop 10 most common genes in filtered dataset:")
    for gene, count in top_genes.items():
        print(f"{gene}: {count} rows")

if __name__ == "__main__":
    main()
