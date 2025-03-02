import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_gene_stats(input_file, output_file):
    """
    Generate statistics for each gene.
    
    Args:
        input_file (str): Path to filtered CSV file
        output_file (str): Path to output gene stats CSV file
    """
    # Load the filtered dataset
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Get all score columns
    score_cols = [col for col in df.columns if col.startswith("Score -")]
    print(f"Found {len(score_cols)} score columns")
    
    # Convert score columns to numeric, errors='coerce' will convert non-numeric values to NaN
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Group by Gene Symbol and calculate statistics
    print("Calculating gene statistics...")
    
    # First, handle the cell line separately
    cell_line_mode = df.groupby("Gene Symbol")["Characteristics [Cell Line]"].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else "HeLa"
    ).reset_index()
    
    # Count occurrences
    gene_counts = df.groupby("Gene Symbol").size().reset_index(name="count")
    
    # Calculate mean scores for each column
    score_means = df.groupby("Gene Symbol")[score_cols].mean().reset_index()
    
    # Merge the results
    gene_stats = pd.merge(cell_line_mode, gene_counts, on="Gene Symbol")
    gene_stats = pd.merge(gene_stats, score_means, on="Gene Symbol")
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gene_stats.to_csv(output_path, index=False)
    
    print(f"Saved gene statistics to {output_file}")
    
    # Print some example statistics
    print("\nExample statistics for common genes:")
    for gene in ["KIF11", "INCENP", "COPB1"]:
        if gene in gene_stats["Gene Symbol"].values:
            gene_row = gene_stats.loc[gene_stats["Gene Symbol"] == gene].iloc[0]
            print(f"\n{gene} statistics:")
            print(f"  Cell line: {gene_row['Characteristics [Cell Line]']}")
            print(f"  Count: {gene_row['count']}")
            
            # Print top 3 highest scores
            scores = [(col.replace('Score - ', ''), gene_row[col]) for col in score_cols if pd.notnull(gene_row[col])]
            scores.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top phenotypes by score:")
            for phenotype, score in scores[:3]:
                print(f"    {phenotype}: {score:.4f}")
    
    return gene_stats

def main():
    parser = argparse.ArgumentParser(description="Generate gene statistics")
    parser.add_argument("--input", type=str, default="filtered.csv", help="Path to filtered CSV file")
    parser.add_argument("--output", type=str, default="gene_stats.csv", help="Path to output gene stats CSV file")
    args = parser.parse_args()
    
    # Generate gene statistics
    gene_stats = generate_gene_stats(args.input, args.output)
    
    # Print some additional statistics
    print("\nGene statistics summary:")
    print(f"Total number of genes: {len(gene_stats)}")
    
    # Find the genes with the highest scores for some key phenotypes
    key_phenotypes = [
        "mitotic delay/arrest (automatic)",
        "cell death (automatic)",
        "increased proliferation (automatic)"
    ]
    
    print("\nGenes with highest scores for key phenotypes:")
    for phenotype in key_phenotypes:
        col = f"Score - {phenotype}"
        if col in gene_stats.columns:
            top_genes = gene_stats.sort_values(col, ascending=False).head(3)
            print(f"\nTop genes for {phenotype}:")
            for _, row in top_genes.iterrows():
                print(f"  {row['Gene Symbol']}: {row[col]:.4f}")

if __name__ == "__main__":
    main()
