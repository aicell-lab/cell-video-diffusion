import pandas as pd
import argparse
from pathlib import Path

def generate_technical_prompts(stats_file, output_file, high_threshold=0.5, low_threshold=-0.5, max_phenotypes=3):
    """
    Generate technical prompts for gene knockdowns based on gene statistics.
    
    Args:
        stats_file (str): Path to gene statistics CSV file
        output_file (str): Path to output technical prompts CSV file
        high_threshold (float): Threshold for considering a high score significant
        low_threshold (float): Threshold for considering a low score significant
        max_phenotypes (int): Maximum number of phenotypes to include in prompt (for each direction)
    """
    # Load the gene statistics
    print(f"Loading gene statistics from {stats_file}")
    gene_stats = pd.read_csv(stats_file)
    
    # Get all score columns
    score_cols = [col for col in gene_stats.columns if col.startswith("Score -")]
    print(f"Found {len(score_cols)} score columns")
    
    # Function to get significant phenotypes for a gene (both high and low)
    def get_significant_phenotypes(row, high_thresh=high_threshold, low_thresh=low_threshold, max_pheno=max_phenotypes):
        """Get significant phenotypes for a gene based on scores."""
        # Get phenotypes with high scores
        high_phenotypes = []
        for col in score_cols:
            if pd.notnull(row[col]) and row[col] > high_thresh:
                # Extract phenotype name from column name
                phenotype = col.replace("Score - ", "").replace(" (manual)", "").replace(" (automatic)", "")
                # Add score value for sorting
                high_phenotypes.append((phenotype, row[col], "high"))
        
        # Get phenotypes with low scores
        low_phenotypes = []
        for col in score_cols:
            if pd.notnull(row[col]) and row[col] < low_thresh:
                # Extract phenotype name from column name
                phenotype = col.replace("Score - ", "").replace(" (manual)", "").replace(" (automatic)", "")
                # Add score value for sorting
                low_phenotypes.append((phenotype, row[col], "low"))
        
        # Sort by absolute score value (highest first) and take top max_phenotypes for each
        high_phenotypes.sort(key=lambda x: x[1], reverse=True)
        low_phenotypes.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Combine the results, with high phenotypes first
        significant_phenotypes = [(p[0], p[2]) for p in high_phenotypes[:max_pheno]]
        significant_phenotypes.extend([(p[0], p[2]) for p in low_phenotypes[:max_pheno]])
        
        return significant_phenotypes
    
    # Apply to get significant phenotypes for each gene
    gene_stats["significant_phenotypes"] = gene_stats.apply(get_significant_phenotypes, axis=1)
    
    # Create technical prompts
    def create_technical_prompt(row):
        cell_line = row["Characteristics [Cell Line]"]
        gene = row["Gene Symbol"]
        phenotypes = row["significant_phenotypes"]
        
        if pd.isna(gene) or not phenotypes:
            return ""
        
        # Basic prompt structure
        prompt = f"{cell_line} cells with {gene} knockdown"
        
        if phenotypes:
            # Separate high and low phenotypes
            high = [p[0] for p in phenotypes if p[1] == "high"]
            low = [p[0] for p in phenotypes if p[1] == "low"]
            
            # Add high phenotypes
            if high:
                if len(high) == 1:
                    prompt += f" showing high {high[0]}"
                else:
                    prompt += f" showing high {', '.join(high[:-1])} and {high[-1]}"
            
            # Add low phenotypes
            if low:
                connector = " and low " if high else " showing low "
                if len(low) == 1:
                    prompt += f"{connector}{low[0]}"
                else:
                    prompt += f"{connector}{', '.join(low[:-1])} and {low[-1]}"
        
        return prompt
    
    # Generate technical prompts
    gene_stats["technical_prompt"] = gene_stats.apply(create_technical_prompt, axis=1)
    
    # Create output dataframe with just gene symbol and technical prompt
    output_df = gene_stats[["Gene Symbol", "technical_prompt"]].copy()
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"Saved technical prompts to {output_file}")
    
    # Print some examples
    print("\nExample technical prompts:")
    for gene in ["KIF11", "INCENP", "COPB1"]:
        if gene in output_df["Gene Symbol"].values:
            prompt = output_df.loc[output_df["Gene Symbol"] == gene, "technical_prompt"].values[0]
            print(f"{gene}: {prompt}")
    
    # Print statistics
    print(f"\nTotal genes with prompts: {len(output_df)}")
    print(f"Genes with non-empty prompts: {len(output_df[output_df['technical_prompt'] != ''])}")
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Generate technical prompts for gene knockdowns")
    parser.add_argument("--input", type=str, default="gene_stats.csv", help="Path to gene statistics CSV file")
    parser.add_argument("--output", type=str, default="technical_prompts_1.csv", help="Path to output technical prompts CSV file")
    parser.add_argument("--high_threshold", type=float, default=0.5, help="High score threshold for phenotypes")
    parser.add_argument("--low_threshold", type=float, default=-0.5, help="Low score threshold for phenotypes")
    parser.add_argument("--max_phenotypes", type=int, default=3, help="Maximum number of phenotypes to include (each direction)")
    args = parser.parse_args()
    
    # Generate technical prompts
    generate_technical_prompts(args.input, args.output, args.high_threshold, args.low_threshold, args.max_phenotypes)

if __name__ == "__main__":
    main()
