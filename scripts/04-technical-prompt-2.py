import pandas as pd
import requests
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

def get_gene_info(gene_symbol: Optional[str] = None, gene_id: Optional[str] = None) -> Dict[str, str]:
    """
    Retrieve gene information from the UniProt API.
    
    This function queries the UniProt API for reviewed entries matching the given gene symbol or gene identifier 
    and extracts:
      - description: Derived from the 'proteinDescription' field (using recommendedName or submissionNames).
      - function: A text found in the FUNCTION comment section.
      - biological_processes: GO annotations for Biological Processes.
    
    Args:
        gene_symbol (str, optional): Gene symbol to query (e.g., "TP53").
        gene_id (str, optional): Gene identifier to query (e.g., "ENSG00000134057").
        At least one of gene_symbol or gene_id must be provided.
    
    Returns:
        dict: A dictionary with keys "gene", "description", "function", and "biological_processes".
              If no data is found or errors occur, empty strings are returned for each key.
    """
    if gene_symbol is None and gene_id is None:
        raise ValueError("Either gene_symbol or gene_id must be provided")
    
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    
    # Construct the query based on which parameter is provided
    if gene_symbol:
        query = f"(gene:{gene_symbol})"
        gene_identifier = gene_symbol
    else:
        # For gene IDs like Ensembl IDs (ENSG...)
        query = f"(xref:{gene_id})"
        gene_identifier = gene_id
        
    # Add the reviewed filter
    query += " AND (reviewed:true)"
    
    params = {
        "query": query,
        "format": "json",
        "fields": "protein_name,cc_function,go",  
        "size": 1
    }
    
    out_dict = {
        "gene": gene_identifier,
        "description": "",
        "function": "",
        "biological_processes": []
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            result = data["results"][0]
            # Extract protein description from recommended or submission names
            protein_desc = result.get("proteinDescription", {})
            if "recommendedName" in protein_desc:
                description = protein_desc["recommendedName"]["fullName"]["value"]
            elif "submissionNames" in protein_desc:
                description = protein_desc["submissionNames"][0]["fullName"]["value"]
            out_dict["description"] = description
            
            # Extract protein function from the comments section
            comments = result.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_text = texts[0].get("value", "")
                        break
            out_dict["function"] = function_text
            
            # Extract GO Biological Process annotations from uniProtKBCrossReferences
            cross_references = result.get("uniProtKBCrossReferences", [])
            bio_processes = []
            for ref in cross_references:
                if ref.get("database") == "GO":
                    properties = ref.get("properties", [])
                    for prop in properties:
                        if prop.get("key") == "GoTerm":
                            go_term = prop.get("value")
                            if go_term:
                                # Expect value in form "P:term" where 'P' indicates Biological Process
                                parts = go_term.split(":", 1)
                                if len(parts) == 2 and parts[0] == "P":
                                    bio_processes.append(parts[1].strip())
            out_dict["biological_processes"] = bio_processes
        
        return out_dict
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for gene {gene_identifier}: {str(e)}")
        return out_dict
    except (KeyError, IndexError) as e:
        print(f"Error parsing data for gene {gene_identifier}: {str(e)}")
        return out_dict

def generate_uniprot_prompts(input_file, output_file, max_retries=3, sleep_time=1):
    """
    Generate technical prompts with UniProt information.
    
    Args:
        input_file (str): Path to input technical prompts CSV file
        output_file (str): Path to output enhanced technical prompts CSV file
        max_retries (int): Maximum number of retries for API calls
        sleep_time (float): Time to sleep between API calls
    """
    # Load the technical prompts
    print(f"Loading technical prompts from {input_file}")
    tech_prompts = pd.read_csv(input_file)
    
    # Create a new column for the UniProt-only prompts
    tech_prompts["uniprot_prompt"] = ""
    
    # Process each gene
    print("Querying UniProt for gene information...")
    for i, row in tech_prompts.iterrows():
        gene_symbol = row["Gene Symbol"]
        
        if pd.isna(gene_symbol) or gene_symbol == "":
            continue
            
        print(f"Processing gene {i+1}/{len(tech_prompts)}: {gene_symbol}")
        
        # Try to get gene info with retries
        gene_info = None
        for attempt in range(max_retries):
            try:
                gene_info = get_gene_info(gene_symbol=gene_symbol)
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {gene_symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(sleep_time)
        
        if not gene_info:
            continue
            
        # Create the UniProt-only prompt
        uniprot_prompt = ""
        
        # Add function information if available
        if gene_info["function"]:
            uniprot_prompt = f"{gene_symbol} normally functions as {gene_info['function']}"
        # If no function info, try to use description
        elif gene_info["description"]:
            uniprot_prompt = f"{gene_symbol} is {gene_info['description']}"
        
        # Update the dataframe
        tech_prompts.at[i, "uniprot_prompt"] = uniprot_prompt
        
        # Sleep to avoid overwhelming the API
        time.sleep(sleep_time)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tech_prompts.to_csv(output_path, index=False)
    
    print(f"Saved technical prompts with UniProt information to {output_file}")
    
    # Print some examples
    print("\nExample UniProt prompts:")
    for gene in ["KIF11", "INCENP", "COPB1"]:
        if gene in tech_prompts["Gene Symbol"].values:
            prompt = tech_prompts.loc[tech_prompts["Gene Symbol"] == gene, "uniprot_prompt"].values[0]
            print(f"{gene}: {prompt}")
    
    # Print statistics
    print(f"\nTotal genes processed: {len(tech_prompts)}")
    print(f"Genes with UniProt prompts: {len(tech_prompts[tech_prompts['uniprot_prompt'] != ''])}")
    
    return tech_prompts

def main():
    parser = argparse.ArgumentParser(description="Generate UniProt prompts for genes")
    parser.add_argument("--input", type=str, default="technical_prompts_1.csv", help="Path to input technical prompts CSV file")
    parser.add_argument("--output", type=str, default="technical_prompts_2.csv", help="Path to output technical prompts CSV file")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for API calls")
    parser.add_argument("--sleep_time", type=float, default=1.0, help="Time to sleep between API calls")
    args = parser.parse_args()
    
    # Generate UniProt prompts
    generate_uniprot_prompts(args.input, args.output, args.max_retries, args.sleep_time)

if __name__ == "__main__":
    main()
