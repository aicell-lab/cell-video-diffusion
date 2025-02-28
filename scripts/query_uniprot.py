#%%
import requests
import time
from typing import Dict, Optional

def get_gene_info(gene_symbol: Optional[str] = None, gene_id: Optional[str] = None) -> Dict[str, str]:
    """
    Retrieve gene information from the UniProt API.
    
    This function queries the UniProt API for reviewed entries matching the given gene symbol or gene identifier 
    and extracts:
      - description: Derived from the 'proteinDescription' field (using recommendedName or submissionNames).
      - function: A text found in the FUNCTION comment section.
      - biological_processes: GO annotations for Biological Processes.
    
    GO Annotation Details:
      The UniProt cross-references under "uniProtKBCrossReferences" provide GO terms in the format "X:Term"
      where:
        F -> Molecular Function,
        P -> Biological Process, and
        C -> Cellular Component.
      Only terms with category "P" (Biological Process) are extracted.
    
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
        "biological_processes": ""
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        description = ""
        function_text = ""
        go_bio_process = ""
        
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
            
            # Extract GO Biological Process annotations from uniProtKBCrossReferences.
            # Each GO term is given as "Category:Term". Only extract if Category equals "P".
            cross_references = result.get("uniProtKBCrossReferences", [])
            bio_processes = []
            for ref in cross_references:
                if ref.get("database") == "GO":
                    properties = ref.get("properties", [])
                    for prop in properties:
                        if prop.get("key") == "GoTerm":
                            go_term = prop.get("value")
                            if go_term:
                                # Expect value in form "P:term" where 'P' indicates Biological Process.
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

#%%
# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Test with gene symbols
    test_genes = ["TP53", "BRCA1", "EGFR"]
    for gene in test_genes:
        info = get_gene_info(gene_symbol=gene)
        print(f"Using gene symbol: {gene}")
        print(info, end="\n\n")
        time.sleep(1)  # Be nice to the API
        
    # Test with gene identifier
    gene_id = "ENSG00000134057"  # Cyclin B1 / CCNB1
    info = get_gene_info(gene_id=gene_id)
    print(f"Using gene ID: {gene_id}")
    print(info, end="\n\n")
    time.sleep(1)  # Be nice to the API

    # Load plate annotation data
    csv_path = "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/idr0013-screenA-annotation.csv"
    df = pd.read_csv(csv_path)

    # Select a subset of columns for gene information
    subset_cols = ["Plate", "Well Number", "Well", "Gene Identifier", "Gene Symbol", "Potential Mitotic Hit At Gene Level", "Validated Mitotic Hit At Gene Level"]
    df = df[subset_cols]

    # Filter out rows with missing gene identifiers
    df = df.dropna(subset=["Gene Identifier"])

    # Select rows that match "ENSG00000134057" in the gene identifier
    ccnb1_rows = df[df["Gene Identifier"].str.contains("ENSG00000134057", case=False)]
    
