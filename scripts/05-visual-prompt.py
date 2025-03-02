import pandas as pd
import argparse
import time
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

def create_visual_prompt(technical_prompt, uniprot_prompt):
    """
    Create a system prompt for GPT-4o to generate a visual prompt.
    
    Args:
        technical_prompt (str): The technical prompt describing phenotypes
        uniprot_prompt (str): The UniProt prompt describing gene function
        
    Returns:
        str: The system prompt for GPT-4o
    """
    system_prompt = f"""You are an expert in cell biology and microscopy imaging. Your task is to transform technical descriptions of gene knockdown experiments into visually descriptive prompts for a text-to-video diffusion model.

INPUT:
1. Technical phenotype description: "{technical_prompt}"
2. Gene function information: "{uniprot_prompt}"

TASK:
Transform these technical descriptions into a detailed visual prompt that:
1. Focuses on observable visual features in microscopy videos
2. Describes cell appearance, shape, movement, and temporal changes
3. Uses concrete, descriptive language about what would be visible in a video
4. Avoids technical jargon when possible, or explains it visually
5. Includes temporal aspects (what happens over time in the video)
6. Describes the visual contrast between normal cells and these affected cells

FORMAT YOUR RESPONSE:
- Start with a brief overview of what's happening to the cells
- Then provide specific visual details about cell appearance, movement, and behavior
- Include a temporal progression of what happens over the video duration
- Keep your response to 3-5 sentences, focusing on the most visually distinctive features
- Use language that would help a text-to-video model generate accurate cell biology videos

IMPORTANT: Focus only on visual aspects that would be observable in microscopy videos. Do not include molecular mechanisms or processes that wouldn't be directly visible.
"""
    return system_prompt

def generate_visual_prompts(input_file, output_file, api_key=None, model="gpt-4o", max_retries=3, sleep_time=1):
    """
    Generate visual prompts using GPT-4o.
    
    Args:
        input_file (str): Path to input technical prompts CSV file
        output_file (str): Path to output visual prompts CSV file
        api_key (str): OpenAI API key
        model (str): OpenAI model to use
        max_retries (int): Maximum number of retries for API calls
        sleep_time (float): Time to sleep between API calls
    """
    # Load the technical prompts
    print(f"Loading technical prompts from {input_file}")
    prompts_df = pd.read_csv(input_file)
    
    # Create a new column for the visual prompts
    prompts_df["visual_prompt"] = ""
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Process each gene
    print(f"Generating visual prompts using {model}...")
    for i, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
        gene_symbol = row["Gene Symbol"]
        technical_prompt = row["technical_prompt"]
        uniprot_prompt = row["uniprot_prompt"]
        
        # Skip if either prompt is missing
        if pd.isna(technical_prompt) or technical_prompt == "" or pd.isna(uniprot_prompt) or uniprot_prompt == "":
            continue
            
        # Create the system prompt for GPT-4o
        system_prompt = create_visual_prompt(technical_prompt, uniprot_prompt)
        
        # Try to get response with retries
        visual_prompt = ""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Generate a visual prompt for this gene knockdown."}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                
                visual_prompt = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {gene_symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(sleep_time * (attempt + 1))  # Exponential backoff
        
        # Update the dataframe
        prompts_df.at[i, "visual_prompt"] = visual_prompt
        
        # Sleep to avoid rate limits
        time.sleep(sleep_time)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_df.to_csv(output_path, index=False)
    
    print(f"Saved visual prompts to {output_file}")
    
    # Print some examples
    print("\nExample visual prompts:")
    for gene in ["KIF11", "INCENP", "COPB1"]:
        if gene in prompts_df["Gene Symbol"].values:
            prompt = prompts_df.loc[prompts_df["Gene Symbol"] == gene, "visual_prompt"].values[0]
            if prompt:
                print(f"\n{gene}:")
                print(prompt)
    
    # Print statistics
    print(f"\nTotal genes processed: {len(prompts_df)}")
    print(f"Genes with visual prompts: {len(prompts_df[prompts_df['visual_prompt'] != ''])}")
    
    return prompts_df

def main():
    parser = argparse.ArgumentParser(description="Generate visual prompts using GPT-4o")
    parser.add_argument("--input", type=str, default="technical_prompts_2.csv", help="Path to input technical prompts CSV file")
    parser.add_argument("--output", type=str, default="visual_prompts.csv", help="Path to output visual prompts CSV file")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for API calls")
    parser.add_argument("--sleep_time", type=float, default=1.0, help="Time to sleep between API calls")
    args = parser.parse_args()
    
    # Generate visual prompts
    generate_visual_prompts(args.input, args.output, args.api_key, args.model, args.max_retries, args.sleep_time)

if __name__ == "__main__":
    main()
