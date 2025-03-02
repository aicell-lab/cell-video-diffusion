import pandas as pd

# Read the original CSV file
input_file = "scripts/idr0013/prompt_mapping_all.csv"
df = pd.read_csv(input_file)

# Select only the required columns
df_filtered = df[["plate_well_id", "video_path"]]

# Write to a new CSV file
output_file = "scripts/idr0013/video_paths.csv"
df_filtered.to_csv(output_file, index=False)

print(f"Created {output_file} with {len(df_filtered)} rows")