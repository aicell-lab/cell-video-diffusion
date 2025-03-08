import pandas as pd

# Read the CSV file
df = pd.read_csv('output/extreme_phenotypes_with_prompts.csv')

# Define the columns and their possible labels we want to create subsets for
categories = {
    'initial_cell_count_label': ['LOW', 'MED', 'HIGH'],
    'cell_death_label': ['LOW', 'MED', 'HIGH'],
    'migration_speed_label': ['LOW', 'MED', 'HIGH'],
    'proliferation_label': ['LOW', 'MED', 'HIGH']
}

# For each category and label, create a separate txt file
for category, labels in categories.items():
    for label in labels:
        # Filter the dataframe for this label
        subset = df[df[category] == label]
        
        # Create filename like 'initial_cell_count_LOW.txt'
        output_filename = f'videos_{category}_{label}.txt'
        
        # Write video paths to file
        with open(output_filename, 'w') as f:
            for path in subset['video_path']:
                f.write(f'{path}\n')
        
        print(f'Created {output_filename} with {len(subset)} videos')