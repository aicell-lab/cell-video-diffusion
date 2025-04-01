import pandas as pd

df = pd.read_csv('output/extreme_phenotypes_with_prompts.csv')

categories = {
    'initial_cell_count_label': ['LOW', 'MED', 'HIGH'],
    'cell_death_label': ['LOW', 'MED', 'HIGH'],
    'migration_speed_label': ['LOW', 'MED', 'HIGH'],
    'proliferation_label': ['LOW', 'MED', 'HIGH']
}


for category, labels in categories.items():
    for label in labels:

        subset = df[df[category] == label]
        
        output_filename = f'videos_{category}_{label}.txt'
        
        with open(output_filename, 'w') as f:
            for path in subset['video_path']:
                f.write(f'{path}\n')
        
        print(f'Created {output_filename} with {len(subset)} videos')