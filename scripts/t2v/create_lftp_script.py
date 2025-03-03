# Match plate IDs with their full directory names and create an lftp script

# Read plate IDs from plates.txt
with open('plates.txt', 'r') as f:
    plate_ids = [line.strip() for line in f if line.strip()]

# Read full directory names from long_plates.txt
with open('long_plates.txt', 'r') as f:
    full_dir_lines = f.readlines()

# Create a map from plate ID to full directory name
plate_to_dir = {}
for line in full_dir_lines:
    # Skip empty lines
    if not line.strip():
        continue
    
    # Extract the directory name from the line
    parts = line.strip().split()
    if len(parts) < 9:  # Ensure there are enough parts
        continue
    
    full_dir = parts[8]  # The directory name is in the 9th column
    
    # Check if this directory matches any of our plate IDs
    for plate_id in plate_ids:
        if full_dir.startswith(plate_id + "--"):
            if plate_id not in plate_to_dir:
                plate_to_dir[plate_id] = []
            plate_to_dir[plate_id].append(full_dir)

# Create the lftp script with exact directory names
with open('download_plates.txt', 'w') as out:
    out.write("# LFTP script to download hdf5 folders\n\n")
    
    for plate_id in plate_ids:
        if plate_id in plate_to_dir:
            for full_dir in plate_to_dir[plate_id]:
                # Mirror while preserving the directory structure
                out.write(f"mirror {full_dir}/hdf5 {full_dir}/hdf5\n")
        else:
            out.write(f"# WARNING: No match found for {plate_id}\n")

# Print summary
matched = sum(1 for plate_id in plate_ids if plate_id in plate_to_dir)
print(f"Found matches for {matched} out of {len(plate_ids)} plate IDs")
print("Created download_plates.txt for use with lftp")
print("Run 'source download_plates.txt' in your lftp session") 