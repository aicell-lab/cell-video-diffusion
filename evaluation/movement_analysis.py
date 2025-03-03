# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.measure import regionprops, label as skimage_label
from tqdm import tqdm


def track_nuclei(masks):
    """Track nuclei across frames and calculate their centroids.
    
    Args:
        masks (np.ndarray): Binary masks of shape (T, H, W)
        
    Returns:
        dict: Dictionary of tracks where keys are track_ids and values are lists of
              dictionaries with 'frame', 'centroid', and 'area' for each detection
    """
    print("\nTracking nuclei across frames...")
    tracks = {}
    next_track_id = 1
    
    for t in tqdm(range(masks.shape[0]), desc="Processing frames"):
        # Get properties for all nuclei in this frame
        props = regionprops(skimage_label(masks[t]))
        
        # First frame: initialize tracks
        if t == 0:
            for prop in props:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [{
                    'frame': t,
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'label': prop.label
                }]
            continue
        
        # Get previous frame nuclei and current nuclei
        curr_nuclei = [(prop.centroid, prop.area, prop.label) for prop in props]
        
        # Skip if no nuclei in current frame
        if not curr_nuclei:
            continue
            
        # Extract existing tracks that were active in the previous frame
        active_tracks = {}
        for track_id, track_data in tracks.items():
            if track_data[-1]['frame'] == t-1:
                active_tracks[track_id] = track_data[-1]
        
        # Skip if no active tracks from previous frame
        if not active_tracks:
            # Create new tracks for all current nuclei
            for centroid, area, label in curr_nuclei:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [{
                    'frame': t,
                    'centroid': centroid,
                    'area': area,
                    'label': label
                }]
            continue
        
        # Calculate distance matrix between previous and current nuclei
        prev_centroids = np.array([data['centroid'] for data in active_tracks.values()])
        curr_centroids = np.array([c[0] for c in curr_nuclei])
        
        dist_matrix = cdist(prev_centroids, curr_centroids)
        
        # Track assignment using greedy nearest-neighbor algorithm
        prev_indices, curr_indices = np.unravel_index(
            np.argsort(dist_matrix.ravel()), dist_matrix.shape)
        
        assigned_curr = set()
        assigned_prev = set()
        active_track_ids = list(active_tracks.keys())
        
        for p_idx, c_idx in zip(prev_indices, curr_indices):
            # Skip if either nucleus is already assigned
            if p_idx in assigned_prev or c_idx in assigned_curr:
                continue
                
            # Skip if distance is too large (maximum expected displacement)
            if dist_matrix[p_idx, c_idx] > 40:  # Pixel distance threshold
                continue
            
            # Assign current nucleus to existing track
            track_id = active_track_ids[p_idx]
            centroid, area, label = curr_nuclei[c_idx]
            
            tracks[track_id].append({
                'frame': t,
                'centroid': centroid,
                'area': area,
                'label': label
            })
            
            # Mark as assigned
            assigned_prev.add(p_idx)
            assigned_curr.add(c_idx)
        
        # Create new tracks for unassigned current nuclei
        for c_idx, (centroid, area, label) in enumerate(curr_nuclei):
            if c_idx not in assigned_curr:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [{
                    'frame': t,
                    'centroid': centroid,
                    'area': area,
                    'label': label
                }]
    
    # Filter short tracks (likely tracking errors)
    tracks = {k: v for k, v in tracks.items() if len(v) >= 3}
    
    print(f"Created {len(tracks)} tracks spanning ≥3 frames")
    return tracks


def compute_movement_metrics(tracks):
    """Compute movement metrics for each track.
    
    Args:
        tracks (dict): Dictionary of tracks
        
    Returns:
        pd.DataFrame: DataFrame with movement metrics for each track
    """
    print("\nComputing movement metrics...")
    
    metrics = []
    for track_id, track_data in tqdm(tracks.items(), desc="Analyzing tracks"):
        # Need at least 2 points to calculate speed
        if len(track_data) < 2:
            continue
            
        # Sort by frame
        track_data = sorted(track_data, key=lambda x: x['frame'])
        
        # Calculate frame-to-frame displacements
        displacements = []
        for i in range(1, len(track_data)):
            p1 = track_data[i-1]['centroid']
            p2 = track_data[i]['centroid']
            displacement = np.sqrt(((p2[0] - p1[0])**2) + ((p2[1] - p1[1])**2))
            displacements.append(displacement)  # In pixels
        
        # Total path length
        total_distance = sum(displacements)  # pixels
        
        # Time elapsed (in frames)
        total_frames = track_data[-1]['frame'] - track_data[0]['frame']
        
        # Average speed
        avg_speed = total_distance / total_frames if total_frames > 0 else 0  # pixels/frame
        
        # Net displacement (start to end)
        p_start = track_data[0]['centroid']
        p_end = track_data[-1]['centroid']
        net_displacement = np.sqrt(((p_end[0] - p_start[0])**2) + 
                                  ((p_end[1] - p_start[1])**2))  # pixels
        
        # Directness ratio (net displacement / total path length)
        directness = net_displacement / total_distance if total_distance > 0 else 0
        
        # Instantaneous speeds for each frame-to-frame movement
        frame_speeds = displacements  # pixels/frame (since frame interval is 1)
        
        # Create metrics dictionary
        metrics.append({
            'track_id': track_id,
            'avg_speed': avg_speed,  # pixels/frame
            'max_speed': max(frame_speeds) if frame_speeds else 0,  # pixels/frame
            'total_distance': total_distance,  # pixels
            'net_displacement': net_displacement,  # pixels
            'directness': directness,  # unitless ratio
            'track_duration': total_frames,  # frames
            'start_frame': track_data[0]['frame'],
            'end_frame': track_data[-1]['frame']
        })
    
    return pd.DataFrame(metrics)


def plot_movement_metrics(metrics_df, title, output_path):
    """Plot histograms of movement metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with movement metrics
        title (str): Base title for plots
        output_path (str): Base path for saving plots
    """
    print(f"\nGenerating movement metrics plots: {output_path}")
    
    # Create 2x3 plot of all metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot distributions of key metrics
    axes[0, 0].hist(metrics_df['avg_speed'].dropna(), bins=20, alpha=0.7)
    axes[0, 0].set_title(f"{title} - Average Speed (pixels/frame)")
    axes[0, 0].set_xlabel("Speed (pixels/frame)")
    
    axes[0, 1].hist(metrics_df['max_speed'].dropna(), bins=20, alpha=0.7)
    axes[0, 1].set_title(f"{title} - Maximum Speed (pixels/frame)")
    axes[0, 1].set_xlabel("Speed (pixels/frame)")
    
    axes[0, 2].hist(metrics_df['total_distance'].dropna(), bins=20, alpha=0.7)
    axes[0, 2].set_title(f"{title} - Total Distance (pixels)")
    axes[0, 2].set_xlabel("Distance (pixels)")
    
    axes[1, 0].hist(metrics_df['net_displacement'].dropna(), bins=20, alpha=0.7)
    axes[1, 0].set_title(f"{title} - Net Displacement (pixels)")
    axes[1, 0].set_xlabel("Displacement (pixels)")
    
    axes[1, 1].hist(metrics_df['directness'].dropna(), bins=20, alpha=0.7)
    axes[1, 1].set_title(f"{title} - Directness Ratio")
    axes[1, 1].set_xlabel("Directness (0-1)")
    
    axes[1, 2].hist(metrics_df['track_duration'].dropna(), bins=20, alpha=0.7)
    axes[1, 2].set_title(f"{title} - Track Duration (frames)")
    axes[1, 2].set_xlabel("Duration (frames)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    # Also save key metrics as separate files for detailed inspection
    for metric in ['avg_speed', 'total_distance', 'directness']:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_df[metric].dropna(), bins=25, alpha=0.7)
        plt.title(f"{title} - {metric.replace('_', ' ').title()}")
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        metric_path = output_path.replace('.png', f'_{metric}.png')
        plt.savefig(metric_path)
        plt.close()
    
    print(f"Saved movement metrics plots to {output_path}")


def compare_movement_metrics(real_metrics, gen_metrics, output_path):
    """Compare movement metrics between real and generated videos.
    
    Args:
        real_metrics (pd.DataFrame): Movement metrics for real video
        gen_metrics (pd.DataFrame): Movement metrics for generated video
        output_path (str): Path to save comparison plots
    """
    print(f"\nGenerating movement metrics comparison plots: {output_path}")
    
    # Create comparison plots for key metrics
    metrics_to_compare = ['avg_speed', 'total_distance', 'directness', 'track_duration']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_to_compare):
        real_values = real_metrics[metric].dropna()
        gen_values = gen_metrics[metric].dropna()
        
        axes[i].hist(real_values, bins=20, alpha=0.5, label="Real", density=True)
        axes[i].hist(gen_values, bins=20, alpha=0.5, label="Generated", density=True)
        
        # Calculate Wasserstein distance
        from scipy.stats import wasserstein_distance
        if len(real_values) > 0 and len(gen_values) > 0:
            wd = wasserstein_distance(real_values, gen_values)
            title = f"{metric.replace('_', ' ').title()} (EMD={wd:.2f})"
        else:
            title = f"{metric.replace('_', ' ').title()}"
            
        axes[i].set_title(title)
        axes[i].set_xlabel(metric.replace('_', ' ').title())
        axes[i].set_ylabel("Density")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    print(f"Saved movement comparison plots to {output_path}")


def plot_tracks(masks, tracks, output_path, max_tracks=20):
    """Plot nucleus tracks overlaid on first frame.
    
    Args:
        masks (np.ndarray): Binary masks of shape (T, H, W)
        tracks (dict): Dictionary of tracks
        output_path (str): Path to save track visualization
        max_tracks (int): Maximum number of tracks to display
    """
    print(f"\nGenerating track visualization: {output_path}")
    
    # Create background image - projection of all frames
    background = np.mean(masks, axis=0)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(background, cmap='gray')
    
    # Filter to longest tracks and limit number of tracks to display
    track_lengths = {k: len(v) for k, v in tracks.items()}
    longest_tracks = sorted(track_lengths.items(), key=lambda x: x[1], reverse=True)
    
    # Plot tracks
    num_tracks = min(max_tracks, len(longest_tracks))
    cmap = plt.cm.jet
    colors = [cmap(i/num_tracks) for i in range(num_tracks)]
    
    for i, (track_id, _) in enumerate(longest_tracks[:num_tracks]):
        track_data = tracks[track_id]
        track_data = sorted(track_data, key=lambda x: x['frame'])
        
        # Extract coordinates for the track
        x_coords = [p['centroid'][1] for p in track_data]  # x corresponds to column (width)
        y_coords = [p['centroid'][0] for p in track_data]  # y corresponds to row (height)
        
        # Plot track
        plt.plot(x_coords, y_coords, '-', color=colors[i], linewidth=2, alpha=0.7)
        
        # Mark start and end
        plt.plot(x_coords[0], y_coords[0], 'o', color=colors[i], markersize=8)
        plt.plot(x_coords[-1], y_coords[-1], 's', color=colors[i], markersize=8)
    
    plt.title(f"Nucleus Tracks (Top {num_tracks} longest tracks)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved track visualization to {output_path}")


def calculate_movement_statistics(real_metrics, gen_metrics):
    """Calculate statistics comparing movement patterns.
    
    Args:
        real_metrics (pd.DataFrame): Movement metrics for real video
        gen_metrics (pd.DataFrame): Movement metrics for generated video
        
    Returns:
        dict: Dictionary containing movement statistics
    """
    stats = {}
    
    # Calculate statistics for key metrics
    for prefix, df in [("real", real_metrics), ("gen", gen_metrics)]:
        for metric in ['avg_speed', 'max_speed', 'total_distance', 'net_displacement', 'directness', 'track_duration']:
            values = df[metric].dropna()
            if len(values) > 0:
                stats[f"{prefix}_{metric}_mean"] = values.mean()
                stats[f"{prefix}_{metric}_median"] = values.median()
                stats[f"{prefix}_{metric}_std"] = values.std()
                stats[f"{prefix}_{metric}_min"] = values.min()
                stats[f"{prefix}_{metric}_max"] = values.max()
    
    # Calculate Wasserstein distances between distributions
    for metric in ['avg_speed', 'max_speed', 'total_distance', 'net_displacement', 'directness', 'track_duration']:
        real_values = real_metrics[metric].dropna()
        gen_values = gen_metrics[metric].dropna()
        
        if len(real_values) > 0 and len(gen_values) > 0:
            from scipy.stats import wasserstein_distance
            stats[f"wasserstein_{metric}"] = wasserstein_distance(real_values, gen_values)
    
    return stats


# %%
if __name__ == "__main__":
    preview_dir = os.path.join(os.path.dirname(__file__), "preview")
    segmentation_dir = os.path.join(preview_dir, "segmentation")
    analysis_dir = os.path.join(preview_dir, "movement_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    real_masks_path = os.path.join(
        segmentation_dir,
        "masks_validation-real-600-1-<ALEXANDER>-Time-lapse-mi-23565_20250228_164150.npy",
    )
    gen_masks_path = os.path.join(
        segmentation_dir,
        "masks_validation-gen-600-1-<ALEXANDER>-Time-lapse-mi-23565_20250228_163511.npy",
    )

    print("\nLoading mask sequences...")
    real_masks = np.load(real_masks_path)
    gen_masks = np.load(gen_masks_path)
    print(f"Loaded masks with shapes: {real_masks.shape} and {gen_masks.shape}")

    if real_masks.shape[0] != gen_masks.shape[0]:
        print("Warning: Mask sequences have different lengths, truncating to match")
        min_length = min(real_masks.shape[0], gen_masks.shape[0])
        real_masks = real_masks[:min_length]
        gen_masks = gen_masks[:min_length]
        print(f"Truncated masks to {min_length} frames")

    # Track nuclei in both videos
    real_tracks = track_nuclei(real_masks)
    gen_tracks = track_nuclei(gen_masks)
    
    # Compute movement metrics
    real_metrics = compute_movement_metrics(real_tracks)
    gen_metrics = compute_movement_metrics(gen_tracks)
    
    # Save metrics to CSV
    real_metrics.to_csv(os.path.join(analysis_dir, "movement_metrics_real.csv"), index=False)
    gen_metrics.to_csv(os.path.join(analysis_dir, "movement_metrics_generated.csv"), index=False)
    
    # Generate individual plots
    plot_movement_metrics(
        real_metrics,
        "Real Video",
        os.path.join(analysis_dir, "movement_metrics_real.png")
    )
    
    plot_movement_metrics(
        gen_metrics,
        "Generated Video",
        os.path.join(analysis_dir, "movement_metrics_generated.png")
    )
    
    # Generate comparison plots
    compare_movement_metrics(
        real_metrics,
        gen_metrics,
        os.path.join(analysis_dir, "movement_metrics_comparison.png")
    )
    
    # Generate track visualizations
    plot_tracks(
        real_masks,
        real_tracks,
        os.path.join(analysis_dir, "tracks_real.png")
    )
    
    plot_tracks(
        gen_masks,
        gen_tracks,
        os.path.join(analysis_dir, "tracks_generated.png")
    )
    
    # Calculate and save movement statistics
    stats = calculate_movement_statistics(real_metrics, gen_metrics)
    
    # Create a DataFrame and save to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    stats_df.index.name = 'Metric'
    stats_df.to_csv(os.path.join(analysis_dir, "movement_statistics.csv"))
    
    print("\nMovement analysis completed. Results saved to:", analysis_dir)

# %% 