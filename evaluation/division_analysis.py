# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from skimage.measure import label
from tqdm import tqdm


def count_nuclei(segmentation_mask):
    """Count the number of nuclei in a segmentation mask.
    
    Args:
        segmentation_mask (np.ndarray): Binary mask of shape (H, W)
        
    Returns:
        int: Number of nuclei (connected components)
    """
    labeled_mask = label(segmentation_mask)
    return labeled_mask.max()  # The maximum label value is the number of objects


def compute_nuclei_counts(masks):
    """Compute the number of nuclei in each frame of a video.
    
    Args:
        masks (np.ndarray): Binary masks of shape (T, H, W)
        
    Returns:
        np.ndarray: Array containing the number of nuclei in each frame
    """
    print("\nCounting nuclei over time...")
    nuclei_counts = []
    
    for t in tqdm(range(masks.shape[0]), desc="Processing frames"):
        count = count_nuclei(masks[t])
        nuclei_counts.append(count)
        
    print(f"Processed {len(nuclei_counts)} frames successfully")
    return np.array(nuclei_counts)


def plot_nuclei_counts(counts, title, output_path):
    """Plot the number of nuclei over time.
    
    Args:
        counts (np.ndarray): Array containing nuclei counts
        title (str): Plot title
        output_path (str): Path to save the output plot
    """
    print(f"\nGenerating nuclei count plot: {output_path}")
    plt.figure(figsize=(12, 6))
    plt.plot(counts, 'o-', linewidth=2)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Number of Nuclei")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compare_nuclei_counts(real_counts, gen_counts, output_path):
    """Compare nuclei counts between real and generated videos.
    
    Args:
        real_counts (np.ndarray): Nuclei counts for real video
        gen_counts (np.ndarray): Nuclei counts for generated video
        output_path (str): Path to save the output plot
    """
    print(f"\nGenerating nuclei count comparison plot: {output_path}")
    plt.figure(figsize=(12, 6))
    
    plt.plot(real_counts, 'o-', label="Real", linewidth=2)
    plt.plot(gen_counts, 'o-', label="Generated", linewidth=2)
    
    plt.title("Comparison of Nuclei Counts")
    plt.xlabel("Frame")
    plt.ylabel("Number of Nuclei")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def detect_division_events(counts, window=3, threshold=0.5):
    """Detect potential division events based on increases in nuclei counts.
    
    Args:
        counts (np.ndarray): Array containing nuclei counts
        window (int): Size of the window for smoothing
        threshold (float): Minimum increase to be considered a division event
        
    Returns:
        list: Frames where division events likely occurred
    """
    print("\nDetecting potential division events...")
    
    # Smooth the counts to reduce noise
    smoothed = signal.savgol_filter(counts, window_length=window, polyorder=1)
    
    # Calculate first derivative (rate of change)
    derivative = np.diff(smoothed)
    
    # Find points where derivative exceeds threshold (significant increase)
    division_frames = np.where(derivative > threshold)[0]
    
    print(f"Detected {len(division_frames)} potential division events")
    return division_frames


def plot_division_events(counts, division_frames, title, output_path):
    """Plot nuclei counts with highlighted division events.
    
    Args:
        counts (np.ndarray): Array containing nuclei counts
        division_frames (np.ndarray): Frames where division events occurred
        title (str): Plot title
        output_path (str): Path to save the output plot
    """
    print(f"\nGenerating division events plot: {output_path}")
    plt.figure(figsize=(12, 6))
    
    plt.plot(counts, 'o-', linewidth=2, label="Nuclei Count")
    
    if len(division_frames) > 0:
        plt.scatter(division_frames, counts[division_frames], color='red', s=80, 
                    marker='*', label="Division Event")
    
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Number of Nuclei")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_division_statistics(real_counts, gen_counts, real_divisions, gen_divisions):
    """Calculate statistics about division events.
    
    Args:
        real_counts (np.ndarray): Nuclei counts for real video
        gen_counts (np.ndarray): Nuclei counts for generated video
        real_divisions (np.ndarray): Division frames for real video
        gen_divisions (np.ndarray): Division frames for generated video
        
    Returns:
        dict: Dictionary containing division statistics
    """
    stats = {}
    
    # Total number of nuclei (start and end)
    stats["real_initial_count"] = real_counts[0]
    stats["real_final_count"] = real_counts[-1]
    stats["gen_initial_count"] = gen_counts[0]
    stats["gen_final_count"] = gen_counts[-1]
    
    # Growth metrics
    stats["real_growth_ratio"] = real_counts[-1] / max(real_counts[0], 1)
    stats["gen_growth_ratio"] = gen_counts[-1] / max(gen_counts[0], 1)
    stats["real_growth_absolute"] = real_counts[-1] - real_counts[0]
    stats["gen_growth_absolute"] = gen_counts[-1] - gen_counts[0]
    
    # Division events statistics
    stats["real_division_events_count"] = len(real_divisions)
    stats["gen_division_events_count"] = len(gen_divisions)
    
    if len(real_divisions) > 1:
        stats["real_avg_division_interval"] = np.mean(np.diff(real_divisions))
    else:
        stats["real_avg_division_interval"] = None
        
    if len(gen_divisions) > 1:
        stats["gen_avg_division_interval"] = np.mean(np.diff(gen_divisions))
    else:
        stats["gen_avg_division_interval"] = None
    
    return stats


# %%
if __name__ == "__main__":
    preview_dir = os.path.join(os.path.dirname(__file__), "preview")
    segmentation_dir = os.path.join(preview_dir, "segmentation")
    analysis_dir = os.path.join(preview_dir, "division_analysis")
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

    # Compute nuclei counts for both videos
    real_counts = compute_nuclei_counts(real_masks)
    gen_counts = compute_nuclei_counts(gen_masks)
    
    # Individual plots
    plot_nuclei_counts(
        real_counts, 
        "Number of Nuclei Over Time (Real Video)", 
        os.path.join(analysis_dir, "nuclei_counts_real.png")
    )
    
    plot_nuclei_counts(
        gen_counts, 
        "Number of Nuclei Over Time (Generated Video)", 
        os.path.join(analysis_dir, "nuclei_counts_generated.png")
    )
    
    # Comparison plot
    compare_nuclei_counts(
        real_counts, 
        gen_counts, 
        os.path.join(analysis_dir, "nuclei_counts_comparison.png")
    )
    
    # Detect and visualize division events
    real_divisions = detect_division_events(real_counts)
    gen_divisions = detect_division_events(gen_counts)
    
    plot_division_events(
        real_counts, 
        real_divisions,
        "Division Events (Real Video)", 
        os.path.join(analysis_dir, "division_events_real.png")
    )
    
    plot_division_events(
        gen_counts, 
        gen_divisions,
        "Division Events (Generated Video)", 
        os.path.join(analysis_dir, "division_events_generated.png")
    )
    
    # Calculate and save division statistics
    stats = calculate_division_statistics(
        real_counts, 
        gen_counts, 
        real_divisions, 
        gen_divisions
    )
    
    # Create a DataFrame and save to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    stats_df.index.name = 'Metric'
    stats_df.to_csv(os.path.join(analysis_dir, "division_statistics.csv"))
    
    print("\nDivision analysis completed. Results saved to:", analysis_dir)

# %%
