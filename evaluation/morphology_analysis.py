# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage.measure import label, regionprops
from tqdm import tqdm


def get_nucleus_morphology(segmentation_mask):
    labeled_mask = label(segmentation_mask)
    properties = regionprops(labeled_mask)
    return [
        {
            "area": prop.area,
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
            "perimeter": prop.perimeter,
        }
        for prop in properties
    ]


def plot_nucleus_morphology(morphology, output_path):
    df = pd.DataFrame(morphology)
    df.hist(bins=20, figsize=(10, 10))
    plt.savefig(output_path)
    plt.close()


def compute_temporal_morphology(masks):
    """Compute morphology statistics over time for a sequence of masks.

    Args:
        masks (np.ndarray): Binary masks of shape (T, H, W)

    Returns:
        pd.DataFrame: DataFrame containing mean, median, and std for each morphological parameter
    """
    print("\nAnalyzing temporal morphology...")
    time_series = []

    for t in tqdm(range(masks.shape[0]), desc="Processing frames"):
        frame_morphology = get_nucleus_morphology(masks[t])
        if frame_morphology:
            df = pd.DataFrame(frame_morphology)
            stats_dict = {}
            for column in df.columns:
                stats_dict[f"mean_{column}"] = df[column].mean()
                stats_dict[f"median_{column}"] = df[column].median()
                stats_dict[f"std_{column}"] = df[column].std()
            time_series.append(stats_dict)

    print(f"Processed {len(time_series)} frames successfully")
    return pd.DataFrame(time_series)


def plot_temporal_morphology(temporal_stats, output_path):
    """Plot temporal evolution of morphology parameters."""
    print(f"\nGenerating temporal morphology plot: {output_path}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    parameters = ["area", "eccentricity", "solidity", "perimeter"]

    for ax, param in zip(axes.flat, parameters):
        mean_values = temporal_stats[f"mean_{param}"]
        std_values = temporal_stats[f"std_{param}"]
        median_values = temporal_stats[f"median_{param}"]

        time_points = range(len(mean_values))
        ax.errorbar(
            time_points, mean_values, yerr=std_values, label="Mean ± STD", alpha=0.5
        )
        ax.plot(time_points, median_values, "r--", label="Median")

        ax.set_title(f"{param.capitalize()} over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel(param)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_emd_over_time(real_masks, gen_masks):
    """Compare morphology distributions between two videos using Earth Mover's Distance.

    Args:
        real_masks (np.ndarray): First sequence of binary masks
        gen_masks (np.ndarray): Second sequence of binary masks

    Returns:
        pd.DataFrame: DataFrame containing EMD values for each parameter over time
    """
    print("\nComputing Earth Mover's Distance between videos...")
    emd_series = []
    parameters = ["area", "eccentricity", "solidity", "perimeter"]

    n_frames = min(real_masks.shape[0], gen_masks.shape[0])

    for t in tqdm(range(n_frames), desc="Computing EMD"):
        morph1 = get_nucleus_morphology(real_masks[t])
        morph2 = get_nucleus_morphology(gen_masks[t])

        if not morph1 or not morph2:
            print(f"Warning: No nuclei detected in frame {t}")
            continue

        df1 = pd.DataFrame(morph1)
        df2 = pd.DataFrame(morph2)

        frame_emd = {}
        for param in parameters:
            emd = wasserstein_distance(df1[param], df2[param])
            frame_emd[param] = emd

        emd_series.append(frame_emd)

    print(f"Successfully computed EMD for {len(emd_series)} frames")
    return pd.DataFrame(emd_series)


def plot_emd_comparison(emd_df, output_path):
    """Plot Earth Mover's Distance comparison over time.

    Args:
        emd_df (pd.DataFrame): DataFrame containing EMD values
        output_path (str): Path to save the output plot
    """
    print(f"\nGenerating EMD comparison plot: {output_path}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    parameters = ["area", "eccentricity", "solidity", "perimeter"]

    for ax, param in zip(axes.flat, parameters):
        ax.plot(range(len(emd_df)), emd_df[param], "-o")
        ax.set_title(f"{param.capitalize()} EMD over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Earth Mover Distance")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# %%
if __name__ == "__main__":
    preview_dir = os.path.join(os.path.dirname(__file__), "preview")
    segmentation_dir = os.path.join(preview_dir, "segmentation")
    analysis_dir = os.path.join(preview_dir, "morphology_analysis")
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

    # Compute and plot EMD comparison
    emd_df = compute_emd_over_time(real_masks, gen_masks)
    plot_emd_comparison(emd_df, os.path.join(analysis_dir, "emd_comparison.png"))

    # Static morphology analysis
    morphology = get_nucleus_morphology(real_masks[0])
    plot_nucleus_morphology(
        morphology, os.path.join(analysis_dir, "morphology_frame1_real_masks.png")
    )
    morphology = get_nucleus_morphology(gen_masks[0])
    plot_nucleus_morphology(
        morphology, os.path.join(analysis_dir, "morphology_frame1_gen_masks.png")
    )

    # Temporal morphology analysis
    temporal_stats = compute_temporal_morphology(real_masks)
    plot_temporal_morphology(
        temporal_stats, os.path.join(analysis_dir, "temporal_morphology_real_video.png")
    )
    temporal_stats = compute_temporal_morphology(gen_masks)
    plot_temporal_morphology(
        temporal_stats,
        os.path.join(analysis_dir, "temporal_morphology_generated_video.png"),
    )

# %%
