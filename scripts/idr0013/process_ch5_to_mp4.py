import os
import sys
import glob
import h5py
import numpy as np
import imageio
def find_ch5_image_datasets(h5file):
    """
    Search an open HDF5 file (CellH5) for datasets likely to contain raw images,
    typically (1, T, 1, Y, X) with dtype=uint8.

    Returns a list of dataset paths that match these criteria.
    """
    found_paths = []

    def visit_dataset(name, node):
        if isinstance(node, h5py.Dataset):
            shape = node.shape
            dtype = node.dtype
            # Check for 5D shape: (1, T, 1, Y, X) and dtype=uint8
            if (len(shape) == 5 
                and shape[0] == 1
                and shape[2] == 1
                and dtype == np.uint8):
                found_paths.append(name)

    h5file.visititems(visit_dataset)
    return found_paths

def load_ch5_as_timeseries(ch5_path):
    """
    Attempt to load the first matching image dataset from a MitoCheck CH5 file.
    Returns a 3D NumPy array of shape (T, Y, X) if found, else None.
    """
    with h5py.File(ch5_path, "r") as f:
        image_paths = find_ch5_image_datasets(f)
        if not image_paths:
            print(f"  No raw image dataset found in {ch5_path}")
            return None
        
        # We'll just take the first found path
        ds_path = image_paths[0]
        ds = f[ds_path]
        data_5d = ds[()]  # shape: (1, T, 1, Y, X)
        data_squeezed = np.squeeze(data_5d)  # (T, Y, X)
        return data_squeezed


def make_mp4(timeseries, out_path, fps=10):
    """
    Write a 3D numpy array (T, Y, X) to an MP4 file.
    No per-frame normalization—use raw pixel values.
    """    
    print(f"Creating MP4 with data shape={timeseries.shape}, dtype={timeseries.dtype}")

    # Optional check: warn if not uint8
    if timeseries.dtype != np.uint8:
        print("WARNING: Data is not uint8. Consider scaling or casting to avoid unexpected clipping.")

    T = timeseries.shape[0]
    writer = imageio.get_writer(out_path, fps=fps)

    for t in range(T):
        frame_8bit = timeseries[t]
        writer.append_data(frame_8bit)
    
    writer.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_ch5_to_mp4.py /path/to/hdf5_folder [fps=10]")
        sys.exit(1)
    
    hdf5_folder = sys.argv[1]
    fps = 10
    if len(sys.argv) >= 3:
        fps = int(sys.argv[2])
    
    # The plate directory might look like "LT0001_02--ex2005_11_16--sp2005_02_17--tt17--c3/hdf5"
    # We create a simpler output folder named after the first segment (e.g., "LT0001_02") or the entire path?
    
    plate_name = os.path.basename(os.path.dirname(hdf5_folder))
    # E.g. if hdf5_folder = ".../LT0001_02--ex2005_11_16--sp2005_02_17--tt17--c3/hdf5"
    # then plate_name might be "LT0001_02--ex2005_11_16--sp2005_02_17--tt17--c3"
    # If you'd rather parse just "LT0001_02" from that string, you'd do:
    plate_name = plate_name.split('--')[0]
    
    out_dir = plate_name  # or "output_" + plate_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Find all .ch5 in the input folder
    ch5_files = sorted(glob.glob(os.path.join(hdf5_folder, "*.ch5")))
    
    if not ch5_files:
        print(f"No .ch5 files found in {hdf5_folder}")
        sys.exit(0)
    
    for ch5_file in ch5_files:
        file_stem = os.path.splitext(os.path.basename(ch5_file))[0]
        # e.g. "00046_01"
        out_mp4_path = os.path.join(out_dir, file_stem + ".mp4")
        
        print(f"Processing {ch5_file} -> {out_mp4_path}")
        data_squeezed = load_ch5_as_timeseries(ch5_file)
        if data_squeezed is None:
            # skip
            continue
        
        print(f"  shape={data_squeezed.shape}, dtype={data_squeezed.dtype}")
        
        make_mp4(data_squeezed, out_mp4_path, fps=fps)
        print(f"  Wrote {out_mp4_path}")

if __name__ == "__main__":
    main()