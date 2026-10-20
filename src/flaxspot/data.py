import os
import glob
import numpy as np


def kepler_data_generator(data_dir, seq_len=4000, batch_size=32, shuffle=True):
  
    file_list = glob.glob(os.path.join(data_dir, "*.npy"))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    if shuffle:
        np.random.shuffle(file_list)

    batch_flux, batch_mask = [], []

    for file_path in file_list:
        try:
            data_array = np.load(file_path)
            flux = data_array[1]

            flux_mean = np.nanmean(flux)
            flux_std = np.nanstd(flux)
            if flux_std == 0 or np.isnan(flux_std):
                continue
            flux_scaled = np.nan_to_num((flux - flux_mean) / flux_std, nan=0.0)

            num_segments = len(flux_scaled) // seq_len
            for i in range(num_segments):
                segment = flux_scaled[i * seq_len : (i + 1) * seq_len]
                batch_flux.append(segment.reshape(seq_len, 1))
                batch_mask.append(np.ones(seq_len, dtype=np.float32))

                if len(batch_flux) == batch_size:
                    yield np.array(batch_flux, dtype=np.float32), np.array(
                        batch_mask, dtype=np.float32
                    )
                    batch_flux, batch_mask = [], []
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
            continue


def prepare_single_star(file_path, seq_len=4000):

    data_array = np.load(file_path)
    real_time = data_array[0]
    real_flux = data_array[1]

    flux_mean = np.nanmean(real_flux)
    flux_std = np.nanstd(real_flux)
    flux_scaled = np.nan_to_num((real_flux - flux_mean) / flux_std, nan=0.0)

    num_segments = len(flux_scaled) // seq_len
    flux_segments = flux_scaled[: num_segments * seq_len].reshape(
        (num_segments, seq_len, 1)
    )
    time_segments = real_time[: num_segments * seq_len].reshape((num_segments, seq_len))
    mask_segments = np.ones((num_segments, seq_len), dtype=np.float32)

    return {
        "kic_id": os.path.basename(file_path).replace(".npy", ""),
        "flux_segments": flux_segments,
        "time_segments": time_segments,
        "mask_segments": mask_segments,
        "flux_mean": flux_mean,
        "flux_std": flux_std,
        "num_segments": num_segments,
    }
