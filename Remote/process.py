#!/usr/bin/env python3
# Build a waterfall plot from per-sweep tinySA CSV files in the current folder.

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_WATERFALL_INDEX = "waterfall_by_index.png"
OUTPUT_WATERFALL_FREQ = "waterfall_by_frequency.png"
OUTPUT_MATRIX_CSV = "waterfall_matrix.csv"


def find_sweep_csv_files(folder):
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    sweep_files = []
    for path in csv_files:
        name = os.path.basename(path).lower()

        if "summary" in name:
            continue
        if "waterfall" in name:
            continue

        sweep_files.append(path)

    if not sweep_files:
        raise FileNotFoundError("No per-sweep CSV files found in the current folder.")

    return sweep_files


def load_single_sweep_csv(path):
    df = pd.read_csv(path)

    required_columns = {"Frequency_Hz", "Power_dBm"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing required columns: {sorted(missing)}")

    freq = df["Frequency_Hz"].to_numpy(dtype=float)
    power = df["Power_dBm"].to_numpy(dtype=float)

    return freq, power


def load_all_sweeps(folder):
    sweep_files = find_sweep_csv_files(folder)

    freq_reference = None
    all_power = []
    used_files = []

    for path in sweep_files:
        freq, power = load_single_sweep_csv(path)

        if freq_reference is None:
            freq_reference = freq
        else:
            if len(freq) != len(freq_reference):
                raise ValueError(
                    f"{os.path.basename(path)} has {len(freq)} points, expected {len(freq_reference)}"
                )

            if not np.allclose(freq, freq_reference, rtol=0, atol=1e-6):
                raise ValueError(
                    f"{os.path.basename(path)} frequency axis does not match the first sweep"
                )

        all_power.append(power)
        used_files.append(os.path.basename(path))

    power_matrix = np.vstack(all_power)

    return freq_reference, power_matrix, used_files


def save_waterfall_matrix_csv(freq, power_matrix, used_files, folder):
    num_sweeps, num_points = power_matrix.shape

    columns = ["Sweep_File", "Sweep_Index"] + [f"Bin_{i:03d}" for i in range(num_points)]
    rows = []

    for sweep_idx in range(num_sweeps):
        row = [used_files[sweep_idx], sweep_idx]
        row.extend(power_matrix[sweep_idx, :].tolist())
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    metadata_path = os.path.join(folder, OUTPUT_MATRIX_CSV)
    df.to_csv(metadata_path, index=False)

    return metadata_path


def plot_waterfall_by_index(power_matrix, folder):
    num_sweeps, num_points = power_matrix.shape

    fig, ax = plt.subplots(figsize=(12, 7))

    image = ax.imshow(
        power_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Sweep Index")
    ax.set_title("Waterfall Plot by Band Index")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Power (dBm)")

    fig.tight_layout()

    out_path = os.path.join(folder, OUTPUT_WATERFALL_INDEX)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


def plot_waterfall_by_frequency(freq, power_matrix, folder):
    num_sweeps, _ = power_matrix.shape

    fig, ax = plt.subplots(figsize=(12, 7))

    extent = [freq[0], freq[-1], 0, num_sweeps - 1]

    image = ax.imshow(
        power_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Sweep Index")
    ax.set_title("Waterfall Plot by Frequency")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Power (dBm)")

    fig.tight_layout()

    out_path = os.path.join(folder, OUTPUT_WATERFALL_FREQ)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


def main():
    folder = os.getcwd()

    freq, power_matrix, used_files = load_all_sweeps(folder)

    matrix_csv_path = save_waterfall_matrix_csv(freq, power_matrix, used_files, folder)
    waterfall_index_path = plot_waterfall_by_index(power_matrix, folder)
    waterfall_freq_path = plot_waterfall_by_frequency(freq, power_matrix, folder)

    print(f"[INFO] Folder: {folder}")
    print(f"[INFO] Sweep files loaded: {len(used_files)}")
    print(f"[INFO] Points per sweep: {power_matrix.shape[1]}")
    print(f"[INFO] Power matrix shape: {power_matrix.shape}")
    print(f"[INFO] Saved matrix CSV: {matrix_csv_path}")
    print(f"[INFO] Saved waterfall by index: {waterfall_index_path}")
    print(f"[INFO] Saved waterfall by frequency: {waterfall_freq_path}")


if __name__ == "__main__":
    main()
