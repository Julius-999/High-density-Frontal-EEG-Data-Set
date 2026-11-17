import numpy as np
from scipy.signal import stft
import pandas as pd
import glob
import os


def extract_stft_de_features(
        eeg_window,  # [T, C]
        fs,
        nperseg,  # approx 0.5s
        noverlap,
        detrend_mean=True
):
    """
    Output: [T', C * len(bands)] -> Directly usable as Transformer input
    """
    eps = 1e-8
    bands = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50))
    X = np.asarray(eeg_window, dtype=np.float64)  # [T, C]
    T, C = X.shape

    if detrend_mean:
        X = X - X.mean(axis=0, keepdims=True)

    all_ch_feats = []

    for ch in range(C):
        sig = X[:, ch]

        freqs, times, Zxx = stft(
            sig,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None
        )

        power = np.abs(Zxx) ** 2  # [F, T']

        band_feats = []
        for (lo, hi) in bands:
            idx = np.where((freqs >= lo) & (freqs <= hi))[0]

            if len(idx) == 0:
                de = np.zeros(power.shape[1])
            else:
                band_power = power[idx, :]  # [F_band, T']
                band_energy = np.sum(band_power, axis=0)  # Band total power
                de = np.log(band_energy + eps)  # STFT-DE

            band_feats.append(de)

        band_feats = np.stack(band_feats, axis=0)  # [B, T']
        all_ch_feats.append(band_feats)

    X_feat = np.stack(all_ch_feats, axis=0)  # [C, B, T']
    X_feat = X_feat.transpose(2, 0, 1)  # [T', C, B]
    X_feat = X_feat.reshape(X_feat.shape[0], -1)  # [T', C * B]

    return X_feat


def sliding_window_create_features(eeg, window_size, step_size, fs, nperseg, noverlap, save_path):
    """
    eeg: [N, C]
    return: [NumSamples, T', C * len(bands)]
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    features = []
    N = eeg.shape[0]

    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size
        window = eeg[start:end, :]
        de_feature = extract_stft_de_features(window, fs, nperseg, noverlap)
        features.append(de_feature)

    features = np.array(features, dtype=np.float32)
    np.save(save_path, features)
    print(f"✅ STFT-DE features saved to: {save_path}")
    print(f"✅ Feature shape: {features.shape}")


# Parameter settings
WINDOW_SIZE = 500
STEP_SIZE = 125
FREQUENCY = 250
NPERSEG = 128
NOVERLAP = 64

# Directory settings
input_dir = r"HF-Subjects\Sub1"
output_dir = r"HF-EEG\STFT-features\Sub1"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Find all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

if not csv_files:
    print(f"⚠️ No CSV files found in {input_dir}")
else:
    for input_file in csv_files:
        file_name = os.path.basename(input_file)
        file_name_without_ext = os.path.splitext(file_name)[0]
        save_file_path = os.path.join(output_dir, f"{file_name_without_ext}.npy")

        # Read CSV file
        df = pd.read_csv(input_file)
        # Extract features and save as .npy
        sliding_window_create_features(
            df, WINDOW_SIZE, STEP_SIZE, FREQUENCY, NPERSEG, NOVERLAP, save_file_path
        )