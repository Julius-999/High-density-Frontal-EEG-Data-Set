import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch


# -------------------------
# 1) Core Feature Extraction Logic
# -------------------------
def extract_features(
        eeg_window: np.ndarray,
        fs: int = 250,
        eps: float = 1e-12
) -> np.ndarray:
    assert eeg_window.ndim == 2, "eeg_window must be 2D [window_len, num_channels]"
    n = eeg_window.shape[0]
    if n < 2:
        raise ValueError("window_len is too short, at least 2 samples are required")

    nperseg = min(n, 256)
    noverlap = nperseg // 2

    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 50.0),
    }

    def band_power(psd, f, lo, hi):
        idx = (f >= lo) & (f < hi)
        if not np.any(idx):
            return 0.0
        return float(np.trapz(psd[idx], f[idx]))

    features = []
    c = 2.0 * np.pi * np.e

    for ch in range(eeg_window.shape[1]):
        x = eeg_window[:, ch].astype(np.float64)

        mean_val = float(np.mean(x))
        var_val = float(np.var(x, ddof=0))
        rms_val = float(np.sqrt(np.mean(x * x)))
        ptp_val = float(np.ptp(x))

        f, psd = welch(
            x, fs=fs, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            detrend="constant", scaling="density"
        )

        p_delta = band_power(psd, f, *bands["delta"])
        p_theta = band_power(psd, f, *bands["theta"])
        p_alpha = band_power(psd, f, *bands["alpha"])
        p_beta = band_power(psd, f, *bands["beta"])
        p_gamma = band_power(psd, f, *bands["gamma"])

        de_delta = 0.5 * np.log(c * (p_delta + eps))
        de_theta = 0.5 * np.log(c * (p_theta + eps))
        de_alpha = 0.5 * np.log(c * (p_alpha + eps))
        de_beta = 0.5 * np.log(c * (p_beta + eps))
        de_gamma = 0.5 * np.log(c * (p_gamma + eps))

        features.extend([
            mean_val, var_val, rms_val, ptp_val,
            de_delta, de_theta, de_alpha, de_beta, de_gamma
        ])

    return np.asarray(features, dtype=np.float64)


# -------------------------
# 2) Single CSV to Feature .npy
# -------------------------
def csv_to_feature_npy(
        csv_path: str,
        out_npy_path: str,
        fs: int = 250,
        window_sec: float = 2.0,
        step_sec: float = 0.5,
):
    df = pd.read_csv(csv_path)

    # Automatically read all numerical columns by default
    X = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        raise ValueError(f"NaN/Inf in {csv_path}, first bad (row={bad[0][0]}, col={bad[1][0]})")

    win_len = int(round(window_sec * fs))  # 500 samples
    hop = int(round(step_sec * fs))  # 125 samples

    if X.shape[0] < win_len:
        raise ValueError(f"Too short: {csv_path} has {X.shape[0]} samples < win_len {win_len}")

    feats = []
    for start in range(0, X.shape[0] - win_len + 1, hop):
        window = X[start:start + win_len, :]
        feats.append(extract_features(window, fs=fs))

    feats = np.vstack(feats).astype(np.float64)

    out_dir = os.path.dirname(out_npy_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_npy_path, feats)


# -------------------------
# 3) Batch Process Directory -> Output to Feature Root
# -------------------------
def batch_extract_all(
        data_root: str = "data",
        feature_root: str = "feature",
        fs: int = 250,
        window_sec: float = 2.0,
        step_sec: float = 0.5
):
    data_root = Path(data_root)
    feature_root = Path(feature_root)

    csv_files = sorted(data_root.rglob("*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV found under: {data_root.resolve()}")
        return

    print(f"[INFO] Found {len(csv_files)} CSV files under {data_root.resolve()}")

    ok, fail = 0, 0
    for csv_path in csv_files:
        rel = csv_path.relative_to(data_root)
        out_path = (feature_root / rel).with_suffix(".npy")

        try:
            csv_to_feature_npy(
                csv_path=str(csv_path),
                out_npy_path=str(out_path),
                fs=fs,
                window_sec=window_sec,
                step_sec=step_sec,
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {csv_path} -> {out_path} | {type(e).__name__}: {e}")

    print(f"[DONE] success={ok}, failed={fail}")
    print(f"[OUT ] feature root: {feature_root.resolve()}")


if __name__ == "__main__":
    batch_extract_all(
        data_root=r"HF-Subjects\Sub1",
        feature_root=r"HF-EEG\ML-features\Sub1",
        fs=250,
        window_sec=2.0,
        step_sec=0.5
    )