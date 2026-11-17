import os
import numpy as np
import pandas as pd
import mne
import glob

# Subject and path settings
subject = "sub1"
base = r"SEED"
base_2 = r"SEED-Subjects"
sfreq_raw = 1000
sfreq_target = 250

ch_names = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO5","PO3","POZ","PO4","PO6","PO8",
    "CB1","O1","OZ","O2","CB2"
]
ch_num = len(ch_names)

base_in = os.path.join(base, subject)
base_out = os.path.join(base_2, subject)
os.makedirs(base_out, exist_ok=True)

# Find all CSV files in the input directory for sub1
csv_files = glob.glob(os.path.join(base_in, "*.csv"))

if not csv_files:
    print(f"⚠️ No CSV files found in {base_in}")
else:
    for input_file in csv_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(base_out, filename)

        # Read only the 62 columns in the order specified by ch_names
        df = pd.read_csv(input_file, usecols=ch_names)
        assert list(df.columns) == ch_names, "CSV column order/names do not match ch_names"

        X = df.to_numpy(dtype=np.float64).T  # (62, N), unit: V

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq_raw, ch_types=["eeg"] * ch_num)
        raw = mne.io.RawArray(X, info, verbose="ERROR")
        raw.load_data()

        raw.resample(sfreq_target, npad="auto", verbose="ERROR")
        raw.set_eeg_reference(ref_channels="average", verbose="ERROR")
        raw.notch_filter(freqs=[50], verbose="ERROR")
        raw.filter(l_freq=0.5, h_freq=50,
                   l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                   verbose="ERROR")

        Y = raw.get_data().T   # (N, 62), unit: V
        Y_uV = Y * 1e6         # Convert to microvolts (µV)

        out_df = pd.DataFrame(Y_uV, columns=ch_names)
        out_df.to_csv(output_file, index=False)
        print("Saved:", output_file, "shape:", out_df.shape)