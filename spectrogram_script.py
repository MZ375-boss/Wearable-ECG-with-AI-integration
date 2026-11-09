import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # FIX: non-GUI backend, faster for bulk saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal
import wfdb

# Configuration
FS = 360
LOCAL_DATASET_PATH = Path(r"C:\Users\Admin\Desktop\Fyp_2026\Dataset\mit-bih-arrhythmia-database-1.0.0")

# Define dataset record groups
ds1_records = {
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230   # FIX: 212 removed, 230 added
}
ds2_records = {
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213,   # FIX: 212 added
    214, 219, 221, 222, 228, 231, 232, 233, 234
}
all_records = ds1_records.union(ds2_records)

PRE_PEAK = 0.35  # 350ms before
POST_PEAK = 0.45  # 450ms after
SAMPLES_BEFORE = round(PRE_PEAK * FS)   # FIX: round instead of int
SAMPLES_AFTER = round(POST_PEAK * FS)   # FIX: round instead of int

VALID_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "/", "f", "Q",
}

# Directory path setup for output images
base_dir = Path(r"C:\Users\Admin\Desktop\spectrograms2")
ds1_train_dir = base_dir / "DS1_train"
ds2_test_dir = base_dir / "DS2_test"

ds1_train_dir.mkdir(parents=True, exist_ok=True)
ds2_test_dir.mkdir(parents=True, exist_ok=True)

# --- Processing Loop ---
for rec_id_int in sorted(all_records):
    record_id = str(rec_id_int)
    record_path = LOCAL_DATASET_PATH / record_id
    
    if not (LOCAL_DATASET_PATH / f"{record_id}.dat").exists() or not (LOCAL_DATASET_PATH / f"{record_id}.hea").exists():
        print(f"Skipping Record {record_id}: Files not found in {LOCAL_DATASET_PATH}")
        continue
    
    print(f"\nProcessing Record ID: {record_id}")

    try:
        # Load record and annotations locally (must be string for wfdb)
        record = wfdb.rdrecord(str(record_path))
        annotation = wfdb.rdann(str(record_path), "atr")
    except Exception as e:
        print(f"Error loading record {record_id}: {e}")
        continue

    # Verify if p_signal exists and loaded correctly
    if hasattr(record, "p_signal") and record.p_signal is not None:
        ch = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0  # FIX: pick MLII by name
        ecg_signal = record.p_signal[:, ch]  # Lead MLII
    else:
        print(f"Warning: 'p_signal' missing for record {record_id}. Skipping.")
        continue

    peak_samples = annotation.sample
    symbols = annotation.symbol

    # Determine target directory based on record group
    target_dir = ds1_train_dir if rec_id_int in ds1_records else ds2_test_dir

    saved_count = 0
    # Iterate through each annotated R-peak
    for idx, (peak_sample, symbol) in enumerate(zip(peak_samples, symbols)):
        if symbol not in VALID_SYMBOLS:
            continue

        if (
            peak_sample - SAMPLES_BEFORE < 0
            or peak_sample + SAMPLES_AFTER >= len(ecg_signal)
        ):
            continue

        beat_segment = ecg_signal[
            peak_sample - SAMPLES_BEFORE : peak_sample + SAMPLES_AFTER
        ]

        # Compute Spectrogram for the single beat
        f, t, Sxx = signal.spectrogram(beat_segment, fs=FS, nperseg=32, noverlap=16)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)

        Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)
        Sxx_img = (Sxx_norm * 255).astype(np.uint8)

        # Save cleanly as a 64x64 RGB image
        fig, ax = plt.subplots(figsize=(0.64, 0.64), dpi=100)
        ax.axis("off")
        ax.pcolormesh(t, f, Sxx_img, shading="gouraud", cmap="viridis")
        ax.margins(0)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        temp_path = f"temp_{record_id}_{idx}.png"
        plt.savefig(temp_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        img = Image.open(temp_path).convert("RGB")
        img = img.resize((64, 64))

        safe_symbol = symbol.replace("/", "PACE").replace("|", "PIPE").replace('"', "Quote")
        final_filename = f"{record_id}_{idx}_beat_{safe_symbol}.png"
        final_path = target_dir / final_filename

        img.save(final_path)
        saved_count += 1

        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"Successfully generated and saved {saved_count} beat spectrograms for Record {record_id} into {target_dir.name}/")

print("\nSuccessfully generated and sorted beat spectrograms into 'spectrograms2/' folder.")