import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal
import wfdb

# Configuration
RECORD_ID = "234"
FS = 360

# Point directly to your local dataset directory
LOCAL_DATA_DIR = Path(
    r"C:\Users\Admin\Desktop\Fyp_2026\Dataset\mit-bih-arrhythmia-database-1.0.0"
)
record_path = str(LOCAL_DATA_DIR / RECORD_ID)

# 1. Load record and annotations locally
print(f"Loading record from: {record_path}")
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, "atr")

# --- SAFETY CHECK ---
# Verify if p_signal exists and loaded correctly
if hasattr(record, "p_signal") and record.p_signal is not None:
  ecg_signal = record.p_signal[:, 0]  # Lead MLII
  print(f"Successfully loaded ECG signal with shape: {record.p_signal.shape}")
else:
  raise ValueError(
      f"Error: 'p_signal' not found in record object. Check if '{RECORD_ID}.dat'"
      f" and '{RECORD_ID}.hea' exist in: {LOCAL_DATA_DIR}"
  )
# --------------------

PRE_PEAK = 0.2  # 200ms before
POST_PEAK = 0.4  # 400ms after
SAMPLES_BEFORE = int(PRE_PEAK * FS)
SAMPLES_AFTER = int(POST_PEAK * FS)
symbol= {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "/", "f", "Q",
}

# Directory path setup for output images
base_dir = "binary_dataset"
normal_dir = os.path.join(base_dir, "normal")
anomaly_dir = os.path.join(base_dir, "anomalous")

os.makedirs(normal_dir, exist_ok=True)
os.makedirs(anomaly_dir, exist_ok=True)

peak_samples = annotation.sample
symbols = annotation.symbol
print(f"Successfully loaded {len(peak_samples)} beat annotations.")

# 2. Iterate through each annotated R-peak
for idx, (peak_sample, symbol) in enumerate(zip(peak_samples, symbols)):
  if (
      peak_sample - SAMPLES_BEFORE < 0
      or peak_sample + SAMPLES_AFTER >= len(ecg_signal)
  ):
    continue

  beat_segment = ecg_signal[
      peak_sample - SAMPLES_BEFORE : peak_sample + SAMPLES_AFTER
  ]

  # 3. Compute Spectrogram for the single beat
  f, t, Sxx = signal.spectrogram(beat_segment, fs=FS, nperseg=32, noverlap=16)
  Sxx_log = 10 * np.log10(Sxx + 1e-10)

  Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)
  Sxx_img = (Sxx_norm * 255).astype(np.uint8)

  # 4. Save cleanly as a 224x224 RGB image
  fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
  ax.axis("off")
  ax.pcolormesh(t, f, Sxx_img, shading="gouraud", cmap="viridis")
  ax.margins(0)
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

  temp_path = f"temp_{RECORD_ID}_{idx}.png"
  plt.savefig(temp_path, bbox_inches="tight", pad_inches=0)
  plt.close(fig)

  img = Image.open(temp_path).convert("RGB")
  img = img.resize((224, 224))

  # 5. Route to 'normal' or 'anomalous' folder
  safe_symbol = symbol.replace("/", "PACE").replace("|", "PIPE").replace('"', "Quote")
  if symbol == "N":
    final_path = os.path.join(normal_dir, f"{RECORD_ID}_{idx}_beat_N.png")
  else:
    final_path = os.path.join(anomaly_dir, f"{RECORD_ID}_{idx}_beat_{safe_symbol}.png")

  img.save(final_path)

  if os.path.exists(temp_path):
    os.remove(temp_path)

print(
    "Successfully generated and sorted beat spectrograms from local files into"
    f" '{base_dir}/'."
)