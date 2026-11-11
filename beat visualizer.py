from pathlib import Path
import wfdb
import matplotlib.pyplot as plt

# 1. Point to your dataset folder and record 100
dataset_dir = Path(r"C:\Users\Admin\Desktop\Fyp_2026\Dataset\mit-bih-arrhythmia-database-1.0.0")
record_path = str(dataset_dir / '101')

# 2. Read record and annotations
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

print(f"Sampling Frequency: {record.fs} Hz")
print(f"Number of Leads: {record.n_sig}")
print(f"Signal Shape: {record.p_signal.shape}")

# 3. Plot the first 10 seconds manually using matplotlib
fs = record.fs
samples_to_plot = int(fs * 10)
time_axis = [i / fs for i in range(samples_to_plot)]

plt.figure(figsize=(12, 6))
# Plot Lead 0
plt.plot(time_axis, record.p_signal[:samples_to_plot, 0], label=record.sig_name[0], color='black')

plt.title("MIT-BIH Record 100 - Lead 0 (First 10 Seconds)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.legend()
plt.show()