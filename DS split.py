from pathlib import Path
import wfdb
import numpy as np

# 1. Point to your dataset folder
dataset_dir = Path(r"C:\Users\Admin\Desktop\Fyp_2026\Dataset\mit-bih-arrhythmia-database-1.0.0")

# 2. Standard MIT-BIH DS1 and DS2 patient record splits
DS1_records = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', 
               '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', 
               '223', '230']

DS2_records = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', 
               '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', 
               '233', '234']

# Valid AAMI mapping function integrated directly during extraction
def map_to_aami(symbol):
    # N: Normal, LBBB, RBBB, Atrial escape, Nodal escape
    if symbol in ['N', 'L', 'R', 'e', 'j']:
        return 'N'
    # S: Atrial premature, Aberrated atrial premature, Nodal premature, Supraventricular premature
    elif symbol in ['A', 'a', 'J', 'S']:
        return 'S'
    # V: Premature ventricular contraction, Ventricular escape
    elif symbol in ['V', 'E']:
        return 'V'
    # F: Fusion of ventricular and normal beat
    elif symbol in ['F']:
        return 'F'
    # Q: Paced, Fusion of paced and normal, Unclassified
    elif symbol in ['/', 'f', 'Q']:
        return 'Q'
    else:
        return None # Ignore non-beat annotations (noise, artifacts, rhythm markers)

def process_records(record_list, dataset_name):
    all_heartbeats = []
    all_labels = []
    
    window_before = 120
    window_after = 180
    
    print(f"\n--- Processing {dataset_name} ({len(record_list)} records) ---")
    
    for rec_id in record_list:
        record_path = str(dataset_dir / rec_id)
        
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            signal = record.p_signal[:, 0]  # Using Lead 0
            beat_samples = annotation.sample
            beat_symbols = annotation.symbol
            
            count = 0
            for i, peak_idx in enumerate(beat_samples):
                # Map symbol to AAMI class first; skip if it's an invalid/non-beat symbol
                aami_class = map_to_aami(beat_symbols[i])
                if aami_class is None:
                    continue
                
                if peak_idx - window_before >= 0 and peak_idx + window_after < len(signal):
                    heartbeat_window = signal[peak_idx - window_before : peak_idx + window_after]
                    all_heartbeats.append(heartbeat_window)
                    all_labels.append(aami_class)
                    count += 1
                    
            print(f"Record {rec_id}: Sliced {count} valid beats.")
        except Exception as e:
            print(f"Skipping {rec_id}: {e}")
            
    return np.array(all_heartbeats), np.array(all_labels)

# 3. Process DS1 (Training Set)
X_train, y_train = process_records(DS1_records, "DS1 (Training)")

# 4. Process DS2 (Testing Set)
X_test, y_test = process_records(DS2_records, "DS2 (Testing)")

# 5. Save both datasets into your Processed_Data folder
output_dir = Path(r"C:\Users\Admin\Desktop\Fyp_2026\Processed_Data")
output_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(output_dir / 'mitbih_DS1_train.npz', X=X_train, y=y_train)
np.savez_compressed(output_dir / 'mitbih_DS2_test.npz', X=X_test, y=y_test)

print("\n========================================")
print(" Extraction and Saving Complete! ")
print(f"DS1 Training Set saved: {X_train.shape} beats -> mitbih_DS1_train.npz")
print(f"DS2 Testing Set saved:  {X_test.shape} beats -> mitbih_DS2_test.npz")
print("========================================")
