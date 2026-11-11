from pathlib import Path
import wfdb

# 1. Point to your dataset folder
dataset_dir = Path(r"C:\Users\Admin\Desktop\Fyp_2026\Dataset\mit-bih-arrhythmia-database-1.0.0")

# 2. Automatically find all record IDs by looking for .hea files in the folder
# (Each record has a .hea header file)
record_ids = sorted(list(set(f.stem for f in dataset_dir.glob('*.hea'))))

print(f"Successfully found {len(record_ids)} patient records in your folder!\n")

# 3. Loop through every single record
for rec_id in record_ids:
    record_path = str(dataset_dir / rec_id)
    
    try:
        # Load record and annotations
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        print(f"Record {rec_id} | Sampling Rate: {record.fs} Hz | Leads: {record.n_sig} | Beats: {len(annotation.symbol)}")
        
    except Exception as e:
        print(f"Skipping {rec_id} (not a standard record file): {e}")