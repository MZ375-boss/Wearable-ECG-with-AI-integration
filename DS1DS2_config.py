import os
import shutil
from pathlib import Path

# 1. Define standard, clinically rigorous DS1 and DS2 record groups(Records 102, 104, 107, and 217 are usually excluded )
ds1_records = {101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 212, 215, 220, 223}
ds2_records = {100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234}

# 2. Configure paths to match your current system setup
root_source_dir = Path(r"C:\Users\Admin\Desktop\binary_dataset") # Folder containing 'normal' and 'anomalous'
output_train_dir = Path(r"C:\Users\Admin\Desktop\Spectrogram\DS1_Train")
output_test_dir = Path(r"C:\Users\Admin\Desktop\Spectrogram\DS2_Test")

# 3. Iterate through 'normal' and 'anomalous' subfolders
for class_folder in root_source_dir.iterdir():
    if not class_folder.is_dir():
        continue
    
    class_name = class_folder.name  # Evaluates to 'normal' or 'anomalous'
    print(f"Processing subfolder class: {class_name}...")
    
    # Create corresponding subfolders inside the new train/test destinations
    target_train_class_dir = output_train_dir / class_name
    target_test_class_dir = output_test_dir / class_name
    
    target_train_class_dir.mkdir(parents=True, exist_ok=True)
    target_test_class_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Scan all generated spectrogram images inside the current class folder
    for img_path in class_folder.glob("*.png"):
        filename = img_path.name
        
        try:
            # Matches your exact format from the image: "100_1_beat_N.png"
            # split('_') yields: ['100', '1', 'beat', 'N.png']
            parts = filename.split('_')
            
            # Extract the 1st item (index 0) as the integer Record ID
            record_id = int(parts[0])
            
            if record_id in ds1_records:
                shutil.copy(img_path, target_train_class_dir / filename)
            elif record_id in ds2_records:
                shutil.copy(img_path, target_test_class_dir / filename)
                
        except (ValueError, IndexError):
            print(f"Skipping file {filename} - Unable to parse Record ID from the start of the filename.")

print("Success! Dataset successfully split into DS1_Train and DS2_Test while preserving your structure!")
