import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
DATA_DIR = r"C:\Users\Admin\Desktop\Spectrogram" # Contains 'ds1' and 'ds2' subfolders
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
TIME_STEPS = 10  # Number of consecutive heartbeats in a sequence
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

# --- 2. MODEL ARCHITECTURE (2D-CNN-LSTM Hybrid) ---
def build_cnn_lstm_model(time_steps, height, width, channels):
    """
    TimeDistributed applies the 2D-CNN feature extractor to each 
    individual heartbeat spectrogram independently.
    The LSTM then analyzes the sequence of features across time steps.
    """
    model = models.Sequential([
        # Input shape: (Batch, TimeSteps, Height, Width, Channels)
        layers.Input(shape=(time_steps, height, width, channels)),
        
        # Spatial Feature Extraction via 2D CNN (applied per heartbeat frame)
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        
        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        
        layers.TimeDistributed(layers.Flatten()),
        
        # Temporal Feature Modeling across consecutive heartbeats
        layers.LSTM(64, return_sequences=False),
        
        # Classification Head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Binary classification (ds1 vs ds2)
    ])
    
    return model

# --- 3. DATA PREPARATION UTILITIES ---
def load_dataset_sequences(base_dir, time_steps=10):
    # Paths for ds1 and ds2, each containing 'normal' and 'anomalous' subfolders
    ds1_normal = os.path.join(base_dir, "DS1_Train", "normal")
    ds1_anomaly = os.path.join(base_dir, "DS1_Train", "anomalous")
    
    ds2_normal = os.path.join(base_dir, "DS2_Test", "normal")
    ds2_anomaly = os.path.join(base_dir, "DS2_Test", "anomalous")
    
    # Helper to gather all image file paths from a directory safely
    def get_files(directory):
        if not os.path.exists(directory):
            return []
        return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])

    ds1_files = get_files(ds1_normal) + get_files(ds1_anomaly)
    ds2_files = get_files(ds2_normal) + get_files(ds2_anomaly)
    
    X, y = [], []
    
    # Helper to chunk files into sequential windows of length `time_steps`
    def process_files(file_list, label):
        for i in range(0, len(file_list) - time_steps + 1):
            sequence_paths = file_list[i : i + time_steps]
            seq_imgs = []
            for path in sequence_paths:
                img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_arr = img_to_array(img) / 255.0  # Normalize to [0, 1]
                seq_imgs.append(img_arr)
            X.append(seq_imgs)
            y.append(label)

    print("Loading ds1 sequences...")
    process_files(ds1_files, label=0)
    
    print("Loading ds2 sequences...")
    process_files(ds2_files, label=1)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

if __name__ == "__main__":
    # Load sequences from ds1 and ds2
    if os.path.exists(DATA_DIR):
        X, y = load_dataset_sequences(DATA_DIR, time_steps=TIME_STEPS)
        print(f"Dataset shapes -> X: {X.shape}, y: {y.shape}")
        
        if len(X) > 0:
            # Train/Test Split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Build model
            model = build_cnn_lstm_model(TIME_STEPS, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
            )
            model.summary()
            
            # Callbacks for training
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint("best_ecg_model.keras", monitor="val_accuracy", mode="max", save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=callbacks
            )
            print("Training complete! Best model saved as 'best_ecg_model.keras'.")
        else:
            print("Error: Not enough spectrogram images found in ds1/ds2 subfolders to form sequences. Generate more data first.")
    else:
        print(f"Error: Directory '{DATA_DIR}' not found. Run your spectrogram generator script first.")