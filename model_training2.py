import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from scipy.signal import spectrogram
from biosppy.signals import ecg

# --- 1. CONFIGURATION ---
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
        layers.Dense(1, activation='sigmoid') # Binary classification (0: Normal, 1: Anomalous)
    ])
    
    return model

# --- 3. DATA PREPARATION: SPECTROGRAM FOLDERS ---
def load_dataset_sequences_from_folders(base_dir, time_steps=10):
    """
    Loads pre-generated spectrogram PNG images from structured subfolders
    and groups them into sequences of length `time_steps`.
    """
    subfolders = [
        ("DS1_Train/normal", 0),
        ("DS1_Train/anomalous", 1),
        ("DS2_Test/normal", 0),
        ("DS2_Test/anomalous", 1)
    ]
    
    X, y = [], []
    
    for subfolder, label in subfolders:
        dir_path = os.path.join(base_dir, subfolder)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found -> {dir_path}")
            continue
            
        file_list = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')])
        
        # Chunk files into sequential windows
        for i in range(0, len(file_list) - time_steps + 1):
            sequence_paths = file_list[i : i + time_steps]
            seq_imgs = []
            for path in sequence_paths:
                img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_arr = img_to_array(img) / 255.0  # Normalize to [0, 1]
                seq_imgs.append(img_arr)
            X.append(seq_imgs)
            y.append(label)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# --- 4. DATA PREPARATION: RAW ECG SIGNALS ---
def raw_ecg_to_spectrogram_sequence(raw_signal, sampling_rate=360, time_steps=10):
    """
    Takes a raw 1D ECG voltage array, segments individual heartbeats via R-peaks,
    converts them into STFT spectrograms resized to (IMG_HEIGHT, IMG_WIDTH, CHANNELS),
    and groups them into sliding sequences of `time_steps`.
    """
    try:
        out = ecg.ecg(signal=raw_signal, sampling_rate=sampling_rate, show=False)
        rpeaks = out['rpeaks']
    except Exception as e:
        print(f"Error in R-peak detection: {e}")
        return None

    # Define window size around R-peak (e.g., 0.2s before, 0.4s after)
    before = int(0.2 * sampling_rate)
    after = int(0.4 * sampling_rate)
    
    spectrogram_sequence = []
    
    for r in rpeaks:
        start = r - before
        end = r + after
        if start >= 0 and end < len(raw_signal):
            beat_signal = raw_signal[start:end]
            
            # Compute Short-Time Fourier Transform (STFT)
            _, _, Sxx = spectrogram(beat_signal, fs=sampling_rate, nperseg=32, noverlap=16)
            
            # Convert single channel spectrogram to 3-channel and resize to model specs
            Sxx_normalized = np.stack([Sxx] * CHANNELS, axis=-1)
            resized_spec = tf.image.resize(Sxx_normalized, [IMG_HEIGHT, IMG_WIDTH]).numpy() / 255.0
            spectrogram_sequence.append(resized_spec)
            
    spectrogram_sequence = np.array(spectrogram_sequence, dtype=np.float32)
    
    if len(spectrogram_sequence) < time_steps:
        print(f"Error: Found only {len(spectrogram_sequence)} heartbeats, but {time_steps} are required.")
        return None

    # Chunk into sliding windows of length `time_steps`
    X_inference = []
    for i in range(0, len(spectrogram_sequence) - time_steps + 1):
        X_inference.append(spectrogram_sequence[i : i + time_steps])
        
    return np.array(X_inference, dtype=np.float32)

# --- 5. UNIFIED CONTROLLER INTERFACE ---
class ECGAnalyzerSystem:
    def __init__(self, model_path="best_ecg_model.keras"):
        self.model_path = model_path
        self.model = None

    def train_from_directory(self, data_dir):
        """Trains the model using dataset directories containing spectrogram PNGs."""
        print(f"Loading spectrogram sequences from: {data_dir}")
        X, y = load_dataset_sequences_from_folders(data_dir, time_steps=TIME_STEPS)
        print(f"Dataset shapes -> X: {X.shape}, y: {y.shape}")
        
        if len(X) == 0:
            print("Error: No data found to train on.")
            return

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = build_cnn_lstm_model(TIME_STEPS, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        self.model.summary()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(self.model_path, monitor="val_accuracy", mode="max", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        print(f"Training complete! Model saved to '{self.model_path}'.")

    def predict_raw_ecg(self, raw_signal, sampling_rate=360):
        """Loads a saved model and predicts anomaly probabilities on a raw patient ECG signal."""
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading trained model from '{self.model_path}'...")
                self.model = tf.keras.models.load_model(self.model_path)
            else:
                print("Error: No trained model found. Train the model first or provide a valid model path.")
                return None

        # Convert raw ECG signal into model-compatible sequence windows
        X_infer = raw_ecg_to_spectrogram_sequence(raw_signal, sampling_rate=sampling_rate, time_steps=TIME_STEPS)
        if X_infer is None:
            return None

        # Run inference
        predictions = self.model.predict(X_infer)
        return predictions.flatten() # Returns array of probabilities (0 = Normal, 1 = Anomalous)

# --- 6. EXAMPLE EXECUTION ---
if __name__ == "__main__":
    DATA_DIR = r"C:\Users\Admin\Desktop\Spectrogram"
    
    system = ECGAnalyzerSystem(model_path="best_ecg_model.keras")
    
    # Mode A: Train the model using your spectrogram image folders
    if os.path.exists(DATA_DIR):
        system.train_from_directory(DATA_DIR)
    
    # Mode B: Analyze a raw patient ECG signal directly (Future-proofing demonstration)
    simulated_patient_signal = np.sin(np.linspace(0, 100, 3600)) # Placeholder raw data array
    results = system.predict_raw_ecg(simulated_patient_signal, sampling_rate=360)
    if results is not None:
        print("Patient ECG Window Predictions (Anomaly Probabilities):", results)