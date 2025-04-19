# processing_data.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# -------------------------------
# Config
# -------------------------------
DATASET_FOLDER = 'dataset'
WINDOW_SIZE = 150  # 3 seconds if sampling rate is 50Hz
STRIDE = 30        # 80% overlap

# -------------------------------
# Sensor columns (based on your app)
# -------------------------------
sensor_cols = ['X_Acc','Y_Acc','Z_Acc','X_Gyro','Y_Gyro','Z_Gyro']

# -------------------------------
# Combine all CSV files
# -------------------------------
def load_and_concatenate_csv(folder):
    all_data = []
    all_labels = []
    
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            label = file.split("_")[1]  # Ex: Bitumen_1.csv → "Bitumen"
            df['label'] = label
            all_data.append(df)
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

# -------------------------------
# Create sliding windows
# -------------------------------
def create_windows(df, window_size, stride):
    X, y = [], []
    for i in range(0, len(df) - window_size, stride):
        window = df.iloc[i:i+window_size]
        if len(window['label'].unique()) == 1:
            X.append(window[sensor_cols].values)
            y.append(window['label'].iloc[0])
    return np.array(X), np.array(y)

# -------------------------------
# Main pipeline
# -------------------------------
print("🔄 Loading and processing data...")

df = load_and_concatenate_csv(DATASET_FOLDER)

X, y = create_windows(df, WINDOW_SIZE, STRIDE)

# Normalize sensor data
scaler = StandardScaler()
num_samples, num_timesteps, num_features = X.shape
X_reshaped = X.reshape(-1, num_features)
X_scaled = scaler.fit_transform(X_reshaped).reshape(num_samples, num_timesteps, num_features)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# Save scaler and label encoder
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save train/test splits
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("✅ Data processing complete. Saved all files.")


print("Dataset", df)

