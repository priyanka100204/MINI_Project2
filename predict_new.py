# predict_new.py

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load model & preprocessing objects
model = load_model('road_classifier_lstm.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load new 3-second CSV (150 rows)
new_data = pd.read_csv('preprocessed_sample.csv')  # Replace with your new CSV
sensor_cols = ['X_Acc','Y_Acc','Z_Acc','X_Gyro','Y_Gyro','Z_Gyro']
new_data = new_data[sensor_cols]

# Scale and reshape
X_input = scaler.transform(new_data.values).reshape(1, 150, 6)

# Predict
prediction = model.predict(X_input)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
print("Predicted Road Type:", predicted_class[0])
