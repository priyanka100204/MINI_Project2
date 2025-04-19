# visualize_data.py

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load label encoder and y_train
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

y_train = np.load('y_train.npy')
X_train = np.load('X_train.npy')

# Convert back to string labels
labels = label_encoder.inverse_transform(np.argmax(y_train, axis=1))
print("labels", labels[0])
# Plot average sample from 2 labels
def compare_labels(label1, label2):
    idx1 = np.where(labels == label1)[0]
    idx2 = np.where(labels == label2)[0]

    sample1 = X_train[idx1]
    sample2 = X_train[idx2]

    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(sample1[:, i], label=label1)
        plt.plot(sample2[:, i], label=label2)
        plt.title(['X_Acc','Y_Acc','Z_Acc','X_Gyro','Y_Gyro','Z_Gyro'][i])
        plt.legend()
    plt.tight_layout()
    plt.show()
# print("Available labels:", np.unique(labels))

# Example usage
compare_labels('Bitumen', 'Kankar')
