# utils.py

import os
import pandas as pd

def list_all_files(folder, extension=".csv"):
    return [f for f in os.listdir(folder) if f.endswith(extension)]

def read_and_label(filepath):
    df = pd.read_csv(filepath)
    label = os.path.basename(filepath).split('_')[0]
    df['label'] = label
    return df


