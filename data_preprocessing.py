# -*- coding: utf-8 -*-
"""
Human Activity Recognition - Data Preprocessing
Dataset: UCI HAR Dataset
"""

import numpy as np
import pandas as pd
import os
import zipfile
import urllib.request
from scipy import signal
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
np.random.seed(42)

class HARDataProcessor:
    def __init__(self, data_dir='UCI HAR Dataset'):
        # FIXED FOLDER NAME
        self.data_dir = data_dir  
        self.sampling_rate = 50
        self.window_size = 128
        self.overlap = 0.5

    def download_dataset(self):
        """Download UCI HAR Dataset safely."""
        url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
        zip_path = 'uci_har.zip'

        if not os.path.exists(self.data_dir):
            print("Downloading UCI HAR Dataset...")
            urllib.request.urlretrieve(url, zip_path)

            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove(zip_path)

            # FIX: Extracted folder name
            if os.path.exists("UCI HAR Dataset"):
                print("✓ Dataset downloaded & extracted successfully")
            else:
                raise Exception("Extraction failed — folder not found.")
        else:
            print("Dataset already exists!")

    def load_raw_signals(self, set_type='train'):
        """Load raw accelerometer and gyroscope signals."""
        signals_path = os.path.join(self.data_dir, set_type, 'Inertial Signals')

        signal_types = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        signals_data = {}
        for sig in signal_types:
            file_path = os.path.join(signals_path, f"{sig}_{set_type}.txt")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} NOT FOUND")

            signals_data[sig] = np.loadtxt(file_path)

        labels = np.loadtxt(os.path.join(self.data_dir, set_type, f"y_{set_type}.txt"), dtype=int) - 1
        subjects = np.loadtxt(os.path.join(self.data_dir, set_type, f"subject_{set_type}.txt"), dtype=int)

        return signals_data, labels, subjects

    def apply_butterworth_filter(self, data, cutoff=20, order=4):
        nyquist = 0.5 * self.sampling_rate
        b, a = signal.butter(order, cutoff / nyquist, btype='low')
        return signal.filtfilt(b, a, data, axis=1)

    def prepare_dataset(self):
        print("Loading training data...")
        train_signals, train_labels, train_subjects = self.load_raw_signals("train")

        print("Loading test data...")
        test_signals, test_labels, test_subjects = self.load_raw_signals("test")

        print("Applying filters...")
        for k in train_signals:
            train_signals[k] = self.apply_butterworth_filter(train_signals[k])
            test_signals[k] = self.apply_butterworth_filter(test_signals[k])

        print("Stacking signals...")
        train_X = np.stack([train_signals[k] for k in sorted(train_signals.keys())], axis=-1)
        test_X = np.stack([test_signals[k] for k in sorted(test_signals.keys())], axis=-1)

        print("Normalizing data...")
        scaler = StandardScaler()
        train_X_flat = train_X.reshape(-1, train_X.shape[-1])
        test_X_flat = test_X.reshape(-1, test_X.shape[-1])

        train_X_scaled = scaler.fit_transform(train_X_flat).reshape(train_X.shape)
        test_X_scaled = scaler.transform(test_X_flat).reshape(test_X.shape)

        print(f"Train shape: {train_X_scaled.shape}")
        print(f"Test shape: {test_X_scaled.shape}")

        return {
            "train_X": train_X_scaled,
            "train_y": train_labels,
            "train_subjects": train_subjects,
            "test_X": test_X_scaled,
            "test_y": test_labels,
            "test_subjects": test_subjects,
            "scaler": scaler
        }


# MAIN
if __name__ == "__main__":
    processor = HARDataProcessor()

    processor.download_dataset()  # FIXED

    dataset = processor.prepare_dataset()

    print("\nSaving dataset...")
    np.savez_compressed("har_processed_data.npz", **dataset)

    print("\n✓ DONE!")
