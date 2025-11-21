"""
Data preprocessing for HAR using RAW inertial signals (not pre-computed features).
This allows the Android app to send similar raw accelerometer data.
"""
import os
import urllib.request
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split

class HARRawDataProcessor:
    def __init__(self, dataset_dir="UCI HAR Dataset"):
        self.dataset_dir = dataset_dir
        self.download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        self.zip_path = "UCI_HAR_Dataset.zip"
        
        # Window parameters - MUST match Android app
        self.window_size = 128  # 128 timesteps (2.56 sec at 50Hz)
        self.n_features = 3     # ax, ay, az (total acceleration)

    def download_file_with_resume(self, url, filename):
        CHUNK_SIZE = 1024 * 1024
        if os.path.exists(filename):
            if zipfile.is_zipfile(filename):
                print("✔ Valid ZIP file already exists.")
                return
            else:
                print("⚠ Existing file is corrupted. Re-downloading...")
                os.remove(filename)

        req = urllib.request.urlopen(url)
        total_size = int(req.headers.get("Content-Length", 0))
        print(f"Downloading {filename} ({total_size/1024/1024:.2f} MB)...")

        downloaded = 0
        with open(filename, "wb") as f:
            while True:
                chunk = req.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded: {downloaded/1024/1024:.2f} MB", end="\r")
        print("\n✔ Download complete.")

    def download_dataset(self):
        if os.path.exists(self.dataset_dir):
            print("✔ Dataset already exists.")
            return
        print("Downloading UCI HAR Dataset...")
        self.download_file_with_resume(self.download_url, self.zip_path)
        if not zipfile.is_zipfile(self.zip_path):
            raise Exception("❌ Invalid ZIP file.")
        print("Extracting...")
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            zf.extractall()
        print("✔ Extraction complete.")

    def load_inertial_signals(self, subset="train"):
        """Load raw accelerometer data (total_acc) from Inertial Signals folder."""
        path = os.path.join(self.dataset_dir, subset, "Inertial Signals")
        
        # Load total acceleration (includes gravity) - this is what phone sensors give
        acc_x = np.loadtxt(os.path.join(path, f"total_acc_x_{subset}.txt"))
        acc_y = np.loadtxt(os.path.join(path, f"total_acc_y_{subset}.txt"))
        acc_z = np.loadtxt(os.path.join(path, f"total_acc_z_{subset}.txt"))
        
        # Stack into shape (samples, timesteps, features)
        # Each row is 128 readings, we have 3 axes
        X = np.stack([acc_x, acc_y, acc_z], axis=2)
        
        return X

    def load_labels(self, subset="train"):
        path = os.path.join(self.dataset_dir, subset, f"y_{subset}.txt")
        y = np.loadtxt(path).astype(int)
        return y

    def preprocess(self):
        print("Loading raw inertial signals...")
        
        X_train = self.load_inertial_signals("train")
        y_train = self.load_labels("train")
        X_test = self.load_inertial_signals("test")
        y_test = self.load_labels("test")

        print(f"Train: X={X_train.shape}, y={y_train.shape}")
        print(f"Test:  X={X_test.shape}, y={y_test.shape}")

        # Combine for unified processing
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        # CRITICAL: Compute normalization stats BEFORE splitting
        # These will be used in Android app too!
        self.mean = X.mean(axis=(0, 1))  # mean per axis
        self.std = X.std(axis=(0, 1))    # std per axis
        
        print(f"\n=== NORMALIZATION CONSTANTS (copy to Android app) ===")
        print(f"MEAN = [{self.mean[0]:.6f}f, {self.mean[1]:.6f}f, {self.mean[2]:.6f}f]")
        print(f"STD  = [{self.std[0]:.6f}f, {self.std[1]:.6f}f, {self.std[2]:.6f}f]")
        print("=" * 55)

        # Normalize: (x - mean) / std
        X = (X - self.mean) / self.std

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Convert labels from 1-6 to 0-5
        y_train = y_train - 1
        y_test = y_test - 1

        print(f"\nFinal shapes:")
        print(f"X_train: {X_train.shape}")  # (samples, 128, 3)
        print(f"X_test:  {X_test.shape}")
        print(f"y_train: {y_train.shape}, classes: {np.unique(y_train)}")

        # Save
        np.save("X_train_raw.npy", X_train)
        np.save("X_test_raw.npy", X_test)
        np.save("y_train_raw.npy", y_train)
        np.save("y_test_raw.npy", y_test)
        
        # Save normalization constants
        np.save("norm_mean.npy", self.mean)
        np.save("norm_std.npy", self.std)

        print("\n✔ Saved: X_train_raw.npy, X_test_raw.npy, y_train_raw.npy, y_test_raw.npy")
        print("✔ Saved: norm_mean.npy, norm_std.npy")

if __name__ == "__main__":
    processor = HARRawDataProcessor()
    processor.download_dataset()
    processor.preprocess()