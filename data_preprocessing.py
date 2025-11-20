import os
import urllib.request
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split

class HARDataProcessor:
    def __init__(self, dataset_dir="UCI HAR Dataset"):
        self.dataset_dir = dataset_dir
        self.download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        self.zip_path = "UCI_HAR_Dataset.zip"

    # ------------------------------------------------------------
    # SAFE DOWNLOADER (Fixes IncompleteRead / Broken ZIP issues)
    # ------------------------------------------------------------
    def download_file_with_resume(self, url, filename):
        """Downloads file with resume capability and prevents incomplete ZIP."""
        CHUNK_SIZE = 1024 * 1024  # 1 MB chunks

        # Check if file exists and is valid ZIP
        if os.path.exists(filename):
            if zipfile.is_zipfile(filename):
                print("✔ Valid ZIP file already exists.")
                return
            else:
                print("⚠ Existing file is corrupted. Deleting and re-downloading...")
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
                print(f"Downloaded: {downloaded/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB", end="\r")

        print("\n✔ Download complete.")

    # ------------------------------------------------------------
    def download_dataset(self):
        """Downloads and extracts the UCI HAR dataset safely."""
        if os.path.exists(self.dataset_dir):
            print("✔ Dataset already exists.")
            return

        print("Downloading UCI HAR Dataset...")
        self.download_file_with_resume(self.download_url, self.zip_path)

        # Validate ZIP (this should always pass now)
        if not zipfile.is_zipfile(self.zip_path):
            raise Exception("❌ Downloaded file is not a valid ZIP. Try deleting it and rerun.")

        print("Extracting dataset...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall()

        print("✔ Extraction complete.")
        
        # Optional: Clean up ZIP file after extraction
        # os.remove(self.zip_path)

    # ------------------------------------------------------------
    # Helper to load dataset files
    # ------------------------------------------------------------
    def load_file(self, filepath):
        return np.loadtxt(filepath)

    # ------------------------------------------------------------
    def load_dataset(self):
        print("Loading dataset...")

        # Load train data
        X_train = self.load_file(os.path.join(self.dataset_dir, "train", "X_train.txt"))
        y_train = self.load_file(os.path.join(self.dataset_dir, "train", "y_train.txt")).astype(int)

        # Load test data
        X_test = self.load_file(os.path.join(self.dataset_dir, "test", "X_test.txt"))
        y_test = self.load_file(os.path.join(self.dataset_dir, "test", "y_test.txt")).astype(int)

        # Combine for unified splitting
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        print(f"✔ Loaded: X={X.shape}, y={y.shape}")
        return X, y

    # ------------------------------------------------------------
    def preprocess(self):
        X, y = self.load_dataset()

        # Normalize features
        X = X / X.max()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Reshape for LSTM / 1D-CNN → (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        print("✔ Preprocessing complete.")
        print(f"Final Shapes → X_train: {X_train.shape}, X_test: {X_test.shape}")

        # Save files for training
        np.save("X_train.npy", X_train)
        np.save("X_test.npy", X_test)
        np.save("y_train.npy", y_train)
        np.save("y_test.npy", y_test)

        print("✔ Saved preprocessed numpy files.")

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    processor = HARDataProcessor()
    processor.download_dataset()
    processor.preprocess()