import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Paths
DATASET_DIR = "UCI HAR Dataset"
MODEL_PATH = "best_har_raw_model.h5"   # raw-signal model
MEAN_PATH = "norm_mean.npy"
STD_PATH = "norm_std.npy"

# Activity labels (0–5)
ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

def load_raw_test_with_subjects(dataset_dir=DATASET_DIR):
    """
    Load raw inertial test signals + subject IDs, then normalize using
    the SAME mean/std that were used during training.
    Test split here is the ORIGINAL UCI HAR 'test' subset.
    """

    inertial_path = os.path.join(dataset_dir, "test", "Inertial Signals")

    # total acceleration (same as in HARRawDataProcessor)
    acc_x = np.loadtxt(os.path.join(inertial_path, "total_acc_x_test.txt"))
    acc_y = np.loadtxt(os.path.join(inertial_path, "total_acc_y_test.txt"))
    acc_z = np.loadtxt(os.path.join(inertial_path, "total_acc_z_test.txt"))

    # Shape: (samples, 128, 3)
    X_test = np.stack([acc_x, acc_y, acc_z], axis=2)

    # Labels (1–6 in original UCI HAR)
    y_test = np.loadtxt(os.path.join(dataset_dir, "test", "y_test.txt")).astype(int)
    y_test_zero = y_test - 1   # → 0–5

    # Subject IDs
    subj_test = np.loadtxt(os.path.join(dataset_dir, "test", "subject_test.txt")).astype(int)

    # Load normalization constants computed during preprocessing_raw
    if not (os.path.exists(MEAN_PATH) and os.path.exists(STD_PATH)):
        raise FileNotFoundError(
            f"Normalization files {MEAN_PATH} / {STD_PATH} not found. "
            f"Run your raw preprocessing script first to generate them."
        )

    mean = np.load(MEAN_PATH)   # shape (3,)
    std = np.load(STD_PATH)     # shape (3,)

    # Normalize: (x - mean) / std, axis-wise
    X_test_norm = (X_test - mean) / std

    # Final shape: (samples, 128, 3) – exactly what your CNN/LSTM expects
    return X_test_norm, y_test_zero, subj_test


def per_user_accuracy():
    print("Loading raw test data with subject IDs...")
    X_test, y_test, subj_test = load_raw_test_with_subjects()
    print(f"✔ X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print("\nLoading trained raw model...")
    model = load_model(MODEL_PATH)

    print("Predicting on full test set...")
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Overall accuracy on this test split
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy on UCI-HAR TEST users: {overall_acc:.4f}")

    # Group by subject
    user_correct = defaultdict(int)
    user_total = defaultdict(int)

    for y_true, y_hat, sid in zip(y_test, y_pred, subj_test):
        user_total[sid] += 1
        if y_true == y_hat:
            user_correct[sid] += 1

    print("\nPer-user accuracy (UCI HAR test subjects):")
    for sid in sorted(user_total.keys()):
        acc = user_correct[sid] / user_total[sid]
        print(f"Subject {sid:2d}: accuracy = {acc:.3f}  (n={user_total[sid]})")

if __name__ == "__main__":
    per_user_accuracy()
