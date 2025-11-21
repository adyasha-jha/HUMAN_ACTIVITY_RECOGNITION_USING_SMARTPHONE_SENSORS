import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from collections import defaultdict

DATASET_DIR = "UCI HAR Dataset"
MODEL_PATH = "best_har_model.h5"

def load_raw_uci_with_subjects(dataset_dir=DATASET_DIR):
    # Load original UCI HAR features (same as in data_preprocessing.py)
    X_train = np.loadtxt(os.path.join(dataset_dir, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(dataset_dir, "train", "y_train.txt")).astype(int)
    subj_train = np.loadtxt(os.path.join(dataset_dir, "train", "subject_train.txt")).astype(int)

    X_test = np.loadtxt(os.path.join(dataset_dir, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(dataset_dir, "test", "y_test.txt")).astype(int)
    subj_test = np.loadtxt(os.path.join(dataset_dir, "test", "subject_test.txt")).astype(int)

    # Reproduce your normalization: global max over train+test
    X_all = np.vstack((X_train, X_test))
    max_val = X_all.max()
    X_test_norm = X_test / max_val

    # Reshape like training: (samples, timesteps, features)
    X_test_norm = X_test_norm.reshape((X_test_norm.shape[0], X_test_norm.shape[1], 1))

    # Convert labels to 0–5
    y_test_zero = y_test - 1

    return X_test_norm, y_test_zero, subj_test

def per_user_accuracy():
    print("Loading test data with subject IDs...")
    X_test, y_test, subj_test = load_raw_uci_with_subjects()

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Predicting on full test set...")
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Group results by subject
    user_correct = defaultdict(int)
    user_total = defaultdict(int)

    for y_true, y_hat, sid in zip(y_test, y_pred, subj_test):
        user_total[sid] += 1
        if y_true == y_hat:
            user_correct[sid] += 1

    print("\nPer-user accuracy (only test subjects):")
    for sid in sorted(user_total.keys()):
        acc = user_correct[sid] / user_total[sid]
        print(f"Subject {sid:2d}: accuracy = {acc:.3f}  (n={user_total[sid]})")

if __name__ == "__main__":
    per_user_accuracy()
