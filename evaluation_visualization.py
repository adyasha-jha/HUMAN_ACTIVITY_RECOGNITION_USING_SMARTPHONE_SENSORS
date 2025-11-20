# -*- coding: utf-8 -*-
"""
Human Activity Recognition - Evaluation & Visualization Script
Evaluates trained models and generates performance visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)
import tensorflow as tf

np.random.seed(42)

ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ACTIVITY_LABELS,
        yticklabels=ACTIVITY_LABELS
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — HAR")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Confusion matrix saved as {save_path}")
    plt.show()


def plot_class_distribution(y_true, save_path="class_distribution.png"):
    """Plot distribution of actual class labels."""
    plt.figure(figsize=(9, 5))
    sns.countplot(x=y_true, palette="viridis")
    plt.title("Class Distribution in Test Data")
    plt.xlabel("Activity Label")
    plt.ylabel("Count")
    plt.xticks(ticks=range(6), labels=ACTIVITY_LABELS, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Class distribution saved as {save_path}")
    plt.show()


def plot_prediction_confidence(predicted_probs, save_path="prediction_confidence.png"):
    """Plot confidence (max probability) of predictions."""
    confidences = np.max(predicted_probs, axis=1)

    plt.figure(figsize=(9, 5))
    sns.histplot(confidences, bins=20, kde=True)
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Prediction confidence plot saved as {save_path}")
    plt.show()


def evaluate_model(model_path, data_path="har_processed_data.npz"):
    """
    Load trained model & dataset and compute all metrics.
    """

    print("\nLoading processed dataset...")
    data = np.load(data_path, allow_pickle=True)
    X_test = data["test_X"]
    y_test = data["test_y"]

    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)

    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\n==============================")
    print("⭐ MODEL EVALUATION RESULTS ⭐")
    print("==============================")
    print(f"✓ Accuracy: {accuracy:.4f}")

    # F1 Score
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"✓ Weighted F1 Score: {f1:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=ACTIVITY_LABELS))

# -*- coding: utf-8 -*-
"""
HAR Evaluation & Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.models import load_model


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.show()


def plot_class_distribution(labels):
    plt.figure(figsize=(7, 5))
    sns.countplot(x=labels)
    plt.title("Class Distribution")
    plt.savefig("class_distribution.png")
    plt.show()


def plot_prediction_confidence(pred_probs):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=pred_probs)
    plt.title("Prediction Confidence Distribution")
    plt.savefig("confidence.png")
    plt.show()


def evaluate_model(model_file):
    print("\n=== Evaluating Model:", model_file)

    model = load_model(model_file)

    data = np.load("har_processed_data.npz", allow_pickle=True)
    X_test = data["test_X"]
    y_test = data["test_y"]

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    acc = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nAccuracy:", acc)
    print("F1 Score:", f1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)
    plot_class_distribution(y_test)
    plot_prediction_confidence(predictions)

    print("\n✓ Evaluation Complete!")
    return acc, f1


if __name__ == "__main__":
    model_file = "HAR_LSTM_final.h5"
    evaluate_model(model_file)
