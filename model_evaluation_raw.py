"""
Evaluation script for the raw-data HAR model.
"""
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class HAREvaluatorRaw:
    def __init__(self, model_path="best_har_raw_model.h5"):
        self.model_path = model_path
        self.num_classes = 6
        self.activity_labels = [
            "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
            "SITTING", "STANDING", "LAYING"
        ]

    def load_data(self):
        print("Loading test data...")
        X_test = np.load("X_test_raw.npy")
        y_test = np.load("y_test_raw.npy")  # Already 0-5
        print(f"✔ X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_test, y_test

    def evaluate(self):
        X_test, y_test = self.load_data()

        print("Loading model...")
        model = load_model(self.model_path)

        print("Running predictions...")
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
        print(f"{'='*50}")

        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.activity_labels,
            digits=4
        ))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.activity_labels,
            yticklabels=self.activity_labels
        )
        plt.title(f"HAR Confusion Matrix (Accuracy: {acc:.2%})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("confusion_matrix_raw.png", dpi=150)
        plt.show()

        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        for i, label in enumerate(self.activity_labels):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == i).mean()
                print(f"  {label:20s}: {class_acc:.4f} ({mask.sum()} samples)")

        return acc

if __name__ == "__main__":
    evaluator = HAREvaluatorRaw()
    evaluator.evaluate()