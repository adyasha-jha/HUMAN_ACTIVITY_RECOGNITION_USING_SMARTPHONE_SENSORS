import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class HAREvaluator:
    def __init__(self, model_path="best_har_model.h5"):
        self.model_path = model_path
        self.num_classes = 6
        self.activity_labels = [
            "WALKING", 
            "WALKING_UPSTAIRS", 
            "WALKING_DOWNSTAIRS",
            "SITTING", 
            "STANDING", 
            "LAYING"
        ]

    # ------------------------------------------------------------
    def load_data(self):
        print("Loading preprocessed test data...")

        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy") - 1    # convert labels 1-6 → 0-5

        print(f"✔ Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_test, y_test

    # ------------------------------------------------------------
    def evaluate(self):
        X_test, y_test = self.load_data()

        print("Loading trained model...")
        model = load_model(self.model_path)

        print("Running predictions...")
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"\n✔ Test Accuracy: {acc:.4f}")

        # Classification report (includes per-class F1)
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.activity_labels
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=self.activity_labels,
                    yticklabels=self.activity_labels,
                    cmap="Blues")
        plt.title("HAR Model Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # Save confusion matrix
        np.save("confusion_matrix.npy", cm)
        print("✔ Confusion matrix saved as confusion_matrix.npy")

# ------------------------------------------------------------
if __name__ == "__main__":
    evaluator = HAREvaluator(model_path="best_har_model.h5")
    evaluator.evaluate()
