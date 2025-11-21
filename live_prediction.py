import numpy as np
from tensorflow.keras.models import load_model


class HARLivePredictor:
    def __init__(self, model_path="best_har_raw_model.h5", window_size=128):
        # Load trained CNN/LSTM model
        self.model = load_model(model_path)
        self.window_size = window_size

        # Load normalization constants (same as training)
        self.mean = np.load("norm_mean.npy")     # shape (3,)
        self.std = np.load("norm_std.npy")       # shape (3,)

        # Labels (0–5)
        self.activity_labels = [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING"
        ]

        # Live sliding window (stores 3-axis readings)
        self.buffer = []   # each value = [ax, ay, az]


    # ------------------------------------------------------------
    def preprocess_window(self, window):
        """
        Preprocess the 128×3 window exactly like training data.
        """
        window = np.array(window, dtype=np.float32)     # (128, 3)

        # Normalize axis-wise:  (x - mean) / std
        window = (window - self.mean) / self.std

        # Reshape for model → (1, timesteps, features)
        window = window.reshape(1, window.shape[0], window.shape[1])

        return window


    # ------------------------------------------------------------
    def predict_activity(self, new_xyz):
        """
        new_xyz must be a list or array:
        [ax, ay, az]
        """
        if len(new_xyz) != 3:
            raise ValueError("Input must be [ax, ay, az]")

        self.buffer.append(new_xyz)

        # Need full window first
        if len(self.buffer) < self.window_size:
            return None

        # Take last 128 samples
        window = self.buffer[-self.window_size:]

        # Preprocess
        input_data = self.preprocess_window(window)

        # Predict
        probs = self.model.predict(input_data, verbose=0)
        pred_class = int(np.argmax(probs))
        pred_label = self.activity_labels[pred_class]

        return pred_label


# ------------------------------------------------------------
if __name__ == "__main__":
    predictor = HARLivePredictor(
        model_path="best_har_raw_model.h5",
        window_size=128
    )

    # Simulate live 3-axis accelerometer stream
    simulated_stream = np.random.randn(500, 3)  # each sample = [ax, ay, az]

    for xyz in simulated_stream:
        result = predictor.predict_activity(xyz)
        if result is not None:
            print("Predicted Activity →", result)
