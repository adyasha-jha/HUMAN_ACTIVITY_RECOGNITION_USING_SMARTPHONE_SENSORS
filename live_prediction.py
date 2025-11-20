import numpy as np
from tensorflow.keras.models import load_model

class HARLivePredictor:
    def __init__(self, model_path="../best_har_model.h5", window_size=128):
        self.model = load_model(model_path)
        self.window_size = window_size
        self.activity_labels = [
            "WALKING", 
            "WALKING_UPSTAIRS", 
            "WALKING_DOWNSTAIRS",
            "SITTING", 
            "STANDING", 
            "LAYING"
        ]

        self.buffer = []  # stores live sensor values until window fills

    # ------------------------------------------------------------
    def preprocess_window(self, window):
        """Preprocess exactly like training pipeline."""
        window = np.array(window)

        # Normalize using max scaling (same as training)
        window = window / np.max(window)

        # Reshape → (1, timesteps, features)
        window = window.reshape(1, window.shape[0], 1)
        return window

    # ------------------------------------------------------------
    def predict_activity(self, new_value):
        """
        Add new accelerometer/gyro reading and predict
        once we reach one full window.
        """
        self.buffer.append(new_value)

        # Not enough data yet
        if len(self.buffer) < self.window_size:
            return None  

        # Take last window_size samples
        window = self.buffer[-self.window_size:]

        # Preprocess window
        input_data = self.preprocess_window(window)

        # Predict
        probs = self.model.predict(input_data)
        pred_class = np.argmax(probs)
        pred_label = self.activity_labels[pred_class]

        return pred_label

# ------------------------------------------------------------
if __name__ == "__main__":
    predictor = HARLivePredictor()

    # Example usage
    simulated_stream = np.random.randn(500)  # simulate sensor values

    for value in simulated_stream:
        result = predictor.predict_activity(value)
        if result is not None:
            print("Predicted Activity →", result)
