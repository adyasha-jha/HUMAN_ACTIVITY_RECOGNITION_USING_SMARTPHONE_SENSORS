"""
Model training for HAR using raw accelerometer data.
Input shape: (batch, 128 timesteps, 3 features)
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class HARRawModelTrainer:
    def __init__(self, model_type="lstm"):
        self.model_type = model_type.lower()
        self.num_classes = 6
        self.activity_labels = [
            "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
            "SITTING", "STANDING", "LAYING"
        ]

    def load_data(self):
        print("Loading raw preprocessed data...")
        X_train = np.load("X_train_raw.npy")
        X_test = np.load("X_test_raw.npy")
        y_train = np.load("y_train_raw.npy")
        y_test = np.load("y_test_raw.npy")

        # One-hot encode
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)

        print(f"✔ X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape):
        """LSTM model for (128, 3) input."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(self.num_classes, activation="softmax")
        ])
        return model

    def build_cnn_model(self, input_shape):
        """1D CNN model - often faster and equally accurate."""
        model = Sequential([
            Conv1D(64, kernel_size=5, activation="relu", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(128, kernel_size=5, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(64, kernel_size=3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.4),
            Dense(self.num_classes, activation="softmax")
        ])
        return model

    def train_model(self):
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Input shape: (timesteps=128, features=3)
        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"Input shape: {input_shape}")

        print(f"Building {self.model_type.upper()} model...")
        if self.model_type == "lstm":
            model = self.build_lstm_model(input_shape)
        else:
            model = self.build_cnn_model(input_shape)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint("best_har_raw_model.h5", save_best_only=True, verbose=1)
        ]

        print("\nTraining...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        # Final evaluation
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✔ Test Accuracy: {acc:.4f}")

        model.save("final_har_raw_model.h5")
        print("✔ Saved: best_har_raw_model.h5, final_har_raw_model.h5")

        self.plot_history(history)
        return model

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Val')
        ax2.set_title('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.show()

if __name__ == "__main__":
    # Use CNN for faster inference on mobile
    trainer = HARRawModelTrainer(model_type="cnn")
    trainer.train_model()
