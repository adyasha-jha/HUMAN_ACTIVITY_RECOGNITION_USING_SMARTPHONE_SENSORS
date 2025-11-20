import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class HARModelTrainer:
    def __init__(self, model_type="lstm"):
        self.model_type = model_type.lower()   # "lstm" or "cnn"
        self.num_classes = 6  # UCI HAR has 6 activities

    # ------------------------------------------------------------
    def load_data(self):
        print("Loading preprocessed numpy files...")

        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
        y_train = np.load("y_train.npy") - 1    # UCI labels start from 1
        y_test = np.load("y_test.npy") - 1

        # One-hot encode labels
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)

        print(f"✔ Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------
    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),

            LSTM(32),
            Dropout(0.3),

            Dense(32, activation="relu"),
            Dense(self.num_classes, activation="softmax")
        ])
        return model

    # ------------------------------------------------------------
    def build_cnn_model(self, input_shape):
        model = Sequential([
            Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),

            Conv1D(32, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),

            Flatten(),
            Dense(64, activation="relu"),
            Dense(self.num_classes, activation="softmax")
        ])
        return model

    # ------------------------------------------------------------
    def train_model(self):
        X_train, X_test, y_train, y_test = self.load_data()
        input_shape = X_train.shape[1], X_train.shape[2]

        print("Building model...")
        if self.model_type == "lstm":
            model = self.build_lstm_model(input_shape)
        else:
            model = self.build_cnn_model(input_shape)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print(model.summary())

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint("best_har_model.h5", save_best_only=True)
        ]

        print("Training started...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=25,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        print("✔ Training complete.")
        model.save("final_har_model.h5")
        print("✔ Model saved as final_har_model.h5")

        self.plot_history(history)

    # ------------------------------------------------------------
    def plot_history(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

# ------------------------------------------------------------
if __name__ == "__main__":
    trainer = HARModelTrainer(model_type="lstm")    # or "cnn"
    trainer.train_model()
