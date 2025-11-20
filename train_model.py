# -*- coding: utf-8 -*-
"""
HAR Training Script
Connects preprocessed data → model architectures → training engine
"""

import numpy as np
from tensorflow.keras.utils import to_categorical
from model_architectures import HARModels, ModelTrainer

MODEL_TYPE = "lstm"   # options: "lstm", "gru", "cnn1d", "hybrid", "bilstm"
MODEL_NAME = "HAR_LSTM"

def build_selected_model(model_builder):
    if MODEL_TYPE == "lstm":
        return model_builder.build_lstm_model()
    elif MODEL_TYPE == "gru":
        return model_builder.build_gru_model()
    elif MODEL_TYPE == "bilstm":
        return model_builder.build_bilstm_model()
    elif MODEL_TYPE == "cnn1d":
        return model_builder.build_cnn1d_model()
    elif MODEL_TYPE == "hybrid":
        return model_builder.build_cnn_lstm_hybrid()
    else:
        raise ValueError("Invalid model type selected!")

if __name__ == "__main__":
    print("=== HAR MODEL TRAINING ===")

    # Load processed dataset
    data = np.load("har_processed_data.npz", allow_pickle=True)

    X_train = data["train_X"]
    y_train = to_categorical(data["train_y"], 6)

    X_test = data["test_X"]
    y_test = to_categorical(data["test_y"], 6)

    print("Data Loaded!")
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_builder = HARModels(input_shape=input_shape, n_classes=6)

    model = build_selected_model(model_builder)
    model.summary()

    # Train model
    trainer = ModelTrainer(model, model_name=MODEL_NAME)
    trainer.compile_model(learning_rate=0.001)

    trainer.train(X_train, y_train, X_test, y_test,
                  epochs=60, batch_size=64)

    trainer.plot_training_history()
    trainer.save_model()

    print("\n✓ Training completed successfully!")
