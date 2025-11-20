# -*- coding: utf-8 -*-
"""
Human Activity Recognition - Model Architectures
LSTM, GRU, 1D-CNN, and Ensemble Models
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Conv1D, 
                                     MaxPooling1D, Flatten, BatchNormalization,
                                     Bidirectional, GlobalAveragePooling1D,
                                     Concatenate, Input, Add, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)


class HARModels:
    """Collection of HAR model architectures"""
    
    def __init__(self, input_shape, n_classes=6):
        self.input_shape = input_shape  # (window_size, n_features)
        self.n_classes = n_classes
    
    def build_lstm_model(self, lstm_units=[128, 64], dropout=0.3):
        """
        Build stacked LSTM model
        """
        model = Sequential([
            LSTM(lstm_units[0], return_sequences=True, 
                 input_shape=self.input_shape, name='lstm_1'),
            Dropout(dropout),
            BatchNormalization(),
            
            LSTM(lstm_units[1], return_sequences=False, name='lstm_2'),
            Dropout(dropout),
            BatchNormalization(),
            
            Dense(64, activation='relu', name='dense_1'),
            Dropout(dropout),
            
            Dense(self.n_classes, activation='softmax', name='output')
        ], name='LSTM_Model')
        
        return model
    
    def build_gru_model(self, gru_units=[128, 64], dropout=0.3):
        """
        Build stacked GRU model (faster than LSTM)
        """
        model = Sequential([
            GRU(gru_units[0], return_sequences=True, 
                input_shape=self.input_shape, name='gru_1'),
            Dropout(dropout),
            BatchNormalization(),
            
            GRU(gru_units[1], return_sequences=False, name='gru_2'),
            Dropout(dropout),
            BatchNormalization(),
            
            Dense(64, activation='relu', name='dense_1'),
            Dropout(dropout),
            
            Dense(self.n_classes, activation='softmax', name='output')
        ], name='GRU_Model')
        
        return model
    
    def build_bilstm_model(self, lstm_units=[128, 64], dropout=0.3):
        """
        Build Bidirectional LSTM model for better context
        """
        model = Sequential([
            Bidirectional(LSTM(lstm_units[0], return_sequences=True),
                         input_shape=self.input_shape, name='bilstm_1'),
            Dropout(dropout),
            BatchNormalization(),
            
            Bidirectional(LSTM(lstm_units[1], return_sequences=False),
                         name='bilstm_2'),
            Dropout(dropout),
            BatchNormalization(),
            
            Dense(64, activation='relu', name='dense_1'),
            Dropout(dropout),
            
            Dense(self.n_classes, activation='softmax', name='output')
        ], name='BiLSTM_Model')
        
        return model
    
    def build_cnn1d_model(self, filters=[64, 128, 256], kernel_size=3, dropout=0.3):
        """
        Build 1D-CNN model for temporal pattern recognition
        """
        model = Sequential([
            # Block 1
            Conv1D(filters[0], kernel_size, activation='relu', 
                   input_shape=self.input_shape, name='conv1d_1'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, name='pool_1'),
            Dropout(dropout),
            
            # Block 2
            Conv1D(filters[1], kernel_size, activation='relu', name='conv1d_2'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, name='pool_2'),
            Dropout(dropout),
            
            # Block 3
            Conv1D(filters[2], kernel_size, activation='relu', name='conv1d_3'),
            BatchNormalization(),
            GlobalAveragePooling1D(name='global_pool'),
            
            # Dense layers
            Dense(128, activation='relu', name='dense_1'),
            Dropout(dropout),
            Dense(64, activation='relu', name='dense_2'),
            Dropout(dropout),
            
            Dense(self.n_classes, activation='softmax', name='output')
        ], name='CNN1D_Model')
        
        return model
    
    def build_cnn_lstm_hybrid(self, cnn_filters=[64, 128], lstm_units=64, dropout=0.3):
        """
        Build hybrid CNN-LSTM model
        CNN for feature extraction, LSTM for temporal learning
        """
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(cnn_filters[0], 3, activation='relu', 
                   input_shape=self.input_shape, name='conv1d_1'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, name='pool_1'),
            Dropout(dropout),
            
            Conv1D(cnn_filters[1], 3, activation='relu', name='conv1d_2'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, name='pool_2'),
            Dropout(dropout),
            
            # LSTM layers for temporal patterns
            LSTM(lstm_units, return_sequences=False, name='lstm_1'),
            Dropout(dropout),
            
            # Dense layers
            Dense(64, activation='relu', name='dense_1'),
            Dropout(dropout),
            
            Dense(self.n_classes, activation='softmax', name='output')
        ], name='CNN_LSTM_Hybrid')
        
        return model
    
    def build_resnet1d_model(self, filters=[64, 128, 256], dropout=0.3):
        """
        Build ResNet-inspired 1D-CNN with residual connections
        """
        inputs = Input(shape=self.input_shape)
        
        # Initial convolution
        x = Conv1D(filters[0], 7, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Residual block 1
        shortcut = x
        x = Conv1D(filters[1], 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters[1], 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Match dimensions for shortcut
        shortcut = Conv1D(filters[1], 1, padding='same')(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(dropout)(x)
        
        # Residual block 2
        shortcut = x
        x = Conv1D(filters[2], 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters[2], 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        shortcut = Conv1D(filters[2], 1, padding='same')(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout)(x)
        
        # Classification head
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout)(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ResNet1D_Model')
        return model
    
    def build_ensemble_model(self, base_models):
        """
        Build ensemble model by averaging predictions from multiple models
        """
        inputs = Input(shape=self.input_shape)
        
        # Get predictions from each model
        predictions = []
        for i, model in enumerate(base_models):
            # Make base models non-trainable in ensemble
            for layer in model.layers:
                layer.trainable = False
                layer._name = f'{layer.name}_model{i}'
            
            pred = model(inputs)
            predictions.append(pred)
        
        # Average predictions
        if len(predictions) > 1:
            avg_output = tf.keras.layers.Average()(predictions)
        else:
            avg_output = predictions[0]
        
        ensemble = Model(inputs=inputs, outputs=avg_output, name='Ensemble_Model')
        return ensemble


class ModelTrainer:
    """Handle model training with callbacks and history tracking"""
    
    def __init__(self, model, model_name='har_model'):
        self.model = model
        self.model_name = model_name
        self.history = None
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with optimizer and loss"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"\n{self.model_name} compiled successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
    
    def get_callbacks(self, patience=10):
        """Define training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=64):
        """Train the model"""
        print(f"\nTraining {self.model_name}...")
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print(f"\n✓ Training complete for {self.model_name}!")
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title(f'{self.model_name} - Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_title(f'{self.model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved as '{self.model_name}_training_history.png'")
        plt.show()
    
    def save_model(self, filepath=None):
        """Save trained model"""
        if filepath is None:
            filepath = f'{self.model_name}_final.h5'
        self.model.save(filepath)
        print(f"Model saved as '{filepath}'")


# Main execution example
if __name__ == "__main__":
    # Load processed data
    data = np.load('har_processed_data.npz', allow_pickle=True)
    X_train = data['train_X']
    y_train = data['train_y']
    X_test = data['test_X']
    y_test = data['test_y']
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=6)
    y_test_cat = to_categorical(y_test, num_classes=6)
    
    print(f"Data loaded successfully!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Initialize model builder
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_builder = HARModels(input_shape=input_shape, n_classes=6)
    
    # Build and visualize different models
    print("\n" + "="*60)
    print("Building LSTM Model...")
    print("="*60)
    lstm_model = model_builder.build_lstm_model()
    lstm_model.summary()
    
    print("\n" + "="*60)
    print("Building 1D-CNN Model...")
    print("="*60)
    cnn_model = model_builder.build_cnn1d_model()
    cnn_model.summary()
    
    print("\n" + "="*60)
    print("Building CNN-LSTM Hybrid Model...")
    print("="*60)
    hybrid_model = model_builder.build_cnn_lstm_hybrid()
    hybrid_model.summary()
    
    # Train LSTM model (example)
    print("\n" + "="*60)
    print("Training LSTM Model (Example)...")
    print("="*60)
    
    trainer = ModelTrainer(lstm_model, model_name='LSTM_HAR')
    trainer.compile_model(learning_rate=0.001)
    
    # Train for fewer epochs as example
    history = trainer.train(
        X_train, y_train_cat,
        X_test, y_test_cat,
        epochs=50,
        batch_size=64
    )
    
    # Plot results
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model()
    
    print("\n✓ Model building and training example complete!")