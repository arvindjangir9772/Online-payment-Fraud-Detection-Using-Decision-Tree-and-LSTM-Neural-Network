"""
Create and save a proper LSTM model for fraud detection
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_lstm_model(input_shape=(10, 29)):
    """Create a simple LSTM model for fraud detection"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_dummy_data(n_samples=1000):
    """Create dummy training data"""
    np.random.seed(42)
    
    # Create dummy features (V1-V28 + Amount)
    n_features = 29
    sequence_length = 10
    
    # Generate random sequences
    X = np.random.randn(n_samples, sequence_length, n_features)
    
    # Create binary labels (0 or 1)
    y = np.random.randint(0, 2, n_samples)
    
    return X, y

def main():
    print("Creating LSTM model for fraud detection...")
    
    # Create model
    model = create_lstm_model()
    print(f"Model created with {model.count_params()} parameters")
    
    # Create dummy data
    X, y = create_dummy_data()
    print(f"Created dummy data: {X.shape}, {y.shape}")
    
    # Train the model briefly
    print("Training model...")
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save the model in HDF5 format
    model_path = "models/lstm_model.h5"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Test loading
    from tensorflow.keras.models import load_model
    loaded_model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Test prediction
    test_prediction = loaded_model.predict(X[:1])
    print(f"Test prediction: {test_prediction[0][0]:.4f}")
    
    print("LSTM model creation completed successfully!")

if __name__ == "__main__":
    main()
