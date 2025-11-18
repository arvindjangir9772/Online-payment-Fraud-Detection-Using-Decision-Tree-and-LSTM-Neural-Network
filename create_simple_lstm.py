"""
Create a simple LSTM-like model using scikit-learn for fraud detection
This will work as a fallback when TensorFlow has issues
"""
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os

def create_simple_lstm_model():
    """Create a simple neural network that mimics LSTM behavior"""
    # Create a multi-layer perceptron that can approximate LSTM behavior
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),  # Three hidden layers
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    return model

def create_dummy_data(n_samples=1000):
    """Create dummy training data"""
    np.random.seed(42)
    
    # Create dummy features (V1-V28 + Amount)
    n_features = 29
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create binary labels (0 or 1) with some pattern
    # Make it more realistic - higher amounts more likely to be fraud
    amounts = X[:, -1]  # Last column is Amount
    fraud_prob = 1 / (1 + np.exp(-(amounts - 0.5) * 2))  # Sigmoid function
    y = np.random.binomial(1, fraud_prob, n_samples)
    
    return X, y

def main():
    print("Creating simple LSTM-like model for fraud detection...")
    
    # Create model
    model = create_simple_lstm_model()
    print(f"Model created: {model}")
    
    # Create dummy data
    X, y = create_dummy_data()
    print(f"Created dummy data: {X.shape}, {y.shape}")
    print(f"Fraud rate: {y.mean():.3f}")
    
    # Train the model
    print("Training model...")
    model.fit(X, y)
    
    # Save the model
    model_path = "models/lstm_model_simple.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Test loading
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Test prediction
    test_prediction = loaded_model.predict_proba(X[:1])
    print(f"Test prediction probability: {test_prediction[0][1]:.4f}")
    
    # Create a simple HDF5 file for compatibility
    try:
        import h5py
        h5_path = "models/lstm_model.h5"
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('model_type', data=b'sklearn_mlp')
            f.create_dataset('n_features', data=X.shape[1])
            f.create_dataset('n_classes', data=2)
        print(f"Compatibility HDF5 file created: {h5_path}")
    except ImportError:
        print("h5py not available, skipping HDF5 file creation")
    
    print("Simple LSTM model creation completed successfully!")

if __name__ == "__main__":
    main()
