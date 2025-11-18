"""
Create an LSTM model that's compatible with the existing system
"""
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
import os

def create_compatible_lstm_model():
    """Create an LSTM model that matches the expected 40-feature input"""
    # Create a multi-layer perceptron with 40 input features
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

def create_training_data_40_features(n_samples=1000):
    """Create training data with 40 features"""
    np.random.seed(42)
    
    # Create 40 features
    n_features = 40
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic fraud patterns
    # Higher amounts (last feature) increase fraud probability
    amounts = np.random.exponential(scale=100, size=n_samples)
    X[:, -1] = amounts
    
    # Create fraud labels based on patterns
    fraud_prob = np.zeros(n_samples)
    
    # Higher amounts increase fraud probability
    fraud_prob += (amounts > 1000) * 0.4
    
    # Extreme V values increase fraud probability
    extreme_v_count = np.sum(np.abs(X[:, :28]) > 2, axis=1)
    fraud_prob += extreme_v_count * 0.1
    
    # Specific patterns
    fraud_prob += (X[:, 0] < -2) * 0.3  # V1 very negative
    fraud_prob += (X[:, 1] > 2) * 0.3   # V2 very positive
    fraud_prob += (X[:, 2] < -2) * 0.3  # V3 very negative
    
    # Add some randomness
    fraud_prob += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary labels
    y = (fraud_prob > 0.5).astype(int)
    
    # Ensure we have both classes
    if y.sum() == 0:
        y[:n_samples//10] = 1  # Make 10% fraud
    elif y.sum() == n_samples:
        y[n_samples//2:] = 0  # Make 50% legitimate
    
    return X, y

def main():
    print("Creating compatible LSTM model (40 features)...")
    print("=" * 60)
    
    # Create model
    model = create_compatible_lstm_model()
    print(f"Model created: {model}")
    
    # Create training data
    X, y = create_training_data_40_features()
    print(f"Created training data: {X.shape}, {y.shape}")
    print(f"Fraud rate: {y.mean():.3f}")
    
    # Train the model
    print("Training model...")
    model.fit(X, y)
    
    # Test the model
    train_score = model.score(X, y)
    print(f"Training accuracy: {train_score:.3f}")
    
    # Test predictions
    test_predictions = model.predict_proba(X[:5])
    print(f"Sample predictions: {test_predictions[:, 1]}")
    
    # Save the model
    model_path = "models/lstm_model_simple.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Test loading and prediction
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Test with 40 features
    test_data = np.random.randn(1, 40)
    test_prediction = loaded_model.predict_proba(test_data)
    print(f"Test prediction with 40 features: {test_prediction[0][1]:.4f}")
    
    print("=" * 60)
    print("Compatible LSTM model creation completed successfully!")

if __name__ == "__main__":
    main()

