"""
Create a better LSTM model that matches the training data format
"""
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os

def create_better_lstm_model():
    """Create a neural network that better approximates LSTM behavior for fraud detection"""
    # Create a multi-layer perceptron with more sophisticated architecture
    model = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50, 25),  # Four hidden layers
        activation='relu',
        solver='adam',
        alpha=0.0001,  # Lower L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,  # More iterations
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    return model

def create_realistic_training_data(n_samples=2000):
    """Create more realistic training data that matches fraud patterns"""
    np.random.seed(42)
    
    # Create 40 features (matching the expected input)
    n_features = 40
    
    # Generate features with different patterns for fraud vs legitimate
    X = np.random.randn(n_samples, n_features)
    
    # Create more realistic fraud patterns
    # Fraud transactions tend to have:
    # - Higher amounts (last feature)
    # - More extreme V values
    # - Specific patterns in certain features
    
    # Add amount feature (last column)
    amounts = np.random.exponential(scale=100, size=n_samples)  # Exponential distribution for amounts
    X[:, -1] = amounts
    
    # Create fraud labels based on realistic patterns
    fraud_prob = np.zeros(n_samples)
    
    # Higher amounts increase fraud probability
    fraud_prob += (amounts > 1000) * 0.3
    
    # Extreme V values increase fraud probability
    extreme_v_count = np.sum(np.abs(X[:, :28]) > 2, axis=1)  # Count extreme V1-V28 values
    fraud_prob += extreme_v_count * 0.05
    
    # Specific patterns
    fraud_prob += (X[:, 0] < -2) * 0.2  # V1 very negative
    fraud_prob += (X[:, 1] > 2) * 0.2   # V2 very positive
    fraud_prob += (X[:, 2] < -2) * 0.2  # V3 very negative
    
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
    print("Creating better LSTM model for fraud detection...")
    print("=" * 60)
    
    # Create model
    model = create_better_lstm_model()
    print(f"Model created: {model}")
    
    # Create realistic training data
    X, y = create_realistic_training_data()
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
    
    # Test loading
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Test prediction
    test_prediction = loaded_model.predict_proba(X[:1])
    print(f"Test prediction probability: {test_prediction[0][1]:.4f}")
    
    print("=" * 60)
    print("Better LSTM model creation completed successfully!")

if __name__ == "__main__":
    main()

