"""
Debug LSTM model loading and prediction
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.model_manager import model_manager
import numpy as np
import pandas as pd

def debug_lstm():
    print("Debugging LSTM Model...")
    print("=" * 40)
    
    # Check if LSTM model is loaded
    if 'lstm' in model_manager.models:
        lstm_model = model_manager.models['lstm']
        print(f"LSTM model type: {type(lstm_model)}")
        print(f"LSTM model: {lstm_model}")
    else:
        print("No LSTM model found!")
        return
    
    # Create test data
    test_df = pd.DataFrame({
        'Amount': [25.50],
        'V1': [0.0], 'V2': [0.0], 'V3': [0.0], 'V4': [0.0], 'V5': [0.0],
        'V6': [0.0], 'V7': [0.0], 'V8': [0.0], 'V9': [0.0], 'V10': [0.0],
        'V11': [0.0], 'V12': [0.0], 'V13': [0.0], 'V14': [0.0], 'V15': [0.0],
        'V16': [0.0], 'V17': [0.0], 'V18': [0.0], 'V19': [0.0], 'V20': [0.0],
        'V21': [0.0], 'V22': [0.0], 'V23': [0.0], 'V24': [0.0], 'V25': [0.0],
        'V26': [0.0], 'V27': [0.0], 'V28': [0.0]
    })
    
    # Preprocess data
    X_seq, X_flat, feature_list = model_manager.preprocess_input(test_df)
    print(f"X_seq shape: {X_seq.shape}")
    print(f"X_flat shape: {X_flat.shape}")
    
    # Test LSTM prediction directly
    try:
        if hasattr(lstm_model, 'predict_proba'):
            probs = lstm_model.predict_proba(X_flat)
            print(f"LSTM probabilities: {probs}")
            print(f"Fraud probability: {probs[0][1]:.4f}")
        else:
            pred = lstm_model.predict(X_flat)
            print(f"LSTM prediction: {pred}")
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the hybrid model's LSTM method
    print("\nTesting Hybrid Model LSTM Method:")
    print("-" * 40)
    
    try:
        lstm_probs = model_manager.hybrid._safe_predict_lstm(X_seq)
        print(f"Hybrid LSTM probabilities: {lstm_probs}")
    except Exception as e:
        print(f"Hybrid LSTM error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lstm()

