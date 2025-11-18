"""
Debug script to check if the new LSTM model is being used correctly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.model_manager import model_manager
import numpy as np
import pandas as pd

def debug_new_model():
    print("Debugging New LSTM Model Usage...")
    print("=" * 50)
    
    # Check what LSTM model is loaded
    if 'lstm' in model_manager.models:
        lstm_model = model_manager.models['lstm']
        print(f"LSTM model type: {type(lstm_model)}")
        print(f"LSTM model: {lstm_model}")
        print(f"LSTM model n_features_in_: {getattr(lstm_model, 'n_features_in_', 'Not available')}")
    else:
        print("No LSTM model found!")
        return
    
    # Create test data with extreme values to trigger fraud detection
    test_df = pd.DataFrame({
        'Amount': [10000.0],  # High amount
        'V1': [-3.0], 'V2': [3.0], 'V3': [-3.0], 'V4': [3.0], 'V5': [-3.0],
        'V6': [3.0], 'V7': [-3.0], 'V8': [3.0], 'V9': [-3.0], 'V10': [3.0],
        'V11': [-3.0], 'V12': [3.0], 'V13': [-3.0], 'V14': [3.0], 'V15': [-3.0],
        'V16': [3.0], 'V17': [-3.0], 'V18': [3.0], 'V19': [-3.0], 'V20': [3.0],
        'V21': [-3.0], 'V22': [3.0], 'V23': [-3.0], 'V24': [3.0], 'V25': [-3.0],
        'V26': [3.0], 'V27': [-3.0], 'V28': [3.0]
    })
    
    print(f"\nTest data Amount: {test_df['Amount'].iloc[0]}")
    print(f"Test data V1-V5: {test_df[['V1', 'V2', 'V3', 'V4', 'V5']].iloc[0].tolist()}")
    
    # Preprocess data
    X_seq, X_flat, feature_list = model_manager.preprocess_input(test_df)
    print(f"\nPreprocessed data shape: {X_flat.shape}")
    print(f"Amount after preprocessing: {X_flat[0][-1]:.2f}")
    
    # Test LSTM prediction directly
    print("\nTesting LSTM Model Directly:")
    print("-" * 30)
    
    try:
        if hasattr(lstm_model, 'predict_proba'):
            probs = lstm_model.predict_proba(X_flat)
            print(f"LSTM probabilities: {probs}")
            print(f"Fraud probability: {probs[0][1]:.4f}")
            
            # Check if this is different from fallback
            if probs[0][1] != 0.5:
                print("✓ LSTM model is working with real predictions!")
            else:
                print("⚠ LSTM model returning fallback value (0.5)")
        else:
            pred = lstm_model.predict(X_flat)
            print(f"LSTM prediction: {pred}")
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the hybrid model's LSTM method
    print("\nTesting Hybrid Model LSTM Method:")
    print("-" * 30)
    
    try:
        lstm_probs = model_manager.hybrid._safe_predict_lstm(X_seq)
        print(f"Hybrid LSTM probabilities: {lstm_probs}")
        
        if lstm_probs[0] != 0.5:
            print("✓ Hybrid LSTM method working with real predictions!")
        else:
            print("⚠ Hybrid LSTM method returning fallback value (0.5)")
    except Exception as e:
        print(f"Hybrid LSTM error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with different amounts to see if the model responds
    print("\nTesting with Different Amounts:")
    print("-" * 30)
    
    amounts = [25.0, 1000.0, 5000.0, 10000.0]
    for amount in amounts:
        test_df_amount = test_df.copy()
        test_df_amount['Amount'] = amount
        
        X_seq, X_flat, _ = model_manager.preprocess_input(test_df_amount)
        
        try:
            probs = lstm_model.predict_proba(X_flat)
            fraud_prob = probs[0][1]
            print(f"Amount: ${amount:6.0f} -> Fraud Probability: {fraud_prob:.4f}")
        except Exception as e:
            print(f"Amount: ${amount:6.0f} -> Error: {e}")

if __name__ == "__main__":
    debug_new_model()

