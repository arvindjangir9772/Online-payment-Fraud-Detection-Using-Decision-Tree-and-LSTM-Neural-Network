"""
Test script to verify both LSTM and Decision Tree models are working with correct preprocessing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.model_manager import model_manager
import numpy as np
import pandas as pd

def test_models():
    print("Testing Hybrid Fraud Detection Models (Fixed)...")
    print("=" * 60)
    
    # Test data
    test_data = {
        'Amount': [25.50, 5000.0, 100.0],
        'V1': [0.0, -1.36, 0.5],
        'V2': [0.0, -0.07, -0.2],
        'V3': [0.0, 2.54, 0.3],
        'V4': [0.0, 1.38, -0.1],
        'V5': [0.0, -0.34, 0.4],
        'V6': [0.0, 0.46, -0.3],
        'V7': [0.0, 0.24, 0.2],
        'V8': [0.0, 0.10, 0.1],
        'V9': [0.0, 0.36, -0.2],
        'V10': [0.0, 0.09, 0.3],
        'V11': [0.0, -0.55, -0.1],
        'V12': [0.0, -0.62, 0.2],
        'V13': [0.0, -0.99, -0.3],
        'V14': [0.0, -0.31, 0.1],
        'V15': [0.0, 1.47, 0.4],
        'V16': [0.0, -0.47, -0.2],
        'V17': [0.0, 0.21, 0.3],
        'V18': [0.0, 0.03, -0.1],
        'V19': [0.0, 0.40, 0.2],
        'V20': [0.0, 0.25, -0.3],
        'V21': [0.0, -0.02, 0.1],
        'V22': [0.0, 0.28, 0.4],
        'V23': [0.0, -0.11, -0.2],
        'V24': [0.0, 0.07, 0.3],
        'V25': [0.0, 0.13, -0.1],
        'V26': [0.0, -0.19, 0.2],
        'V27': [0.0, 0.13, -0.3],
        'V28': [0.0, -0.02, 0.1]
    }
    
    df = pd.DataFrame(test_data)
    
    print("Test Data:")
    print(f"Amounts: {df['Amount'].tolist()}")
    print()
    
    # Test preprocessing first
    print("Testing Preprocessing:")
    print("-" * 30)
    
    try:
        X_seq, X_flat, feature_list = model_manager.preprocess_input(df)
        print(f"[OK] X_seq shape: {X_seq.shape if X_seq is not None else 'None'}")
        print(f"[OK] X_flat shape: {X_flat.shape if X_flat is not None else 'None'}")
        print(f"[OK] Feature list length: {len(feature_list) if feature_list else 'None'}")
        print()
    except Exception as e:
        print(f"[ERROR] Preprocessing error: {e}")
        return
    
    # Test individual models with correct preprocessed data
    print("Testing Individual Models:")
    print("-" * 30)
    
    # Test Decision Tree
    try:
        dt_probs = model_manager.hybrid._safe_predict_dt(X_flat)
        print(f"[OK] Decision Tree probabilities: {dt_probs}")
    except Exception as e:
        print(f"[ERROR] Decision Tree error: {e}")
    
    # Test LSTM
    try:
        lstm_probs = model_manager.hybrid._safe_predict_lstm(X_seq)
        print(f"[OK] LSTM probabilities: {lstm_probs}")
    except Exception as e:
        print(f"[ERROR] LSTM error: {e}")
    
    print()
    
    # Test hybrid prediction
    print("Testing Hybrid Predictions:")
    print("-" * 30)
    
    try:
        results = model_manager.hybrid_predict(df, with_explanation=True)
        
        for i, result in enumerate(results):
            print(f"Transaction {i+1} (Amount: ${df['Amount'].iloc[i]}):")
            print(f"  Prediction: {'FRAUD' if result['prediction'] == 1 else 'LEGITIMATE'}")
            print(f"  Probability: {result['probability']:.3f}")
            print(f"  LSTM Prob: {result.get('lstm_probability', 'N/A'):.3f}")
            print(f"  DT Prob: {result.get('dt_probability', 'N/A'):.3f}")
            print(f"  Alpha Used: {result.get('alpha_used', 'N/A'):.3f}")
            print(f"  Dominant Model: {result.get('dominant_model', 'N/A')}")
            print()
    except Exception as e:
        print(f"[ERROR] Hybrid prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("Model Testing Complete!")

if __name__ == "__main__":
    test_models()
