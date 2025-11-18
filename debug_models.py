"""
Debug script to check what's wrong with the models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.model_manager import model_manager
import numpy as np
import pandas as pd
import joblib

def debug_models():
    print("Debugging Models...")
    print("=" * 50)
    
    # Check what models are loaded
    print("Available models:", list(model_manager.models.keys()))
    print("Available scalers:", list(model_manager.scalers.keys()))
    print("Feature names:", model_manager.feature_names[:5], "...")
    print()
    
    # Test Decision Tree directly
    print("Testing Decision Tree directly:")
    print("-" * 30)
    
    if 'decision_tree' in model_manager.models:
        dt_model = model_manager.models['decision_tree']
        print(f"Decision Tree type: {type(dt_model)}")
        print(f"Decision Tree has predict_proba: {hasattr(dt_model, 'predict_proba')}")
        
        # Create simple test data
        test_data = np.array([[0.0] * 29])  # 29 features
        print(f"Test data shape: {test_data.shape}")
        
        try:
            # Test prediction
            if hasattr(dt_model, 'predict_proba'):
                probs = dt_model.predict_proba(test_data)
                print(f"Decision Tree probabilities: {probs}")
            else:
                pred = dt_model.predict(test_data)
                print(f"Decision Tree prediction: {pred}")
        except Exception as e:
            print(f"Decision Tree prediction error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # Test LSTM directly
    print("Testing LSTM directly:")
    print("-" * 30)
    
    if 'lstm' in model_manager.models:
        lstm_model = model_manager.models['lstm']
        print(f"LSTM type: {type(lstm_model)}")
        print(f"LSTM has predict_proba: {hasattr(lstm_model, 'predict_proba')}")
        
        # Create simple test data
        test_data = np.array([[0.0] * 29])  # 29 features
        print(f"Test data shape: {test_data.shape}")
        
        try:
            # Test prediction
            if hasattr(lstm_model, 'predict_proba'):
                probs = lstm_model.predict_proba(test_data)
                print(f"LSTM probabilities: {probs}")
            else:
                pred = lstm_model.predict(test_data)
                print(f"LSTM prediction: {pred}")
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # Test preprocessing
    print("Testing Preprocessing:")
    print("-" * 30)
    
    test_df = pd.DataFrame({
        'Amount': [25.50],
        'V1': [0.0], 'V2': [0.0], 'V3': [0.0], 'V4': [0.0], 'V5': [0.0],
        'V6': [0.0], 'V7': [0.0], 'V8': [0.0], 'V9': [0.0], 'V10': [0.0],
        'V11': [0.0], 'V12': [0.0], 'V13': [0.0], 'V14': [0.0], 'V15': [0.0],
        'V16': [0.0], 'V17': [0.0], 'V18': [0.0], 'V19': [0.0], 'V20': [0.0],
        'V21': [0.0], 'V22': [0.0], 'V23': [0.0], 'V24': [0.0], 'V25': [0.0],
        'V26': [0.0], 'V27': [0.0], 'V28': [0.0]
    })
    
    try:
        X_seq, X_flat, feature_list = model_manager.preprocess_input(test_df)
        print(f"X_seq shape: {X_seq.shape if X_seq is not None else 'None'}")
        print(f"X_flat shape: {X_flat.shape if X_flat is not None else 'None'}")
        print(f"Feature list length: {len(feature_list) if feature_list else 'None'}")
        print(f"X_flat sample: {X_flat[0][:5] if X_flat is not None else 'None'}")
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_models()

