# re_save_models.py
"""
Re-save existing ML models using the current scikit-learn version (1.7.2)
to fix dtype incompatibility issues.
"""

import joblib
import os
from keras.models import load_model

# ✅ Path to your existing model files
base_path = os.path.join(os.path.dirname(__file__), "models")

# Create models directory if not exists
os.makedirs(base_path, exist_ok=True)

# --- Load your existing models ---
print("🔄 Loading old models...")

try:
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    dt_model = joblib.load(os.path.join(base_path, "decision_tree.pkl"))
    features = joblib.load(os.path.join(base_path, "feature_names.pkl"))
    lstm_model = load_model(os.path.join(base_path, "lstm_model.h5"))
except Exception as e:
    print(f"❌ Failed to load old models: {e}")
    exit(1)

# --- Re-save them using current sklearn (1.7.2) ---
print("💾 Re-saving models with scikit-learn 1.7.2...")

try:
    joblib.dump(scaler, os.path.join(base_path, "scaler.pkl"))
    joblib.dump(dt_model, os.path.join(base_path, "decision_tree.pkl"))
    joblib.dump(features, os.path.join(base_path, "feature_names.pkl"))
    lstm_model.save(os.path.join(base_path, "lstm_model.h5"))
    print("✅ All models successfully re-saved using scikit-learn 1.7.2.")
except Exception as e:
    print(f"❌ Error while saving: {e}")
