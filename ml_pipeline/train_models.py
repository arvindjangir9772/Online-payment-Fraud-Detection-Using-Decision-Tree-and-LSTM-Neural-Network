# ml_pipeline/train_models.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ------------------------- DATA HANDLING -------------------------

def load_data(file_path="D:\online-payment-fraud-detection\data\processed\processed_data.csv"):
    df = pd.read_csv(file_path)
    logger.info(f"✅ Data loaded successfully with shape {df.shape}")
    return df

def preprocess_data(df):
    df = df.drop_duplicates().dropna()
    df["Amount_Log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] // 3600) % 24

    scaler_amount = RobustScaler()
    scaler_v = StandardScaler()

    df["Amount_Scaled"] = scaler_amount.fit_transform(df["Amount"].values.reshape(-1, 1))
    v_cols = [f"V{i}" for i in range(1, 29)]
    df[v_cols] = scaler_v.fit_transform(df[v_cols])

    X = df.drop(columns=["Time", "Amount", "Class"])
    y = df["Class"]

    return X, y, scaler_amount, scaler_v

# ------------------------- MODEL DEFINITIONS -------------------------

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, MODELS_DIR / "decision_tree.pkl")
        logger.info("🌳 Decision Tree model trained and saved.")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class LSTMModel:
    """Simulated LSTM using MLPClassifier (for lightweight compatibility)"""
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, MODELS_DIR / "lstm_model.pkl")
        logger.info("🧠 LSTM (MLP substitute) trained and saved.")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class HybridModel:
    """Combine Decision Tree + LSTM outputs"""
    def __init__(self, lstm_model, tree_model, lstm_weight=0.6, tree_weight=0.4):
        self.lstm_model = lstm_model
        self.tree_model = tree_model
        self.lstm_weight = lstm_weight
        self.tree_weight = tree_weight

    def predict_proba(self, X):
        p1 = self.lstm_model.predict_proba(X)
        p2 = self.tree_model.predict_proba(X)
        return (self.lstm_weight * p1) + (self.tree_weight * p2)

    def predict(self, X, threshold=0.5):
        p_final = self.predict_proba(X)
        return (p_final >= threshold).astype(int)

    def save(self):
        joblib.dump({
            "lstm_weight": self.lstm_weight,
            "tree_weight": self.tree_weight
        }, MODELS_DIR / "hybrid_weights.pkl")
        logger.info("🔗 Hybrid model configuration saved.")


# ------------------------- PIPELINE -------------------------

def train_pipeline():
    logger.info("🚀 Starting hybrid fraud detection training pipeline...")

    df = load_data()
    X, y, scaler_amount, scaler_v = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Train both models
    tree_model = DecisionTreeModel().train(X_train, y_train)
    lstm_model = LSTMModel().train(X_train, y_train)

    # Hybrid fusion
    hybrid = HybridModel(lstm_model, tree_model)
    hybrid.save()

    # Evaluate
    y_pred_proba = hybrid.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"🏁 Hybrid AUC Score: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    # Save preprocessors
    joblib.dump(scaler_amount, MODELS_DIR / "amount_scaler.pkl")
    joblib.dump(scaler_v, MODELS_DIR / "v_scaler.pkl")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_names.pkl")

    logger.info("✅ All models and preprocessors saved successfully.")
    logger.info("📁 You can now restart the FastAPI app to use the hybrid model.")

    return auc


if __name__ == "__main__":
    train_pipeline()
