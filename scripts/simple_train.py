# simple_train.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_train():
    """Simple training script without complex imports"""
    logger.info("🚀 Starting simple training...")
    
    # Check if data exists
    data_path = Path("data/raw/creditcard.csv")
    if not data_path.exists():
        logger.error(f"❌ Dataset not found at {data_path}")
        logger.info("💡 Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.info("📁 And place it in: data/raw/creditcard.csv")
        return
    
    # Load data
    logger.info("📥 Loading data...")
    df = pd.read_csv(data_path)
    logger.info(f"✅ Data loaded: {df.shape}")
    
    # Preprocess
    logger.info("⚙️ Preprocessing data...")
    df_processed = df.copy()
    
    # Handle Time feature
    df_processed['Time_Hour'] = (df_processed['Time'] // 3600) % 24
    
    # Scale Amount
    scaler = RobustScaler()
    df_processed['Amount_Scaled'] = scaler.fit_transform(df_processed['Amount'].values.reshape(-1, 1))
    
    # Drop original columns
    df_processed = df_processed.drop(['Time', 'Amount'], axis=1)
    
    # Prepare features and target
    X = df_processed.drop('Class', axis=1)
    y = df_processed['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle imbalance
    logger.info("⚖️ Handling class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train model
    logger.info("🤖 Training Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"✅ Model trained! AUC: {auc_score:.4f}")
    
    # Save model and preprocessor
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, models_dir / "decision_tree.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(X.columns.tolist(), models_dir / "feature_names.pkl")
    
    logger.info("💾 Models saved to models/ directory")
    logger.info("🎉 Training completed! Restart your web app.")

if __name__ == "__main__":
    simple_train()