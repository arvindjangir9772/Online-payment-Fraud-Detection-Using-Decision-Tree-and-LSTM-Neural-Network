# app/models/model_manager.py
"""
ModelManager: loads models + scalers + feature_names and provides preprocessing + hybrid predict.
Designed to work with your repo layout where 'models/' is at project root.
"""
import os
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Any, Tuple, List, Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# local import for hybrid class; avoid circular import issues
from app.models.hybrid_model import HybridFraudNet

class ModelManager:
    def __init__(self, models_dir: Optional[str] = None):
        # default to a top-level 'models' directory
        if models_dir:
            self.models_dir = Path(models_dir).resolve()
        else:
            self.models_dir = Path(os.getcwd()).resolve() / "models"

        self.models = {}
        self.scalers = {}
        self.feature_names: Optional[List[str]] = None
        self.hybrid: Optional[HybridFraudNet] = None
        self._load_models()

    def _try_load_keras(self, path: Path):
        try:
            import tensorflow as tf
            # tf.keras.models.load_model works for directories or .h5 files
            model = tf.keras.models.load_model(str(path))
            logger.info(f"Keras model loaded from {path}")
            return model
        except Exception as e:
            logger.info(f"Keras load failed for {path}: {e}")
            return None

    def _safe_joblib_load(self, path: Path):
        try:
            return joblib.load(str(path))
        except Exception as e:
            logger.error(f"joblib load failed for {path}: {e}")
            return None

    def _load_models(self):
        logger.info(f"Loading models from: {self.models_dir}")
        try:
            # paths
            dt_path = self.models_dir / "decision_tree.pkl"
            lstm_path_candidates = [self.models_dir / "lstm_model.h5", self.models_dir / "lstm_model.pkl", self.models_dir / "lstm_model"]
            scaler_path = self.models_dir / "scaler.pkl"
            amount_scaler_path = self.models_dir / "amount_scaler.pkl"
            feature_names_path = self.models_dir / "feature_names.pkl"

            # feature names
            if feature_names_path.exists():
                try:
                    self.feature_names = joblib.load(str(feature_names_path))
                    logger.info("feature_names loaded")
                except Exception as e:
                    logger.error(f"Failed to load feature_names: {e}")

            # scalers
            if scaler_path.exists():
                s = self._safe_joblib_load(scaler_path)
                if s is not None:
                    self.scalers['scaler'] = s
            if amount_scaler_path.exists():
                s = self._safe_joblib_load(amount_scaler_path)
                if s is not None:
                    self.scalers['amount_scaler'] = s

            # decision tree
            if dt_path.exists():
                self.models['decision_tree'] = self._safe_joblib_load(dt_path)

            # lstm: try simple LSTM model first, then keras, then other joblib models
            lstm_model = None
            
            # First try the simple LSTM model (compatible with 40 features)
            simple_lstm_path = self.models_dir / "lstm_model_simple.pkl"
            if simple_lstm_path.exists():
                lstm_model = self._safe_joblib_load(simple_lstm_path)
                if lstm_model is not None:
                    self.models['lstm'] = lstm_model
                    logger.info("Simple LSTM model loaded (40 features)")
            
            # If simple LSTM not found, try keras and other models
            if lstm_model is None:
                for p in lstm_path_candidates:
                    if p.exists():
                        lstm_model = self._try_load_keras(p) or self._safe_joblib_load(p)
                        if lstm_model is not None:
                            self.models['lstm'] = lstm_model
                            logger.info(f"LSTM model loaded from {p}")
                            break

            # initialize hybrid if both present
            if 'decision_tree' in self.models and self.models['decision_tree'] is not None:
                try:
                    # Use LSTM if available, otherwise None (Decision Tree only mode)
                    lstm_model = self.models.get('lstm') if 'lstm' in self.models else None
                    self.hybrid = HybridFraudNet(
                        lstm_model, 
                        self.models['decision_tree'], 
                        alpha=0.6,
                        feature_names=self.feature_names
                    )
                    logger.info("Hybrid model created with SHAP explainability")
                except Exception as e:
                    logger.error(f"Failed to create hybrid: {e}")

        except Exception as e:
            logger.error(f"Error while loading models: {e}")

    def _ensure_default_feature_names(self) -> List[str]:
        # fallback default for the Kaggle-like dataset
        v_cols = [f"V{i}" for i in range(1, 29)]
        defaults = v_cols + ['V_Sum', 'V_Mean', 'V_Std', 'V_Max', 'V_Min', 'Amount']
        return defaults

    def preprocess_input(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], np.ndarray, List[str]]:
        """
        Prepare X_seq (for LSTM) and X_flat (for DT/hybrid) from raw df.
        - ensures feature names exist
        - builds aggregates if missing (V_Sum etc.)
        - scales Amount if amount_scaler exists
        - returns:
            X_seq: (n, seq_len, n_features) or None (if no lstm loaded)
            X_flat: (n, n_features_flat)
            feature_list: list of columns used for X_flat
        """
        if self.feature_names is None:
            self.feature_names = self._ensure_default_feature_names()

        df_proc = df.copy()

        # ensure V1..V28 exist (fill with 0 if not)
        for i in range(1, 29):
            col = f"V{i}"
            if col not in df_proc.columns:
                df_proc[col] = 0.0

        # create aggregates if missing
        v_cols = [f"V{i}" for i in range(1, 29)]
        if not all(c in df_proc.columns for c in ['V_Sum', 'V_Mean', 'V_Std', 'V_Max', 'V_Min']):
            df_proc['V_Sum'] = df_proc[v_cols].sum(axis=1)
            df_proc['V_Mean'] = df_proc[v_cols].mean(axis=1)
            df_proc['V_Std'] = df_proc[v_cols].std(axis=1)
            df_proc['V_Max'] = df_proc[v_cols].max(axis=1)
            df_proc['V_Min'] = df_proc[v_cols].min(axis=1)

        # fill missing feature_cols with 0.0
        for col in self.feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0.0

        # scale Amount
        if 'amount_scaler' in self.scalers and 'Amount' in df_proc.columns:
            try:
                df_proc['Amount_Scaled'] = self.scalers['amount_scaler'].transform(df_proc[['Amount']])
            except Exception:
                df_proc['Amount_Scaled'] = df_proc['Amount']
        else:
            if 'Amount' in df_proc.columns and 'Amount_Scaled' not in df_proc.columns:
                df_proc['Amount_Scaled'] = df_proc['Amount']

        # Build flat feature matrix
        flat_features = [f for f in self.feature_names if f in df_proc.columns]
        X_flat = df_proc[flat_features].fillna(0.0).values.astype(float)

        # Build X_seq for LSTM: if lstm model exists, try to form (n, seq_len, n_features)
        if 'lstm' in self.models:
            lstm = self.models['lstm']
            # try to infer seq_len from model input_shape if possible
            seq_len = 1
            n_feat = X_flat.shape[1]
            try:
                ishape = getattr(lstm, 'input_shape', None)
                if ishape and len(ishape) == 3:
                    seq_len = ishape[1] or 1
                    n_feat = ishape[2] or n_feat
            except Exception:
                seq_len = 1
            # for simplicity: repeat each flat vector seq_len times -> shape (n, seq_len, n_feat)
            X_seq = np.repeat(X_flat.reshape((X_flat.shape[0], 1, X_flat.shape[1])), seq_len, axis=1)
        else:
            X_seq = None

        return X_seq, X_flat, flat_features

    def hybrid_predict(self, df: pd.DataFrame, threshold: float = 0.5, 
                      with_explanation: bool = False):
        """
        Enhanced hybrid prediction with optional explainability
        """
        if self.hybrid is None:
            raise RuntimeError("Hybrid model not initialized / models missing")

        X_seq, X_flat, feature_list = self.preprocess_input(df)
        
        # Extract transaction amounts if available
        transaction_amounts = None
        if 'Amount' in df.columns:
            transaction_amounts = df['Amount'].values

        if with_explanation:
            # Use the comprehensive prediction method
            result = self.hybrid.predict_with_explanation(
                X_seq, X_flat, feature_list, transaction_amounts, threshold
            )
            
            # Format results for API response
            results = []
            for i in range(len(result['predictions'])):
                results.append({
                    "prediction": int(result['predictions'][i]),
                    "probability": float(result['probabilities'][i]),
                    "lstm_probability": float(result['lstm_probabilities'][i]),
                    "dt_probability": float(result['dt_probabilities'][i]),
                    "alpha_used": float(result['alpha_used']),
                    "explanation": result['explanations'][i],
                    "feature_contributions": result['feature_contributions'][i] if i < len(result['feature_contributions']) else [],
                    "threshold_used": float(result['threshold_used']),
                    "dominant_model": "LSTM" if result['lstm_probabilities'][i] > result['dt_probabilities'][i] else "Decision Tree"
                })
            return results
        else:
            # Simple prediction for backward compatibility
            probs = self.hybrid.predict_proba(X_seq, X_flat, transaction_amounts)
            preds = (probs >= threshold).astype(int)

            # Get contribution dominance
            try:
                p_lstm = self.hybrid._safe_predict_lstm(X_seq) if X_seq is not None else np.full_like(probs, 0.5)
                p_dt = self.hybrid._safe_predict_dt(X_flat)
                dominant = np.where(p_lstm >= p_dt, 'lstm', 'dt').tolist()
            except Exception:
                dominant = ['unknown'] * len(probs)

            results = []
            for i in range(len(probs)):
                results.append({
                    "prediction": int(preds[i]),
                    "probability": float(probs[i]),
                    "dominant_model": dominant[i]
                })
            return results

    def get_model_analytics(self) -> Dict:
        """Get comprehensive model analytics and information"""
        if self.hybrid is None:
            return {"error": "Hybrid model not available"}
        
        analytics = {
            "hybrid_model_info": self.hybrid.get_model_info(),
            "available_models": list(self.models.keys()),
            "available_scalers": list(self.scalers.keys()),
            "feature_names": self.feature_names,
            "total_features": len(self.feature_names) if self.feature_names else 0
        }
        
        return analytics

# create singleton for import convenience
model_manager = ModelManager()
