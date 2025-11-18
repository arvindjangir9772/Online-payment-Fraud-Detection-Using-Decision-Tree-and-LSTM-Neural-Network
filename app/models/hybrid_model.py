# app/models/hybrid_model.py
"""
Advanced Hybrid Fraud Detection Model
Combines LSTM (temporal patterns) and Decision Tree (rule-based interpretability)
with dynamic weighting, explainability, and adaptive thresholds.
"""
import numpy as np
import pandas as pd
from typing import Any, Optional, Sequence, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Import SHAP explainer
try:
    from app.models.shap_explainer import SHAPExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Advanced explainability features will be limited.")

class HybridFraudNet:
    def __init__(self, lstm_model: Any, dt_model: Any, alpha: float = 0.6, 
                 adaptive_threshold: bool = True, explainability: bool = True,
                 feature_names: Optional[List[str]] = None):
        self.lstm_model = lstm_model
        self.dt_model = dt_model
        self.base_alpha = float(alpha)
        self.adaptive_threshold = adaptive_threshold
        self.explainability = explainability
        self.feature_names = feature_names or []
        
        # For adaptive thresholding
        self.threshold_history = []
        self.optimal_threshold = 0.5
        
        # Feature importance tracking
        self.feature_importance = None
        self._initialize_feature_importance()
        
        # SHAP explainers
        self.shap_explainer_dt = None
        self.shap_explainer_lstm = None
        self.shap_explainer_hybrid = None
        self._initialize_shap_explainers()

    def _initialize_feature_importance(self):
        """Initialize feature importance tracking"""
        try:
            if hasattr(self.dt_model, 'feature_importances_'):
                self.feature_importance = self.dt_model.feature_importances_
            else:
                self.feature_importance = None
        except Exception as e:
            logger.warning(f"Could not initialize feature importance: {e}")
            self.feature_importance = None
    
    def _initialize_shap_explainers(self):
        """Initialize SHAP explainers for advanced explainability"""
        if not SHAP_AVAILABLE or not self.explainability:
            return
        
        try:
            # Initialize SHAP explainer for Decision Tree
            if self.dt_model is not None:
                self.shap_explainer_dt = SHAPExplainer(
                    self.dt_model, 
                    self.feature_names
                )
            
            # Initialize SHAP explainer for LSTM (if possible)
            if self.lstm_model is not None:
                self.shap_explainer_lstm = SHAPExplainer(
                    self.lstm_model, 
                    self.feature_names
                )
            
            logger.info("SHAP explainers initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainers: {e}")
            self.shap_explainer_dt = None
            self.shap_explainer_lstm = None

    def _safe_predict_lstm(self, X_seq: np.ndarray) -> np.ndarray:
        """Return 1D array of lstm probabilities with error handling."""
        if X_seq is None or self.lstm_model is None:
            return None
        try:
            # Handle both Keras and scikit-learn models
            if hasattr(self.lstm_model, 'predict_proba'):
                # Scikit-learn model (MLPClassifier)
                if len(X_seq.shape) == 3:
                    # X_seq is (n_samples, sequence_length, n_features)
                    # Take the last timestep for prediction
                    X_flat = X_seq[:, -1, :]  # Shape: (n_samples, n_features)
                else:
                    X_flat = X_seq
                p = self.lstm_model.predict_proba(X_flat)[:, 1]  # Get fraud probability
            else:
                # Keras model
                p = self.lstm_model.predict(X_seq, verbose=0)
                p = np.array(p).reshape(-1)
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}, using fallback")
            p = np.full((X_seq.shape[0],), 0.5)
        return np.clip(p, 0.0, 1.0)

    def _safe_predict_dt(self, X_flat: np.ndarray) -> np.ndarray:
        """Return 1D array of dt probabilities with error handling."""
        try:
            p = self.dt_model.predict_proba(X_flat)[:, 1]
        except Exception:
            try:
                preds = self.dt_model.predict(X_flat)
                p = np.array(preds).astype(float)
            except Exception:
                logger.warning("Decision Tree prediction failed, using fallback")
                p = np.full((X_flat.shape[0],), 0.5)
        return np.clip(p, 0.0, 1.0)

    def dynamic_alpha(self, X_flat: Optional[np.ndarray] = None, 
                     X_seq: Optional[np.ndarray] = None,
                     transaction_amounts: Optional[np.ndarray] = None) -> float:
        """
        Dynamic alpha calculation based on:
        - Transaction amount anomaly
        - Feature variance
        - Model confidence levels
        """
        base_alpha = self.base_alpha
        
        if X_flat is None:
            return base_alpha
            
        try:
            # Calculate feature variance
            feature_variance = np.var(X_flat, axis=1).mean()
            
            # Amount-based adjustment
            amount_factor = 1.0
            if transaction_amounts is not None:
                amount_std = np.std(transaction_amounts)
                if amount_std > 0:
                    # Higher variance in amounts -> trust LSTM more for temporal patterns
                    amount_factor = min(1.2, 1.0 + (amount_std / 1000))
            
            # Variance-based adjustment
            variance_factor = 1.0
            if feature_variance > 0.1:  # High variance in features
                variance_factor = 1.1  # Slightly favor LSTM for complex patterns
            
            # Combine factors
            dynamic_alpha = base_alpha * amount_factor * variance_factor
            return np.clip(dynamic_alpha, 0.3, 0.9)  # Keep within reasonable bounds
            
        except Exception as e:
            logger.warning(f"Dynamic alpha calculation failed: {e}")
            return base_alpha

    def calculate_feature_contributions(self, X_flat: np.ndarray, 
                                      feature_names: List[str]) -> List[Dict]:
        """
        Calculate feature contributions for explainability
        """
        if not self.explainability or self.feature_importance is None:
            return []
            
        try:
            contributions = []
            for i in range(X_flat.shape[0]):
                sample_contributions = {}
                for j, feature_name in enumerate(feature_names):
                    if j < len(self.feature_importance):
                        # Weight by feature importance and value magnitude
                        contribution = (X_flat[i, j] * self.feature_importance[j]) 
                        sample_contributions[feature_name] = float(contribution)
                
                # Sort by absolute contribution
                sorted_contributions = sorted(
                    sample_contributions.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                contributions.append(sorted_contributions[:5])  # Top 5 features
                
            return contributions
        except Exception as e:
            logger.warning(f"Feature contribution calculation failed: {e}")
            return []

    def generate_explanation(self, prob_lstm: float, prob_dt: float, 
                           final_prob: float, feature_contributions: List[Tuple],
                           threshold: float = 0.5) -> str:
        """
        Generate human-readable explanation for the prediction
        """
        is_fraud = final_prob >= threshold
        status = "FRAUDULENT" if is_fraud else "LEGITIMATE"
        
        # Determine dominant model
        dominant = "LSTM" if prob_lstm > prob_dt else "Decision Tree"
        
        # Get top contributing features
        top_features = [f"{feat}: {contrib:.3f}" for feat, contrib in feature_contributions[:3]]
        
        explanation = f"""
        Transaction Status: {status}
        Fraud Probability: {final_prob:.1%}
        Dominant Model: {dominant}
        LSTM Confidence: {prob_lstm:.1%}
        Decision Tree Confidence: {prob_dt:.1%}
        Key Factors: {', '.join(top_features)}
        """
        
        return explanation.strip()

    def adaptive_threshold_calculation(self, recent_predictions: List[float]) -> float:
        """
        Calculate adaptive threshold based on recent prediction patterns
        """
        if not self.adaptive_threshold or len(recent_predictions) < 10:
            return self.optimal_threshold
            
        try:
            # Simple adaptive logic: adjust threshold based on recent fraud rate
            recent_frauds = sum(1 for p in recent_predictions if p >= self.optimal_threshold)
            fraud_rate = recent_frauds / len(recent_predictions)
            
            # If fraud rate is too high, increase threshold (be more conservative)
            if fraud_rate > 0.3:
                self.optimal_threshold = min(0.8, self.optimal_threshold + 0.05)
            # If fraud rate is too low, decrease threshold (be more sensitive)
            elif fraud_rate < 0.05:
                self.optimal_threshold = max(0.2, self.optimal_threshold - 0.05)
                
            return self.optimal_threshold
        except Exception as e:
            logger.warning(f"Adaptive threshold calculation failed: {e}")
            return self.optimal_threshold

    def predict_proba(self, X_seq: Optional[np.ndarray], X_flat: np.ndarray,
                     transaction_amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enhanced prediction with dynamic alpha and error handling
        """
        n = X_flat.shape[0]
        p_lstm = self._safe_predict_lstm(X_seq) if X_seq is not None else np.full((n,), 0.5)
        p_dt = self._safe_predict_dt(X_flat)
        
        # Dynamic alpha calculation
        alpha = self.dynamic_alpha(X_flat=X_flat, X_seq=X_seq, 
                                 transaction_amounts=transaction_amounts)
        
        # Weighted combination
        p_final = alpha * p_lstm + (1.0 - alpha) * p_dt
        return np.clip(p_final, 0.0, 1.0)

    def predict_with_explanation(self, X_seq: Optional[np.ndarray], X_flat: np.ndarray,
                               feature_names: List[str], 
                               transaction_amounts: Optional[np.ndarray] = None,
                               threshold: float = 0.5) -> Dict:
        """
        Comprehensive prediction with full explainability
        """
        # Get individual model predictions
        p_lstm = self._safe_predict_lstm(X_seq) if X_seq is not None else np.full((X_flat.shape[0],), 0.5)
        p_dt = self._safe_predict_dt(X_flat)
        
        # Get final probabilities
        p_final = self.predict_proba(X_seq, X_flat, transaction_amounts)
        
        # Calculate dynamic alpha
        alpha = self.dynamic_alpha(X_flat=X_flat, X_seq=X_seq, 
                                 transaction_amounts=transaction_amounts)
        
        # Get feature contributions
        feature_contributions = self.calculate_feature_contributions(X_flat, feature_names)
        
        # Generate explanations
        explanations = []
        for i in range(len(p_final)):
            explanation = self.generate_explanation(
                p_lstm[i], p_dt[i], p_final[i], 
                feature_contributions[i] if i < len(feature_contributions) else [],
                threshold
            )
            explanations.append(explanation)
        
        # Predictions
        preds = (p_final >= threshold).astype(int)
        
        return {
            'predictions': preds,
            'probabilities': p_final,
            'lstm_probabilities': p_lstm,
            'dt_probabilities': p_dt,
            'alpha_used': alpha,
            'explanations': explanations,
            'feature_contributions': feature_contributions,
            'threshold_used': threshold
        }

    def predict(self, X_seq: Optional[np.ndarray], X_flat: np.ndarray, 
                threshold: float = 0.5, transaction_amounts: Optional[np.ndarray] = None):
        """Simple prediction interface for backward compatibility"""
        probs = self.predict_proba(X_seq, X_flat, transaction_amounts)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    def get_shap_explanation(self, X_flat: np.ndarray, model_type: str = 'hybrid') -> Dict[str, Any]:
        """
        Get SHAP explanation for a prediction
        
        Args:
            X_flat: Input features (n_samples, n_features)
            model_type: 'dt', 'lstm', or 'hybrid'
        
        Returns:
            SHAP explanation dictionary
        """
        if not self.explainability:
            return {'error': 'Explainability disabled'}
        
        try:
            if model_type == 'dt' and self.shap_explainer_dt is not None:
                return self.shap_explainer_dt.explain_prediction(X_flat)
            elif model_type == 'lstm' and self.shap_explainer_lstm is not None:
                return self.shap_explainer_lstm.explain_prediction(X_flat)
            elif model_type == 'hybrid':
                # For hybrid, use Decision Tree SHAP as primary explainer
                if self.shap_explainer_dt is not None:
                    return self.shap_explainer_dt.explain_prediction(X_flat)
                else:
                    return self._fallback_explanation(X_flat)
            else:
                return self._fallback_explanation(X_flat)
                
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return self._fallback_explanation(X_flat)
    
    def get_batch_shap_explanations(self, X_flat: np.ndarray, model_type: str = 'hybrid', 
                                   max_samples: int = 100) -> Dict[str, Any]:
        """
        Get SHAP explanations for a batch of predictions
        
        Args:
            X_flat: Input features (n_samples, n_features)
            model_type: 'dt', 'lstm', or 'hybrid'
            max_samples: Maximum number of samples to explain
        
        Returns:
            Batch SHAP explanations dictionary
        """
        if not self.explainability:
            return {'error': 'Explainability disabled'}
        
        try:
            if model_type == 'dt' and self.shap_explainer_dt is not None:
                return self.shap_explainer_dt.explain_batch(X_flat, max_samples)
            elif model_type == 'lstm' and self.shap_explainer_lstm is not None:
                return self.shap_explainer_lstm.explain_batch(X_flat, max_samples)
            elif model_type == 'hybrid':
                if self.shap_explainer_dt is not None:
                    return self.shap_explainer_dt.explain_batch(X_flat, max_samples)
                else:
                    return self._fallback_batch_explanation(X_flat, max_samples)
            else:
                return self._fallback_batch_explanation(X_flat, max_samples)
                
        except Exception as e:
            logger.error(f"Error generating batch SHAP explanations: {e}")
            return self._fallback_batch_explanation(X_flat, max_samples)
    
    def _fallback_explanation(self, X_flat: np.ndarray) -> Dict[str, Any]:
        """Fallback explanation when SHAP is not available"""
        if len(X_flat.shape) == 1:
            X_flat = X_flat.reshape(1, -1)
        
        # Simple feature importance based on feature values
        feature_values = X_flat[0]
        
        feature_importance = []
        for i, (feature_name, value) in enumerate(zip(self.feature_names, feature_values)):
            importance = abs(value)
            feature_importance.append({
                'feature': feature_name,
                'shap_value': float(value),
                'abs_shap_value': float(importance),
                'contribution': float(value),
                'impact': 'positive' if value > 0 else 'negative' if value < 0 else 'neutral'
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        explanation = self._generate_fallback_explanation(feature_importance, feature_values)
        
        return {
            'shap_values': feature_values.tolist(),
            'feature_importance': feature_importance,
            'explanation': explanation,
            'base_value': 0.0,
            'prediction_value': float(np.sum(feature_values))
        }
    
    def _fallback_batch_explanation(self, X_flat: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """Fallback batch explanation when SHAP is not available"""
        X_subset = X_flat[:max_samples] if len(X_flat) > max_samples else X_flat
        
        explanations = []
        for i in range(len(X_subset)):
            explanation = self._fallback_explanation(X_subset[i:i+1])
            explanations.append(explanation)
        
        # Calculate global importance
        global_importance = []
        for i, feature_name in enumerate(self.feature_names):
            if i < X_subset.shape[1]:
                feature_values = X_subset[:, i]
                importance = np.mean(np.abs(feature_values))
                global_importance.append({
                    'feature': feature_name,
                    'importance': float(importance),
                    'rank': i + 1
                })
        
        global_importance.sort(key=lambda x: x['importance'], reverse=True)
        for i, item in enumerate(global_importance):
            item['rank'] = i + 1
        
        return {
            'global_importance': global_importance,
            'explanations': explanations,
            'base_value': 0.0
        }
    
    def _generate_fallback_explanation(self, feature_importance: List[Dict], X: np.ndarray) -> str:
        """Generate fallback explanation text"""
        if not feature_importance:
            return "No explanation available."
        
        # Get top 3 most important features
        top_features = feature_importance[:3]
        
        explanation_parts = []
        
        for feature in top_features:
            feature_name = feature['feature']
            shap_value = feature['shap_value']
            impact = feature['impact']
            
            # Get the actual feature value
            feature_idx = self.feature_names.index(feature_name) if feature_name in self.feature_names else 0
            feature_value = X[feature_idx] if feature_idx < len(X) else 0.0
            
            if impact == 'positive':
                explanation_parts.append(
                    f"High {feature_name} value ({feature_value:.3f}) increases fraud risk"
                )
            elif impact == 'negative':
                explanation_parts.append(
                    f"Low {feature_name} value ({feature_value:.3f}) increases fraud risk"
                )
        
        if explanation_parts:
            return "This transaction appears suspicious because: " + "; ".join(explanation_parts) + "."
        else:
            return "This transaction shows normal patterns across all features."
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        shap_info = {
            'shap_available': SHAP_AVAILABLE,
            'shap_dt_available': self.shap_explainer_dt is not None,
            'shap_lstm_available': self.shap_explainer_lstm is not None
        }
        
        return {
            'model_type': 'HybridFraudNet',
            'base_alpha': self.base_alpha,
            'adaptive_threshold': self.adaptive_threshold,
            'explainability_enabled': self.explainability,
            'optimal_threshold': self.optimal_threshold,
            'feature_importance_available': self.feature_importance is not None,
            'lstm_loaded': self.lstm_model is not None,
            'dt_loaded': self.dt_model is not None,
            'feature_names': self.feature_names,
            'shap_info': shap_info
        }
