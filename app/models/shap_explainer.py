# app/models/shap_explainer.py
"""
SHAP (SHapley Additive exPlanations) integration for advanced model explainability
Provides feature importance and reasoning for fraud detection decisions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self, model, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer for model interpretability
        
        Args:
            model: Trained model (Decision Tree, LSTM, or Hybrid)
            feature_names: List of feature names
            background_data: Background dataset for SHAP (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type"""
        try:
            import shap
            
            # Determine model type and initialize appropriate explainer
            if hasattr(self.model, 'predict_proba'):
                # For models with predict_proba (Decision Tree, etc.)
                if self.background_data is not None:
                    self.explainer = shap.TreeExplainer(self.model, self.background_data)
                else:
                    self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'predict'):
                # For neural networks (LSTM, etc.)
                if self.background_data is not None:
                    self.explainer = shap.DeepExplainer(self.model, self.background_data)
                else:
                    # Use a simple background if none provided
                    background = np.zeros((1, len(self.feature_names)))
                    self.explainer = shap.DeepExplainer(self.model, background)
            else:
                logger.warning("Model type not supported for SHAP explanation")
                self.explainer = None
                
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
            self.explainer = None
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, X: np.ndarray, prediction_index: int = 0) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            X: Input features (n_samples, n_features)
            prediction_index: Index of prediction to explain
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        if self.explainer is None:
            return self._fallback_explanation(X, prediction_index)
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle different model types
            if isinstance(shap_values, list):
                # Multi-class case - use the positive class (fraud)
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Get values for the specific prediction
            if len(shap_values.shape) > 1:
                prediction_shap = shap_values[prediction_index]
            else:
                prediction_shap = shap_values
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(prediction_shap)
            
            # Generate explanation text
            explanation = self._generate_explanation_text(feature_importance, X[prediction_index])
            
            return {
                'shap_values': prediction_shap.tolist(),
                'feature_importance': feature_importance,
                'explanation': explanation,
                'base_value': getattr(self.explainer, 'expected_value', 0.0),
                'prediction_value': float(np.sum(prediction_shap) + getattr(self.explainer, 'expected_value', 0.0))
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return self._fallback_explanation(X, prediction_index)
    
    def explain_batch(self, X: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a batch of predictions
        
        Args:
            X: Input features (n_samples, n_features)
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing batch SHAP explanations
        """
        if self.explainer is None:
            return self._fallback_batch_explanation(X, max_samples)
        
        try:
            # Limit samples for performance
            X_subset = X[:max_samples] if len(X) > max_samples else X
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(X_subset)
            
            # Handle different model types
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Calculate global feature importance
            global_importance = self._calculate_global_importance(shap_values)
            
            # Generate explanations for each sample
            explanations = []
            for i in range(len(X_subset)):
                prediction_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values
                feature_importance = self._calculate_feature_importance(prediction_shap)
                explanation = self._generate_explanation_text(feature_importance, X_subset[i])
                
                explanations.append({
                    'shap_values': prediction_shap.tolist(),
                    'feature_importance': feature_importance,
                    'explanation': explanation
                })
            
            return {
                'global_importance': global_importance,
                'explanations': explanations,
                'base_value': getattr(self.explainer, 'expected_value', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error generating batch SHAP explanation: {e}")
            return self._fallback_batch_explanation(X, max_samples)
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate feature importance from SHAP values"""
        importance = []
        
        for i, (feature_name, shap_value) in enumerate(zip(self.feature_names, shap_values)):
            importance.append({
                'feature': feature_name,
                'shap_value': float(shap_value),
                'abs_shap_value': float(abs(shap_value)),
                'contribution': float(shap_value),
                'impact': 'positive' if shap_value > 0 else 'negative' if shap_value < 0 else 'neutral'
            })
        
        # Sort by absolute SHAP value (most important first)
        importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        return importance
    
    def _calculate_global_importance(self, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate global feature importance across all samples"""
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        global_importance = []
        for i, (feature_name, importance) in enumerate(zip(self.feature_names, mean_abs_shap)):
            global_importance.append({
                'feature': feature_name,
                'importance': float(importance),
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by importance and assign ranks
        global_importance.sort(key=lambda x: x['importance'], reverse=True)
        for i, item in enumerate(global_importance):
            item['rank'] = i + 1
        
        return global_importance
    
    def _generate_explanation_text(self, feature_importance: List[Dict], X: np.ndarray) -> str:
        """Generate human-readable explanation text"""
        if not feature_importance:
            return "No explanation available."
        
        # Get top 3 most important features
        top_features = feature_importance[:3]
        
        explanation_parts = []
        
        for i, feature in enumerate(top_features):
            feature_name = feature['feature']
            shap_value = feature['shap_value']
            impact = feature['impact']
            
            # Get the actual feature value
            feature_idx = self.feature_names.index(feature_name)
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
    
    def _fallback_explanation(self, X: np.ndarray, prediction_index: int = 0) -> Dict[str, Any]:
        """Fallback explanation when SHAP is not available"""
        # Simple feature importance based on feature values
        feature_values = X[prediction_index] if len(X.shape) > 1 else X
        
        feature_importance = []
        for i, (feature_name, value) in enumerate(zip(self.feature_names, feature_values)):
            # Simple heuristic: higher absolute values are more important
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
        
        explanation = self._generate_explanation_text(feature_importance, feature_values)
        
        return {
            'shap_values': feature_values.tolist(),
            'feature_importance': feature_importance,
            'explanation': explanation,
            'base_value': 0.0,
            'prediction_value': float(np.sum(feature_values))
        }
    
    def _fallback_batch_explanation(self, X: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """Fallback batch explanation when SHAP is not available"""
        X_subset = X[:max_samples] if len(X) > max_samples else X
        
        explanations = []
        for i in range(len(X_subset)):
            explanation = self._fallback_explanation(X_subset, i)
            explanations.append(explanation)
        
        # Calculate global importance
        all_values = X_subset.flatten()
        global_importance = []
        for i, feature_name in enumerate(self.feature_names):
            feature_values = X_subset[:, i] if len(X_subset.shape) > 1 else X_subset
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
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get a summary of feature importance across all features"""
        if self.explainer is None:
            return {
                'available': False,
                'message': 'SHAP explainer not available'
            }
        
        return {
            'available': True,
            'explainer_type': type(self.explainer).__name__,
            'feature_count': len(self.feature_names),
            'background_samples': len(self.background_data) if self.background_data is not None else 0
        }
