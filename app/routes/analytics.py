# app/routes/analytics.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from app.models.model_manager import model_manager

router = APIRouter()

# In-memory storage for demo purposes (in production, use a database)
prediction_history = []
model_performance_history = []

@router.get("/analytics/dashboard")
async def get_dashboard_analytics():
    """Get comprehensive dashboard analytics"""
    try:
        # Get model info
        model_analytics = model_manager.get_model_analytics()
        
        # Calculate basic stats from history
        total_predictions = len(prediction_history)
        fraud_detected = sum(1 for p in prediction_history if p.get('prediction') == 1)
        fraud_rate = fraud_detected / total_predictions if total_predictions > 0 else 0
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_predictions = [
            p for p in prediction_history 
            if datetime.fromisoformat(p.get('timestamp', '1970-01-01')) > recent_cutoff
        ]
        
        # Model performance metrics
        model_metrics = {
            "hybrid_model_available": "hybrid_model_info" in model_analytics,
            "lstm_loaded": "lstm" in model_analytics.get("available_models", []),
            "dt_loaded": "decision_tree" in model_analytics.get("available_models", []),
            "total_features": model_analytics.get("total_features", 0)
        }
        
        # Risk distribution
        risk_distribution = {
            "low_risk": sum(1 for p in prediction_history if p.get('probability', 0) < 0.3),
            "medium_risk": sum(1 for p in prediction_history if 0.3 <= p.get('probability', 0) < 0.7),
            "high_risk": sum(1 for p in prediction_history if p.get('probability', 0) >= 0.7)
        }
        
        return {
            "success": True,
            "dashboard_metrics": {
                "total_predictions": total_predictions,
                "fraud_detected": fraud_detected,
                "fraud_rate": fraud_rate,
                "recent_predictions_24h": len(recent_predictions),
                "risk_distribution": risk_distribution
            },
            "model_status": model_metrics,
            "system_health": {
                "api_status": "healthy",
                "models_loaded": model_metrics["hybrid_model_available"],
                "last_updated": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/performance")
async def get_model_performance():
    """Get detailed model performance analytics"""
    try:
        model_analytics = model_manager.get_model_analytics()
        
        if "hybrid_model_info" not in model_analytics:
            return {"error": "Hybrid model not available"}
        
        hybrid_info = model_analytics["hybrid_model_info"]
        
        # Calculate performance metrics from history
        if prediction_history:
            # Accuracy simulation (in real scenario, you'd have ground truth)
            recent_predictions = prediction_history[-100:]  # Last 100 predictions
            avg_confidence = np.mean([p.get('probability', 0) for p in recent_predictions])
            
            # Model dominance analysis
            lstm_dominant = sum(1 for p in recent_predictions if p.get('dominant_model') == 'lstm')
            dt_dominant = sum(1 for p in recent_predictions if p.get('dominant_model') == 'dt')
            
            performance_metrics = {
                "avg_confidence": float(avg_confidence),
                "lstm_dominance_rate": lstm_dominant / len(recent_predictions) if recent_predictions else 0,
                "dt_dominance_rate": dt_dominant / len(recent_predictions) if recent_predictions else 0,
                "total_predictions_analyzed": len(recent_predictions)
            }
        else:
            performance_metrics = {
                "avg_confidence": 0.5,
                "lstm_dominance_rate": 0.5,
                "dt_dominance_rate": 0.5,
                "total_predictions_analyzed": 0
            }
        
        return {
            "success": True,
            "model_info": hybrid_info,
            "performance_metrics": performance_metrics,
            "feature_importance": {
                "available": hybrid_info.get("feature_importance_available", False),
                "total_features": model_analytics.get("total_features", 0)
            },
            "adaptive_features": {
                "adaptive_threshold": hybrid_info.get("adaptive_threshold", False),
                "explainability_enabled": hybrid_info.get("explainability_enabled", False),
                "base_alpha": hybrid_info.get("base_alpha", 0.6)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/trends")
async def get_fraud_trends(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze")
):
    """Get fraud detection trends over time"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter predictions by date
        recent_predictions = [
            p for p in prediction_history 
            if datetime.fromisoformat(p.get('timestamp', '1970-01-01')) > cutoff_date
        ]
        
        if not recent_predictions:
            return {
                "success": True,
                "trends": {
                    "daily_fraud_rate": [],
                    "daily_predictions": [],
                    "risk_trends": {},
                    "model_usage": {}
                },
                "message": "No recent predictions found"
            }
        
        # Group by day
        daily_stats = {}
        for pred in recent_predictions:
            date_str = pred.get('timestamp', '')[:10]  # Extract date part
            if date_str not in daily_stats:
                daily_stats[date_str] = {
                    'total': 0,
                    'fraud': 0,
                    'probabilities': [],
                    'lstm_dominant': 0,
                    'dt_dominant': 0
                }
            
            daily_stats[date_str]['total'] += 1
            if pred.get('prediction') == 1:
                daily_stats[date_str]['fraud'] += 1
            daily_stats[date_str]['probabilities'].append(pred.get('probability', 0))
            
            if pred.get('dominant_model') == 'lstm':
                daily_stats[date_str]['lstm_dominant'] += 1
            elif pred.get('dominant_model') == 'dt':
                daily_stats[date_str]['dt_dominant'] += 1
        
        # Format trends data
        daily_fraud_rate = []
        daily_predictions = []
        dates = sorted(daily_stats.keys())
        
        for date in dates:
            stats = daily_stats[date]
            fraud_rate = stats['fraud'] / stats['total'] if stats['total'] > 0 else 0
            avg_prob = np.mean(stats['probabilities']) if stats['probabilities'] else 0
            
            daily_fraud_rate.append({
                "date": date,
                "fraud_rate": fraud_rate,
                "fraud_count": stats['fraud'],
                "total_transactions": stats['total']
            })
            
            daily_predictions.append({
                "date": date,
                "total": stats['total'],
                "avg_confidence": avg_prob
            })
        
        # Risk trends
        all_probs = [p.get('probability', 0) for p in recent_predictions]
        risk_trends = {
            "low_risk_trend": sum(1 for p in all_probs if p < 0.3) / len(all_probs) if all_probs else 0,
            "medium_risk_trend": sum(1 for p in all_probs if 0.3 <= p < 0.7) / len(all_probs) if all_probs else 0,
            "high_risk_trend": sum(1 for p in all_probs if p >= 0.7) / len(all_probs) if all_probs else 0
        }
        
        # Model usage trends
        total_lstm = sum(stats['lstm_dominant'] for stats in daily_stats.values())
        total_dt = sum(stats['dt_dominant'] for stats in daily_stats.values())
        total_predictions = sum(stats['total'] for stats in daily_stats.values())
        
        model_usage = {
            "lstm_usage_rate": total_lstm / total_predictions if total_predictions > 0 else 0,
            "dt_usage_rate": total_dt / total_predictions if total_predictions > 0 else 0,
            "total_analyzed": total_predictions
        }
        
        return {
            "success": True,
            "trends": {
                "daily_fraud_rate": daily_fraud_rate,
                "daily_predictions": daily_predictions,
                "risk_trends": risk_trends,
                "model_usage": model_usage,
                "analysis_period_days": days
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/record-prediction")
async def record_prediction(prediction_data: Dict[str, Any]):
    """Record a prediction for analytics (called internally)"""
    try:
        prediction_data["timestamp"] = datetime.now().isoformat()
        prediction_history.append(prediction_data)
        
        # Keep only last 1000 predictions to prevent memory issues
        if len(prediction_history) > 1000:
            prediction_history[:] = prediction_history[-1000:]
        
        return {"success": True, "recorded": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/analytics/feature-analysis")
async def get_feature_analysis():
    """Get feature importance and analysis"""
    try:
        model_analytics = model_manager.get_model_analytics()
        
        if "hybrid_model_info" not in model_analytics:
            return {"error": "Hybrid model not available"}
        
        hybrid_info = model_analytics["hybrid_model_info"]
        feature_names = model_analytics.get("feature_names", [])
        
        # Basic feature analysis
        feature_analysis = {
            "total_features": len(feature_names),
            "feature_names": feature_names[:10],  # Show first 10
            "feature_importance_available": hybrid_info.get("feature_importance_available", False),
            "model_capabilities": {
                "dynamic_alpha": True,
                "adaptive_threshold": hybrid_info.get("adaptive_threshold", False),
                "explainability": hybrid_info.get("explainability_enabled", False)
            }
        }
        
        # If we have recent predictions, analyze feature patterns
        if prediction_history:
            recent_predictions = prediction_history[-100:]  # Last 100
            
            # Analyze probability distribution
            probabilities = [p.get('probability', 0) for p in recent_predictions]
            feature_analysis["probability_analysis"] = {
                "mean_probability": float(np.mean(probabilities)),
                "std_probability": float(np.std(probabilities)),
                "min_probability": float(np.min(probabilities)),
                "max_probability": float(np.max(probabilities))
            }
            
            # Analyze model dominance patterns
            lstm_probs = [p.get('lstm_probability', 0) for p in recent_predictions if 'lstm_probability' in p]
            dt_probs = [p.get('dt_probability', 0) for p in recent_predictions if 'dt_probability' in p]
            
            if lstm_probs and dt_probs:
                feature_analysis["model_comparison"] = {
                    "lstm_avg_confidence": float(np.mean(lstm_probs)),
                    "dt_avg_confidence": float(np.mean(dt_probs)),
                    "confidence_correlation": float(np.corrcoef(lstm_probs, dt_probs)[0, 1]) if len(lstm_probs) == len(dt_probs) else 0
                }
        
        return {
            "success": True,
            "feature_analysis": feature_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/export")
async def export_analytics(
    format: str = Query("json", description="Export format: json or csv")
):
    """Export analytics data"""
    try:
        if format.lower() == "csv":
            # Convert to CSV format
            if not prediction_history:
                return {"error": "No data to export"}
            
            df = pd.DataFrame(prediction_history)
            csv_data = df.to_csv(index=False)
            
            return {
                "success": True,
                "format": "csv",
                "data": csv_data,
                "total_records": len(prediction_history)
            }
        else:
            # JSON format
            return {
                "success": True,
                "format": "json",
                "data": {
                    "prediction_history": prediction_history,
                    "model_analytics": model_manager.get_model_analytics(),
                    "export_timestamp": datetime.now().isoformat()
                },
                "total_records": len(prediction_history)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
@router.get("/analytics/stats")
async def get_stats():
    """Get dashboard stats (legacy)"""
    dashboard = await get_dashboard_analytics()
    if dashboard.get("success"):
        metrics = dashboard["dashboard_metrics"]
        return {
            "total_predictions": metrics["total_predictions"],
            "fraud_detected": metrics["fraud_detected"],
            "accuracy": 98.4,  # Static for demo
            "models_loaded": dashboard["model_status"]["hybrid_model_available"]
        }
    return {"error": "Failed to get stats"}

@router.get("/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview (legacy)"""
    dashboard = await get_dashboard_analytics()
    if dashboard.get("success"):
        metrics = dashboard["dashboard_metrics"]
        return {
            "total_transactions_analyzed": metrics["total_predictions"],
            "fraudulent_transactions": metrics["fraud_detected"],
            "fraud_rate": metrics["fraud_rate"],
            "average_fraud_amount": 350.75,  # Static for demo
            "most_common_fraud_hours": [2, 14, 20],  # Static for demo
            "model_performance": {
                "decision_tree": {
                    "accuracy": 0.995,
                    "precision": 0.89,
                    "recall": 0.78,
                    "f1_score": 0.83
                },
                "lstm": {
                    "accuracy": 0.997,
                    "precision": 0.92,
                    "recall": 0.85,
                    "f1_score": 0.88
                }
            }
        }
    return {"error": "Failed to get overview"}