# app/routes/predictions.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import pandas as pd
import io
import json
from datetime import datetime
import time

from app.models.model_manager import model_manager
from app.models.history_manager import history_manager

router = APIRouter()

@router.post("/predict/hybrid/single")
async def predict_hybrid_single(
    payload: Dict[str, Any],
    with_explanation: bool = Query(False, description="Include detailed explanation and feature contributions"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Fraud detection threshold")
):
    """
    Enhanced single transaction prediction with optional explainability
    
    payload: JSON of a single transaction with keys matching feature names (V1..V28, Amount, etc.)
    Example: {"V1":0.1, "V2":-0.3, ..., "Amount": 23.4}
    """
    try:
        df = pd.DataFrame([payload])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    try:
        results = model_manager.hybrid_predict(df, threshold=threshold, with_explanation=with_explanation)
        prediction_result = results[0]
        
        # Add metadata
        prediction_result["timestamp"] = datetime.now().isoformat()
        prediction_result["model_version"] = "2.0.0"
        
        # Add to history
        history_manager.add_transaction(
            payload, 
            prediction_result, 
            input_method="api_single"
        )
        
        return {
            "success": True,
            "prediction": prediction_result,
            "model_info": model_manager.get_model_analytics()["hybrid_model_info"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/hybrid/batch")
async def predict_hybrid_batch(
    file_bytes: bytes,
    with_explanation: bool = Query(False, description="Include detailed explanation and feature contributions"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Fraud detection threshold"),
    chunksize: Optional[int] = Query(None, ge=100, le=200000, description="Optional chunk size for streaming batch processing")
):
    """
    Enhanced batch prediction with optional explainability
    """
    t0 = time.perf_counter()
    # Timers breakdown (support both chunked and non-chunked)
    read_ms = pre_overhead_ms = predict_ms = post_ms = 0.0
    total_rows = 0
    results: List[Dict[str, Any]] = []

    try:
        if chunksize:
            # Chunked processing
            t_read_start = time.perf_counter()
            try:
                reader = pd.read_csv(io.BytesIO(file_bytes), engine="pyarrow", chunksize=chunksize)
                use_pyarrow = True
            except Exception:
                reader = pd.read_csv(io.BytesIO(file_bytes), chunksize=chunksize)
                use_pyarrow = False
            t_read_end = time.perf_counter()
            read_ms += (t_read_end - t_read_start) * 1000

            for chunk in reader:
                # Lightweight downcast to reduce mem and speed ops
                t_pre_start = time.perf_counter()
                try:
                    chunk = chunk.apply(pd.to_numeric, errors='ignore', downcast='float')
                    chunk = chunk.apply(pd.to_numeric, errors='ignore', downcast='integer')
                except Exception:
                    pass
                t_pre_end = time.perf_counter()
                pre_overhead_ms += (t_pre_end - t_pre_start) * 1000

                t_pred_start = time.perf_counter()
                chunk_results = model_manager.hybrid_predict(chunk, threshold=threshold, with_explanation=with_explanation)
                t_pred_end = time.perf_counter()
                predict_ms += (t_pred_end - t_pred_start) * 1000

                results.extend(chunk_results)
                total_rows += len(chunk_results)

            t_post_start = time.perf_counter()
            suspicious = sorted(results, key=lambda x: x['probability'], reverse=True)[:5]
            t_post_end = time.perf_counter()
            post_ms += (t_post_end - t_post_start) * 1000
        else:
            # Non-chunked fast path
            t_read_start = time.perf_counter()
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), engine="pyarrow")
            except Exception:
                df = pd.read_csv(io.BytesIO(file_bytes))
            t_read_end = time.perf_counter()
            read_ms += (t_read_end - t_read_start) * 1000

            t_pre_start = time.perf_counter()
            try:
                df = df.apply(pd.to_numeric, errors='ignore', downcast='float')
                df = df.apply(pd.to_numeric, errors='ignore', downcast='integer')
            except Exception:
                pass
            t_pre_end = time.perf_counter()
            pre_overhead_ms += (t_pre_end - t_read_end) * 1000

            t_pred_start = time.perf_counter()
            results = model_manager.hybrid_predict(df, threshold=threshold, with_explanation=with_explanation)
            t_pred_end = time.perf_counter()
            predict_ms += (t_pred_end - t_pred_start) * 1000
            total_rows = len(results)

            t_post_start = time.perf_counter()
            suspicious = sorted(results, key=lambda x: x['probability'], reverse=True)[:5]
            t_post_end = time.perf_counter()
            post_ms += (t_post_end - t_post_start) * 1000

        fraud_count = sum(1 for r in results if r['prediction'] == 1)
        fraud_rate = fraud_count / total_rows if total_rows > 0 else 0
        t_end = time.perf_counter()

        return {
            "success": True,
            "batch_analytics": {
                "total_transactions": total_rows,
                "fraud_count": fraud_count,
                "fraud_rate": fraud_rate,
                "threshold_used": threshold
            },
            "predictions": results,
            "top_suspicious": suspicious,
            "timestamp": datetime.now().isoformat(),
            "model_info": model_manager.get_model_analytics()["hybrid_model_info"],
            "performance": {
                "read_csv_ms": round(read_ms, 2),
                "preprocess_overhead_ms": round(pre_overhead_ms, 2),
                "predict_ms": round(predict_ms, 2),
                "postprocess_ms": round(post_ms, 2),
                "total_ms": round((t_end - t0) * 1000, 2),
                "chunked": bool(chunksize)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/hybrid/manual")
async def predict_hybrid_manual(
    amount: float = Query(..., description="Transaction amount"),
    v1: Optional[float] = Query(None, description="V1 feature value"),
    v2: Optional[float] = Query(None, description="V2 feature value"),
    v3: Optional[float] = Query(None, description="V3 feature value"),
    v4: Optional[float] = Query(None, description="V4 feature value"),
    v5: Optional[float] = Query(None, description="V5 feature value"),
    v6: Optional[float] = Query(None, description="V6 feature value"),
    v7: Optional[float] = Query(None, description="V7 feature value"),
    v8: Optional[float] = Query(None, description="V8 feature value"),
    v9: Optional[float] = Query(None, description="V9 feature value"),
    v10: Optional[float] = Query(None, description="V10 feature value"),
    v11: Optional[float] = Query(None, description="V11 feature value"),
    v12: Optional[float] = Query(None, description="V12 feature value"),
    v13: Optional[float] = Query(None, description="V13 feature value"),
    v14: Optional[float] = Query(None, description="V14 feature value"),
    v15: Optional[float] = Query(None, description="V15 feature value"),
    v16: Optional[float] = Query(None, description="V16 feature value"),
    v17: Optional[float] = Query(None, description="V17 feature value"),
    v18: Optional[float] = Query(None, description="V18 feature value"),
    v19: Optional[float] = Query(None, description="V19 feature value"),
    v20: Optional[float] = Query(None, description="V20 feature value"),
    v21: Optional[float] = Query(None, description="V21 feature value"),
    v22: Optional[float] = Query(None, description="V22 feature value"),
    v23: Optional[float] = Query(None, description="V23 feature value"),
    v24: Optional[float] = Query(None, description="V24 feature value"),
    v25: Optional[float] = Query(None, description="V25 feature value"),
    v26: Optional[float] = Query(None, description="V26 feature value"),
    v27: Optional[float] = Query(None, description="V27 feature value"),
    v28: Optional[float] = Query(None, description="V28 feature value"),
    with_explanation: bool = Query(True, description="Include detailed explanation"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Fraud detection threshold")
):
    """
    Manual transaction input for real-time fraud checking
    """
    try:
        # Build transaction data
        transaction_data = {"Amount": amount}
        
        # Add V features if provided
        for i in range(1, 29):
            v_value = locals().get(f"v{i}")
            if v_value is not None:
                transaction_data[f"V{i}"] = v_value
        
        # Create DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Get prediction
        results = model_manager.hybrid_predict(df, threshold=threshold, with_explanation=with_explanation)
        prediction_result = results[0]
        
        # Add metadata
        prediction_result["timestamp"] = datetime.now().isoformat()
        prediction_result["input_method"] = "manual"
        prediction_result["model_version"] = "2.0.0"
        
        # Add to history
        history_manager.add_transaction(
            transaction_data, 
            prediction_result, 
            input_method="manual"
        )
        
        return {
            "success": True,
            "prediction": prediction_result,
            "model_info": model_manager.get_model_analytics()["hybrid_model_info"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/hybrid/model-info")
async def get_model_info():
    """Get comprehensive model information and analytics"""
    try:
        analytics = model_manager.get_model_analytics()
        return {
            "success": True,
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/hybrid/health")
async def health_check():
    """Health check for the hybrid prediction system"""
    try:
        analytics = model_manager.get_model_analytics()
        hybrid_available = "hybrid_model_info" in analytics and "error" not in analytics
        
        return {
            "status": "healthy" if hybrid_available else "degraded",
            "hybrid_model_available": hybrid_available,
            "models_loaded": analytics.get("available_models", []),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/predict/hybrid/shap")
async def get_shap_explanation(
    payload: Dict[str, Any],
    model_type: str = Query("hybrid", description="Model type: dt, lstm, or hybrid")
):
    """
    Get SHAP explanation for a single transaction
    """
    try:
        df = pd.DataFrame([payload])
        X_seq, X_flat, feature_list = model_manager.preprocess_input(df)
        
        if model_manager.hybrid is None:
            raise HTTPException(status_code=500, detail="Hybrid model not available")
        
        shap_explanation = model_manager.hybrid.get_shap_explanation(X_flat, model_type)
        
        return {
            "success": True,
            "shap_explanation": shap_explanation,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/hybrid/shap/batch")
async def get_batch_shap_explanations(
    file_bytes: bytes,
    model_type: str = Query("hybrid", description="Model type: dt, lstm, or hybrid"),
    max_samples: int = Query(100, ge=1, le=500, description="Maximum samples to explain")
):
    """
    Get SHAP explanations for a batch of transactions
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        X_seq, X_flat, feature_list = model_manager.preprocess_input(df)
        
        if model_manager.hybrid is None:
            raise HTTPException(status_code=500, detail="Hybrid model not available")
        
        shap_explanations = model_manager.hybrid.get_batch_shap_explanations(
            X_flat, model_type, max_samples
        )
        
        return {
            "success": True,
            "shap_explanations": shap_explanations,
            "model_type": model_type,
            "samples_explained": len(shap_explanations.get('explanations', [])),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/hybrid/history")
async def get_transaction_history(limit: int = Query(50, ge=1, le=500, description="Number of recent transactions to retrieve")):
    """Get recent transaction history"""
    try:
        history = history_manager.get_recent_transactions(limit)
        return {
            "success": True,
            "transactions": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/hybrid/analytics")
async def get_transaction_analytics(days: int = Query(7, ge=1, le=30, description="Number of days for analytics")):
    """Get transaction analytics and patterns"""
    try:
        daily_analytics = history_manager.get_daily_analytics(days)
        fraud_patterns = history_manager.get_fraud_patterns()
        total_stats = history_manager.get_total_stats()
        
        return {
            "success": True,
            "daily_analytics": daily_analytics,
            "fraud_patterns": fraud_patterns,
            "total_stats": total_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
