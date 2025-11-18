# app/routes/data_upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Dict, Any, List, Optional
import pandas as pd
import io
import numpy as np
from datetime import datetime
import json

from app.models.model_manager import model_manager

router = APIRouter()

def advanced_clean_on_spot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Advanced data cleaning with detailed reporting
    """
    original_shape = df.shape
    cleaning_report = {
        "original_rows": original_shape[0],
        "original_columns": original_shape[1],
        "cleaning_steps": []
    }
    
    df_clean = df.copy()
    
    # Step 1: Remove completely empty rows
    empty_rows = df_clean.isnull().all(axis=1).sum()
    if empty_rows > 0:
        df_clean = df_clean.dropna(how='all')
        cleaning_report["cleaning_steps"].append(f"Removed {empty_rows} completely empty rows")
    
    # Step 2: Remove duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        cleaning_report["cleaning_steps"].append(f"Removed {duplicates} duplicate rows")
    
    # Step 3: Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    missing_before = df_clean[numeric_cols].isnull().sum().sum()
    
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            # Use median for numeric columns
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0  # Fallback to 0 if all values are NaN
            df_clean[col] = df_clean[col].fillna(median_val)
    
    missing_after = df_clean[numeric_cols].isnull().sum().sum()
    if missing_before > 0:
        cleaning_report["cleaning_steps"].append(f"Filled {missing_before - missing_after} missing numeric values with median")
    
    # Step 4: Handle outliers (optional - just flag them)
    outlier_count = 0
    for col in numeric_cols:
        if col != 'Amount':  # Skip Amount for now
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outlier_count += outliers
    
    if outlier_count > 0:
        cleaning_report["cleaning_steps"].append(f"Detected {outlier_count} potential outliers (not removed)")
    
    # Step 5: Ensure required columns exist
    required_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    missing_required = [col for col in required_cols if col not in df_clean.columns]
    if missing_required:
        for col in missing_required:
            df_clean[col] = 0.0
        cleaning_report["cleaning_steps"].append(f"Added missing required columns: {missing_required}")
    
    cleaning_report["final_rows"] = df_clean.shape[0]
    cleaning_report["final_columns"] = df_clean.shape[1]
    cleaning_report["rows_removed"] = original_shape[0] - df_clean.shape[0]
    
    return df_clean, cleaning_report

def generate_data_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate insights about the uploaded data"""
    insights = {
        "data_summary": {
            "total_transactions": len(df),
            "date_range": "Not available",  # Could be enhanced with time features
            "amount_stats": {}
        },
        "feature_analysis": {},
        "data_quality": {}
    }
    
    # Amount statistics
    if 'Amount' in df.columns:
        insights["data_summary"]["amount_stats"] = {
            "min": float(df['Amount'].min()),
            "max": float(df['Amount'].max()),
            "mean": float(df['Amount'].mean()),
            "median": float(df['Amount'].median()),
            "std": float(df['Amount'].std())
        }
    
    # V features analysis
    v_cols = [f'V{i}' for i in range(1, 29) if f'V{i}' in df.columns]
    if v_cols:
        v_data = df[v_cols]
        insights["feature_analysis"] = {
            "v_features_available": len(v_cols),
            "v_features_mean": float(v_data.mean().mean()),
            "v_features_std": float(v_data.std().mean()),
            "v_features_range": float(v_data.max().max() - v_data.min().min())
        }
    
    # Data quality metrics
    insights["data_quality"] = {
        "completeness": float((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
        "duplicate_rate": float(df.duplicated().sum() / len(df) * 100),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns)
    }
    
    return insights

@router.post("/upload/hybrid")
async def upload_and_check_hybrid(
    file: UploadFile = File(...),
    with_explanation: bool = Query(False, description="Include detailed explanations"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Fraud detection threshold"),
    include_insights: bool = Query(True, description="Include data insights and analytics")
):
    """
    Enhanced CSV upload with advanced cleaning, hybrid prediction, and comprehensive analytics
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV upload: {e}")

    # Advanced cleaning
    df_clean, cleaning_report = advanced_clean_on_spot(df)

    # Generate data insights
    data_insights = generate_data_insights(df_clean) if include_insights else {}

    # Run hybrid predictions
    try:
        preds = model_manager.hybrid_predict(df_clean, threshold=threshold, with_explanation=with_explanation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid prediction failed: {e}")

    # Attach predictions to df for readability
    df_out = df_clean.copy()
    df_out['fraud_prediction'] = [p['prediction'] for p in preds]
    df_out['fraud_probability'] = [p['probability'] for p in preds]
    df_out['dominant_model'] = [p['dominant_model'] for p in preds]

    # Enhanced analytics
    total = len(preds)
    frauds = sum([p['prediction'] for p in preds])
    fraud_rate = frauds / total if total > 0 else 0.0
    
    # Risk distribution
    risk_levels = {
        "low_risk": sum(1 for p in preds if p['probability'] < 0.3),
        "medium_risk": sum(1 for p in preds if 0.3 <= p['probability'] < 0.7),
        "high_risk": sum(1 for p in preds if p['probability'] >= 0.7)
    }
    
    # Top suspicious transactions
    top_suspicious = sorted(preds, key=lambda x: x['probability'], reverse=True)[:10]
    
    # Model performance insights
    model_insights = {
        "lstm_dominant": sum(1 for p in preds if p.get('dominant_model') == 'lstm'),
        "dt_dominant": sum(1 for p in preds if p.get('dominant_model') == 'dt'),
        "avg_alpha": np.mean([p.get('alpha_used', 0.6) for p in preds if 'alpha_used' in p]) if any('alpha_used' in p for p in preds) else 0.6
    }

    return {
        "success": True,
        "upload_info": {
            "filename": file.filename,
            "file_size_bytes": len(content),
            "upload_timestamp": datetime.now().isoformat()
        },
        "cleaning_report": cleaning_report,
        "data_insights": data_insights,
        "prediction_analytics": {
            "total_transactions": int(total),
            "fraud_count": int(frauds),
            "fraud_rate": float(fraud_rate),
            "risk_distribution": risk_levels,
            "threshold_used": threshold
        },
        "model_insights": model_insights,
        "top_suspicious": top_suspicious,
        "predictions": df_out.to_dict(orient='records'),
        "model_info": model_manager.get_model_analytics()["hybrid_model_info"]
    }

@router.post("/upload/validate")
async def validate_upload_file(file: UploadFile = File(...)):
    """
    Validate uploaded file without running predictions
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")
    
    # Basic validation
    validation_result = {
        "valid": True,
        "issues": [],
        "recommendations": []
    }
    
    # Check required columns
    required_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_result["issues"].append(f"Missing required columns: {missing_cols}")
        validation_result["recommendations"].append("Add missing columns with default values (0.0)")
    
    # Check data types
    if 'Amount' in df.columns and not pd.api.types.is_numeric_dtype(df['Amount']):
        validation_result["issues"].append("Amount column should be numeric")
        validation_result["recommendations"].append("Convert Amount column to numeric type")
    
    # Check for empty file
    if df.empty:
        validation_result["valid"] = False
        validation_result["issues"].append("File is empty")
    
    # Check data quality
    missing_data_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_data_pct > 50:
        validation_result["issues"].append(f"High percentage of missing data: {missing_data_pct:.1f}%")
        validation_result["recommendations"].append("Consider cleaning the data before upload")
    
    validation_result["file_info"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_data_percentage": missing_data_pct,
        "file_size_bytes": len(content)
    }
    
    return validation_result
