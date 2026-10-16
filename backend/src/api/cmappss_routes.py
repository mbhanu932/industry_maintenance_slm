"""
NASA C-MAPSS API Routes

Extended API endpoints for NASA C-MAPSS turbofan engine data.

Provides endpoints for:
- Engine health monitoring
- Sensor data visualization
- RUL prediction
- Anomaly detection
- Fleet overview
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

router = APIRouter(prefix="/api/cmappss", tags=["C-MAPSS"])

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class EngineHealthResponse(BaseModel):
    """Engine health status response."""
    engine_id: int
    current_cycle: int
    rul: float
    health_score: float
    risk_level: str
    anomaly_score: float
    anomaly_status: str
    failure_probability_24h: float
    failure_probability_48h: float
    max_cycle: int


class SensorDataResponse(BaseModel):
    """Sensor data response."""
    engine_id: int
    cycle: int
    sensors: Dict[str, float]
    operational_settings: Dict[str, float]


class FleetOverviewResponse(BaseModel):
    """Fleet overview response."""
    total_engines: int
    healthy_engines: int
    attention_engines: int
    high_risk_engines: int
    critical_engines: int
    avg_health_score: float
    avg_rul: float
    total_anomalies: int


class RULPredictionRequest(BaseModel):
    """Request for RUL prediction."""
    engine_id: int
    sensor_data: List[Dict[str, float]]


class RULPredictionResponse(BaseModel):
    """RUL prediction response."""
    engine_id: int
    predicted_rul: float
    confidence: float
    risk_level: str
    recommended_action: str


# ═══════════════════════════════════════════════════════════════════════════════
# SENSOR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# NASA C-MAPSS sensor names and descriptions
SENSOR_METADATA = {
    "sensor_1": {"name": "T30", "description": "Total temperature at HPC outlet", "unit": "°R"},
    "sensor_2": {"name": "T50", "description": "Total temperature at LPT outlet", "unit": "°R"},
    "sensor_3": {"name": "P30", "description": "Total pressure at HPC outlet", "unit": "psia"},
    "sensor_4": {"name": "Ps30", "description": "Static pressure at HPC outlet", "unit": "psia"},
    "sensor_5": {"name": "phi", "description": "Ratio of fuel flow to Ps30", "unit": "pps/psi"},
    "sensor_6": {"name": "Nr", "description": "Corrected core speed", "unit": "rpm"},
    "sensor_7": {"name": "BPR", "description": "Bypass Ratio", "unit": ""},
    "sensor_8": {"name": "htBleed", "description": "Bleed Enthalpy", "unit": ""},
    "sensor_9": {"name": "W31", "description": "HPT coolant bleed", "unit": "lbm/s"},
    "sensor_10": {"name": "W32", "description": "LPT coolant bleed", "unit": "lbm/s"},
    "sensor_11": {"name": "T24", "description": "Total temperature at LPC outlet", "unit": "°R"},
    "sensor_12": {"name": "T50_alt", "description": "Total temperature at LPT outlet (alt)", "unit": "°R"},
    "sensor_13": {"name": "P20", "description": "Total pressure in bypass-duct", "unit": "psia"},
    "sensor_14": {"name": "P50", "description": "Total pressure at LPT outlet", "unit": "psia"},
    "sensor_15": {"name": "Nf", "description": "Physical fan speed", "unit": "rpm"},
    "sensor_16": {"name": "Nc", "description": "Physical core speed", "unit": "rpm"},
    "sensor_17": {"name": "epr", "description": "Engine pressure ratio", "unit": ""},
    "sensor_18": {"name": "farB", "description": "Burner fuel-air ratio", "unit": ""},
    "sensor_19": {"name": "Nf_dmd", "description": "Demanded fan speed", "unit": "rpm"},
    "sensor_20": {"name": "PCNfR_dmd", "description": "Demanded corrected fan speed", "unit": "rpm"},
    "sensor_21": {"name": "smFan", "description": "Fan stall margin", "unit": ""},
}

# Key sensors for health monitoring
KEY_SENSORS = [
    "sensor_2",   # T50 - LPT temperature
    "sensor_3",   # P30 - HPC pressure
    "sensor_4",   # Ps30 - HPC static pressure
    "sensor_7",   # BPR - Bypass ratio
    "sensor_11",  # T24 - LPC temperature
]


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL DATA STORE (In production, use database)
# ═══════════════════════════════════════════════════════════════════════════════

_data_store = {
    'train_df': None,
    'test_df': None,
    'rul_df': None
}


def load_cmappss_data():
    """Load C-MAPSS data into memory."""
    data_path = Path("data/processed")
    
    # Try to load processed data
    train_file = data_path / "cmappss_fd001_train.parquet"
    test_file = data_path / "cmappss_fd001_test.parquet"
    rul_file = data_path / "cmappss_fd001_rul.parquet"
    
    if train_file.exists() and test_file.exists():
        _data_store['train_df'] = pd.read_parquet(train_file)
        _data_store['test_df'] = pd.read_parquet(test_file)
        
        if rul_file.exists():
            _data_store['rul_df'] = pd.read_parquet(rul_file)
        
        return True
    
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/engines")
async def get_engines():
    """Get list of all engine IDs."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    train_engines = sorted(_data_store['train_df']['unit_number'].unique().tolist())
    test_engines = sorted(_data_store['test_df']['unit_number'].unique().tolist()) if _data_store['test_df'] is not None else []
    
    return {
        "train_engines": train_engines,
        "test_engines": test_engines,
        "total_train": len(train_engines),
        "total_test": len(test_engines)
    }


@router.get("/engines/{engine_id}/health", response_model=EngineHealthResponse)
async def get_engine_health(engine_id: int):
    """Get current health status of an engine."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    # Get latest cycle data for this engine
    engine_data = _data_store['train_df'][_data_store['train_df']['unit_number'] == engine_id]
    
    if engine_data.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")
    
    latest = engine_data.iloc[-1]
    max_cycle = engine_data['cycle'].max()
    
    return EngineHealthResponse(
        engine_id=int(engine_id),
        current_cycle=int(latest['cycle']),
        rul=float(latest.get('rul', 0)),
        health_score=float(latest.get('health_score', 100)),
        risk_level=str(latest.get('risk_level', 'NORMAL')),
        anomaly_score=float(latest.get('anomaly_score', 0)),
        anomaly_status=str(latest.get('anomaly_status', 'NORMAL')),
        failure_probability_24h=float(latest.get('failure_probability_24h', 0)),
        failure_probability_48h=float(latest.get('failure_probability_48h', 0)),
        max_cycle=int(max_cycle)
    )


@router.get("/engines/{engine_id}/sensors")
async def get_engine_sensors(engine_id: int, cycle: Optional[int] = None):
    """Get sensor data for an engine at a specific cycle."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    engine_data = _data_store['train_df'][_data_store['train_df']['unit_number'] == engine_id]
    
    if engine_data.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")
    
    # Get specific cycle or latest
    if cycle is not None:
        cycle_data = engine_data[engine_data['cycle'] == cycle]
        if cycle_data.empty:
            raise HTTPException(status_code=404, detail=f"Cycle {cycle} not found for engine {engine_id}")
        data = cycle_data.iloc[0]
    else:
        data = engine_data.iloc[-1]
    
    # Extract sensor values
    sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
    sensors = {col: float(data[col]) for col in sensor_cols if col in data}
    
    # Extract operational settings
    op_settings = {
        'op_setting_1': float(data.get('op_setting_1', 0)),
        'op_setting_2': float(data.get('op_setting_2', 0)),
        'op_setting_3': float(data.get('op_setting_3', 0))
    }
    
    return {
        "engine_id": engine_id,
        "cycle": int(data['cycle']),
        "sensors": sensors,
        "operational_settings": op_settings,
        "sensor_metadata": SENSOR_METADATA
    }


@router.get("/engines/{engine_id}/sensors/trend")
async def get_sensor_trend(engine_id: int, sensor_id: str = "sensor_2"):
    """Get historical trend for a specific sensor."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    engine_data = _data_store['train_df'][_data_store['train_df']['unit_number'] == engine_id]
    
    if engine_data.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")
    
    if sensor_id not in engine_data.columns:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
    
    trend_data = engine_data[['cycle', sensor_id]].to_dict('records')
    
    return {
        "engine_id": engine_id,
        "sensor_id": sensor_id,
        "sensor_name": SENSOR_METADATA.get(sensor_id, {}).get('name', sensor_id),
        "unit": SENSOR_METADATA.get(sensor_id, {}).get('unit', ''),
        "trend": trend_data
    }


@router.get("/engines/{engine_id}/sensors/all/trend")
async def get_all_sensors_trend(engine_id: int):
    """Get historical trend for all sensors."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    engine_data = _data_store['train_df'][_data_store['train_df']['unit_number'] == engine_id]
    
    if engine_data.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")
    
    # Get all sensor columns
    sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
    
    trends = {}
    for sensor in sensor_cols[:14]:  # First 14 sensors
        trends[sensor] = {
            "name": SENSOR_METADATA.get(sensor, {}).get('name', sensor),
            "unit": SENSOR_METADATA.get(sensor, {}).get('unit', ''),
            "data": engine_data[['cycle', sensor]].to_dict('records')
        }
    
    return {
        "engine_id": engine_id,
        "sensors": trends
    }


@router.get("/fleet/overview", response_model=FleetOverviewResponse)
async def get_fleet_overview():
    """Get overall fleet health statistics."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    df = _data_store['train_df']
    
    # Get latest data for each engine
    latest_data = df.groupby('unit_number').last().reset_index()
    
    # Count by risk level
    risk_counts = latest_data['risk_level'].value_counts().to_dict()
    
    return FleetOverviewResponse(
        total_engines=int(latest_data['unit_number'].nunique()),
        healthy_engines=int(risk_counts.get('NORMAL', 0)),
        attention_engines=int(risk_counts.get('ATTENTION', 0)),
        high_risk_engines=int(risk_counts.get('HIGH', 0)),
        critical_engines=int(risk_counts.get('CRITICAL', 0)),
        avg_health_score=float(latest_data['health_score'].mean()),
        avg_rul=float(latest_data['rul'].mean()),
        total_anomalies=int((latest_data['anomaly_status'] == 'ANOMALOUS').sum())
    )


@router.get("/fleet/heatmap")
async def get_fleet_heatmap():
    """Get fleet heatmap data for visualization."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    df = _data_store['train_df']
    
    # Get latest data for each engine
    latest_data = df.groupby('unit_number').last().reset_index()
    
    engines = []
    for _, row in latest_data.iterrows():
        engines.append({
            "id": int(row['unit_number']),
            "cycle": int(row['cycle']),
            "rul": float(row['rul']),
            "health_score": float(row['health_score']),
            "risk_level": str(row['risk_level']),
            "anomaly_score": float(row['anomaly_score'])
        })
    
    return {
        "engines": engines,
        "total": len(engines)
    }


@router.get("/fleet/alerts")
async def get_fleet_alerts():
    """Get active alerts across the fleet."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            raise HTTPException(status_code=503, detail="C-MAPSS data not loaded")
    
    df = _data_store['train_df']
    latest_data = df.groupby('unit_number').last().reset_index()
    
    alerts = []
    
    # Critical RUL alerts
    critical = latest_data[latest_data['rul'] < 20]
    for _, row in critical.iterrows():
        alerts.append({
            "id": f"ALERT_{row['unit_number']}_CRITICAL",
            "engine_id": int(row['unit_number']),
            "type": "CRITICAL_RUL",
            "message": f"Engine {int(row['unit_number'])} has RUL of {int(row['rul'])} cycles",
            "severity": "critical",
            "timestamp": "2024-10-05T00:00:00Z"
        })
    
    # High risk alerts
    high_risk = latest_data[latest_data['risk_level'] == 'HIGH']
    for _, row in high_risk.iterrows():
        alerts.append({
            "id": f"ALERT_{row['unit_number']}_HIGH",
            "engine_id": int(row['unit_number']),
            "type": "HIGH_RISK",
            "message": f"Engine {int(row['unit_number'])} classified as HIGH risk",
            "severity": "high",
            "timestamp": "2024-10-05T00:00:00Z"
        })
    
    # Anomaly alerts
    anomalies = latest_data[latest_data['anomaly_status'] == 'ANOMALOUS']
    for _, row in anomalies.iterrows():
        alerts.append({
            "id": f"ALERT_{row['unit_number']}_ANOMALY",
            "engine_id": int(row['unit_number']),
            "type": "ANOMALY_DETECTED",
            "message": f"Engine {int(row['unit_number'])} showing anomalous behavior",
            "severity": "medium",
            "timestamp": "2024-10-05T00:00:00Z"
        })
    
    return {
        "alerts": alerts,
        "total": len(alerts)
    }


@router.post("/predict/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """Predict RUL for an engine based on sensor data."""
    # This would integrate with the RUL predictor model
    # For now, return a placeholder response
    
    return RULPredictionResponse(
        engine_id=request.engine_id,
        predicted_rul=75.5,  # Placeholder
        confidence=0.85,
        risk_level="ATTENTION",
        recommended_action="Schedule inspection within next 50 cycles"
    )


@router.get("/metadata/sensors")
async def get_sensor_metadata():
    """Get metadata for all sensors."""
    return {
        "sensors": SENSOR_METADATA,
        "key_sensors": KEY_SENSORS
    }


@router.get("/dataset/info")
async def get_dataset_info():
    """Get information about the loaded C-MAPSS dataset."""
    if _data_store['train_df'] is None:
        if not load_cmappss_data():
            return {
                "loaded": False,
                "message": "No C-MAPSS data loaded. Run preprocessor first."
            }
    
    df = _data_store['train_df']
    
    return {
        "loaded": True,
        "dataset": "NASA C-MAPSS FD001",
        "total_engines": int(df['unit_number'].nunique()),
        "total_cycles": int(len(df)),
        "avg_cycles_per_engine": float(df.groupby('unit_number')['cycle'].max().mean()),
        "columns": list(df.columns),
        "risk_distribution": df.groupby('unit_number').last()['risk_level'].value_counts().to_dict()
    }
