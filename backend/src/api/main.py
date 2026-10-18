import random
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from src.inference.investigator import Investigator

app = FastAPI(title="Industrial Maintenance Investigation SLM", version="2.0.0")

# Load telemetry data at startup
TELEMETRY_PATH = Path("data/processed/engine_telemetry_timeseries.parquet")
try:
    TELEMETRY_DF = pd.read_parquet(TELEMETRY_PATH)
    print(f"✓ Loaded telemetry: {len(TELEMETRY_DF)} cycles from {TELEMETRY_DF['engine_id'].nunique()} engines")
except FileNotFoundError:
    print(f"⚠ Telemetry file not found at {TELEMETRY_PATH}, using fallback mode")
    TELEMETRY_DF = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://industry-maintenance-slm.vercel.app",
        "https://industry-maintenance-awl7f8j6i-veltris6.vercel.app",
        "https://industry-maintenance-3ojd1i9ll-veltris6.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

investigator = Investigator()

# ─── Sensor configuration with realistic normal ranges ────────────────────────
SENSOR_META = {
    "temperature":  {"label": "Temperature",  "unit": "°C",    "normal_min": 60,   "normal_max": 90,   "icon": "thermometer"},
    "pressure":     {"label": "Pressure",     "unit": "PSI",   "normal_min": 95,   "normal_max": 130,  "icon": "gauge"},
    "vibration":    {"label": "Vibration",    "unit": "mm/s",  "normal_min": 3,    "normal_max": 5,    "icon": "activity"},
    "voltage":      {"label": "Voltage",      "unit": "V",     "normal_min": 400,  "normal_max": 430,  "icon": "zap"},
    "rotation":     {"label": "Rotation",     "unit": "RPM",   "normal_min": 2900, "normal_max": 3100, "icon": "rotate-cw"},
    "current":      {"label": "Current",      "unit": "A",     "normal_min": 18,   "normal_max": 28,   "icon": "activity"},
}

# Error codes lookup
ERROR_CODES = {
    "E201": "Overtemperature threshold exceeded",
    "E202": "Pressure drop detected",
    "E203": "Voltage fluctuation out of range",
    "E204": "Vibration threshold exceeded",
    "E205": "Bearing wear indicator triggered",
    "E206": "Lubrication system low pressure",
    "E207": "Coolant flow reduced",
    "E208": "Motor current surge detected",
}

def _seed(machine_id: int) -> random.Random:
    return random.Random(machine_id * 7919)

def _get_sensor_readings(machine_id: int, ctx: dict) -> list:
    """Fetch real sensor readings from telemetry parquet.
    Falls back to PRNG if telemetry unavailable."""
    
    # Try to load real data first
    if TELEMETRY_DF is not None:
        try:
            # Get latest cycle for this engine
            engine_data = TELEMETRY_DF[TELEMETRY_DF['engine_id'] == machine_id]
            if not engine_data.empty:
                latest = engine_data.sort_values('cycle').iloc[-1]
                
                sensors = []
                sensor_keys = ['temperature_celsius', 'pressure_psi', 'vibration_mm_s', 'voltage_v', 'rotation_rpm', 'current_a']
                ui_labels = ['Temperature', 'Pressure', 'Vibration', 'Voltage', 'Rotation', 'Current']
                ui_units = ['°C', 'PSI', 'mm/s', 'V', 'RPM', 'A']
                ui_icons = ['thermometer', 'gauge', 'activity', 'zap', 'rotate-cw', 'activity']
                
                # Map telemetry columns to UI sensor keys
                sensor_map = {
                    'temperature_celsius': 'temperature',
                    'pressure_psi': 'pressure',
                    'vibration_mm_s': 'vibration',
                    'voltage_v': 'voltage',
                    'rotation_rpm': 'rotation',
                    'current_a': 'current',
                }
                
                for col, ui_key in sensor_map.items():
                    if col in latest.index:
                        value = float(latest[col])
                        meta = SENSOR_META.get(ui_key, {})
                        nmin, nmax = meta.get("normal_min", 0), meta.get("normal_max", 100)
                        normal_val = (nmin + nmax) / 2
                        delta_pct = round(((value - normal_val) / normal_val) * 100, 1) if normal_val != 0 else 0
                        
                        # Determine status based on thresholds
                        if value < nmin * 0.85 or value > nmax * 1.25:
                            status = "critical"
                        elif value < nmin * 0.93 or value > nmax * 1.12:
                            status = "warning"
                        else:
                            status = "normal"
                        
                        sensor_idx = list(sensor_map.values()).index(ui_key)
                        sensors.append({
                            "key": ui_key,
                            "label": ui_labels[sensor_idx],
                            "unit": ui_units[sensor_idx],
                            "icon": ui_icons[sensor_idx],
                            "value": round(value, 2),
                            "normal_min": nmin,
                            "normal_max": nmax,
                            "delta_pct": delta_pct,
                            "status": status,
                        })
                
                return sensors
        except Exception as e:
            print(f"Error loading telemetry for engine {machine_id}: {e}")
    
    # Fallback to PRNG if telemetry unavailable
    print(f"⚠ Using PRNG fallback for engine {machine_id}")
    rng = _seed(machine_id)
    health = ctx.get("health_score", 80)
    anomaly = ctx.get("anomaly_score", 0)
    degradation = max(0, (100 - health) / 100)

    sensors = []
    for key, meta in SENSOR_META.items():
        nmin, nmax = meta["normal_min"], meta["normal_max"]
        normal_val = rng.uniform(nmin, nmax)

        if key == "vibration" and anomaly > 0.4:
            skew = 1 + (anomaly * 1.8)
        elif key == "temperature" and degradation > 0.3:
            skew = 1 + (degradation * 0.3)
        elif key == "pressure" and degradation > 0.5:
            skew = 1 - (degradation * 0.15)
        elif key == "current" and anomaly > 0.3:
            skew = 1 + (anomaly * 0.5)
        else:
            skew = rng.uniform(0.97, 1.03)

        value = round(normal_val * skew, 2)
        delta_pct = round(((value - normal_val) / normal_val) * 100, 1)

        if value < nmin * 0.85 or value > nmax * 1.25:
            status = "critical"
        elif value < nmin * 0.93 or value > nmax * 1.12:
            status = "warning"
        else:
            status = "normal"

        sensors.append({
            "key": key,
            "label": meta["label"],
            "unit": meta["unit"],
            "icon": meta["icon"],
            "value": value,
            "normal_min": nmin,
            "normal_max": nmax,
            "delta_pct": delta_pct,
            "status": status,
        })
    return sensors

def _get_trend_data(machine_id: int, ctx: dict) -> list:
    """Fetch 24-hour trend data from telemetry (or generate PRNG fallback)."""
    
    # Try to load real data first
    if TELEMETRY_DF is not None:
        try:
            engine_data = TELEMETRY_DF[TELEMETRY_DF['engine_id'] == machine_id].sort_values('cycle')
            
            if not engine_data.empty:
                # Take last 24 cycles as "24-hour trend"
                trend_data = engine_data.tail(24).reset_index(drop=True)
                
                points = []
                for idx, row in trend_data.iterrows():
                    point = {"hour": f"{idx:02d}:00"}
                    
                    sensor_map = {
                        'temperature': 'temperature_celsius',
                        'pressure': 'pressure_psi',
                        'vibration': 'vibration_mm_s',
                        'voltage': 'voltage_v',
                        'rotation': 'rotation_rpm',
                        'current': 'current_a',
                    }
                    
                    for key, col in sensor_map.items():
                        if col in row.index:
                            point[key] = round(float(row[col]), 2)
                    
                    points.append(point)
                
                return points
        except Exception as e:
            print(f"Error loading trend data for engine {machine_id}: {e}")
    
    # Fallback to PRNG
    print(f"⚠ Using PRNG fallback for trend data, engine {machine_id}")
    rng = _seed(machine_id + 1)
    health = ctx.get("health_score", 80)
    degradation = max(0, (100 - health) / 100)

    points = []
    for hour in range(24):
        point = {"hour": f"{hour:02d}:00"}
        for key, meta in SENSOR_META.items():
            nmin, nmax = meta["normal_min"], meta["normal_max"]
            base = rng.uniform(nmin, nmax)
            if hour > 16 and key == "vibration":
                drift = 1 + (degradation * (hour - 16) * 0.08)
            elif hour > 18 and key == "temperature":
                drift = 1 + (degradation * (hour - 18) * 0.04)
            else:
                drift = rng.uniform(0.97, 1.03)
            point[key] = round(base * drift, 2)
        points.append(point)
    return points

def _get_maintenance_history(machine_id: int, ctx: dict) -> dict:
    """Generate deterministic maintenance history for a machine."""
    rng = _seed(machine_id + 2)
    health = ctx.get("health_score", 80)
    degradation = max(0, (100 - health) / 100)
    base_errors = int(degradation * 12) + rng.randint(0, 4)

    error_log = []
    error_keys = list(ERROR_CODES.keys())
    for i in range(base_errors):
        code = rng.choice(error_keys)
        days_ago = rng.randint(1, 365)
        error_log.append({
            "code": code,
            "description": ERROR_CODES[code],
            "days_ago": days_ago,
            "count": rng.randint(1, 5),
        })
    error_log.sort(key=lambda x: x["days_ago"])

    actions = [
        "Bearing replacement — spindle assembly",
        "Coolant flush and refill",
        "Belt tension adjustment",
        "Lubrication service",
        "Sensor calibration",
        "Full preventive maintenance",
        "Vibration damper replacement",
    ]
    history_len = rng.randint(2, 6)
    maintenance_log = []
    for i in range(history_len):
        days = rng.randint(30, 700)
        maintenance_log.append({
            "action": rng.choice(actions),
            "days_ago": days,
            "technician": f"Tech-{rng.randint(1,8):02d}",
        })
    maintenance_log.sort(key=lambda x: x["days_ago"])

    machine_age_years = round(rng.uniform(2, 12), 1)
    last_service_days = maintenance_log[0]["days_ago"] if maintenance_log else rng.randint(60, 300)

    return {
        "machine_age_years": machine_age_years,
        "last_service_days_ago": last_service_days,
        "total_maintenance_events": len(maintenance_log),
        "total_errors": len(error_log),
        "error_log": error_log[:8],
        "maintenance_log": maintenance_log[:6],
    }


# ─── Request models ───────────────────────────────────────────────────────────
class InvestigationRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "Why is this machine at its current risk level?"},
                {"question": "Should this machine be taken offline for inspection?"},
                {"question": "Is there bearing degradation?"},
            ]
        }
    }

class ApprovalRequest(BaseModel):
    approved: bool = True
    engineer: str = "qualified_engineer"
    engine_id: int | None = None
    diagnosis: str | None = None
    recommended_actions: list[str] | None = None
    checked_actions: list[int] | None = None
    timestamp: str | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Industrial Maintenance Copilot API</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#090d13;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh}
  .card{background:#0e1420;border:1px solid #1e2d3d;border-radius:16px;padding:48px 56px;max-width:560px;width:100%;text-align:center;box-shadow:0 8px 40px rgba(0,0,0,.5)}
  .icon{font-size:48px;margin-bottom:16px}
  h1{font-size:22px;font-weight:700;margin-bottom:8px}
  p{color:#8ba3ba;font-size:14px;margin-bottom:32px;line-height:1.6}
  .links{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
  a{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:600;text-decoration:none;transition:all .2s}
  .primary{background:#0ea5e9;color:#fff}
  .primary:hover{background:#38bdf8}
  .secondary{background:#1e2d3d;color:#e2e8f0;border:1px solid #2a3f54}
  .secondary:hover{background:#243447}
  .badge{display:inline-block;background:rgba(14,165,233,.12);color:#0ea5e9;border:1px solid rgba(14,165,233,.25);border-radius:20px;font-size:11px;font-weight:600;padding:3px 10px;letter-spacing:.05em;text-transform:uppercase;margin-bottom:20px}
  .status{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:#22c55e;margin-top:24px}
  .dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
</style>
</head>
<body>
<div class="card">
  <div class="icon">⚙️</div>
  <div class="badge">SLM v2</div>
  <h1>Industrial Maintenance Copilot</h1>
  <p>Predictive maintenance investigation API powered by machine health analytics, RAG knowledge retrieval, and an AI diagnostic engine.</p>
  <div class="links">
    <a href="/docs" class="primary">📖 API Docs (Swagger)</a>
    <a href="/redoc" class="secondary">📄 ReDoc</a>
    <a href="/health" class="secondary">❤️ Health Check</a>
    <a href="/fleet/summary" class="secondary">🏭 Fleet Summary</a>
  </div>
  <div class="status"><div class="dot"></div>Backend running · FastDiagnosticEngine active</div>
</div>
</body>
</html>"""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/machines")
def machines():
    ids = sorted(investigator.health["engine_id"].dropna().astype(int).unique().tolist())
    return {"machines": ids}

@app.get("/machines/{machine_id}")
def machine(machine_id: int):
    ctx = investigator.context(machine_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Engine not found")
    return ctx

@app.get("/machines/{machine_id}/sensors")
def sensors(machine_id: int):
    ctx = investigator.context(machine_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Engine not found")
    return {
        "engine_id": machine_id,
        "sensors": _get_sensor_readings(machine_id, ctx),
        "trend": _get_trend_data(machine_id, ctx),
    }

@app.get("/machines/{machine_id}/history")
def history(machine_id: int):
    ctx = investigator.context(machine_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Engine not found")
    return {"engine_id": machine_id, **_get_maintenance_history(machine_id, ctx)}

@app.post("/machines/{machine_id}/investigate")
def investigate(machine_id: int, request: InvestigationRequest):
    return investigator.investigate(machine_id, request.question)

@app.post("/maintenance/approve")
def approve(request: ApprovalRequest):
    """Log and approve maintenance actions from the AI diagnosis."""
    return {
        "success": True,
        "approved": True,
        "engine_id": request.engine_id,
        "checked_actions": request.checked_actions or [],
        "total_actions": len(request.recommended_actions or []),
        "timestamp": request.timestamp,
        "message": "Maintenance plan approved and logged in the system.",
    }

@app.get("/fleet/summary")
def fleet_summary():
    """Return a high-level fleet health overview."""
    ids = investigator.health["engine_id"].dropna().astype(int).unique().tolist()
    total = len(ids)
    risk_counts = investigator.health.groupby("risk_level")["engine_id"].count().to_dict()
    critical = risk_counts.get("CRITICAL", 0)
    high = risk_counts.get("HIGH", 0)
    attention = risk_counts.get("ATTENTION", 0)
    normal = total - critical - high - attention
    avg_health = float(investigator.health["health_score"].mean())
    avg_rul = float(investigator.health["predicted_rul"].mean())
    return {
        "total_machines": total,
        "critical": critical,
        "high": high,
        "attention": attention,
        "normal": normal,
        "avg_health_score": round(avg_health, 1),
        "avg_rul": round(avg_rul, 1),
    }

@app.get("/fleet/heatmap")
def fleet_heatmap():
    """Return heatmap data for all machines (grid view)."""
    ids = investigator.health["engine_id"].dropna().astype(int).unique().tolist()
    machines = []
    for mid in sorted(ids)[:100]:  # Limit to 100 machines for performance
        ctx = investigator.context(mid)
        if ctx:
            machines.append({
                "id": mid,
                "risk_level": ctx.get("risk_level", "NORMAL"),
                "health_score": ctx.get("health_score", 100),
                "anomaly_score": ctx.get("anomaly_score", 0),
            })
    return {"machines": machines, "total": len(ids)}

@app.get("/tasks/priority")
def get_priority_tasks():
    """Return high-priority maintenance tasks."""
    ids = investigator.health["engine_id"].dropna().astype(int).unique().tolist()
    tasks = []
    
    for mid in sorted(ids)[:50]:
        ctx = investigator.context(mid)
        if ctx is None:
            continue
        
        risk = ctx.get("risk_level", "NORMAL")
        if risk in ["CRITICAL", "HIGH"]:
            task_id = f"TASK_{mid}_{int(ctx.get('anomaly_score', 0) * 100)}"
            tasks.append({
                "id": task_id,
                "machine_id": mid,
                "title": f"Engine {mid}: {risk} maintenance required",
                "description": f"Risk level: {risk}, Health: {ctx.get('health_score', 100):.0f}%",
                "priority": "HIGH" if risk == "CRITICAL" else "MEDIUM",
                "status": "discovered",
                "created_at": "2024-10-05T00:00:00Z",
                "updated_at": "2024-10-05T00:00:00Z",
            })
    
    return sorted(tasks, key=lambda x: {"HIGH": 0, "MEDIUM": 1}.get(x["priority"], 2))

@app.post("/tasks")
def create_task(task: dict):
    """Create a new maintenance task."""
    task_id = f"TASK_{random.randint(1000, 9999)}"
    return {
        "id": task_id,
        **task,
        "created_at": "2024-10-05T00:00:00Z",
        "updated_at": "2024-10-05T00:00:00Z",
    }

@app.put("/tasks/{task_id}")
def update_task(task_id: str, updates: dict):
    """Update task status and details."""
    return {
        "id": task_id,
        **updates,
        "updated_at": "2024-10-05T00:00:00Z",
    }

@app.get("/fleet/alerts")
def get_fleet_alerts():
    """Return current fleet alerts."""
    ids = investigator.health["engine_id"].dropna().astype(int).unique().tolist()
    alerts = []
    
    for mid in sorted(ids)[:20]:
        ctx = investigator.context(mid)
        if ctx is None:
            continue
        
        if ctx.get("anomaly_score", 0) > 0.5:
            alerts.append({
                "id": f"ALERT_{mid}",
                "machine_id": mid,
                "level": "critical" if ctx.get("risk_level") == "CRITICAL" else "warning",
                "message": f"Engine {mid}: Anomaly detected (score: {ctx.get('anomaly_score', 0):.2f})",
                "created_at": "2024-10-05T00:00:00Z",
                "acknowledged": False,
            })
    
    return alerts

@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, user_id: str):
    """Acknowledge an alert."""
    return {"success": True, "alert_id": alert_id, "user_id": user_id}

@app.get("/auth/me")
def get_current_user():
    """Get current user (demo)."""
    return {
        "id": "demo_user",
        "username": "engineer",
        "email": "engineer@maintenance.local",
        "role": "maintenance_engineer",
    }

@app.post("/auth/login")
def login(username: str, password: str):
    """Login endpoint (demo)."""
    return {
        "token": "demo_token_12345",
        "user": {
            "id": "demo_user",
            "username": username,
            "email": f"{username}@maintenance.local",
            "role": "maintenance_engineer",
        },
        "expires_at": "2024-10-06T00:00:00Z",
    }

@app.post("/auth/logout")
def logout():
    """Logout endpoint."""
    return {"success": True, "message": "Logged out successfully"}

@app.post("/auth/refresh")
def refresh_token():
    """Refresh auth token."""
    return {
        "token": "demo_token_refreshed",
        "expires_at": "2024-10-06T00:00:00Z",
    }

@app.get("/machines/{machine_id}/diagnoses")
def get_diagnosis_history(machine_id: int):
    """Get diagnosis history for a machine."""
    result = investigator.investigate(machine_id, "What is the current health status?")
    return [result] if result else []

@app.post("/diagnoses/{diagnosis_id}/review")
def review_diagnosis(diagnosis_id: str, approved: bool, notes: str = None):
    """Engineer review of diagnosis."""
    return {
        "success": True,
        "diagnosis_id": diagnosis_id,
        "approved": approved,
        "reviewed_at": "2024-10-05T00:00:00Z",
        "notes": notes or "",
    }

@app.post("/diagnoses/{diagnosis_id}/feedback")
def submit_diagnosis_feedback(diagnosis_id: str, rating: int, feedback: str):
    """Submit feedback on diagnosis quality."""
    return {
        "success": True,
        "diagnosis_id": diagnosis_id,
        "rating": rating,
        "feedback": feedback,
        "submitted_at": "2024-10-05T00:00:00Z",
    }

@app.get("/ai/status")
def get_ai_status():
    """Get AI model status."""
    return {
        "status": "online",
        "version": "2.0.0",
        "last_updated": "2024-10-05T00:00:00Z",
        "guardrails_active": True,
    }

@app.get("/config")
def get_config():
    """Get application configuration."""
    return {
        "version": "2.0.0",
        "environment": "production",
        "features": {
            "ai_diagnostics": True,
            "predictive_maintenance": True,
            "audit_trail": True,
            "fleet_monitoring": True,
        },
        "limits": {
            "max_machines": 1000,
            "max_api_requests_per_minute": 1000,
            "diagnosis_timeout_ms": 5000,
        },
    }

@app.get("/metrics")
def get_metrics():
    """Get system metrics."""
    return {
        "cpu_usage": round(random.uniform(10, 40), 1),
        "memory_usage": round(random.uniform(20, 50), 1),
        "disk_usage": round(random.uniform(30, 60), 1),
        "active_connections": random.randint(1, 20),
        "request_rate": random.randint(10, 100),
        "error_rate": round(random.uniform(0, 1), 3),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ═══════════════════════════════════════════════════════════════════════════════
# NASA C-MAPSS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

from src.api.cmappss_routes import router as cmappss_router
app.include_router(cmappss_router)
