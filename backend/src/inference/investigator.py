import os
import random
import pandas as pd
from .rag import Retriever
from .slm import SLM, FastDiagnosticEngine, _USE_MODEL

HEALTH_DATA_PATH = os.getenv("HEALTH_DATA_PATH", "data/processed/machine_health.parquet")

# ── Sensor normal ranges (mirrors main.py, kept here for self-contained context) ──
SENSOR_META = {
    "temperature": {"normal_min": 60,   "normal_max": 90,   "unit": "°C"},
    "pressure":    {"normal_min": 95,   "normal_max": 130,  "unit": "PSI"},
    "vibration":   {"normal_min": 3,    "normal_max": 5,    "unit": "mm/s"},
    "voltage":     {"normal_min": 400,  "normal_max": 430,  "unit": "V"},
    "rotation":    {"normal_min": 2900, "normal_max": 3100, "unit": "RPM"},
    "current":     {"normal_min": 18,   "normal_max": 28,   "unit": "A"},
}

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


def _build_sensor_snapshot(machine_id: int, ctx: dict) -> list[dict]:
    """
    Re-derive the same deterministic sensor readings that main.py serves
    so the engine receives per-sensor context without an extra HTTP call.
    """
    rng = _seed(machine_id)
    health  = ctx.get("health_score", 80)
    anomaly = ctx.get("anomaly_score", 0)
    degradation = max(0, (100 - health) / 100)

    sensors = []
    for key, meta in SENSOR_META.items():
        nmin, nmax = meta["normal_min"], meta["normal_max"]
        nominal = rng.uniform(nmin, nmax)

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

        value = round(nominal * skew, 2)
        delta_pct = round(((value - nominal) / nominal) * 100, 1)

        if value < nmin * 0.85 or value > nmax * 1.25:
            status = "critical"
        elif value < nmin * 0.93 or value > nmax * 1.12:
            status = "warning"
        else:
            status = "normal"

        sensors.append({
            "key":        key,
            "value":      value,
            "unit":       meta["unit"],
            "normal_min": nmin,
            "normal_max": nmax,
            "delta_pct":  delta_pct,
            "status":     status,
        })
    return sensors


def _build_history_snapshot(machine_id: int, ctx: dict) -> dict:
    """
    Re-derive the same deterministic maintenance history that main.py serves.
    """
    rng = _seed(machine_id + 2)
    degradation = max(0, (100 - ctx.get("health_score", 80)) / 100)
    base_errors = int(degradation * 12) + rng.randint(0, 4)

    error_log = []
    for _ in range(base_errors):
        code = rng.choice(list(ERROR_CODES.keys()))
        error_log.append({
            "code":        code,
            "description": ERROR_CODES[code],
            "days_ago":    rng.randint(1, 365),
            "count":       rng.randint(1, 5),
        })
    error_log.sort(key=lambda x: x["days_ago"])

    maintenance_actions = [
        "Bearing replacement — spindle assembly",
        "Coolant flush and refill",
        "Belt tension adjustment",
        "Lubrication service",
        "Sensor calibration",
        "Full preventive maintenance",
        "Vibration damper replacement",
    ]
    maintenance_log = [
        {
            "action":     rng.choice(maintenance_actions),
            "days_ago":   rng.randint(30, 700),
            "technician": f"Tech-{rng.randint(1, 8):02d}",
        }
        for _ in range(rng.randint(2, 6))
    ]
    maintenance_log.sort(key=lambda x: x["days_ago"])

    machine_age_years   = round(rng.uniform(2, 12), 1)
    last_service_days   = maintenance_log[0]["days_ago"] if maintenance_log else rng.randint(60, 300)

    return {
        "machine_age_years":       machine_age_years,
        "last_service_days_ago":   last_service_days,
        "total_maintenance_events": len(maintenance_log),
        "total_errors":            len(error_log),
        "top_errors":              error_log[:3],          # top 3 most recent
        "recent_maintenance":      maintenance_log[:2],    # last 2 actions
    }


class Investigator:
    def __init__(self):
        self.health      = pd.read_parquet(HEALTH_DATA_PATH)
        self.retriever   = Retriever()
        self.slm         = None
        self.fast_engine = FastDiagnosticEngine()

        if _USE_MODEL:
            try:
                self.slm = SLM()
                print("SLM loaded on GPU — using neural inference.")
            except Exception as exc:
                print(f"SLM load failed ({exc}), using FastDiagnosticEngine.")
        else:
            print("No CUDA GPU — using FastDiagnosticEngine (instant responses).")

    def context(self, machine_id: int) -> dict | None:
        rows = self.health[self.health["engine_id"] == int(machine_id)]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return {
            "engine_id":               int(r["engine_id"]),
            "cycle":                   int(r["cycle"]),
            "predicted_rul":           float(r["predicted_rul"]),
            "health_score":            float(r["health_score"]),
            "risk_level":              str(r["risk_level"]),
            "anomaly_score":           float(r["anomaly_score"]),
            "anomaly_status":          str(r["anomaly_status"]),
            "failure_probability_24h": float(r["failure_probability_24h"]),
            "failure_probability_48h": float(r["failure_probability_48h"]),
        }

    def investigate(self, machine_id: int, question: str) -> dict:
        ctx = self.context(machine_id)
        if ctx is None:
            return {"error": "Engine not found"}

        # ── Build enriched context: add live sensor snapshot + history ──────
        sensors = _build_sensor_snapshot(machine_id, ctx)
        history = _build_history_snapshot(machine_id, ctx)

        # Attach to context so engine handlers can reference them
        enriched_ctx = {
            **ctx,
            "sensors":           sensors,
            "machine_age_years": history["machine_age_years"],
            "last_service_days": history["last_service_days_ago"],
            "total_errors":      history["total_errors"],
            "top_errors":        history["top_errors"],
            "recent_maintenance": history["recent_maintenance"],
        }

        knowledge = self.retriever.search(question)
        engine    = self.slm if self.slm else self.fast_engine

        try:
            result = engine.generate(enriched_ctx, question, knowledge)
            result["engine_id"]         = int(machine_id)
            result["knowledge_evidence"] = knowledge
            return result
        except Exception as exc:
            print(f"Engine error ({exc}), using fast fallback.")
            result = self.fast_engine.generate(enriched_ctx, question, knowledge)
            result["engine_id"]         = int(machine_id)
            result["knowledge_evidence"] = knowledge
            return result
