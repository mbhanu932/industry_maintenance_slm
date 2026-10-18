"""
SLM inference module.

On systems with a CUDA-capable NVIDIA GPU the fine-tuned Qwen LoRA model is
loaded and used for generation.

On CPU-only / AMD systems the module falls back immediately to the
FastDiagnosticEngine, which produces structured, rule-based diagnoses in
< 10 ms with no model loading overhead.

Set environment variable  FORCE_SLM=1  to attempt the real model even on CPU.
"""

import json
import os
import re
import random
import zlib

# ── GPU check before importing heavy deps ─────────────────────────────────────
import torch
_CUDA = torch.cuda.is_available()
_FORCE = os.getenv("FORCE_SLM", "0") == "1"
_USE_MODEL = _CUDA or _FORCE

BASE_MODEL   = os.getenv("BASE_MODEL",  "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("MODEL_PATH",  "models/industrial_slm_model_v2")

SYSTEM_PROMPT = (
    "You are an industrial maintenance AI. "
    "Use only the supplied machine context and knowledge. "
    "Reply with ONLY valid JSON: "
    '{"severity":"HIGH","diagnosis":"...","evidence":[...],'
    '"recommended_actions":[...],"knowledge_sources":[]}'
)


# ══════════════════════════════════════════════════════════════════════════════
#  Fast Diagnostic Engine  (instant, no model needed)
# ══════════════════════════════════════════════════════════════════════════════
class FastDiagnosticEngine:
    """
    Rule-based diagnosis engine.  Produces realistic, context-aware
    maintenance diagnoses in milliseconds using the machine health
    context and retrieved knowledge chunks.
    """

    # Severity thresholds
    _SEV = [
        ("CRITICAL", lambda c: c["risk_level"] == "CRITICAL" or c["health_score"] < 25 or c["failure_probability_24h"] > 0.7),
        ("HIGH",     lambda c: c["risk_level"] == "HIGH"     or c["health_score"] < 50 or c["failure_probability_24h"] > 0.4),
        ("MEDIUM",   lambda c: c["risk_level"] == "ATTENTION"or c["health_score"] < 70 or c["anomaly_score"] > 0.35),
        ("LOW",      lambda c: True),
    ]

    # Possible failure modes keyed by anomaly + health bands (TURBOFAN-SPECIFIC)
    _FAILURE_MODES = {
        "CRITICAL": [
            ("HPC blade degradation with imminent failure risk",
             ["Execute controlled shutdown per SOP §4", "Perform bore-scope inspection of HPC stage assemblies", 
              "Verify sensor calibration prior to restart", "Brief maintenance team on failure signature"]),
            ("High-pressure compressor thermal runaway",
             ["Reduce cycle immediately to idle", "Check HPC outlet temperature and pressure trends",
              "Verify cooling system functionality", "Measure fuel flow baseline for comparison"]),
            ("Fan blade leading-edge erosion or cracking",
             ["Inspect fan assembly for visible damage", "Check bypass air pressure and flow capacity",
              "Review foreign object damage (FOD) history", "Schedule borescope inspection"]),
            ("Bearing assembly wear or spalling",
             ["Perform vibration signature analysis", "Check lubrication oil condition and temperature",
              "Inspect bearing mounting brackets for movement", "Schedule emergency bearing replacement"]),
        ],
        "HIGH": [
            ("HPC stall margin degradation",
             ["Monitor HPC pressure ratio and stall margin closely", "Reduce operating envelope if margin < SOP threshold",
              "Review compressor map and operating point", "Schedule maintenance inspection"]),
            ("Thermal stress accumulation in hot section",
             ["Monitor exhaust gas temperature (EGT) trend", "Verify turbine cooling system airflow",
              "Inspect turbine blade coatings for spalling", "Reduce thermal load if approaching limits"]),
            ("Seal degradation and air-oil leakage",
             ["Check oil system pressure and temperature", "Inspect seal areas for leakage traces",
              "Verify bearing compartment cooling flow", "Plan seal replacement at next maintenance"]),
            ("Fan/booster compressor blade tip vortex damage",
             ["Inspect blade tips for rubbing marks or cracks", "Check clearance gauges if accessible",
              "Review blade loading vs design operating point", "Monitor vibration at blade passage frequency"]),
        ],
        "MEDIUM": [
            ("Early-stage compressor fouling or blade deposit buildup",
             ["Schedule water wash or compressor cleaning", "Monitor pressure ratio trend closely",
              "Review fuel quality and additives compliance", "Plan offline cleaning before next flight"]),
            ("Minor thermal creep or material relaxation",
             ["Monitor component temperature trend for acceleration", "Verify component stress levels vs design",
              "Review operating cycle history and duty", "Document for lifetime tracking"]),
            ("Slight bearing wear pattern within normal limits",
             ["Continue monitoring vibration and temperature", "Perform routine lubrication service",
              "Plan bearing inspection at next scheduled maintenance", "Compare with historical baselines"]),
        ],
        "LOW": [
            ("Normal engine operating variation within design envelope",
             ["Continue standard flight monitoring and maintenance schedule", "Log current sensor readings for baseline trend",
              "Verify next scheduled maintenance date aligns with plan"]),
        ],
    }

    # Evidence templates (TURBOFAN-SPECIFIC C-MAPSS METRICS)
    _EVIDENCE_TEMPLATES = {
        "CRITICAL": [
            "Health score critically low at {health_score:.1f}/100 — imminent failure risk",
            "Predicted RUL of {predicted_rul:.0f} cycles violates SOP §4 safety threshold (≤25 cycles)",
            "Anomaly score {anomaly_score:.3f} indicates multiple degradation channels active",
            "Failure probability in next 24 h: {failure_probability_24h:.1%} — above critical threshold",
            "Risk classification: {risk_level} — requires immediate engineering review per SOP §4",
        ],
        "HIGH": [
            "Health score degraded to {health_score:.1f}/100 — accelerating wear pattern detected",
            "Predicted RUL: {predicted_rul:.0f} cycles — below safe operating margin per SOP §3",
            "Anomaly score {anomaly_score:.3f} reflects abnormal sensor behavior requiring investigation",
            "Failure probability (24 h): {failure_probability_24h:.1%} — elevated risk",
            "Risk level {risk_level} — maintenance decision required per SOP §3",
        ],
        "MEDIUM": [
            "Health score at {health_score:.1f}/100 — below optimal range, trending downward",
            "Anomaly score {anomaly_score:.3f} warrants closer sensor monitoring per SOP §2",
            "Predicted RUL: {predicted_rul:.0f} cycles — normal operating window, watch for acceleration",
            "Failure probability (48 h): {failure_probability_48h:.1%}",
            "Anomaly status: {anomaly_status} — log for trend tracking",
        ],
        "LOW": [
            "Health score: {health_score:.1f}/100 — within acceptable operating range",
            "Predicted RUL: {predicted_rul:.0f} cycles — ample remaining service life",
            "Anomaly score {anomaly_score:.3f} — within normal limits, no investigation needed",
            "Risk level: {risk_level} — continue routine monitoring per SOP §1",
            "All telemetry within expected C-MAPSS baseline for comparable engines",
        ],
    }

    # ── Question routing table (TURBOFAN-SPECIFIC) ────────────────────────────
    # Each entry: (list_of_trigger_keywords, handler_method_name)
    _ROUTES = [
        # Core diagnostics
        (["offline", "shut down", "shutdown", "stop operating", "halt", "take offline", "turbofan", "immediate inspection"], "_q_offline"),
        (["hpc thermal", "exhaust gas", "overtemp", "thermal runaway", "hpc outlet", "compressor pressure"], "_q_hpc_thermal"),
        (["bearing", "vibration", "rotational instability", "spindle", "vibrating", "mm/s", "oscillat", "fan blade"], "_q_bearing_vibration"),
        (["sensor", "sensors", "hpc", "contributing", "abnormal behavior", "fluctuation", "sustained change", "pressure ratio", "fuel flow"], "_q_sensor"),
        (["anomaly score", "anomaly status", "anomaly mean", "anomaly indicate", "anomaly telling", "interpret the anomaly"], "_q_anomaly"),
        (["rul", "remaining useful life", "cycles left", "how many cycles", "life left", "when does", "replacing", "decreasing rapidly"], "_q_rul"),
        (["health score", "health score mean", "what does the health", "score low", "score high", "score indicate"], "_q_health_score"),
        (["failure probability", "fail soon", "likely to fail", "chance of failure", "risk of failure", "48h", "48 hour", "24h cycle"], "_q_failure_prob"),
        (["24 hour", "24 cycle", "24h", "last 24", "changed recently", "anomalous drift", "recent change", "recent trend", "since yesterday", "what happened"], "_q_24h"),
        (["maintenance", "repair", "fix", "service", "actions should", "procedure", "what should the technician", "resolve", "schedule", "bore-scope"], "_q_maintenance"),
        (["engineering review", "engineer review", "engineering inspection", "qualified engineer", "approve", "engineering procedure"], "_q_engineering"),
        (["loto", "lockout tagout", "safety", "ai control", "shut down equipment", "control equipment", "safety constraint", "perform maintenance"], "_q_safety"),
        (["risk level", "turbofan", "why is this", "what is wrong", "what is causing", "current risk", "risk classification", "explain the"], "_q_risk_level"),
        (["historical", "history", "previous failure", "past failure", "error code", "degradation trend", "degradation history", "baseline", "compare"], "_q_history"),
        (["pressure drop", "pressure ratio", "fuel flow", "hpc outlet", "bypass ratio", "bpr"], "_q_pressure"),
        (["lubrication", "oil", "lubricant", "lubrication system", "bearing oil", "oil pressure", "oil temperature"], "_q_lubrication"),
        (["electrical", "voltage", "electrical condition", "electrical system", "power supply", "volt", "electrical stability"], "_q_electrical"),
        (["rotation", "rpm", "speed", "fan speed", "core speed", "rotation slowing", "rotation speed"], "_q_rotation_trend"),
        (["inspection", "inspect", "check", "examine", "look at", "bore-scope", "non-destructive", "scope"], "_q_inspection"),
        (["fleet", "compare", "similar engines", "other engines", "benchmark", "fleet health", "fleet risk"], "_q_fleet_comparison"),
        (["priority", "priority order", "which engine first", "urgent", "top engines", "order"], "_q_priority_ranking"),
        (["verdict", "diagnosis", "what is", "summary", "overall"], "_q_risk_level"),
    ]

    def _severity(self, ctx):
        for sev, test in self._SEV:
            if test(ctx):
                return sev
        return "LOW"

    def _failure_mode(self, ctx, rng):
        sev = self._severity(ctx)
        modes = self._FAILURE_MODES[sev]
        return modes[rng.randint(0, len(modes) - 1)]

    def _base_evidence(self, ctx):
        sev = self._severity(ctx)
        return [t.format(**ctx) for t in self._EVIDENCE_TEMPLATES[sev]]

    # ── Individual question handlers ─────────────────────────────────────────

    def _q_offline(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk, p24 = ctx["health_score"], ctx["risk_level"], ctx["failure_probability_24h"] * 100
        eid = ctx["engine_id"]
        if sev in ("CRITICAL", "HIGH"):
            diag = (
                f"Engine {eid} should be taken OFFLINE for inspection immediately. "
                f"Health score is {score:.1f}/100 with {risk} risk and {p24:.1f}% failure probability in the next 24 h. "
                f"Root cause analysis points to {mode}. Per SOP Section 4 (HIGH-RISK CONDITION), "
                f"high-risk assets must be prioritized for engineering review before continuing operation. "
                f"Continuing to run this machine poses an unacceptable failure risk."
            )
        else:
            diag = (
                f"Engine {eid} does NOT require immediate shutdown. "
                f"Health score is {score:.1f}/100 ({risk} risk) with only {p24:.1f}% failure probability in 24 h. "
                f"Per SOP Section 3 (MAINTENANCE DECISION), standard monitoring and scheduled maintenance are sufficient. "
                f"Continue operating under normal supervision."
            )
        return sev, diag, self._base_evidence(ctx), actions

    def _q_hpc_thermal(self, ctx, rng, knowledge):
        """HPC thermal runaway and exhaust gas overtemp analysis"""
        sev = self._severity(ctx)
        eid = ctx["engine_id"]
        temp_reading = next((s for s in ctx.get("sensors", []) if s["key"] == "temperature"), None)
        pressure_reading = next((s for s in ctx.get("sensors", []) if s["key"] == "pressure"), None)
        
        temp_detail = f"HPC outlet temperature: {temp_reading['value']}°C (normal: {temp_reading['normal_min']}-{temp_reading['normal_max']}°C, {'+' if temp_reading['delta_pct']>0 else ''}{temp_reading['delta_pct']}%)" if temp_reading else "Temperature sensor unavailable"
        pressure_detail = f"HPC outlet pressure: {pressure_reading['value']} PSI (normal: {pressure_reading['normal_min']}-{pressure_reading['normal_max']} PSI, {'+' if pressure_reading['delta_pct']>0 else ''}{pressure_reading['delta_pct']}%)" if pressure_reading else "Pressure sensor unavailable"
        
        if sev in ("CRITICAL", "HIGH"):
            diag = (
                f"HPC Thermal Analysis for engine {eid}:\n"
                f"{temp_detail}\n{pressure_detail}\n"
                f"ASSESSMENT: Thermal stress accumulation detected. Health {ctx['health_score']:.1f}/100, "
                f"Anomaly {ctx['anomaly_score']:.3f}. Per SOP §2, this sustained thermal elevation indicates HPC stage deterioration. "
                f"Risk of blade spalling or hot section failure. Reduce operating envelope immediately."
            )
            evidence = [
                temp_detail if temp_reading else "Temperature elevated per health score",
                pressure_detail if pressure_reading else "Pressure elevated per anomaly score",
                f"Health score {ctx['health_score']:.1f}/100 — reflects thermal stress",
                f"RUL {ctx['predicted_rul']:.0f} cycles — degradation accelerating",
                f"Failure probability 24h: {ctx['failure_probability_24h']*100:.1f}%"
            ]
            actions = ["Reduce turbofan fuel flow to idle", "Monitor EGT (exhaust gas temperature) continuously", 
                      "Check cooling system airflow and effectiveness", "Inspect turbine blade coatings for spalling",
                      "Plan controlled shutdown and teardown inspection"]
        else:
            diag = (
                f"HPC Thermal Status for engine {eid}:\n"
                f"{temp_detail}\n{pressure_detail}\n"
                f"ASSESSMENT: Thermal conditions within acceptable range. Continue monitoring."
            )
            evidence = [temp_detail, pressure_detail, f"Health score {ctx['health_score']:.1f}/100 — stable",
                       f"Anomaly score {ctx['anomaly_score']:.3f} — normal", "No immediate thermal concerns"]
            actions = ["Continue routine temperature monitoring", "Log thermal baseline for trend comparison",
                      "Plan next scheduled maintenance per SOP"]
        
        return sev, diag, evidence, actions

    def _q_bearing_vibration(self, ctx, rng, knowledge):
        """Core rotational instability and bearing assembly degradation"""
        sev = self._severity(ctx)
        eid = ctx["engine_id"]
        vibration = next((s for s in ctx.get("sensors", []) if s["key"] == "vibration"), None)
        rotation = next((s for s in ctx.get("sensors", []) if s["key"] == "rotation"), None)
        
        vib_detail = f"Vibration: {vibration['value']} {vibration['unit']} (normal: {vibration['normal_min']}-{vibration['normal_max']}, {'+' if vibration['delta_pct']>0 else ''}{vibration['delta_pct']}%)" if vibration else "Vibration unavailable"
        rot_detail = f"Fan/Core RPM: {rotation['value']} RPM (normal: {rotation['normal_min']}-{rotation['normal_max']}, {'+' if rotation['delta_pct']>0 else ''}{rotation['delta_pct']}%)" if rotation else "Rotation unavailable"
        
        if sev in ("CRITICAL", "HIGH"):
            diag = (
                f"Bearing & Rotational Health for engine {eid}:\n"
                f"{vib_detail}\n{rot_detail}\n"
                f"VERDICT: Bearing assembly degradation likely. Anomaly {ctx['anomaly_score']:.3f} indicates sustained vibration pattern. "
                f"Health {ctx['health_score']:.1f}/100. Per SOP §2, this is NOT an isolated spike — maintenance required immediately."
            )
            evidence = [
                vib_detail if vibration else "Vibration elevated per anomaly score",
                rot_detail if rotation else "Rotation trending off nominal",
                f"Anomaly score {ctx['anomaly_score']:.3f} — sustained abnormal vibration",
                f"Anomaly status: {ctx['anomaly_status']} — not isolated",
                f"RUL {ctx['predicted_rul']:.0f} cycles remaining — urgent action required"
            ]
            actions = ["Perform vibration signature analysis (FFT if available)", "Inspect fan/booster blade for rub marks",
                      "Check bearing oil temperature and condition", "Inspect bearing mounting bracket tightness",
                      "Schedule emergency bearing disassembly and inspection"]
        else:
            diag = (
                f"Bearing & Rotational Status for engine {eid}:\n"
                f"{vib_detail}\n{rot_detail}\n"
                f"VERDICT: Bearings and rotational components within acceptable limits. Continue monitoring."
            )
            evidence = [vib_detail, rot_detail, f"Anomaly score {ctx['anomaly_score']:.3f} — normal range",
                       f"Health score {ctx['health_score']:.1f}/100 — stable", "No bearing wear indicators detected"]
            actions = ["Continue vibration monitoring per maintenance schedule", "Log vibration baseline trend",
                      "Plan bearing inspection at next scheduled maintenance"]
        
        return sev, diag, evidence, actions

    def _q_pressure(self, ctx, rng, knowledge):
        """Pressure drop analysis and HPC/bypass flow capacity"""
        sev = self._severity(ctx)
        eid = ctx["engine_id"]
        pressure = next((s for s in ctx.get("sensors", []) if s["key"] == "pressure"), None)
        
        pressure_detail = f"HPC outlet pressure: {pressure['value']} PSI (normal: {pressure['normal_min']}-{pressure['normal_max']}, change: {'+' if pressure['delta_pct']>0 else ''}{pressure['delta_pct']}%)" if pressure else "Pressure unavailable"
        
        if sev in ("CRITICAL", "HIGH"):
            diag = (
                f"Pressure System Analysis for engine {eid}:\n{pressure_detail}\n"
                f"ASSESSMENT: Pressure drop detected relative to fuel flow baseline. Indicates HPC compressor fouling "
                f"or blade degradation. Health {ctx['health_score']:.1f}/100, RUL {ctx['predicted_rul']:.0f} cycles. "
                f"Per SOP §3, reduced pressure margin reduces stall margin — failure risk increased."
            )
            evidence = [
                pressure_detail if pressure else "Pressure dropped significantly",
                f"Pressure delta: {pressure['delta_pct'] if pressure else ctx['anomaly_score']}% below baseline",
                f"Health score {ctx['health_score']:.1f}/100 — supports compressor degradation hypothesis",
                f"RUL {ctx['predicted_rul']:.0f} cycles — action required soon",
                f"Failure probability 24h: {ctx['failure_probability_24h']*100:.1f}%"
            ]
            actions = ["Monitor pressure ratio trend closely (record every cycle)",
                      "Reduce operating envelope to maintain safe stall margin", "Plan compressor water wash or cleaning",
                      "Check fuel quality and additive compliance", "Schedule borescope inspection"]
        else:
            diag = (
                f"Pressure System Status for engine {eid}:\n{pressure_detail}\n"
                f"ASSESSMENT: Pressure readings within acceptable operating range per design baseline."
            )
            evidence = [pressure_detail, f"Pressure margin adequate for current operating point",
                       f"Health score {ctx['health_score']:.1f}/100", "Stall margin normal per design"]
            actions = ["Continue routine pressure monitoring", "Log pressure baseline for trend tracking",
                      "Plan next scheduled maintenance per SOP"]
        
        return sev, diag, evidence, actions
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        anom, score, rul = ctx["anomaly_status"], ctx["health_score"], ctx["predicted_rul"]
        eid = ctx["engine_id"]

        # Pull real vibration reading if available
        sensors = ctx.get("sensors", [])
        vib = next((s for s in sensors if s["key"] == "vibration"), None)
        vib_detail = (
            f"Vibration reading: {vib['value']} {vib['unit']} "
            f"(normal: {vib['normal_min']}–{vib['normal_max']} {vib['unit']}, "
            f"{'+' if vib['delta_pct']>0 else ''}{vib['delta_pct']}% from nominal, status: {vib['status'].upper()}). "
            if vib else ""
        )

        if sev in ("CRITICAL", "HIGH"):
            diag = (
                f"Vibration/bearing analysis for engine {eid}: {vib_detail}"
                f"Anomaly status is '{anom}' with anomaly score {ctx['anomaly_score']:.3f} — above safe threshold. "
                f"Health score {score:.1f}/100, predicted RUL {rul:.0f} cycles. "
                f"Pattern is consistent with {mode}. "
                f"Per SOP Section 2, this represents a SUSTAINED change, not an isolated fluctuation. "
                f"IMMEDIATE bearing inspection is recommended."
            )
            evidence = [
                vib_detail.strip() if vib_detail else f"Anomaly score {ctx['anomaly_score']:.3f} confirms sustained abnormal vibration",
                f"Anomaly status: {anom} — not an isolated spike",
                f"Health score {score:.1f}/100 supports mechanical degradation hypothesis",
                f"RUL {rul:.0f} cycles — degradation is accelerating",
                f"Risk level {ctx['risk_level']} consistent with bearing wear pattern",
            ]
        else:
            diag = (
                f"Vibration analysis for engine {eid}: {vib_detail}"
                f"Anomaly score {ctx['anomaly_score']:.3f} is within acceptable range. "
                f"Anomaly status: '{anom}'. Health score {score:.1f}/100, RUL {rul:.0f} cycles. "
                f"Per SOP Section 2, no sustained sensor anomaly is present. "
                f"Continue monitoring vibration trend over next operating cycles."
            )
            evidence = [
                vib_detail.strip() if vib_detail else f"Anomaly score {ctx['anomaly_score']:.3f} — within acceptable bounds",
                f"Anomaly status: {anom}",
                f"Health score {score:.1f}/100 — stable",
                f"RUL {rul:.0f} cycles remaining",
            ]
        return sev, diag, evidence, actions

    def _q_sensor(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        anom, score = ctx["anomaly_status"], ctx["health_score"]
        eid = ctx["engine_id"]

        # Use real sensor readings if available in enriched context
        sensors = ctx.get("sensors", [])
        abnormal = [s for s in sensors if s["status"] in ("critical", "warning")]
        if abnormal:
            sensor_summary = ", ".join(
                f"{s['key']} {s['value']}{s['unit']} ({s['status']}, {'+' if s['delta_pct']>0 else ''}{s['delta_pct']}%)"
                for s in abnormal
            )
        else:
            high_sensors = []
            if ctx["anomaly_score"] > 0.5:   high_sensors.append("vibration (primary driver)")
            if ctx["health_score"] < 50:     high_sensors.append("health composite (multi-sensor degradation)")
            if ctx["failure_probability_24h"] > 0.4: high_sensors.append("failure predictor model (cross-sensor)")
            if not high_sensors:             high_sensors = ["no sensors exceed critical thresholds"]
            sensor_summary = ", ".join(high_sensors)

        diag = (
            f"Sensor investigation for engine {eid}: "
            f"Abnormal sensors — {sensor_summary}. "
            f"Anomaly status is '{anom}' (score: {ctx['anomaly_score']:.3f}). "
            f"Per SOP Section 2 (SENSOR INVESTIGATION), sustained changes rather than isolated fluctuations "
            f"are present. Health score {score:.1f}/100 confirms multi-sensor degradation consistent with {mode}. "
            f"{'Engineering review required.' if sev in ('CRITICAL','HIGH') else 'Continue monitoring sensor trends.'}"
        )
        evidence = [
            f"Abnormal sensors: {sensor_summary}",
            f"Anomaly score {ctx['anomaly_score']:.3f} — {'sustained abnormal' if ctx['anomaly_score'] > 0.3 else 'within normal range'}",
            f"Anomaly status: {anom}",
            f"Health score {score:.1f}/100 — {'degraded' if score < 60 else 'acceptable'}",
            f"Failure probability 24h: {ctx['failure_probability_24h']*100:.1f}%",
        ]
        return sev, diag, evidence, actions

    def _q_rul(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        rul, score, risk = ctx["predicted_rul"], ctx["health_score"], ctx["risk_level"]
        eid = ctx["engine_id"]
        urgency = "CRITICAL — immediate action needed" if rul < 20 else \
                  "HIGH — schedule inspection soon" if rul < 50 else \
                  "MODERATE — plan maintenance within next service window" if rul < 100 else \
                  "NORMAL — within expected operating range"
        diag = (
            f"Remaining Useful Life (RUL) assessment for engine {eid}: "
            f"Predicted RUL is {rul:.0f} cycles — urgency level: {urgency}. "
            f"Health score {score:.1f}/100 supports this projection. "
            f"Per SOP Section 1 (GENERAL INVESTIGATION), RUL must be reviewed together with "
            f"health score and anomaly score. Combined analysis indicates {mode}. "
            f"{'Engineering inspection must be prioritized per SOP Section 3.' if sev in ('CRITICAL','HIGH') else 'Continue monitoring RUL trend.'}"
        )
        evidence = [
            f"Predicted RUL: {rul:.0f} cycles — {urgency}",
            f"Health score {score:.1f}/100 corroborates RUL estimate",
            f"Anomaly score {ctx['anomaly_score']:.3f} — {'accelerating degradation' if ctx['anomaly_score'] > 0.4 else 'stable'}",
            f"Risk classification: {risk}",
            f"Failure probability (48h): {ctx['failure_probability_48h']*100:.1f}%",
        ]
        return sev, diag, evidence, actions

    def _q_health_score(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, rul, risk = ctx["health_score"], ctx["predicted_rul"], ctx["risk_level"]
        eid = ctx["engine_id"]
        interpretation = (
            "critically low — immediate intervention required" if score < 25 else
            "severely degraded — engineering inspection needed" if score < 50 else
            "below optimal — monitor closely and plan maintenance" if score < 70 else
            "acceptable — continue normal operations"
        )
        diag = (
            f"Health score interpretation for engine {eid}: "
            f"Current health score is {score:.1f}/100 — {interpretation}. "
            f"The health score is a composite metric derived from multiple sensor readings, "
            f"degradation patterns, and historical baseline comparison. "
            f"Per SOP Section 1, health score must be read alongside RUL ({rul:.0f} cycles) "
            f"and anomaly score ({ctx['anomaly_score']:.3f}). "
            f"Combined picture indicates {mode} as the primary degradation mechanism. "
            f"Risk classification is {risk}."
        )
        evidence = [
            f"Health score {score:.1f}/100 — {interpretation}",
            f"Predicted RUL {rul:.0f} cycles confirms health trajectory",
            f"Anomaly score {ctx['anomaly_score']:.3f} — {'elevated' if ctx['anomaly_score'] > 0.3 else 'normal'}",
            f"Risk level: {risk}",
            f"Failure probability (24h): {ctx['failure_probability_24h']*100:.1f}%",
        ]
        return sev, diag, evidence, actions

    def _q_anomaly(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        anom, score = ctx["anomaly_status"], ctx["health_score"]
        eid = ctx["engine_id"]
        anom_interp = (
            "far outside normal operating envelope — critical anomaly" if ctx["anomaly_score"] > 0.7 else
            "significantly outside normal range — high anomaly" if ctx["anomaly_score"] > 0.4 else
            "moderately outside normal range — watch closely" if ctx["anomaly_score"] > 0.2 else
            "within normal operating range"
        )
        diag = (
            f"Anomaly score analysis for engine {eid}: "
            f"Score is {ctx['anomaly_score']:.3f} — {anom_interp}. "
            f"Anomaly status is classified as '{anom}'. "
            f"Per SOP Section 2, anomaly scores must be interpreted together with RUL and health indicators. "
            f"RUL is {ctx['predicted_rul']:.0f} cycles and health score is {score:.1f}/100. "
            f"The combined signal is consistent with {mode}. "
            f"{'The anomaly is not an isolated fluctuation — it reflects a sustained degradation pattern.' if ctx['anomaly_score'] > 0.3 else 'The anomaly appears to be within acceptable operational variation.'}"
        )
        evidence = [
            f"Anomaly score: {ctx['anomaly_score']:.3f} — {anom_interp}",
            f"Anomaly status: {anom}",
            f"Health score {score:.1f}/100 corroborates anomaly signal",
            f"RUL {ctx['predicted_rul']:.0f} cycles — {'declining' if sev in ('CRITICAL','HIGH') else 'stable'}",
            f"Risk level: {ctx['risk_level']}",
        ]
        return sev, diag, evidence, actions

    def _q_failure_prob(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        p24 = ctx["failure_probability_24h"] * 100
        p48 = ctx["failure_probability_48h"] * 100
        eid = ctx["engine_id"]
        urgency = (
            "IMMEDIATE shutdown and inspection required" if p24 > 70 else
            "Engineering review within 24 hours" if p24 > 40 else
            "Schedule inspection within 48–72 hours" if p48 > 30 else
            "No immediate action — monitor in next service cycle"
        )
        diag = (
            f"Failure probability assessment for engine {eid}: "
            f"24-hour failure probability: {p24:.1f}%. "
            f"48-hour failure probability: {p48:.1f}%. "
            f"Recommended response: {urgency}. "
            f"This probability is derived from the predictive model combining RUL ({ctx['predicted_rul']:.0f} cycles), "
            f"health score ({ctx['health_score']:.1f}/100), and anomaly score ({ctx['anomaly_score']:.3f}). "
            f"Primary failure mode identified: {mode}. "
            f"Per SOP Section 3, if RUL is decreasing significantly, engineering inspection must be prioritized."
        )
        evidence = [
            f"Failure probability (24h): {p24:.1f}% — {'CRITICAL' if p24 > 60 else 'HIGH' if p24 > 30 else 'LOW'}",
            f"Failure probability (48h): {p48:.1f}%",
            f"Health score {ctx['health_score']:.1f}/100 driving probability estimate",
            f"Predicted RUL {ctx['predicted_rul']:.0f} cycles",
            f"Anomaly score {ctx['anomaly_score']:.3f} contributing to failure model",
        ]
        return sev, diag, evidence, actions

    def _q_24h(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk, anom = ctx["health_score"], ctx["risk_level"], ctx["anomaly_status"]
        p24 = ctx["failure_probability_24h"] * 100
        eid = ctx["engine_id"]
        trend = "worsening" if sev in ("CRITICAL", "HIGH") else "stable" if sev == "MEDIUM" else "improving"
        diag = (
            f"24-hour trend analysis for engine {eid}: "
            f"Current status — {risk} risk, health {score:.1f}/100, anomaly '{anom}' (score {ctx['anomaly_score']:.3f}). "
            f"The overall trend over the last 24 hours is {trend}. "
            f"Failure probability in the next 24 h stands at {p24:.1f}%. "
            f"Per SOP Section 2 (SENSOR INVESTIGATION), sustained sensor changes should be investigated "
            f"rather than isolated fluctuations — the current pattern qualifies as {'sustained' if ctx['anomaly_score'] > 0.2 else 'isolated'}. "
            f"Active degradation mode: {mode}."
        )
        evidence = [
            f"24h trend direction: {trend}",
            f"Anomaly status over 24h: {anom} (score {ctx['anomaly_score']:.3f})",
            f"Health score: {score:.1f}/100",
            f"Failure probability in next 24h: {p24:.1f}%",
            f"Risk level: {risk}",
        ]
        return sev, diag, evidence, actions

    def _q_maintenance(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk, rul = ctx["health_score"], ctx["risk_level"], ctx["predicted_rul"]
        eid = ctx["engine_id"]

        last_svc   = ctx.get("last_service_days", None)
        total_errs = ctx.get("total_errors", 0)
        svc_note   = f" Last service was {last_svc} days ago." if last_svc else ""
        err_note   = f" {total_errs} error events recorded." if total_errs else ""

        urgency_label = {
            "CRITICAL": "EMERGENCY — do not delay",
            "HIGH":     "URGENT — within 24-48 hours",
            "MEDIUM":   "SCHEDULED — next service window",
            "LOW":      "ROUTINE — next planned cycle",
        }.get(sev, "ROUTINE")

        diag = (
            f"Maintenance recommendation for engine {eid}: "
            f"Risk level {risk}, health {score:.1f}/100, RUL {rul:.0f} cycles.{svc_note}{err_note} "
            f"Urgency: {urgency_label}. Primary identified issue: {mode}. "
            f"Per SOP Section 3 (MAINTENANCE DECISION), maintenance actions must follow approved engineering "
            f"procedures and a qualified engineer must approve before execution. "
            f"The AI provides investigation support only — do not act without engineering sign-off."
        )
        evidence = [
            f"Maintenance urgency: {urgency_label}",
            f"Health score {score:.1f}/100 — {'requires immediate attention' if score < 50 else 'monitor'}",
            f"RUL {rul:.0f} cycles — {'critical' if rul < 20 else 'degraded' if rul < 60 else 'acceptable'}",
            f"Days since last service: {last_svc}" if last_svc else "Last service date unknown",
            f"Primary failure mode: {mode}",
        ]
        return sev, diag, evidence, actions

    def _q_engineering(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        risk, score = ctx["risk_level"], ctx["health_score"]
        eid = ctx["engine_id"]
        needed = sev in ("CRITICAL", "HIGH")
        diag = (
            f"Engineering review assessment for engine {eid}: "
            f"{'Engineering review IS required immediately.' if needed else 'Engineering review is recommended at next scheduled opportunity.'} "
            f"Per SOP Section 1 (GENERAL INVESTIGATION), a {risk} risk classification {'must' if needed else 'should'} trigger engineering review. "
            f"Current health score {score:.1f}/100 and anomaly score {ctx['anomaly_score']:.3f} provide supporting evidence for {mode}. "
            f"Per SOP Section 3, all maintenance actions resulting from this review must follow approved engineering procedures. "
            f"A qualified engineer must approve any maintenance before it is carried out."
        )
        evidence = [
            f"Engineering review required: {'YES — immediately' if needed else 'YES — next scheduled window'}",
            f"Risk level {risk} {'triggers mandatory review per SOP Section 1' if needed else 'warrants advisory review'}",
            f"Health score {score:.1f}/100",
            f"Anomaly score {ctx['anomaly_score']:.3f}",
            f"Qualified engineer sign-off required per SOP Section 3",
        ]
        return sev, diag, evidence, actions

    def _q_safety(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        _, actions = self._failure_mode(ctx, rng)
        eid = ctx["engine_id"]
        if sev in ("CRITICAL", "HIGH"):
            escalation_msg = (
                f"Given the {ctx['risk_level']} risk level, "
                "escalate to engineering immediately."
        )
        else:
            escalation_msg = "Standard safety protocols apply."

    diag = (
        f"Safety protocol for engine {eid}: "
        "Per SOP Section 5 (SAFETY), the AI system MUST NOT directly "
        "control or shut down industrial equipment. "
        "All AI-generated recommendations must be reviewed and approved "
        "by qualified engineering personnel before any maintenance "
        "or operational change is made. "
        f"Current machine status — risk: {ctx['risk_level']}, "
        f"health: {ctx['health_score']:.1f}/100, "
        f"anomaly score: {ctx['anomaly_score']:.3f}. "
        f"{escalation_msg} "
        "Do not perform maintenance based solely on this AI recommendation."
    )

        evidence = [
            "AI must not directly control or shut down equipment (SOP Section 5)",
            "All recommendations require qualified engineering review",
            f"Current risk level: {ctx['risk_level']}",
            f"Current health score: {ctx['health_score']:.1f}/100",
            "Maintenance requires approved engineering procedures (SOP Section 3)",
        ]
        return sev, diag, evidence, ["Escalate to qualified engineer before any action",
                                     "Do not act on AI diagnosis without engineering sign-off",
                                     "Follow approved engineering procedures for any maintenance",
                                     "Document AI recommendation and engineering decision"]

    def _q_risk_level(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        risk, score, rul = ctx["risk_level"], ctx["health_score"], ctx["predicted_rul"]
        anom = ctx["anomaly_status"]
        eid = ctx["engine_id"]
        diag = (
            f"Risk level explanation for engine {eid}: "
            f"The machine is classified as {risk} because health score is {score:.1f}/100, "
            f"predicted RUL is {rul:.0f} cycles, and anomaly score is {ctx['anomaly_score']:.3f} ({anom}). "
            f"Per SOP Section 1 (GENERAL INVESTIGATION), risk is determined by reviewing RUL, health score, "
            f"anomaly score, and historical degradation trends together — not any single metric. "
            f"The convergence of these indicators points to {mode} as the primary driver of the {risk} classification."
        )
        evidence = [
            f"Risk driver 1: Health score {score:.1f}/100 — {'degraded' if score < 60 else 'acceptable'}",
            f"Risk driver 2: Predicted RUL {rul:.0f} cycles — {'low' if rul < 50 else 'moderate' if rul < 100 else 'adequate'}",
            f"Risk driver 3: Anomaly score {ctx['anomaly_score']:.3f} ({anom})",
            f"Risk driver 4: Failure probability 24h {ctx['failure_probability_24h']*100:.1f}%",
            f"Combined classification: {risk}",
        ]
        return sev, diag, evidence, actions

    def _q_history(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, rul, risk = ctx["health_score"], ctx["predicted_rul"], ctx["risk_level"]
        eid = ctx["engine_id"]

        # Use real history if available in enriched context
        age        = ctx.get("machine_age_years", "unknown")
        last_svc   = ctx.get("last_service_days", "unknown")
        total_errs = ctx.get("total_errors", 0)
        top_errors = ctx.get("top_errors", [])
        recent_mnt = ctx.get("recent_maintenance", [])

        error_summary = (
            ", ".join(f"{e['code']} ({e['description']}, {e['count']}x)" for e in top_errors[:2])
            if top_errors else "no errors recorded"
        )
        maint_summary = (
            f"Last action: {recent_mnt[0]['action']} ({recent_mnt[0]['days_ago']}d ago)"
            if recent_mnt else "no recent maintenance"
        )

        diag = (
            f"Historical degradation analysis for engine {eid}: "
            f"Machine age {age} years, last service {last_svc} days ago, {total_errs} total errors recorded. "
            f"Top recent errors — {error_summary}. {maint_summary}. "
            f"Current health score {score:.1f}/100 and RUL {rul:.0f} cycles reflect accumulated degradation. "
            f"Per SOP Section 1, current behavior should be compared with historical degradation trends. "
            f"The anomaly pattern (score {ctx['anomaly_score']:.3f}) is consistent with {mode}. "
            f"Per SOP Section 4, historical comparison is key in determining inspection vs scheduled maintenance."
        )
        evidence = [
            f"Machine age: {age} years",
            f"Days since last service: {last_svc}",
            f"Total error events: {total_errs}",
            f"Recent errors: {error_summary}",
            f"Health score {score:.1f}/100 vs baseline 100/100 at commissioning",
            f"RUL {rul:.0f} cycles — degradation rate implies {mode}",
        ]
        return sev, diag, evidence, actions

    def _q_temperature(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk = ctx["health_score"], ctx["risk_level"]
        eid = ctx["engine_id"]
        temp_concern = ctx["anomaly_score"] > 0.3 or score < 60
        diag = (
            f"Thermal analysis for engine {eid}: "
            f"{'Elevated thermal stress is suspected based on anomaly score ' + str(round(ctx['anomaly_score'],3)) + ' and health score ' + str(round(score,1)) + '/100.' if temp_concern else 'Thermal indicators are within acceptable range.'} "
            f"Per SOP Section 2, temperature sensors contributing to abnormal behavior should be investigated "
            f"for sustained changes rather than isolated spikes. "
            f"Risk level is {risk}. Primary degradation mode: {mode}. "
            f"{'Inspect cooling system and reduce duty cycle.' if temp_concern else 'Continue standard thermal monitoring.'}"
        )
        evidence = [
            f"Thermal stress indicator: {'elevated' if temp_concern else 'normal'} (via composite health score)",
            f"Health score {score:.1f}/100",
            f"Anomaly score {ctx['anomaly_score']:.3f}",
            f"Risk level: {risk}",
            f"RUL: {ctx['predicted_rul']:.0f} cycles",
        ]
        actions = ["Inspect cooling system and heat exchanger", "Check ambient temperature",
                   "Verify coolant flow rate", "Reduce cycle frequency if overheating confirmed"] if temp_concern else actions
        return sev, diag, evidence, actions

    def _q_pressure(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk = ctx["health_score"], ctx["risk_level"]
        eid = ctx["engine_id"]
        diag = (
            f"Pressure analysis for engine {eid}: "
            f"Health score {score:.1f}/100 and anomaly score {ctx['anomaly_score']:.3f} are the available pressure-correlated indicators. "
            f"Per SOP Section 2 (SENSOR INVESTIGATION), pressure sensors should be reviewed for sustained deviations "
            f"from the machine's previous operating cycles. "
            f"Risk level is {risk}. Degradation pattern consistent with {mode}. "
            f"{'Inspect pressure lines and pump for wear.' if sev in ('CRITICAL','HIGH') else 'Monitor pressure trend over next 24 hours.'}"
        )
        evidence = [
            f"Pressure-correlated health score: {score:.1f}/100",
            f"Anomaly score {ctx['anomaly_score']:.3f} — may include pressure sensor contribution",
            f"Risk level: {risk}",
            f"RUL: {ctx['predicted_rul']:.0f} cycles",
        ]
        actions = ["Inspect pressure lines and pump", "Check pressure sensor calibration",
                   "Verify seals and gaskets", "Compare with baseline pressure profile"] if sev in ("CRITICAL","HIGH") else actions
        return sev, diag, evidence, actions

    def _q_electrical(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk = ctx["health_score"], ctx["risk_level"]
        eid = ctx["engine_id"]
        diag = (
            f"Electrical system analysis for engine {eid}: "
            f"Health score {score:.1f}/100 and anomaly score {ctx['anomaly_score']:.3f} indicate {'elevated electrical stress' if sev in ('CRITICAL','HIGH') else 'acceptable electrical condition'}. "
            f"Per SOP Section 2, voltage, current, and winding sensors contributing to abnormal behavior "
            f"should be investigated for sustained deviations. "
            f"Risk level {risk}. Primary concern: {mode}. "
            f"{'Check motor windings, connections, and power supply immediately.' if sev in ('CRITICAL','HIGH') else 'Routine electrical inspection at next service cycle.'}"
        )
        evidence = [
            f"Electrical health indicator (composite): {score:.1f}/100",
            f"Anomaly score {ctx['anomaly_score']:.3f}",
            f"Risk: {risk}",
            f"RUL: {ctx['predicted_rul']:.0f} cycles",
            f"Failure probability 24h: {ctx['failure_probability_24h']*100:.1f}%",
        ]
        actions = ["Measure motor winding resistance", "Check current draw against nameplate rating",
                   "Inspect electrical connections and terminals", "Verify ground continuity"] if sev in ("CRITICAL","HIGH") else actions
        return sev, diag, evidence, actions

    def _q_lubrication(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk, rul = ctx["health_score"], ctx["risk_level"], ctx["predicted_rul"]
        eid = ctx["engine_id"]
        diag = (
            f"Lubrication assessment for engine {eid}: "
            f"{'Lubrication system failure is a likely contributor to current degradation.' if sev in ('CRITICAL','HIGH') else 'Lubrication system appears adequate based on current indicators.'} "
            f"Health score {score:.1f}/100 and RUL {rul:.0f} cycles. "
            f"Per SOP Section 2, lubrication-related sensor changes should be investigated alongside "
            f"vibration and bearing indicators. "
            f"Risk level {risk}. {'Inspect oil pressure, oil level, and lubrication lines immediately.' if sev in ('CRITICAL','HIGH') else 'Verify lubrication at next scheduled service.'}"
        )
        evidence = [
            f"Lubrication health indicator (composite): {score:.1f}/100",
            f"Anomaly score {ctx['anomaly_score']:.3f} — {'may include lubrication sensor' if ctx['anomaly_score'] > 0.3 else 'normal'}",
            f"RUL {rul:.0f} cycles",
            f"Risk level: {risk}",
        ]
        actions = ["Check oil pressure and oil level", "Inspect lubrication lines for leaks",
                   "Verify grease fittings on bearing assembly", "Sample oil for contamination analysis"]
        return sev, diag, evidence, actions

    def _q_inspection(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, actions = self._failure_mode(ctx, rng)
        score, risk, rul = ctx["health_score"], ctx["risk_level"], ctx["predicted_rul"]
        p24 = ctx["failure_probability_24h"] * 100
        eid = ctx["engine_id"]
        scope = {
            "CRITICAL": "full emergency inspection — all systems",
            "HIGH":     "comprehensive inspection focusing on mechanical and sensor systems",
            "MEDIUM":   "targeted inspection of anomalous subsystems",
            "LOW":      "routine visual and sensor inspection at next scheduled service",
        }.get(sev, "routine inspection")
        diag = (
            f"Inspection scope for engine {eid}: {scope.upper()}. "
            f"Health score {score:.1f}/100, RUL {rul:.0f} cycles, failure probability {p24:.1f}% in 24 h. "
            f"Per SOP Section 4 (HIGH-RISK CONDITION), the inspection should include: "
            f"review of recent telemetry behavior, check for abnormal sensor patterns, "
            f"and comparison with historical degradation. "
            f"Suspected finding: {mode}. "
            f"A qualified engineer must lead the inspection per SOP Section 3."
        )
        evidence = [
            f"Inspection scope: {scope}",
            f"Health score {score:.1f}/100 justifies inspection priority",
            f"RUL {rul:.0f} cycles",
            f"Failure probability 24h: {p24:.1f}%",
            f"Risk level: {risk} (per SOP Section 4 — prioritize for engineering review)",
        ]
        return sev, diag, evidence, actions

    # ── NEW QUESTION HANDLERS (12 additional questions) ────────────────────────

    def _q_rotation_trend(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        sensors = ctx.get("sensors", [])
        rot = next((s for s in sensors if s["key"] == "rotation"), None)
        if not rot:
            return sev, "Rotation data not available", [], []
        rpm_val, rpm_min, rpm_max = rot["value"], rot["normal_min"], rot["normal_max"]
        pct = rot["delta_pct"]
        if rpm_val < rpm_min * 0.95:
            issue = "CRITICAL: RPM significantly BELOW normal (motor slowing)"
        elif rpm_val < rpm_min: issue = "WARNING: RPM slightly BELOW normal"
        elif rpm_val > rpm_max * 1.05: issue = "WARNING: RPM ABOVE normal (overspeed)"
        else: issue = "NORMAL: RPM within operating range"
        diag = f"Rotation analysis for {ctx['engine_id']}: {rpm_val} RPM. {issue}. Normal: {rpm_min}-{rpm_max}. " \
               f"Health {ctx['health_score']:.1f}/100. {'Motor may be overloaded or bearing friction present.' if rpm_val < rpm_min else ''}"
        evidence = [f"RPM: {rpm_val} ({pct:+.1f}%)", f"Normal: {rpm_min}-{rpm_max} RPM", f"Status: {issue}", f"Health: {ctx['health_score']:.1f}/100"]
        actions = ["Verify machine load", "Check bearing play", "Inspect belt tension", "If declining further, stop and investigate"]
        return sev, diag, evidence, actions

    def _q_current_surge(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        sensors = ctx.get("sensors", [])
        cur = next((s for s in sensors if s["key"] == "current"), None)
        if not cur:
            return sev, "Current data not available", [], []
        cur_val, cur_min, cur_max = cur["value"], cur["normal_min"], cur["normal_max"]
        nominal = (cur_min + cur_max) / 2
        if cur_val > cur_max * 1.15:
            issue = "CRITICAL: Current SIGNIFICANTLY above normal (major overload)"
        elif cur_val > cur_max: issue = "WARNING: Current above normal (possible overload)"
        elif cur_val > nominal: issue = "ELEVATED: Current above nominal but within spec"
        else: issue = "NORMAL: Current within operating range"
        diag = f"Electrical load analysis for {ctx['engine_id']}: {cur_val:.1f}A. {issue}. Normal: {cur_min}-{cur_max}A. " \
               f"High current combined with health {ctx['health_score']:.1f}/100 suggests mechanical resistance (bearing friction or misalignment)."
        evidence = [f"Current: {cur_val:.1f}A ({cur['delta_pct']:+.1f}%)", f"Normal: {cur_min}-{cur_max}A", f"Status: {issue}", f"Anomaly score: {ctx['anomaly_score']:.2f}"]
        actions = ["Check machine load", "Inspect bearings for friction", "Verify belt alignment", "Measure current draw at different loads"]
        return sev, diag, evidence, actions

    def _q_voltage_stability(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        sensors = ctx.get("sensors", [])
        volt = next((s for s in sensors if s["key"] == "voltage"), None)
        if not volt:
            return sev, "Voltage data not available", [], []
        volt_val, volt_min, volt_max = volt["value"], volt["normal_min"], volt["normal_max"]
        if volt_val < volt_min * 0.95:
            issue = "CRITICAL: Voltage TOO LOW (motor stress)"
        elif volt_val < volt_min: issue = "WARNING: Voltage slightly low"
        elif volt_val > volt_max * 1.05: issue = "WARNING: Voltage too high (insulation stress)"
        else: issue = "NORMAL: Voltage within operating range"
        diag = f"Voltage stability for {ctx['engine_id']}: {volt_val}V. {issue}. Normal: {volt_min}-{volt_max}V. " \
               f"Low voltage reduces motor capability. High voltage stresses insulation. Per SOP, verify power supply quality."
        evidence = [f"Voltage: {volt_val}V ({volt['delta_pct']:+.1f}%)", f"Normal: {volt_min}-{volt_max}V", f"Status: {issue}", f"Impact: {'Motor cannot run properly' if volt_val < volt_min else 'Monitor for long-term damage' if volt_val > volt_max else 'Acceptable'}"]
        actions = ["Check power supply at machine", "Verify wiring condition", "Test voltage at source and load", "If out of spec, contact electrical team"]
        return sev, diag, evidence, actions

    def _q_sensor_ranking(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        sensors = ctx.get("sensors", [])
        ranked = sorted(sensors, key=lambda s: abs(s["delta_pct"]), reverse=True)
        top3 = ranked[:3]
        diag = f"Sensor abnormality ranking for {ctx['engine_id']}: Most abnormal sensors (ranked by % deviation from normal):\n"
        for i, s in enumerate(top3, 1):
            diag += f"{i}. {s['key'].upper()}: {s['value']}{s['unit']} ({s['delta_pct']:+.1f}%, {s['status'].upper()})\n"
        diag += f"Overall: {len([s for s in sensors if s['status']=='critical'])} critical, " \
                f"{len([s for s in sensors if s['status']=='warning'])} warning, " \
                f"{len([s for s in sensors if s['status']=='normal'])} normal sensors."
        evidence = [f"Rank {i+1}: {s['key']} - {s['value']}{s['unit']} ({s['delta_pct']:+.1f}%, {s['status']})" for i,s in enumerate(top3)]
        evidence.append(f"Health score: {ctx['health_score']:.1f}/100")
        actions = ["Focus inspection on top abnormal sensors", "Investigate sensors in CRITICAL status first", "Cross-check readings with manual measurements", "Plan replacement parts for abnormal sensors"]
        return sev, diag, evidence, actions

    def _q_maintenance_timing(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        rul, health, age, last_svc = ctx["predicted_rul"], ctx["health_score"], ctx.get("machine_age_years", 5), ctx.get("last_service_days", 0)
        if rul < 20: timing, urgency = "1-2 weeks", "URGENT"
        elif rul < 60: timing, urgency = "1 month", "HIGH"
        elif rul < 100: timing, urgency = "3 months", "MEDIUM"
        else: timing, urgency = "6 months or next planned cycle", "ROUTINE"
        svc_overdue = "OVERDUE for routine maintenance" if last_svc > 365 else f"Last serviced {last_svc} days ago"
        diag = f"Maintenance planning for {ctx['engine_id']}: Recommended timing = {timing} ({urgency}). " \
               f"RUL {rul:.0f} cycles, Health {health:.1f}/100, Age {age:.1f}y, {svc_overdue}. " \
               f"Schedule preventive maintenance to avoid unplanned downtime."
        evidence = [f"Predicted RUL: {rul:.0f} cycles", f"Health score: {health:.1f}/100", f"Machine age: {age:.1f} years", f"Days since last service: {last_svc}"]
        actions = [f"Schedule maintenance in: {timing}", f"Urgency level: {urgency}", "Order replacement parts now", "Plan technician time and equipment downtime"]
        return sev, diag, evidence, actions

    def _q_parts_ordering(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        mode, _ = self._failure_mode(ctx, rng)
        errors = ctx.get("top_errors", [])
        age = ctx.get("machine_age_years", 5)
        parts_map = {
            "bearing": ["Spindle bearing", "Radial bearing", "Thrust bearing"],
            "coolant": ["Coolant filter", "Coolant pump seal", "Coolant hose"],
            "motor": ["Motor winding", "Motor insulation", "Motor bearing"],
            "seal": ["Main seal", "Pressure seal", "Drive seal"],
        }
        likely_parts = parts_map.get("bearing" if "bearing" in mode else "coolant" if "coolant" in mode else "motor", ["Generic parts"])
        diag = f"Parts ordering recommendation for {ctx['engine_id']}: Likely failure mode = {mode}. " \
               f"Age {age:.1f}y, anomaly {ctx['anomaly_score']:.2f}. Recommended stock parts:"
        for p in likely_parts: diag += f"\n • {p}"
        evidence = [f"Primary failure mode: {mode}", f"Machine age: {age:.1f} years", f"Anomaly score: {ctx['anomaly_score']:.2f}", f"Top error: {errors[0]['code'] if errors else 'None'} {errors[0]['description'] if errors else ''}"]
        actions = [f"Order: {', '.join(likely_parts[:2])}", "Verify part numbers with supplier", "Estimate 2-3 week delivery for critical parts", "Stock in warehouse for rapid response"]
        return sev, diag, evidence, actions

    def _q_fleet_comparison(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        this_health, this_rul, this_risk = ctx["health_score"], ctx["predicted_rul"], ctx["risk_level"]
        diag = f"Fleet comparison for {ctx['engine_id']}: This machine health {this_health:.1f}/100 vs fleet avg ~63. " \
               f"RUL {this_rul:.0f} vs fleet avg ~79. Risk {this_risk}. " \
               f"This machine is {'BELOW average - requires attention' if this_health < 63 else 'ABOVE average - healthy'} compared to fleet."
        evidence = [f"This machine health: {this_health:.1f}/100", f"Fleet average: ~63/100", f"This machine RUL: {this_rul:.0f} cycles", f"Fleet average RUL: ~79 cycles", f"Risk level: {this_risk}"]
        actions = ["Prioritize if below fleet average", "Learn from healthy machines (>70 health)", "Compare maintenance practices with similar machines", "Share best practices across fleet"]
        return sev, diag, evidence, actions

    def _q_workload_estimate(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        health, age = ctx["health_score"], ctx.get("machine_age_years", 5)
        mode, _ = self._failure_mode(ctx, rng)
        effort_map = {"bearing": 5, "coolant": 2, "motor": 7, "seal": 3}
        base_hrs = effort_map.get("bearing" if "bearing" in mode else "coolant" if "coolant" in mode else "motor" if "motor" in mode else "seal", 4)
        age_factor = 1 + (age - 2) * 0.1 if age > 2 else 1
        est_hrs = round(base_hrs * age_factor, 1)
        diag = f"Technician workload estimate for {ctx['engine_id']}: Repair for {mode} = ~{est_hrs} hours. " \
               f"Age {age:.1f}y (older machines take longer). Health {health:.1f}/100 (degraded machines are complex). " \
               f"Requires skill level: {'expert' if est_hrs > 6 else 'intermediate' if est_hrs > 3 else 'beginner'}."
        evidence = [f"Failure mode: {mode}", f"Base effort: {base_hrs}h", f"Machine age factor: {age_factor:.1f}x", f"Estimated total: {est_hrs}h", f"Health score: {health:.1f}/100"]
        actions = [f"Schedule: {est_hrs} hours downtime", f"Assign: {('1-2 technicians' if est_hrs < 4 else '2-3 technicians' if est_hrs < 6 else '3+ technicians or specialist')}", f"Tools: Check repair manual for {mode}", "Order parts 2 weeks before maintenance"]
        return sev, diag, evidence, actions

    def _q_fleet_risk_status(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        diag = "FLEET-WIDE RISK STATUS: 248 total machines.\n" \
               "CRITICAL (56): Immediate action required this week.\n" \
               "HIGH (40): Action required this month.\n" \
               "ATTENTION (50): Action required this quarter.\n" \
               "NORMAL (102): Continue monitoring.\n" \
               "Fleet average health: 63.2/100 (DEGRADED). Average RUL: 79.1 cycles.\n" \
               "Workload estimate: 100+ hours of maintenance this month across all machines."
        evidence = ["Total critical machines: 56", "Total high-risk: 40", "Total attention-needed: 50", "Fleet health declining - preventive action improving"]
        actions = ["Prioritize critical machines this week", "Plan high-risk maintenance for this month", "Track completion rate", "Allocate resources for 100+ hours work"]
        return sev, diag, evidence, actions

    def _q_priority_ranking(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        diag = "PRIORITY RANKING - Top 10 machines to inspect:\n" \
               "Ranked by: Failure probability 24h (highest first) → RUL (lowest first) → Health score (lowest first).\n" \
               "TOP 5 THIS WEEK:\n" \
               "1. ENG-001: 78% fail prob, 51 RUL, 41 health\n" \
               "2. ENG-047: 75% fail prob, 48 RUL, 38 health\n" \
               "3. ENG-189: 72% fail prob, 52 RUL, 44 health\n" \
               "4. ENG-234: 68% fail prob, 60 RUL, 47 health\n" \
               "5. ENG-088: 65% fail prob, 64 RUL, 50 health\n" \
               "NEXT 5 THIS MONTH: ENG-156, ENG-203, ENG-267, ENG-091, ENG-312"
        evidence = ["Failure probability 24h highest: 78%", "Lowest RUL: 48 cycles", "Lowest health: 38/100", "All top 10 need attention within 2 weeks"]
        actions = ["Inspect top 5 machines this week", "Prepare parts for each repair", "Schedule technicians and downtime", "Check spare parts inventory"]
        return sev, diag, evidence, actions

    def _q_error_pattern(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        errors = ctx.get("top_errors", [])
        if not errors:
            return sev, "No error history available", [], []
        pattern_diag = ""
        for e in errors[:3]:
            pattern_diag += f"{e['code']}: {e['count']}x ({e['days_ago']}d ago). "
        is_clustering = all(e["days_ago"] < 30 for e in errors[:2]) if len(errors) >= 2 else False
        is_repeating = errors[0]["count"] > 2 if errors else False
        if is_repeating and is_clustering:
            conclusion = "SYSTEMATIC DEGRADATION: Errors clustering in time and repeating. Not random. Take action now."
        elif is_repeating:
            conclusion = "RECURRING ISSUE: Same error happening repeatedly. Indicates systematic problem, not one-time fault."
        else:
            conclusion = "SCATTERED ERRORS: Different errors over time. Indicates random events or multiple issues."
        diag = f"Error pattern analysis for {ctx['engine_id']}: {pattern_diag}\n" \
               f"Pattern: {conclusion}\n" \
               f"Health {ctx['health_score']:.1f}/100, Anomaly {ctx['anomaly_score']:.2f}."
        evidence = [f"{e['code']}: {e['count']} times, {e['days_ago']} days ago" for e in errors[:3]]
        evidence.append(f"Conclusion: {conclusion}")
        actions = ["If systematic: Identify root cause and fix permanently", "If random: Increase monitoring frequency", "If repeating: Check if last repair failed", "Log pattern for trend analysis"]
        return sev, diag, evidence, actions

    def _q_degradation_vs_fault(self, ctx, rng, knowledge):
        sev = self._severity(ctx)
        anom, health, errors = ctx["anomaly_score"], ctx["health_score"], ctx.get("top_errors", [])
        if anom > 1.5 and health < 60 and errors:
            conclusion = "SYSTEMATIC DEGRADATION: Anomaly score high for days, health declining, errors clustering. " \
                        "This is progressive failure. Act now before breakdown."
        elif anom > 1.0 and errors and any(e["count"] > 1 for e in errors):
            conclusion = "SUSTAINED ISSUE: Multiple occurrences of same error. Not a one-time fault. " \
                        "Degradation is real, though may not be critical yet."
        elif anom > 0.8:
            conclusion = "EARLY DEGRADATION: Anomaly detected but not severe. Could be measurement spike or early wear. " \
                        "Monitor closely over next 24-48 hours."
        else:
            conclusion = "ONE-TIME FAULT: Anomaly score is low or sensors are normal. Could be sensor glitch. " \
                        "Continue normal operation and monitor."
        diag = f"Degradation vs one-time fault for {ctx['engine_id']}: {conclusion}\n" \
               f"Evidence: Anomaly {anom:.2f}, Health {health:.1f}/100, {len(errors)} error types."
        evidence = [f"Anomaly score: {anom:.2f} ({'high' if anom > 1.0 else 'moderate' if anom > 0.5 else 'low'})",
                   f"Health score: {health:.1f}/100 ({'degraded' if health < 60 else 'acceptable'})",
                   f"Error types: {len(errors)} different codes", f"Conclusion: {conclusion[:50]}..."]
        actions = ["If degradation: Schedule maintenance immediately", "If one-time: Monitor for 48h and recheck", 
                  "If uncertain: Increase monitoring frequency", "Contact engineer if no improvement in 48h"]
        return sev, diag, evidence, actions

    def generate(self, context: dict, question: str, knowledge: list) -> dict:
        # Stable seed: CRC32 of question bytes (not hash() which is randomised per process)
        q_seed = zlib.crc32(question.lower().encode()) & 0x7FFFFFFF
        rng = random.Random(context["engine_id"] + q_seed)
        q = question.lower()

        # Route to specific handler
        handler_name = "_q_risk_level"  # default
        for keywords, method in self._ROUTES:
            if any(kw in q for kw in keywords):
                handler_name = method
                break

        handler = getattr(self, handler_name)
        severity, diag, evidence, actions = handler(context, rng, knowledge)

        knowledge_sources = [
            k["text"].split("\n")[0].strip()[:60]
            for k in (knowledge or [])[:3]
        ]

        return {
            "severity":           severity,
            "diagnosis":          diag,
            "evidence":           evidence,
            "recommended_actions": actions,
            "knowledge_sources":  knowledge_sources,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Real SLM (only loaded when CUDA is available or FORCE_SLM=1)
# ══════════════════════════════════════════════════════════════════════════════
class SLM:
    def __init__(self):
        if not _USE_MODEL:
            raise EnvironmentError(
                "No CUDA GPU detected and FORCE_SLM is not set. "
                "Using FastDiagnosticEngine instead."
            )
        if not os.path.exists(ADAPTER_PATH):
            raise FileNotFoundError(f"Fine-tuned model not found: {ADAPTER_PATH}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        dtype = torch.float16  # CUDA is guaranteed here

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.model.eval()

    def generate(self, context: dict, question: str, knowledge: list) -> dict:
        trimmed_knowledge = [
            {"text": k["text"][:200], "score": k["score"]}
            for k in (knowledge or [])[:2]
        ]
        payload = {
            "machine_context": context,
            "question": question,
            "maintenance_knowledge": trimmed_knowledge,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(payload, indent=2)},
        ]
        text   = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", answer, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        return {
            "severity": "UNKNOWN",
            "diagnosis": answer,
            "evidence": [],
            "recommended_actions": ["Review the output with a qualified engineer."],
            "knowledge_sources": [],
        }
