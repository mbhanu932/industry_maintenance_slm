"""
Evidence-Based Diagnosis Engine (v2.0)

Replaces random failure mode selection with evidence scoring.
Ensures SOP §5 compliance: "Do not diagnose without supporting evidence."

Key principles:
1. Symptoms extracted from sensor data
2. All possible failure modes scored against symptoms
3. Top diagnosis returned WITH evidence (fully traceable)
4. Confidence level based on evidence count
5. Alternative hypotheses included for transparency
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Severity classification"""
    CRITICAL = "CRITICAL"  # Immediate action needed
    HIGH = "HIGH"           # Action needed within 24h
    MEDIUM = "MEDIUM"       # Schedule maintenance within week
    LOW = "LOW"             # Monitor, no immediate action
    NORMAL = "NORMAL"       # No issues detected


class ConfidenceLevel(Enum):
    """Confidence in diagnosis based on evidence count"""
    HIGH = "HIGH"           # 4+ evidence pieces
    MEDIUM = "MEDIUM"       # 2-3 evidence pieces
    LOW = "LOW"             # 1 evidence piece
    UNCERTAIN = "UNCERTAIN" # 0 evidence pieces


@dataclass
class Symptom:
    """A detected symptom (observation from sensor data)"""
    name: str                    # e.g., "high_vibration"
    observation: str             # What was observed
    current_value: float
    baseline_value: float
    deviation_pct: float
    sensor: str                  # Which sensor detected it
    weight: float = 0.5         # Importance (0.0-1.0)
    severity_contribution: str = "medium"  # low, medium, high
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "observation": self.observation,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "deviation_pct": self.deviation_pct,
            "sensor": self.sensor,
            "weight": self.weight
        }


@dataclass
class EvidencePiece:
    """Single piece of evidence supporting/refuting a diagnosis"""
    type: str                           # sensor_deviation, trend_analysis, error_correlation, etc.
    observation: str                    # Human-readable observation
    sensor: Optional[str]               # Which sensor (if applicable)
    current_value: Optional[float]      # Measured value
    baseline_value: Optional[float]     # Expected/normal value
    deviation_pct: float                # Percentage deviation
    supports_diagnosis: bool            # Does this support the hypothesis?
    confidence: float                   # How sure (0.0-1.0)
    sop_section: Optional[str]          # Reference to SOP (e.g., "§3.2")
    reason: str                         # Why this matters
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "observation": self.observation,
            "sensor": self.sensor,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "deviation_pct": self.deviation_pct,
            "supports_diagnosis": self.supports_diagnosis,
            "confidence": self.confidence,
            "sop_section": self.sop_section,
            "reason": self.reason
        }


@dataclass
class DiagnosisHypothesis:
    """A possible diagnosis with supporting evidence"""
    failure_mode: str                   # e.g., "bearing_wear"
    score: float                        # 0.0-1.0, higher is more likely
    evidence_pieces: List[EvidencePiece] = field(default_factory=list)
    supporting_symptoms: List[Symptom] = field(default_factory=list)
    contradicting_observations: List[str] = field(default_factory=list)
    
    @property
    def evidence_count(self) -> int:
        """How many evidence pieces support this"""
        return len([e for e in self.evidence_pieces if e.supports_diagnosis])
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Confidence based on evidence count"""
        if self.evidence_count >= 4:
            return ConfidenceLevel.HIGH
        elif self.evidence_count >= 2:
            return ConfidenceLevel.MEDIUM
        elif self.evidence_count >= 1:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def to_dict(self) -> dict:
        return {
            "failure_mode": self.failure_mode,
            "score": round(self.score, 3),
            "evidence_count": self.evidence_count,
            "confidence_level": self.confidence_level.value,
            "evidence_pieces": [e.to_dict() for e in self.evidence_pieces],
            "contradicting_observations": self.contradicting_observations
        }


@dataclass
class Diagnosis:
    """Final diagnosis result with full traceability"""
    primary_finding: Optional[str]
    severity: SeverityLevel
    confidence: ConfidenceLevel
    confidence_score: float             # 0.0-1.0
    evidence_pieces: List[EvidencePiece]
    hypotheses_considered: List[DiagnosisHypothesis]
    recommended_actions: List[str]
    sop_compliant: bool
    reasoning_trace: Dict
    
    def to_dict(self) -> dict:
        return {
            "primary_finding": self.primary_finding,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "confidence_score": round(self.confidence_score, 3),
            "evidence_count": len([e for e in self.evidence_pieces if e.supports_diagnosis]),
            "evidence_pieces": [e.to_dict() for e in self.evidence_pieces],
            "top_hypotheses": [h.to_dict() for h in sorted(
                self.hypotheses_considered, key=lambda x: x.score, reverse=True
            )[:3]],
            "recommended_actions": self.recommended_actions,
            "sop_compliant": self.sop_compliant,
            "reasoning_trace": self.reasoning_trace
        }


class SymptomExtractor:
    """Extract symptoms from sensor data"""
    
    # Symptom definitions: what triggers each symptom?
    SYMPTOM_DEFINITIONS = {
        "high_temperature": {
            "sensor": "temperature",
            "threshold_pct": 115,  # >15% above normal
            "failure_modes": ["thermal_stress", "cooling_failure", "ambient_high"],
            "weight": 0.8,
            "severity_contribution": "high"
        },
        "high_pressure": {
            "sensor": "pressure",
            "threshold_pct": 120,
            "failure_modes": ["hydraulic_fault", "blockage", "thermal_stress"],
            "weight": 0.7,
            "severity_contribution": "high"
        },
        "high_vibration": {
            "sensor": "vibration",
            "threshold_pct": 200,  # >100% above normal (mm/s)
            "failure_modes": ["bearing_wear", "unbalance", "misalignment"],
            "weight": 0.9,
            "severity_contribution": "high"
        },
        "rising_vibration_trend": {
            "sensor": "vibration",
            "trend_pct_per_day": 2.0,  # Rising 2%+ per day
            "failure_modes": ["bearing_degradation", "progressive_wear"],
            "weight": 0.85,
            "severity_contribution": "high"
        },
        "voltage_anomaly": {
            "sensor": "voltage",
            "threshold_pct": 90,  # <90% nominal or >110%
            "failure_modes": ["supply_fault", "winding_fault", "regulator_failure"],
            "weight": 0.75,
            "severity_contribution": "medium"
        },
        "current_surge": {
            "sensor": "current",
            "threshold_pct": 120,
            "failure_modes": ["electrical_fault", "load_increase", "motor_fault"],
            "weight": 0.7,
            "severity_contribution": "medium"
        },
        "rotation_degradation": {
            "sensor": "rotation",
            "threshold_pct": 95,  # <95% nominal
            "failure_modes": ["bearing_wear", "load_increase", "motor_fault"],
            "weight": 0.8,
            "severity_contribution": "medium"
        },
        "error_history": {
            "source": "error_log",
            "failure_modes": ["sensor_fault", "controller_fault", "bearing_fault"],
            "weight": 0.6,
            "severity_contribution": "medium"
        }
    }
    
    @staticmethod
    def extract(context: dict) -> List[Symptom]:
        """Extract symptoms from machine context"""
        symptoms = []
        
        # Get sensor data
        sensors = context.get("sensors", {})
        
        # Temperature check
        if "temperature" in sensors:
            temp_data = sensors["temperature"]
            if temp_data["delta_pct"] > 15:  # >15% deviation
                symptoms.append(Symptom(
                    name="high_temperature",
                    observation=f"Temperature {temp_data['value']:.1f}°C (normal: {temp_data['normal_max']:.1f}°C)",
                    current_value=temp_data["value"],
                    baseline_value=temp_data["normal_max"],
                    deviation_pct=temp_data["delta_pct"],
                    sensor="temperature",
                    weight=0.8,
                    severity_contribution="high"
                ))
        
        # Vibration check
        if "vibration" in sensors:
            vib_data = sensors["vibration"]
            if vib_data["delta_pct"] > 100:  # >100% deviation
                symptoms.append(Symptom(
                    name="high_vibration",
                    observation=f"Vibration {vib_data['value']:.1f} mm/s (normal: {vib_data['normal_max']:.1f} mm/s)",
                    current_value=vib_data["value"],
                    baseline_value=vib_data["normal_max"],
                    deviation_pct=vib_data["delta_pct"],
                    sensor="vibration",
                    weight=0.9,
                    severity_contribution="high"
                ))
        
        # Voltage check
        if "voltage" in sensors:
            volt_data = sensors["voltage"]
            if volt_data["delta_pct"] < -10 or volt_data["delta_pct"] > 10:
                symptoms.append(Symptom(
                    name="voltage_anomaly",
                    observation=f"Voltage {volt_data['delta_pct']:.1f}% from nominal",
                    current_value=volt_data["value"],
                    baseline_value=(volt_data["normal_min"] + volt_data["normal_max"]) / 2,
                    deviation_pct=volt_data["delta_pct"],
                    sensor="voltage",
                    weight=0.75,
                    severity_contribution="medium"
                ))
        
        # Current check
        if "current" in sensors:
            curr_data = sensors["current"]
            if curr_data["delta_pct"] > 20:
                symptoms.append(Symptom(
                    name="current_surge",
                    observation=f"Current {curr_data['value']:.1f}A ({curr_data['delta_pct']:.1f}% above nominal)",
                    current_value=curr_data["value"],
                    baseline_value=curr_data["normal_max"],
                    deviation_pct=curr_data["delta_pct"],
                    sensor="current",
                    weight=0.7,
                    severity_contribution="medium"
                ))
        
        # Rotation degradation
        if "rotation" in sensors:
            rot_data = sensors["rotation"]
            if rot_data["delta_pct"] < -5:  # >5% below nominal
                symptoms.append(Symptom(
                    name="rotation_degradation",
                    observation=f"Rotation speed {rot_data['delta_pct']:.1f}% below nominal",
                    current_value=rot_data["value"],
                    baseline_value=rot_data["normal_max"],
                    deviation_pct=rot_data["delta_pct"],
                    sensor="rotation",
                    weight=0.8,
                    severity_contribution="medium"
                ))
        
        # Error history
        error_log = context.get("error_log", [])
        if error_log and len(error_log) > 0:
            symptoms.append(Symptom(
                name="error_history",
                observation=f"{len(error_log)} error(s) in history: {', '.join([e.get('code', 'UNKNOWN') for e in error_log[:3]])}",
                current_value=len(error_log),
                baseline_value=0,
                deviation_pct=100 if error_log else 0,
                sensor="system",
                weight=0.6,
                severity_contribution="medium"
            ))
        
        return symptoms


class FailureModeScoringEngine:
    """Score failure modes against observed symptoms"""
    
    # Failure mode definitions: symptoms → likelihood
    FAILURE_MODES = {
        "bearing_wear": {
            "description": "Progressive bearing degradation",
            "supporting_symptoms": {
                "high_vibration": 1.0,
                "rising_vibration_trend": 0.95,
                "rotation_degradation": 0.8,
                "high_temperature": 0.7,
                "error_history": 0.5
            },
            "sop_section": "§3.2 - Vibration Analysis & §4.1 - Bearing Maintenance",
            "typical_timeline": "3-5 days to failure",
            "actions": [
                "Schedule bearing replacement within 24 hours",
                "Increase vibration monitoring to every 4 hours",
                "Inspect lubrication system"
            ]
        },
        "thermal_stress": {
            "description": "Excessive heat generation or cooling failure",
            "supporting_symptoms": {
                "high_temperature": 1.0,
                "high_pressure": 0.7,
                "current_surge": 0.6,
                "error_history": 0.4
            },
            "sop_section": "§3.3 - Temperature Analysis & §4.2 - Cooling System",
            "typical_timeline": "Variable: 1-24 hours",
            "actions": [
                "Check cooling system immediately",
                "Reduce load if possible",
                "Monitor temperature every 15 minutes"
            ]
        },
        "electrical_fault": {
            "description": "Winding, supply, or controller issue",
            "supporting_symptoms": {
                "voltage_anomaly": 1.0,
                "current_surge": 0.9,
                "high_temperature": 0.5,
                "error_history": 0.7
            },
            "sop_section": "§3.4 - Electrical Analysis & §4.3 - Electrical Maintenance",
            "typical_timeline": "Hours to immediate",
            "actions": [
                "Check power supply quality",
                "Inspect wiring and connections",
                "Test winding resistance",
                "May require shutdown"
            ]
        },
        "misalignment": {
            "description": "Shaft or component misalignment",
            "supporting_symptoms": {
                "high_vibration": 0.8,
                "rotation_degradation": 0.6,
                "high_temperature": 0.5,
                "error_history": 0.3
            },
            "sop_section": "§3.5 - Alignment Check",
            "typical_timeline": "1-3 days to damage",
            "actions": [
                "Check mechanical alignment",
                "Inspect couplings",
                "Re-align if needed"
            ]
        },
        "sensor_fault": {
            "description": "Sensor reading anomaly or failure",
            "supporting_symptoms": {
                "error_history": 1.0,
                "voltage_anomaly": 0.5
            },
            "sop_section": "§3.6 - Sensor Diagnostics",
            "typical_timeline": "Immediate (false readings)",
            "actions": [
                "Verify sensor readings manually",
                "Check sensor connections",
                "Replace sensor if defective"
            ]
        }
    }
    
    @staticmethod
    def score_failure_mode(
        failure_mode: str,
        symptoms: List[Symptom]
    ) -> Tuple[float, List[EvidencePiece]]:
        """
        Score a failure mode against observed symptoms.
        
        Returns:
            (score: 0.0-1.0, evidence_pieces: list)
        """
        if failure_mode not in FailureModeScoringEngine.FAILURE_MODES:
            return 0.0, []
        
        mode_def = FailureModeScoringEngine.FAILURE_MODES[failure_mode]
        supporting_symptoms = mode_def["supporting_symptoms"]
        
        score = 0.0
        evidence_pieces = []
        total_weight = 0.0
        
        for symptom in symptoms:
            symptom_weight = supporting_symptoms.get(symptom.name, 0.0)
            
            if symptom_weight > 0:
                # This symptom supports the hypothesis
                weighted_contribution = symptom_weight * symptom.weight
                score += weighted_contribution
                total_weight += symptom.weight
                
                evidence = EvidencePiece(
                    type="sensor_deviation",
                    observation=symptom.observation,
                    sensor=symptom.sensor,
                    current_value=symptom.current_value,
                    baseline_value=symptom.baseline_value,
                    deviation_pct=symptom.deviation_pct,
                    supports_diagnosis=True,
                    confidence=min(symptom_weight, abs(symptom.deviation_pct) / 100),
                    sop_section=mode_def["sop_section"],
                    reason=f"{symptom.name} is consistent with {failure_mode}"
                )
                evidence_pieces.append(evidence)
        
        # Normalize score to 0.0-1.0
        if total_weight > 0:
            score = min(1.0, score / total_weight)
        
        return score, evidence_pieces


class EvidenceBasedDiagnosisEngine:
    """Main diagnosis engine: evidence-based scoring"""
    
    def __init__(self):
        self.symptom_extractor = SymptomExtractor()
        self.scoring_engine = FailureModeScoringEngine()
    
    def diagnose(self, context: dict, question: str) -> Diagnosis:
        """
        Generate evidence-based diagnosis.
        
        Steps:
        1. Extract symptoms from sensor data
        2. Score all possible failure modes
        3. Select top diagnosis with evidence
        4. Return with full traceability
        """
        
        # Step 1: Extract symptoms
        symptoms = self.symptom_extractor.extract(context)
        logger.info(f"Extracted {len(symptoms)} symptoms from machine {context.get('engine_id')}")
        
        # Step 2: Score all failure modes
        hypotheses = []
        for failure_mode in FailureModeScoringEngine.FAILURE_MODES.keys():
            score, evidence = self.scoring_engine.score_failure_mode(failure_mode, symptoms)
            
            hypothesis = DiagnosisHypothesis(
                failure_mode=failure_mode,
                score=score,
                evidence_pieces=evidence,
                supporting_symptoms=symptoms
            )
            hypotheses.append(hypothesis)
        
        # Sort by score
        hypotheses_sorted = sorted(hypotheses, key=lambda h: h.score, reverse=True)
        
        # Step 3: Determine primary finding
        primary_hypothesis = hypotheses_sorted[0]
        primary_finding = None
        evidence_threshold = 0.3  # Minimum score to make diagnosis
        
        if primary_hypothesis.score >= evidence_threshold and primary_hypothesis.evidence_count > 0:
            primary_finding = primary_hypothesis.failure_mode
        
        # Step 4: Determine severity
        severity = self._determine_severity(context, primary_hypothesis, symptoms)
        
        # Step 5: Confidence level
        confidence_level = primary_hypothesis.confidence_level
        confidence_score = min(1.0, primary_hypothesis.score + (primary_hypothesis.evidence_count * 0.15))
        
        # Step 6: Recommended actions
        if primary_finding:
            mode_def = FailureModeScoringEngine.FAILURE_MODES[primary_finding]
            actions = mode_def["actions"]
        else:
            actions = ["Continue monitoring", "Schedule preventive maintenance"]
        
        # Step 7: Traceability
        reasoning_trace = {
            "question": question,
            "symptoms_detected": len(symptoms),
            "symptoms": [s.to_dict() for s in symptoms[:3]],  # Top 3
            "failure_modes_considered": len(hypotheses_sorted),
            "top_hypotheses": [
                {
                    "mode": h.failure_mode,
                    "score": h.score,
                    "evidence_count": h.evidence_count,
                    "confidence": h.confidence_level.value
                }
                for h in hypotheses_sorted[:3]
            ],
            "primary_finding": primary_finding,
            "algorithm": "evidence_based_scoring_v2.0",
            "sop_compliant": True
        }
        
        return Diagnosis(
            primary_finding=primary_finding,
            severity=severity,
            confidence=confidence_level,
            confidence_score=confidence_score,
            evidence_pieces=primary_hypothesis.evidence_pieces,
            hypotheses_considered=hypotheses_sorted,
            recommended_actions=actions,
            sop_compliant=True,
            reasoning_trace=reasoning_trace
        )
    
    def _determine_severity(
        self,
        context: dict,
        hypothesis: DiagnosisHypothesis,
        symptoms: List[Symptom]
    ) -> SeverityLevel:
        """Determine severity based on context and symptoms"""
        
        health_score = context.get("health_score", 50)
        predicted_rul = context.get("predicted_rul", 100)
        anomaly_score = context.get("anomaly_score", 0.0)
        
        # Critical: low health, high anomaly, imminent failure
        if health_score < 30 or predicted_rul < 10 or anomaly_score > 1.5:
            return SeverityLevel.CRITICAL
        
        # High: multiple symptoms, moderate health degradation
        if len(symptoms) >= 3 and health_score < 50 and hypothesis.evidence_count >= 2:
            return SeverityLevel.HIGH
        
        # Medium: some symptoms, degrading
        if len(symptoms) >= 2 and health_score < 70:
            return SeverityLevel.MEDIUM
        
        # Low: minor symptoms only
        if len(symptoms) >= 1 and health_score < 85:
            return SeverityLevel.LOW
        
        # Normal: no significant symptoms
        return SeverityLevel.NORMAL


def create_diagnostic_engine() -> EvidenceBasedDiagnosisEngine:
    """Factory function"""
    return EvidenceBasedDiagnosisEngine()
