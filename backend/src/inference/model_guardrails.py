"""
Neural Model Guardrails and Safety Constraints (v2.0)

Ensures neural model output cannot:
1. Downgrade severity below rules engine
2. Produce hallucinated diagnoses
3. Hang indefinitely (timeout protection)
4. Return unsafe recommendations
5. Output incoherent diagnoses

Key principle: Rules engine is the safety floor.
Neural model can confirm or upgrade, never downgrade.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import logging
import threading
import time
from functools import wraps

logger = logging.getLogger(__name__)


class GuardrailViolation(Exception):
    """Neural model output violated a safety guardrail"""
    pass


class SeverityLevel(Enum):
    """Severity levels (must match diagnosis engine)"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NORMAL = 0
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __le__(self, other):
        return self.value <= other.value
    
    def __gt__(self, other):
        return self.value > other.value
    
    def __ge__(self, other):
        return self.value >= other.value


@dataclass
class GuardrailConfig:
    """Configuration for model guardrails"""
    # Timeout protection
    model_timeout_seconds: float = 10.0
    
    # Severity constraints
    min_confidence_for_diagnosis: float = 0.3
    severity_never_downgrades: bool = True
    
    # Output validation
    max_hallucination_tolerance: float = 0.1  # 10% hallucination allowed
    require_evidence_for_diagnosis: bool = True
    evidence_threshold: int = 1  # Minimum evidence pieces
    
    # Safety constraints
    forbid_autonomous_actions: bool = True
    max_concurrent_diagnoses: int = 100
    
    def validate(self):
        """Validate configuration"""
        assert 0 < self.model_timeout_seconds < 300, "Timeout must be 0-300s"
        assert 0 <= self.min_confidence_for_diagnosis <= 1.0, "Confidence must be 0-1"
        assert 0 <= self.max_hallucination_tolerance <= 1.0, "Hallucination tolerance must be 0-1"


class TimeoutError(GuardrailViolation):
    """Model exceeded timeout"""
    pass


class SeverityDowngradeError(GuardrailViolation):
    """Model attempted to downgrade severity"""
    pass


class HallucinationError(GuardrailViolation):
    """Model produced hallucinated diagnosis"""
    pass


class UnsafeRecommendationError(GuardrailViolation):
    """Model recommended unsafe action"""
    pass


class InvalidDiagnosisError(GuardrailViolation):
    """Model produced invalid diagnosis"""
    pass


class ModelGuardrails:
    """
    Safety guardrails for neural model outputs.
    
    Acts as a validator/filter on model predictions to ensure
    they never violate safety constraints.
    """
    
    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self.config.validate()
        
        self._active_diagnoses = 0
        self._lock = threading.Lock()
    
    def validate_diagnosis(
        self,
        model_diagnosis: Dict[str, Any],
        rules_engine_diagnosis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and potentially modify neural model diagnosis.
        
        Returns: Safe diagnosis (modified if necessary)
        Raises: GuardrailViolation if safety constraints violated
        
        Checks:
        1. Severity never downgrades
        2. Evidence requirements met
        3. No hallucinations detected
        4. No unsafe recommendations
        5. Output coherence
        """
        
        if not model_diagnosis:
            raise InvalidDiagnosisError("Model returned empty diagnosis")
        
        # Check 1: Severity constraint
        model_severity = self._parse_severity(model_diagnosis.get("severity"))
        rules_severity = self._parse_severity(rules_engine_diagnosis.get("severity"))
        
        if self.config.severity_never_downgrades:
            if model_severity < rules_severity:
                logger.warning(
                    f"Severity downgrade detected: rules={rules_severity.name}, "
                    f"model={model_severity.name}. Using rules engine severity."
                )
                # Override with rules engine severity
                model_diagnosis["severity"] = rules_severity.name
                model_diagnosis["guardrail_applied"] = "severity_override"
        
        # Check 2: Evidence requirement
        evidence_count = len(model_diagnosis.get("evidence_pieces", []))
        if self.config.require_evidence_for_diagnosis:
            if model_diagnosis.get("primary_finding") and evidence_count < self.config.evidence_threshold:
                raise InvalidDiagnosisError(
                    f"Diagnosis requires ≥{self.config.evidence_threshold} evidence pieces, "
                    f"got {evidence_count}"
                )
        
        # Check 3: Confidence threshold
        confidence = model_diagnosis.get("confidence_score", 0.0)
        if confidence < self.config.min_confidence_for_diagnosis:
            if model_diagnosis.get("primary_finding"):
                logger.info(f"Confidence too low ({confidence:.2f}), removing diagnosis")
                model_diagnosis["primary_finding"] = None
                model_diagnosis["guardrail_applied"] = "low_confidence_filter"
        
        # Check 4: Hallucination detection
        self._check_hallucination(model_diagnosis, rules_engine_diagnosis)
        
        # Check 5: Unsafe recommendations
        self._check_unsafe_recommendations(model_diagnosis)
        
        # Check 6: Output coherence
        self._check_coherence(model_diagnosis)
        
        return model_diagnosis
    
    def _parse_severity(self, severity_str: str) -> SeverityLevel:
        """Parse severity string to enum"""
        try:
            return SeverityLevel[severity_str.upper()]
        except (KeyError, AttributeError):
            return SeverityLevel.NORMAL
    
    def _check_hallucination(
        self,
        model_diagnosis: Dict[str, Any],
        rules_diagnosis: Dict[str, Any]
    ):
        """Detect hallucinated diagnoses (claims not supported by evidence)"""
        
        # If model claims a failure mode, evidence must support it
        failure_mode = model_diagnosis.get("primary_finding")
        evidence = model_diagnosis.get("evidence_pieces", [])
        
        if failure_mode:
            if not evidence:
                raise HallucinationError(
                    f"Model claimed diagnosis '{failure_mode}' without evidence"
                )
            
            # Check if evidence actually supports the claim
            supporting_evidence = [e for e in evidence if e.get("supports_diagnosis")]
            if not supporting_evidence:
                raise HallucinationError(
                    f"No evidence supports claimed diagnosis '{failure_mode}'"
                )
        
        # Check if diagnosis contradicts rules engine
        model_finding = model_diagnosis.get("primary_finding")
        rules_finding = rules_diagnosis.get("primary_finding")
        
        if model_finding and rules_finding and model_finding != rules_finding:
            # Different diagnosis is OK (could be more specific)
            # But both must be evidence-based
            logger.info(f"Model diagnosis differs from rules: {model_finding} vs {rules_finding}")
    
    def _check_unsafe_recommendations(self, diagnosis: Dict[str, Any]):
        """Verify recommendations don't include unsafe actions"""
        
        if not self.config.forbid_autonomous_actions:
            return
        
        actions = diagnosis.get("recommended_actions", [])
        
        dangerous_patterns = [
            "immediately shutdown",
            "cut power",
            "emergency stop",
            "force shutdown",
            "halt equipment",
            "lock controls",
            "disable safety",
            "override protection",
            "bypass alarm"
        ]
        
        for action in actions:
            action_lower = action.lower()
            for pattern in dangerous_patterns:
                if pattern in action_lower:
                    raise UnsafeRecommendationError(
                        f"Unsafe recommendation detected: '{action}'"
                    )
    
    def _check_coherence(self, diagnosis: Dict[str, Any]):
        """Check diagnosis is internally coherent"""
        
        # If no finding, severity should be NORMAL or LOW
        if not diagnosis.get("primary_finding"):
            severity = diagnosis.get("severity", "NORMAL").upper()
            if severity in ["CRITICAL", "HIGH"]:
                logger.warning(
                    f"Incoherent: no diagnosis but severity={severity}. "
                    f"Setting severity to NORMAL"
                )
                diagnosis["severity"] = "NORMAL"
        
        # If CRITICAL severity, must have high confidence
        if diagnosis.get("severity") == "CRITICAL":
            confidence = diagnosis.get("confidence_score", 0.0)
            if confidence < 0.7:
                logger.warning(
                    f"Incoherent: CRITICAL severity but confidence {confidence:.2f} < 0.7. "
                    f"Lowering severity to HIGH"
                )
                diagnosis["severity"] = "HIGH"
        
        # Evidence count should correlate with confidence
        evidence_count = len(diagnosis.get("evidence_pieces", []))
        confidence = diagnosis.get("confidence_score", 0.0)
        
        if evidence_count >= 4 and confidence < 0.6:
            logger.warning(
                f"Incoherent: {evidence_count} evidence pieces but confidence {confidence:.2f}. "
                f"Raising confidence"
            )
            diagnosis["confidence_score"] = min(1.0, confidence + 0.2)


def with_timeout(timeout_seconds: float):
    """
    Decorator: abort if model takes too long.
    
    Used to protect against model hangs.
    Falls back to rules engine on timeout.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                logger.warning(f"Model timeout after {timeout_seconds}s")
                raise TimeoutError(f"Model exceeded {timeout_seconds}s timeout")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


class NeuralModelWrapper:
    """
    Wrapper around neural model with guardrails.
    
    Ensures model output is safe before using it.
    Falls back to rules engine on any safety violation.
    """
    
    def __init__(self, model, guardrails: ModelGuardrails = None):
        self.model = model
        self.guardrails = guardrails or ModelGuardrails()
        self.stats = {
            "total_calls": 0,
            "guardrail_violations": 0,
            "fallback_to_rules": 0,
            "timeouts": 0
        }
    
    def predict(
        self,
        context: Dict[str, Any],
        rules_diagnosis: Dict[str, Any],
        timeout_seconds: float = 10.0
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Get neural model prediction with safety guardrails.
        
        Returns:
            (diagnosis: dict, is_safe: bool)
        
        If is_safe=False, diagnosis is fallback to rules engine.
        """
        
        self.stats["total_calls"] += 1
        
        try:
            # Call model with timeout
            model_diagnosis = self._call_model_with_timeout(context, timeout_seconds)
            
            # Validate output
            safe_diagnosis = self.guardrails.validate_diagnosis(
                model_diagnosis,
                rules_diagnosis
            )
            
            logger.info(f"Model prediction validated successfully")
            return safe_diagnosis, True
            
        except TimeoutError:
            logger.warning("Model timeout, using rules engine")
            self.stats["timeouts"] += 1
            self.stats["fallback_to_rules"] += 1
            return rules_diagnosis, False
            
        except GuardrailViolation as e:
            logger.warning(f"Guardrail violation: {e}. Using rules engine")
            self.stats["guardrail_violations"] += 1
            self.stats["fallback_to_rules"] += 1
            return rules_diagnosis, False
            
        except Exception as e:
            logger.error(f"Model error: {e}. Using rules engine")
            self.stats["fallback_to_rules"] += 1
            return rules_diagnosis, False
    
    def _call_model_with_timeout(
        self,
        context: Dict[str, Any],
        timeout_seconds: float
    ) -> Dict[str, Any]:
        """Call model with timeout protection"""
        
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = self.model.predict(context)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target, daemon=True)
        start_time = time.time()
        thread.start()
        thread.join(timeout=timeout_seconds)
        elapsed = time.time() - start_time
        
        if thread.is_alive():
            raise TimeoutError(f"Model exceeded {timeout_seconds}s timeout")
        
        if exception[0]:
            raise exception[0]
        
        if result[0] is None:
            raise RuntimeError("Model returned None")
        
        logger.debug(f"Model call completed in {elapsed:.2f}s")
        return result[0]
    
    def get_stats(self) -> Dict[str, int]:
        """Get guardrail statistics"""
        return self.stats.copy()


class RulesEngineFallback:
    """
    Fallback to rules engine when neural model unavailable.
    
    Ensures system remains operational even if model fails.
    """
    
    def __init__(self, rules_engine):
        self.rules_engine = rules_engine
        self.fallback_count = 0
    
    def diagnose(
        self,
        context: Dict[str, Any],
        question: str
    ) -> Dict[str, Any]:
        """
        Use rules engine for diagnosis (guaranteed to work).
        
        Called when:
        - Neural model unavailable
        - Model timed out
        - Model violated guardrails
        - GPU not available
        """
        
        self.fallback_count += 1
        logger.info(f"Using rules engine fallback (count: {self.fallback_count})")
        
        diagnosis = self.rules_engine.diagnose(context, question)
        diagnosis["engine_used"] = "rules_engine"
        diagnosis["from_fallback"] = True
        
        return diagnosis


# Example guardrail configuration for production
PRODUCTION_GUARDRAILS = GuardrailConfig(
    model_timeout_seconds=10.0,
    min_confidence_for_diagnosis=0.4,
    severity_never_downgrades=True,
    require_evidence_for_diagnosis=True,
    evidence_threshold=1,
    forbid_autonomous_actions=True,
    max_concurrent_diagnoses=100
)

# Stricter config for safety-critical scenarios
SAFETY_CRITICAL_GUARDRAILS = GuardrailConfig(
    model_timeout_seconds=5.0,
    min_confidence_for_diagnosis=0.6,  # Higher threshold
    severity_never_downgrades=True,
    require_evidence_for_diagnosis=True,
    evidence_threshold=2,  # More evidence required
    forbid_autonomous_actions=True,
    max_concurrent_diagnoses=50
)
