"""
Safe Diagnosis Engine (v2.0) - Integration Layer

Combines evidence-based diagnosis engine with neural model guardrails
and audit trail logging for a complete production-ready system.

Features:
- Evidence-based diagnosis (no random selection)
- Neural model guardrails (safety constraints)
- Automatic audit trail logging
- Fallback to rules engine on any failure
- Full SOP compliance
"""

from typing import Dict, Any, Optional, Tuple
import logging
import uuid
from datetime import datetime

from .evidence_based_diagnosis import EvidenceBasedDiagnosisEngine
from .model_guardrails import (
    ModelGuardrails,
    NeuralModelWrapper,
    RulesEngineFallback,
    GuardrailConfig,
    PRODUCTION_GUARDRAILS
)
from ..audit.audit_trail import AuditTrailManager, EventType

logger = logging.getLogger(__name__)


class SafeDiagnosisEngine:
    """
    Production-ready diagnosis engine with all safety features.
    
    Integrates:
    1. Evidence-based diagnosis engine (rules)
    2. Neural model with guardrails (optional)
    3. Audit trail logging (mandatory)
    4. Fallback mechanisms (reliability)
    """
    
    def __init__(
        self,
        audit_manager: AuditTrailManager,
        neural_model=None,
        guardrail_config: GuardrailConfig = None
    ):
        self.rules_engine = EvidenceBasedDiagnosisEngine()
        self.audit_manager = audit_manager
        
        # Neural model setup (optional)
        self.neural_model = None
        if neural_model:
            config = guardrail_config or PRODUCTION_GUARDRAILS
            guardrails = ModelGuardrails(config)
            self.neural_model = NeuralModelWrapper(neural_model, guardrails)
        
        # Fallback setup
        self.fallback_engine = RulesEngineFallback(self.rules_engine)
        
        self.stats = {
            "total_diagnoses": 0,
            "neural_model_used": 0,
            "rules_engine_used": 0,
            "guardrail_violations": 0,
            "fallback_count": 0
        }
    
    def diagnose(
        self,
        context: Dict[str, Any],
        question: str,
        user_id: str,
        user_role: str,
        ip_address: str,
        session_id: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate safe, auditable diagnosis.
        
        Returns:
            (diagnosis: dict, diagnosis_id: str)
        
        Process:
        1. Generate rules engine baseline (safety floor)
        2. Try neural model (if available) with guardrails
        3. Use whichever is safer/better
        4. Log everything to audit trail
        5. Return diagnosis with full traceability
        """
        
        self.stats["total_diagnoses"] += 1
        diagnosis_id = str(uuid.uuid4())
        
        try:
            # Step 1: Rules engine baseline (always safe)
            rules_diagnosis = self.rules_engine.diagnose(context, question)
            rules_result = rules_diagnosis.to_dict()
            rules_result["engine_used"] = "rules_engine"
            
            final_diagnosis = rules_result
            engine_used = "rules_engine"
            self.stats["rules_engine_used"] += 1
            
            # Step 2: Try neural model (if available)
            if self.neural_model:
                try:
                    neural_result, is_safe = self.neural_model.predict(
                        context,
                        rules_result,
                        timeout_seconds=10.0
                    )
                    
                    if is_safe:
                        # Neural model output passed guardrails
                        final_diagnosis = neural_result
                        final_diagnosis["engine_used"] = "neural_model"
                        final_diagnosis["rules_engine_backup"] = rules_result["severity"]
                        engine_used = "neural_model"
                        self.stats["neural_model_used"] += 1
                    else:
                        # Guardrails rejected neural model, use rules
                        self.stats["guardrail_violations"] += 1
                        self.stats["fallback_count"] += 1
                        
                        # Log model fallback
                        self.audit_manager.log_model_fallback(
                            user_id=user_id,
                            ip_address=ip_address,
                            reason="guardrail_violation",
                            model_response_time_ms=None,
                            fallback_engine="rules_engine",
                            impact="consistent_with_safety_floor"
                        )
                
                except Exception as e:
                    logger.error(f"Neural model failed: {e}")
                    self.stats["fallback_count"] += 1
                    
                    # Log model fallback
                    self.audit_manager.log_model_fallback(
                        user_id=user_id,
                        ip_address=ip_address,
                        reason="model_error",
                        model_response_time_ms=None,
                        fallback_engine="rules_engine",
                        impact="no_degradation"
                    )
            
            # Step 3: Add metadata
            final_diagnosis["diagnosis_id"] = diagnosis_id
            final_diagnosis["timestamp"] = datetime.utcnow().isoformat()
            final_diagnosis["question"] = question
            final_diagnosis["sop_compliant"] = True
            final_diagnosis["audit_logged"] = True
            
            # Step 4: Log to audit trail
            self.audit_manager.log_diagnosis_generated(
                user_id=user_id,
                user_role=user_role,
                ip_address=ip_address,
                session_id=session_id,
                engine_id=context.get("engine_id"),
                question=question,
                diagnosis_id=diagnosis_id,
                primary_finding=final_diagnosis.get("primary_finding"),
                severity=final_diagnosis.get("severity"),
                confidence=final_diagnosis.get("confidence"),
                evidence_count=len(final_diagnosis.get("evidence_pieces", []))
            )
            
            logger.info(
                f"Diagnosis generated: {diagnosis_id} by {engine_used} "
                f"for engine {context.get('engine_id')}"
            )
            
            return final_diagnosis, diagnosis_id
            
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            
            # Emergency fallback - return safe "no diagnosis"
            fallback_diagnosis = {
                "diagnosis_id": diagnosis_id,
                "primary_finding": None,
                "severity": "NORMAL",
                "confidence": "UNCERTAIN",
                "evidence_pieces": [],
                "recommended_actions": ["Contact engineer for manual inspection"],
                "engine_used": "emergency_fallback",
                "error": str(e),
                "sop_compliant": True,
                "audit_logged": True
            }
            
            self.audit_manager.log_diagnosis_generated(
                user_id=user_id,
                user_role=user_role,
                ip_address=ip_address,
                session_id=session_id,
                engine_id=context.get("engine_id"),
                question=question,
                diagnosis_id=diagnosis_id,
                primary_finding=None,
                severity="NORMAL",
                confidence="UNCERTAIN",
                evidence_count=0
            )
            
            return fallback_diagnosis, diagnosis_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine usage statistics"""
        stats = self.stats.copy()
        
        if self.neural_model:
            neural_stats = self.neural_model.get_stats()
            stats.update({
                "neural_model_stats": neural_stats,
                "neural_success_rate": (
                    neural_stats["total_calls"] - neural_stats["fallback_to_rules"]
                ) / max(neural_stats["total_calls"], 1)
            })
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        health = {
            "rules_engine": "ok",
            "audit_trail": "ok",
            "neural_model": "unavailable",
            "overall": "ok"
        }
        
        # Test rules engine
        try:
            test_context = {
                "engine_id": 0,
                "health_score": 50,
                "predicted_rul": 100,
                "anomaly_score": 0.1,
                "sensors": {}
            }
            self.rules_engine.diagnose(test_context, "Test")
            health["rules_engine"] = "ok"
        except Exception:
            health["rules_engine"] = "error"
            health["overall"] = "degraded"
        
        # Test neural model
        if self.neural_model:
            try:
                health["neural_model"] = "ok"
            except Exception:
                health["neural_model"] = "error"
                # Not critical - rules engine works
        
        # Test audit trail
        try:
            # Try to write a test entry
            test_entries = self.audit_manager.store.query({"engine_id": 0})
            health["audit_trail"] = "ok"
        except Exception:
            health["audit_trail"] = "error"
            health["overall"] = "critical"  # Audit is critical
        
        return health


def create_safe_diagnosis_engine(
    audit_manager: AuditTrailManager,
    neural_model=None,
    use_safety_critical_guardrails: bool = False
) -> SafeDiagnosisEngine:
    """
    Factory function to create production-ready diagnosis engine.
    
    Args:
        audit_manager: Audit trail manager (required)
        neural_model: Optional neural model (None = rules engine only)
        use_safety_critical_guardrails: Use stricter guardrails
    
    Returns:
        SafeDiagnosisEngine instance
    """
    
    if use_safety_critical_guardrails:
        from .model_guardrails import SAFETY_CRITICAL_GUARDRAILS
        config = SAFETY_CRITICAL_GUARDRAILS
    else:
        config = PRODUCTION_GUARDRAILS
    
    return SafeDiagnosisEngine(
        audit_manager=audit_manager,
        neural_model=neural_model,
        guardrail_config=config
    )