"""
Audit Trail Module

Comprehensive event logging for compliance and traceability.
"""

from .audit_trail import (
    AuditTrailManager,
    AuditLogEntry,
    EventType,
    EngineerDecision,
    InMemoryAuditStore,
    create_audit_manager
)

__all__ = [
    "AuditTrailManager",
    "AuditLogEntry",
    "EventType",
    "EngineerDecision",
    "InMemoryAuditStore",
    "create_audit_manager"
]
