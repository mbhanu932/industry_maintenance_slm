"""
Simple Audit Logging
Practical audit trail for compliance
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class SimpleAuditLogger:
    """Simple file-based audit logging."""
    
    def __init__(self, log_dir: str = "./logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Setup file logger
        self.logger = logging.getLogger("audit")
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_action(self, 
                   user_id: str,
                   action: str,
                   resource: str,
                   status: str = "success",
                   details: Optional[Dict[str, Any]] = None):
        """Log an action for audit trail."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "status": status,
            "details": details or {}
        }
        
        self.logger.info(json.dumps(event))
    
    def log_diagnosis(self, user_id: str, machine_id: int, severity: str):
        """Log diagnosis action."""
        self.log_action(
            user_id=user_id,
            action="diagnosis",
            resource=f"machine_{machine_id}",
            details={"severity": severity}
        )
    
    def log_login(self, user_id: str, success: bool):
        """Log login attempt."""
        self.log_action(
            user_id=user_id,
            action="login",
            resource="authentication",
            status="success" if success else "failed"
        )
    
    def log_maintenance(self, user_id: str, machine_id: int, maintenance_type: str):
        """Log maintenance action."""
        self.log_action(
            user_id=user_id,
            action="maintenance",
            resource=f"machine_{machine_id}",
            details={"type": maintenance_type}
        )

# Global audit logger instance
audit_logger = SimpleAuditLogger()