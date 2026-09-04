"""State engine package: trigger routing and compliance state machine.

Public API::

    from src.state_engine import TriggerRouter, ComplianceStateMachine

    router = TriggerRouter()
    machine = ComplianceStateMachine(router=router)
    state = machine.evaluate_frame_state(detections, osm_policy, vlm_audit)
"""

from __future__ import annotations

from src.state_engine.compliance_state_machine import ComplianceStateMachine
from src.state_engine.trigger_router import TriggerRouter

__version__ = "0.1.0"

__all__ = [
    "ComplianceStateMachine",
    "TriggerRouter",
    "__version__",
]
