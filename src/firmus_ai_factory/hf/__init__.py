"""High-fidelity solver adapters (Ansys thermal, ETAP electrical).

Runtime side of the ``firmus_ai_factory.bod.high_fidelity`` contract.
The optimizer imports :func:`run_thermal` and :func:`run_electrical` from
here; the BoD schema stays free of transport concerns.
"""

from firmus_ai_factory.hf.base import (
    HFStatus,
    HFResult,
    ThermalCorrection,
    ElectricalCorrection,
    HFError,
)
from firmus_ai_factory.hf.thermal import run_thermal, AnsysThermalAdapter
from firmus_ai_factory.hf.electrical import run_electrical, ETAPAdapter

__all__ = [
    "HFStatus",
    "HFResult",
    "ThermalCorrection",
    "ElectricalCorrection",
    "HFError",
    "run_thermal",
    "AnsysThermalAdapter",
    "run_electrical",
    "ETAPAdapter",
]
