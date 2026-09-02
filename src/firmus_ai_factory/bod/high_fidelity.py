"""High-fidelity simulation hooks for the AI Factory Digital Twin.

The BoD declares two optional oracles the optimizer may call for each
candidate configuration:

    thermal   -> Ansys Fluent / Icepak (CFD, conjugate heat transfer)
    electrical-> ETAP (load flow, short circuit, protection coordination,
                 arc-flash)

The BoD schema only carries the *contract* (how to reach the tool, what to
send, what to receive). The transport (subprocess / http / file) is
pluggable and lives in ``firmus_ai_factory.hf.adapters`` so the schema
stays pure design intent.

Design principles
-----------------

* **Optional.**  A BoD without ``high_fidelity`` runs entirely on the
  analytic models. The optimizer degrades gracefully.
* **Content-addressed cache.**  Every HF call is keyed by a hash of the
  BoD subset that affects it, so a 200-candidate sweep never re-solves
  the same geometry.
* **Read-only correction, not replacement.**  HF sims return small
  *correction* records (PUE bump, hot-spot flag, arc-flash cal/cm²).
  They never rewrite the BoD.  The BoD stays the single source of truth.
* **Fallback is explicit.**  If a solver is unreachable, times out, or
  disagrees badly, the candidate is scored on the analytic model and the
  result carries ``hf_status='fallback'`` with a reason.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from firmus_ai_factory.bod.schema import _BoDModel


class HFBackend(str, Enum):
    """Transport used to reach a high-fidelity solver.

    * ``subprocess`` - local executable (Ansys/ETAP CLI or journal file)
    * ``http``       - remote solve service (REST)
    * ``file``       - pre-computed lookup on disk (offline runs, CI, DR sites)
    * ``disabled``   - explicitly turned off; optimizer uses the analytic model
    """

    SUBPROCESS = "subprocess"
    HTTP = "http"
    FILE = "file"
    DISABLED = "disabled"


class HFAdapterBoD(_BoDModel):
    """Base class for a high-fidelity adapter declaration.

    Kept intentionally minimal: transport, endpoint / path, timeout,
    optional API key ref (never the secret itself — always a pointer).
    """

    backend: HFBackend
    endpoint: Optional[str] = Field(
        default=None,
        description=(
            "For subprocess: absolute path to the executable or journal file. "
            "For http: base URL of the solve service. "
            "For file: absolute path to the lookup directory / manifest."
        ),
    )
    timeout_s: float = Field(default=300.0, gt=0.0, le=86_400.0)
    credentials_ref: Optional[str] = Field(
        default=None,
        description=(
            "Pointer to a credential (env var name, Vault path). "
            "The BoD never stores the secret itself."
        ),
    )
    cache_enabled: bool = Field(default=True)

    @model_validator(mode="after")
    def _requires_endpoint(self) -> "HFAdapterBoD":
        if self.backend != HFBackend.DISABLED and not self.endpoint:
            raise ValueError(
                f"backend {self.backend!r} requires 'endpoint' to be set"
            )
        return self


class ThermalHFBoD(HFAdapterBoD):
    """Ansys (Fluent / Icepak) thermal oracle configuration."""

    solver: str = Field(
        default="ansys_fluent",
        description="'ansys_fluent' | 'ansys_icepak' | 'custom'.",
    )
    geometry_ref: Optional[str] = Field(
        default=None,
        description="Pointer to the CAD / mesh (SharePoint URL, S3 URI, local path).",
    )
    mesh_size_million_cells: float = Field(default=5.0, gt=0.0, le=200.0)
    turbulence_model: str = Field(default="k_omega_sst")
    coupled_with_cooling_loops: List[str] = Field(
        default_factory=list,
        description="Names of BoD cooling loops the CFD run is coupled to.",
    )


class ElectricalHFBoD(HFAdapterBoD):
    """ETAP electrical oracle configuration."""

    solver: str = Field(
        default="etap",
        description="'etap' | 'digsilent' | 'custom'.",
    )
    project_ref: Optional[str] = Field(
        default=None,
        description="Pointer to the ETAP project file (.etp) or exported archive.",
    )
    studies: List[str] = Field(
        default_factory=lambda: ["load_flow", "short_circuit", "arc_flash"],
        description=(
            "Which ETAP studies to run per call. Supported: 'load_flow', "
            "'short_circuit', 'arc_flash', 'protection_coordination', "
            "'harmonic', 'transient_stability'."
        ),
    )
    check_discrimination_targets: bool = Field(
        default=True,
        description=(
            "If true, ETAP results are checked against "
            "electrical.discrimination_targets and a violation flag is returned."
        ),
    )


class HighFidelityBoD(_BoDModel):
    """Container for high-fidelity solver hooks on a BoD.

    Both fields are optional so a BoD can wire in thermal-only (early
    design), electrical-only (protection review), both (full G3 gate),
    or neither.
    """

    thermal: Optional[ThermalHFBoD] = None
    electrical: Optional[ElectricalHFBoD] = None
    fail_open: bool = Field(
        default=True,
        description=(
            "If true, an unreachable solver falls back to analytic results "
            "and stamps the candidate with hf_status='fallback'. If false, "
            "the candidate is rejected outright."
        ),
    )

    @property
    def is_active(self) -> bool:
        """True iff at least one adapter is non-null and not disabled."""
        for adapter in (self.thermal, self.electrical):
            if adapter is not None and adapter.backend != HFBackend.DISABLED:
                return True
        return False
