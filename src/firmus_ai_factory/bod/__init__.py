"""Basis of Design (BoD) ingestion for the Firmus AI Factory Digital Twin.

The BoD is the machine-readable design intent for a Firmus AI Factory across
nine physical and commercial domains:

    site, climate, mechanical, cooling, electrical, grid, tariff, network,
    nvidia_platform.

Its purpose is one-way and unambiguous:

    BoD (YAML/JSON) ─► BasisOfDesign (validated) ─► FactoryConfig + subsystems
                                                    │
                                                    └─► AIFactoryOptimizer
                                                            │
                                                            ├─► optimal BoD
                                                            ├─► RoI pack
                                                            ├─► Energy pack
                                                            └─► Sensitivity

The BoD schema is version-pinned (see ``BoDMetadata.schema_version``) so we
can evolve it without breaking upstream ETAP / Ansys / OneDrive workflows.
"""

from firmus_ai_factory.bod.schema import (
    BasisOfDesign,
    BoDMetadata,
    SiteBoD,
    ClimateBoD,
    MechanicalBoD,
    CoolingBoD,
    ElectricalBoD,
    GridBoD,
    TariffBoD,
    NetworkBoD,
    NVIDIAPlatformBoD,
    EconomicsBoD,
)
from firmus_ai_factory.bod.loader import (
    load_bod,
    load_bod_dict,
    dump_bod,
    export_json_schema,
)
from firmus_ai_factory.bod.hydrate import (
    hydrate_factory_config,
    hydrate_electricity_tariff,
    hydrate_factory,
)

__all__ = [
    "BasisOfDesign",
    "BoDMetadata",
    "SiteBoD",
    "ClimateBoD",
    "MechanicalBoD",
    "CoolingBoD",
    "ElectricalBoD",
    "GridBoD",
    "TariffBoD",
    "NetworkBoD",
    "NVIDIAPlatformBoD",
    "EconomicsBoD",
    "load_bod",
    "load_bod_dict",
    "dump_bod",
    "export_json_schema",
    "hydrate_factory_config",
    "hydrate_electricity_tariff",
    "hydrate_factory",
]
