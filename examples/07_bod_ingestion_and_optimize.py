"""End-to-end example: BoD (YAML) -> hydrate -> optimize -> RoI/energy/sensitivity.

Run:
    python examples/07_bod_ingestion_and_optimize.py

The script:

    1. Loads examples/bod/bt1_2.yaml through the BoD loader.
    2. Hydrates it into a FirmusAIFactory and prints the design-point report.
    3. Runs the optimizer over the declared free variables.
    4. Writes the optimal BoD back out as examples/bod/bt1_2.optimal.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from firmus_ai_factory.bod import (
    dump_bod,
    hydrate_factory,
    load_bod,
)
from firmus_ai_factory.optimization import optimize


HERE = Path(__file__).resolve().parent
BOD_PATH = HERE / "bod" / "bt1_2.yaml"
OUT_PATH = HERE / "bod" / "bt1_2.optimal.json"


def _fmt_money(x: float, ccy: str) -> str:
    return f"{ccy} {x:,.0f}"


def main() -> None:
    print("=" * 72)
    print(f"  Firmus AI Factory - BoD ingestion demo ({BOD_PATH.name})")
    print("=" * 72)

    bod = load_bod(BOD_PATH)
    print(f"\nLoaded BoD {bod.metadata.bod_version} for site {bod.metadata.site_id}")
    print(f"Platform: {bod.nvidia_platform.platform}  x{bod.nvidia_platform.num_racks} racks")
    print(f"Cooling : {bod.cooling.architecture} @ {bod.cooling.inlet_temp_c} C inlet")
    print(f"Grid    : {bod.grid.region} - {bod.grid.operator}")

    # 1. Design-point hydration
    factory = hydrate_factory(bod)
    report = factory.generate_full_report()
    print("\n--- Design-point report ---")
    print(f"  IT load       : {report['power']['it_power_mw']:.2f} MW")
    print(f"  Facility load : {report['power']['total_facility_power_mw']:.2f} MW")
    print(f"  PUE           : {report['power']['pue']:.3f}")
    print(f"  Total GPUs    : {report['compute']['total_gpus']:,}")

    # 2. Optimize
    print("\n--- Optimizer ---")
    result = optimize(bod, samples_per_var=5, max_candidates=125)

    r = result.roi
    e = result.energy
    ccy = r.currency
    print(f"  Candidates evaluated : {result.candidates_evaluated}")
    print(f"  Objective score      : {result.objective_score:+.3f}")
    print(
        f"  Winning num_racks    : "
        f"{result.optimal_bod.nvidia_platform.num_racks}"
    )
    print(
        f"  Winning inlet_temp_c : "
        f"{result.optimal_bod.cooling.inlet_temp_c:.1f}"
    )
    print(
        f"  Winning util %       : "
        f"{result.optimal_bod.nvidia_platform.utilization_pct:.1f}"
    )

    print("\n  RoI:")
    print(f"    NPV              : {_fmt_money(r.npv, ccy)}")
    print(f"    Payback (yr)     : {r.payback_years:.2f}")
    print(f"    LCOE ({ccy}/MWh) : {r.lcoe_ccy_per_mwh:,.1f}")
    print(f"    Annual EBITDA    : {_fmt_money(r.annual_ebitda, ccy)}")
    print(f"    $ / M tokens     : {r.cost_per_token_usd_micro:,.2f}")

    print("\n  Energy:")
    print(f"    Annual MWh       : {e.annual_energy_mwh:,.0f}")
    print(f"    PUE              : {e.pue:.3f}")
    print(f"    WUE (L/kWh)      : {e.wue_l_per_kwh:.3f}")
    print(f"    Annual tCO2e     : {e.annual_tco2e:,.0f}")

    print("\n  Sensitivity (top 5 NPV levers):")
    for entry in result.sensitivity[:5]:
        print(
            f"    {entry.variable:40s} "
            f"[{entry.low_value:>7.2f} .. {entry.high_value:>7.2f}]  "
            f"NPV swing {_fmt_money(entry.swing, ccy)}"
        )

    # 3. Persist the optimal BoD
    dump_bod(result.optimal_bod, OUT_PATH)
    # Persist the full pack alongside it for dashboards.
    pack_path = OUT_PATH.with_suffix(".pack.json")
    pack_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")

    print(f"\nWrote optimal BoD -> {OUT_PATH.relative_to(HERE.parent)}")
    print(f"Wrote full pack   -> {pack_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
