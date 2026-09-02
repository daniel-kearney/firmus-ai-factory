# Basis of Design (BoD) - Schema and Ingestion

**Status:** v0.1.0 (draft)
**Owner:** Daniel Kearney
**Scope:** Canonical machine-readable design intent for a Firmus AI Factory,
covering nine physical and commercial domains, ingested by the Digital Twin
and the AI Factory Optimizer.

---

## Why a BoD schema?

Every Firmus AI Factory today is described by a mix of PDFs, PowerPoints,
Excel workbooks and vendor datasheets. That is fine for a human reviewer but
brittle for automation - ETAP can't read a PDF, Ansys can't parse a slide,
and the optimizer can't try 200 candidate configurations if the numbers are
locked in prose.

The BoD schema is one canonical machine-readable document that captures the
design intent for a site. It is:

- **One file, nine domains.** A partial BoD is not a BoD.
- **Unit-explicit.** Every field carries its unit in its name (`rack_power_kw`,
  `inlet_temp_c`, `tariff_currency_per_kwh`). No silent conversions.
- **Design intent, not runtime state.** Sensor readings and setpoints live
  elsewhere.
- **Freezable.** Once approved at a JOA gate, `metadata.approval` is populated
  and the document is treated as immutable by the optimizer - the optimizer
  produces a *new* BoD.
- **Optimizer-friendly.** Variables the optimizer may vary are listed
  explicitly under `optimizer.free_variables`. Everything else is a hard
  constraint.

---

## The nine domains

| # | Domain             | Model                     | Captures                                                    |
|---|--------------------|---------------------------|-------------------------------------------------------------|
| 1 | `site`             | `SiteBoD`                 | Location, land, IT hall area, design life                   |
| 2 | `climate`          | `ClimateBoD`              | ASHRAE class, design DB/WB, ambient range                   |
| 3 | `mechanical`       | `MechanicalBoD`           | Containment, floor loading, fire, seismic                   |
| 4 | `cooling`          | `CoolingBoD`              | Architecture, PUE/pPUE/WUE, loops, redundancy               |
| 5 | `electrical`       | `ElectricalBoD`           | MV/LV topology, UPS, gens, protection, arc-flash            |
| 6 | `grid`             | `GridBoD`                 | POI, frequency, emissions factor, DR programs               |
| 7 | `tariff`           | `TariffBoD`               | Structure (Flat/TOU/RTP/PPA), rates, demand charge          |
| 8 | `network`          | `NetworkBoD`              | Fabric, rail topology, egress, peering                      |
| 9 | `nvidia_platform`  | `NVIDIAPlatformBoD`       | Platform code, num racks, power mode, utilisation           |

Plus three service blocks:

- `metadata` - version, site_id, authors, gate approval
- `economics` - capex, opex, revenue/GPU-hour, discount rate, horizon
- `optimizer` - objectives, free variables, fixed constraints

---

## Cross-domain integrity checks

The schema enforces:

- `metadata.site_id == site.site_id`
- `electrical.design_facility_load_mw <= grid.poi_capacity_mw`
- `cooling.design_pue >= cooling.design_ppue`
- `tariff.currency == economics.currency`
- Every cooling loop has `return_temp_c > supply_temp_c`
- Every `(min, max)` tuple has `min <= max`
- Negotiated PPA requires `ppa_ccy_per_mwh`

Extra top-level fields are rejected (typos fail loudly, not silently).

---

## Ingestion pipeline

```
      YAML / JSON on disk
              │
              ▼  loader.load_bod()
     BasisOfDesign (validated)
              │
              ▼  hydrate.hydrate_factory()
      FirmusAIFactory (runnable)
              │
              ▼  optimization.optimize()
     OptimizerResult
              ├─► optimal_bod (BasisOfDesign)
              ├─► roi   (RoIPack)
              ├─► energy (EnergyPack)
              └─► sensitivity (List[SensitivityEntry])
```

Every step is a plain Python function, so the pipeline is easy to script,
schedule, or drop behind a REST endpoint.

---

## Loader usage

```python
from firmus_ai_factory.bod import load_bod, hydrate_factory
from firmus_ai_factory.optimization import optimize

bod = load_bod("examples/bod/bt1_2.yaml")
factory = hydrate_factory(bod)              # runnable Digital Twin
report = factory.generate_full_report()

result = optimize(bod)                       # winning BoD + RoI + energy
print(f"Optimal NPV: {result.roi.currency} {result.roi.npv:,.0f}")
```

Errors surface with dotted paths - e.g. a bad delta-T on the primary loop
reports as:

```
cooling.loops.0.return_temp_c: TCS primary: return_temp_c must exceed supply_temp_c
```

---

## Exporting the JSON Schema

The BoD schema doubles as a JSON Schema so ETAP, Ansys and OneDrive-side
tooling can validate against the same contract without importing Python:

```python
from firmus_ai_factory.bod import export_json_schema
export_json_schema("bod.schema.json")
```

The exported schema is stable per `schema_version`. Breaking changes bump the
version and add a migration note here.

---

## Optimizer contract

The optimizer only touches BoD paths listed in `optimizer.free_variables`.
Anything else is a hard constraint the optimizer must respect. It returns a
new `BasisOfDesign` (re-validated against the full schema) plus the RoI,
energy and sensitivity packs.

Objectives can be composed - the BoD file lists them in priority order:

```yaml
optimizer:
  objectives: ["maximize_npv", "minimize_energy"]
  free_variables:
    "nvidia_platform.num_racks": [16, 48]
    "cooling.inlet_temp_c": [30.0, 40.0]
    "nvidia_platform.utilization_pct": [60.0, 90.0]
```

Supported objectives:

- `maximize_npv`
- `minimize_energy`
- `minimize_lcoe`
- `maximize_pflops_per_mw`

---

## Worked examples

- **BT1_2** (`examples/bod/bt1_2.yaml`) - fully populated, hydrates end-to-end
  with the existing peak-load simulation.
- **Southgate 01** (`examples/bod/southgate_01.yaml`) - draft skeleton proving
  the schema is site-agnostic.

Run the end-to-end demo:

```bash
python examples/07_bod_ingestion_and_optimize.py
```

---

## Versioning

`SCHEMA_VERSION` lives in `src/firmus_ai_factory/bod/schema.py`. Additive
changes (new optional fields) do not bump the version. Any breaking change
(field removed, semantics changed, new required field) requires:

1. Bump `SCHEMA_VERSION`.
2. Add a migration note to this document.
3. Update worked examples.
4. Add a validation test for the new invariant.

---

## Where this connects

- **Model-to-Grid Optimiser** - consumes the BoD, emits an optimal BoD plus
  RoI / energy / sensitivity.
- **AI Factory OS** - reads the BoD at commissioning to seed the runtime
  control plane.
- **Firmus Digital Twin** - Ansys thermal and ETAP electrical simulators
  attach at the BoD level via the optional `high_fidelity` block so their
  high-fidelity results reconcile against the same design intent everyone
  else is using. See [HIGH_FIDELITY_HOOKS.md](HIGH_FIDELITY_HOOKS.md).
- **Dashboards & Reports / Omniverse / Playbooks / Agentic AI training** -
  all downstream of the optimizer pack.
