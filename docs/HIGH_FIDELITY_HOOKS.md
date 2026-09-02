# High-Fidelity Simulation Hooks (Ansys + ETAP)

The AI Factory Optimizer runs entirely on the analytic Digital Twin by
default. For gate-2 and gate-3 sign-off we need the real physics: **Ansys
Fluent / Icepak** for CFD-grade thermal, **ETAP** for load-flow, short-circuit,
protection coordination, and arc-flash.

This document is the contract between the BoD, the optimizer, and those
solvers.

## Design

```
BasisOfDesign
   └── high_fidelity                    (optional block)
        ├── thermal      → AnsysThermalAdapter → ThermalCorrection
        └── electrical   → ETAPAdapter         → ElectricalCorrection
                                                     │
                                       ┌─────────────┴─────────────┐
                                       ▼                           ▼
                              _evaluate_bod folds              OptimizerResult
                              corrections into PUE,            .hf (HFPack) —
                              losses, RoI, energy              status + reason
                                                               + correction
                                                               + violation flags
```

### Correction, not replacement

HF solvers return **narrow correction records** (a PUE bump, a hotspot
temperature, arc-flash cal/cm², discrimination violations), never a
mesh-level state. The BoD stays the single source of truth. This keeps
the pipeline debuggable and the JSON small enough to git-commit.

### Three transports per adapter

| Backend | Endpoint means | Use case |
|---|---|---|
| `subprocess` | Absolute path to CLI / journal script | On-prem workstation with Ansys/ETAP licence |
| `http` | Base URL of a remote solve service | Central solver farm behind the VPN |
| `file` | Directory containing pre-computed `<hash>.json` files or a `manifest.json` | Offline runs, CI, DR sites, reproducibility |
| `disabled` | — | Adapter declared but skipped |

Backends are pluggable by design: swap Ansys for OpenFOAM, ETAP for
DIgSILENT, without touching the schema.

### Content-addressed cache

Every HF call is keyed by a SHA1 over the BoD subset that affects it:

* Thermal key: platform, num_racks, rack power, coolant inlet temp, cooling
  block, climate dry-bulb, mechanical block.
* Electrical key: full electrical block, grid operator + voltage options,
  facility load.

A 200-candidate sweep that varies `num_racks` and `utilization_pct` only
hits Ansys once per unique `(platform, num_racks, rack_power_kw, inlet_temp)`
tuple. The on-disk cache (opt-in, one JSON per hash) makes runs
reproducible across sessions and CI.

### Fallback is explicit

If a solver is unreachable, times out, or returns garbage:

* **`fail_open: true`** (default) → the adapter returns a deterministic
  analytic-shim correction and stamps the candidate with
  `hf_status='fallback'` plus the reason. The optimizer keeps going. Every
  OptimizerResult carries `hf.thermal_reason` / `hf.electrical_reason` so
  fallbacks are visible in the audit trail.
* **`fail_open: false`** → the adapter raises `HFError`. The candidate is
  dropped from the sweep. If no candidate survives, `optimize()` raises.

The analytic shim is **not** a replacement for real CFD or protection
studies. It encodes the *sign* of the coupling (hotter inlet → more pump
work → higher PUE) so tornado plots and integration tests are honest;
it is never a certification artefact.

### Hard constraints from HF

An OK-status HF result can still disqualify a candidate:

* `hotspot_violation` — rack outlet exceeds the NVIDIA envelope.
* `arc_flash_violation` — worst incident energy exceeds
  `electrical.arc_flash_max_incident_energy_cal_cm2`.
* `discrimination_ok == false` — ETAP-simulated time margins fail one or
  more `electrical.discrimination_targets`.

Any of these on a candidate → the candidate is skipped in the sweep. This
is how the BoD's *engineering* limits filter the optimizer's *economic*
search.

## Wiring an HF block into a BoD

Add a `high_fidelity` block. All fields inside `thermal` and `electrical`
are optional except `backend` and (for non-disabled backends) `endpoint`.

```yaml
high_fidelity:
  fail_open: true
  thermal:
    backend: subprocess
    endpoint: /opt/ansys/v242/fluent/bin/fluent
    solver: ansys_fluent
    geometry_ref: sharepoint://firmus/BT1_2/cad/hall_a_v3.step
    mesh_size_million_cells: 12.0
    turbulence_model: k_omega_sst
    coupled_with_cooling_loops: [primary_liquid, secondary_air]
    timeout_s: 900.0
    cache_enabled: true
  electrical:
    backend: http
    endpoint: https://etap.firmus.internal/solve
    solver: etap
    project_ref: etap://projects/BT1_2/rev-c
    studies: [load_flow, short_circuit, arc_flash, protection_coordination]
    check_discrimination_targets: true
    timeout_s: 300.0
    credentials_ref: env:ETAP_API_TOKEN
    cache_enabled: true
```

A fully worked example ships as `examples/bod/bt1_2_with_hf.yaml`.

### Secret handling

`credentials_ref` is a **pointer**, never the secret itself. Supported
schemes today:

* `env:VAR_NAME` — resolved from the process environment at call time.
* `vault:path/to/secret` — resolved via the Firmus secrets client (TBD).

Storing the raw API token in the BoD is a validation error at load time.

## Solver contract (what your endpoint must return)

Both adapters expect a JSON object. Unknown fields are ignored; missing
fields fall back to `ThermalCorrection` / `ElectricalCorrection` defaults.
**Real, working fixtures ship under `examples/hf/` and are exercised by
`tests/unit/test_hf_fixtures.py`** - use those as the reference contract.

### Thermal (Ansys Fluent / Icepak)

The adapter reads five top-level fields. Everything else (mesh_cells,
convergence, loop_summary, per-rack outlet temps) is provenance for
auditors and is ignored by the parser but recommended for the artefact
trail.

**Minimal response:**

```json
{
  "pue_bump": 0.008,
  "rack_hotspot_c": 41.3,
  "rack_hotspot_margin_c": 3.7,
  "cdu_delta_t_c": 11.0,
  "hotspot_violation": false
}
```

**Full worked example:** [`examples/hf/ansys/fluent_nominal.json`](../examples/hf/ansys/fluent_nominal.json)
shows a 12M-cell Fluent v24.2 run against a BT1_2-class BoD with full
provenance (residuals, wall time, licence stamp, per-rack outlet temps,
loop summary, envelope check).

**Violation example:** [`examples/hf/ansys/icepak_hotspot.json`](../examples/hf/ansys/icepak_hotspot.json)
shows an Icepak run reporting `hotspot_violation: true` for a rack row
breaching the NVIDIA envelope. The optimizer disqualifies any candidate
this correction touches.

### Electrical (ETAP)

The adapter reads four top-level fields plus the
`discrimination_margins_s` block. Study-level detail (per-bus load flow,
IEC 60909 SC results, IEEE 1584 arc-flash breakdown by board, time-current
curve review) is provenance and ignored by the parser.

**Minimal response:**

```json
{
  "losses_pct": 2.1,
  "worst_arc_flash_cal_cm2": 6.4,
  "short_circuit_ka": 42.5,
  "discrimination_margins_s": {
    "MV_incomer>LV_incomer": 0.45,
    "LV_incomer>outgoing_feeder": 0.36,
    "outgoing_feeder>PDU": 0.34,
    "PDU>rack_input": 0.26
  }
}
```

**Full worked example:** [`examples/hf/etap/etap_nominal.json`](../examples/hf/etap/etap_nominal.json)
shows a full ETAP v22.6 response with load_flow, short_circuit
(IEC 60909), arc_flash (IEEE 1584-2018), and protection_coordination
study results.

**Violation example:** [`examples/hf/etap/etap_discrimination_miss.json`](../examples/hf/etap/etap_discrimination_miss.json)
shows a run where the LV incomer relay was re-set with a faster
instantaneous element - solver margin 0.22 s vs BoD target 0.40 s. The
adapter produces `discrimination_ok=false` with a specific violation
string naming the boundary and the two numbers, and the optimizer
disqualifies the candidate.

`discrimination_margins_s` keys must mirror
`electrical.discrimination_targets` keys exactly (BT1_2 uses
`MV_incomer > LV_incomer > outgoing_feeder > PDU > rack_input`). Any
boundary the solver doesn't report is treated as unverified, not a
failure - the check is deliberately conservative.

### File-backend manifest format

When `backend: file` and `endpoint` points at a directory, the adapter
looks for `<key>.json` first, then falls back to `manifest.json`:

```json
{
  "$comment": "Maps BoD content hash -> correction payload.",
  "thermal:7a18fb052584a72d53b26136511027d5a7fede4b": {
    "pue_bump": 0.008,
    "rack_hotspot_c": 41.3,
    "rack_hotspot_margin_c": 3.7,
    "cdu_delta_t_c": 11.2,
    "hotspot_violation": false
  }
}
```

Keys are the content hashes produced by
`firmus_ai_factory.hf.thermal._thermal_key` and
`firmus_ai_factory.hf.electrical._electrical_key`. Shipped manifests:

- [`examples/hf/manifest/ansys_thermal/manifest.json`](../examples/hf/manifest/ansys_thermal/manifest.json)
- [`examples/hf/manifest/etap_electrical/manifest.json`](../examples/hf/manifest/etap_electrical/manifest.json)

The example BoD [`examples/bod/bt1_2_with_hf_file_backend.yaml`](../examples/bod/bt1_2_with_hf_file_backend.yaml)
points at both. It runs end-to-end without an Ansys or ETAP licence and
is what CI exercises.

## OptimizerResult.hf

Every `OptimizerResult` — even ones with no HF block — carries an
`HFPack`. On a fully-wired winning candidate it looks like:

```json
{
  "thermal_status": "ok",
  "electrical_status": "ok",
  "thermal_correction": { "pue_bump": 0.008, "rack_hotspot_c": 41.3, ... },
  "electrical_correction": { "losses_pct": 2.1, ... },
  "thermal_reason": "",
  "electrical_reason": "",
  "hotspot_violation": false,
  "arc_flash_violation": false,
  "discrimination_ok": true
}
```

On `absent`, `disabled`, or `fallback` runs the correction dicts still
appear (or are null for `absent`/`disabled`) and the `reason` string
explains what happened — sufficient audit for a G3 gate review.

## Testing without licences

The test suite hermetically exercises every path:

* `ABSENT` — no `high_fidelity` block.
* `DISABLED` — block present, backend `disabled`.
* `FALLBACK` — HTTP backend pointed at an unreachable port, timeout 50 ms.
* `OK` — `file` backend reading a fixture written by the test.
* Cache hit — second call in the same run.
* Hard-constraint disqualification — hotspot violation eliminates the
  hot-inlet candidate from the sweep.

See `tests/unit/test_hf_adapters.py`.

**Fixture-driven contract tests** (`tests/unit/test_hf_fixtures.py`) load
the shipped Ansys and ETAP response fixtures and assert the parsers pull
the right numbers out and apply the BoD-aware discrimination check
correctly. If a fixture stops parsing here, the external solver contract
is broken - fix the code, not the fixture.

## Follow-ups

* MPC controller consumes `HFPack` for online derating decisions.
* Ansys thermal → digital-twin telemetry closed loop (writes recorded
  corrections back to CellarTracker-style history for drift detection).
* Vault-backed `credentials_ref` scheme.
* Sensitivity sweeps optionally re-run HF at each perturbation (currently
  they use the cached winning-BoD correction to keep sweeps fast).
