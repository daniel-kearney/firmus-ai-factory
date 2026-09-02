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

### Thermal

```json
{
  "pue_bump": 0.008,
  "rack_hotspot_c": 41.3,
  "rack_hotspot_margin_c": 3.7,
  "cdu_delta_t_c": 11.0,
  "hotspot_violation": false
}
```

### Electrical

```json
{
  "losses_pct": 2.1,
  "worst_arc_flash_cal_cm2": 6.4,
  "short_circuit_ka": 42.5,
  "discrimination_margins_s": {
    "MV_incomer>LV_incomer": 0.38,
    "LV_incomer>ATS": 0.31
  }
}
```

`discrimination_margins_s` keys must mirror
`electrical.discrimination_targets` keys. The adapter compares them and
raises `discrimination_violations` for any boundary where the solver
margin is below the BoD target.

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
* `FALLBACK` — HTTP backend pointed at an unreachable port, timeout 50ms.
* `OK` — `file` backend reading a fixture written by the test.
* Cache hit — second call in the same run.
* Hard-constraint disqualification — hotspot violation eliminates the
  hot-inlet candidate from the sweep.

See `tests/unit/test_hf_adapters.py`.

## Follow-ups

* MPC controller consumes `HFPack` for online derating decisions.
* Ansys thermal → digital-twin telemetry closed loop (writes recorded
  corrections back to CellarTracker-style history for drift detection).
* Vault-backed `credentials_ref` scheme.
* Sensitivity sweeps optionally re-run HF at each perturbation (currently
  they use the cached winning-BoD correction to keep sweeps fast).
