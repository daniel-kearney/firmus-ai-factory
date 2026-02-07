# Firmus AI Factory - Modules Overview

This document provides an overview of all modules in the Firmus AI Factory digital twin platform.

## Module Architecture

The Firmus AI Factory consists of eight integrated modules that model the complete chip-to-grid energy system:

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTROL & INTEGRATION                     │
│                      (Digital Twin)                          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
│  COMPUTATIONAL │   │     THERMAL     │   │     POWER      │
│   (GPU Models) │   │   (Cooling)     │   │     (PDN)      │
└────────────────┘   └─────────────────┘   └────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
│      GRID      │   │    STORAGE      │   │  OPTIMIZATION  │
│ (Interconnect) │   │   (Battery)     │   │     (MPC)      │
└────────────────┘   └─────────────────┘   └────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                     ┌────────▼────────┐
                     │    ECONOMICS    │
                     │  (Tariff, TCO)  │
                     └─────────────────┘
```

---

## 1. Computational Module

**Purpose**: Model GPU power consumption across different workload phases

**Key Components**:
- `GPUModel`: Phase-aware power decomposition (tensor cores, CUDA cores, memory)
- `GPUSpecs`: GPU specifications (TDP, idle power, component power)

**Capabilities**:
- Training workload power modeling
- Inference workload power modeling
- Idle and memory-only power states
- Support for NVIDIA H100, B200, and custom GPUs

**Example**:
```python
from firmus_ai_factory.computational import GPUModel, GPUSpecs

gpu_specs = GPUSpecs(name="H100 SXM", TDP=700.0, P_idle=50.0)
gpu = GPUModel(gpu_specs)

utilization = {'tensor': 0.9, 'cuda': 0.3, 'memory': 0.8}
power = gpu.calculate_power(utilization)  # Returns power in Watts
```

---

## 2. Thermal Module

**Purpose**: Model cooling systems and thermal management

**Key Components**:
- `ImmersionCoolingSystem`: Two-phase immersion cooling
- `AirCoolingSystem`: Traditional air cooling
- `DirectLiquidCooling`: Cold plate liquid cooling
- `CoolingSpecs`: Cooling system specifications

**Capabilities**:
- Junction temperature calculation
- Cooling power requirements
- pPUE (partial Power Usage Effectiveness) calculation
- Support for 3M Novec fluids, water, and air cooling

**Example**:
```python
from firmus_ai_factory.thermal import ImmersionCoolingSystem, CoolingSpecs

specs = CoolingSpecs(name="Immersion", fluid_type="3M Novec 7100")
cooling = ImmersionCoolingSystem(specs)

result = cooling.analyze(total_power=5600, num_gpus=8)
print(f"Junction Temp: {result.T_junction_max:.1f} °C")
print(f"pPUE: {result.pPUE:.3f}")
```

---

## 3. Power Module

**Purpose**: Model power delivery from utility to GPU voltage rails

**Key Components**:
- `TransformerModel`: Utility transformer (13.8kV/480V, 34.5kV/480V)
- `BuckConverterModel`: DC-DC buck converters (480V→12V, 12V→1V)
- `MultiphaseVRM`: Multi-phase voltage regulator modules

**Capabilities**:
- Voltage regulation and droop analysis
- Efficiency calculations across PDN stages
- Output impedance vs frequency
- Transient response simulation
- Target impedance verification

**Example**:
```python
from firmus_ai_factory.power import TransformerModel, TRANSFORMER_13_8KV_TO_480V

transformer = TransformerModel(TRANSFORMER_13_8KV_TO_480V)
I_load = 10000  # 10 kA load current
v_drop, efficiency = transformer.calculate_voltage_regulation(I_load, pf=0.95)
print(f"Voltage Drop: {v_drop:.2f} V, Efficiency: {efficiency:.3%}")
```

---

## 4. Grid Module

**Purpose**: Enable grid interconnection and demand response participation

**Key Components**:
- `GridInterface`: Frequency regulation, power factor control
- `DemandResponseManager`: DR bid optimization, event evaluation
- `WorkloadDeferralStrategy`: Workload classification and deferral

**Capabilities**:
- Primary frequency response (droop control)
- Demand response bidding and performance measurement
- Workload time-sensitivity classification
- Available reduction capacity calculation

**Example**:
```python
from firmus_ai_factory.grid import GridInterface, GRID_US_480V

grid = GridInterface(GRID_US_480V)

# Frequency response
f_grid = 59.95  # Hz
P_current = 5e6  # 5 MW
P_adjustment = grid.frequency_response(f_grid, P_current)
print(f"Load reduction: {P_adjustment/1000:.1f} kW")
```

---

## 5. Storage Module

**Purpose**: Model battery energy storage and UPS systems

**Key Components**:
- `LithiumIonBattery`: Electrochemical battery model with thermal effects
- `BatterySpecs`: Battery specifications (capacity, voltage, resistance)

**Capabilities**:
- State-of-charge (SOC) evolution
- Open-circuit voltage vs SOC
- Terminal voltage and power loss calculation
- Thermal dynamics
- Capacity fade and degradation modeling

**Example**:
```python
from firmus_ai_factory.storage import LithiumIonBattery, BATTERY_TESLA_MEGAPACK

battery = LithiumIonBattery(BATTERY_TESLA_MEGAPACK, SOC_init=0.7)

# Discharge for 1 hour
I_discharge = 100  # 100 A
dt = 3600  # 1 hour in seconds
V_terminal, P_loss = battery.update_state(I_discharge, dt)
print(f"New SOC: {battery.SOC:.1%}, Voltage: {V_terminal:.1f} V")
```

---

## 6. Optimization Module

**Purpose**: Multi-objective optimization and model predictive control

**Key Components**:
- `ModelPredictiveController`: MPC-based optimization
- `WorkloadJob`: Workload job specification

**Capabilities**:
- Multi-objective optimization (cost, thermal, throughput)
- Constrained optimization with thermal and power limits
- Workload scheduling with deadlines
- 24-hour lookahead optimization

**Dependencies**: Requires `cvxpy` for convex optimization

**Example**:
```python
from firmus_ai_factory.optimization import ModelPredictiveController

mpc = ModelPredictiveController(horizon=24, dt=1.0)

optimal_controls = mpc.optimize(
    current_state={'T_junction': 50.0, 'SOC': 0.7},
    price_forecast=price_array,
    workload_queue=[],
    grid_signals={'frequency': 60.0}
)

print(f"Optimal 24-hour cost: ${optimal_controls['total_cost']:.2f}")
```

---

## 7. Economics Module

**Purpose**: Economic analysis and electricity cost modeling

**Key Components**:
- `ElectricityTariff`: Time-of-use, flat, and real-time pricing models

**Capabilities**:
- Energy charge calculation
- Demand charge calculation
- Time-of-use period classification
- 24-hour cost breakdown

**Example**:
```python
from firmus_ai_factory.economics import ElectricityTariff

tariff = ElectricityTariff(tariff_type="TOU")

power_profile = np.ones(24) * 5e6  # 5 MW constant
timestamps = np.arange(0, 24*3600, 3600) + time.time()

total_cost, breakdown = tariff.calculate_cost(power_profile, timestamps)
print(f"Total 24-hour cost: ${total_cost:.2f}")
```

---

## 8. Control Module

**Purpose**: System-level integration and digital twin coordination

**Key Components**:
- `DigitalTwin`: Complete system integration

**Capabilities**:
- Multi-subsystem state management
- 24-hour scenario simulation
- Real-time state updates
- System-level outputs (power, temperature, cost)

**Example**:
```python
from firmus_ai_factory.control import DigitalTwin

config = {'gpu': gpu_specs, 'thermal': cooling_specs, 'grid': grid_specs}
twin = DigitalTwin(config)

# Run 24-hour simulation
results = twin.run_scenario(duration_hours=24, dt=300)

print(f"Average Power: {np.mean(results['P_total'])/1000:.1f} kW")
print(f"Total Cost: ${results['cumulative_cost'][-1]:.2f}")
```

---

## Integration Example

See `examples/03_complete_system_integration.py` for a comprehensive example that demonstrates all eight modules working together to model a complete AI factory from chip to grid.

**Key Integration Points**:
1. GPU power feeds into thermal analysis
2. Thermal cooling power adds to total facility load
3. Total load flows through PDN (transformer → converter → VRM)
4. Grid interface manages frequency response and demand response
5. Battery provides backup and grid services
6. MPC optimizes across all subsystems
7. Economics quantifies costs and revenues
8. Digital twin coordinates everything in real-time

---

## Module Completion Status

| Module | Status | LOC | Test Coverage |
|--------|--------|-----|---------------|
| Computational | ✅ Complete | 258 | Partial |
| Thermal | ✅ Complete | 612 | Partial |
| Power | ✅ Complete | 450 | Pending |
| Grid | ✅ Complete | 380 | Pending |
| Storage | ✅ Complete | 180 | Pending |
| Optimization | ✅ Complete | 220 | Pending |
| Economics | ✅ Complete | 140 | Pending |
| Control | ✅ Complete | 160 | Pending |

**Total**: 2,400+ lines of production code

---

## Next Steps

1. **Add Unit Tests**: Implement comprehensive unit tests for all new modules
2. **Validation**: Validate models against vendor specifications and operational data
3. **Documentation**: Expand theory documentation for each module
4. **Examples**: Create additional examples for specific use cases
5. **Performance**: Profile and optimize computational performance

---

## References

- NREL Chip-to-Grid Data Center Initiative
- Google Grid-Flexible AI Infrastructure
- Aurora Power-Flexible AI Factory
- IEEE 2030.5 (Smart Energy Profile)
- FERC Order 755 (Frequency Regulation)
