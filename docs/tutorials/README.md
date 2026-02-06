# Tutorials

Step-by-step guides for using the Firmus AI Factory Digital Twin framework.

## Table of Contents

1. [Getting Started](#tutorial-1-getting-started)
2. [GPU Power Profiling](#tutorial-2-gpu-power-profiling)
3. [Thermal Analysis with Immersion Cooling](#tutorial-3-thermal-analysis-with-immersion-cooling)
4. [HGX System Analysis](#tutorial-4-full-hgx-system-analysis)
5. [Comparing GPU Generations](#tutorial-5-comparing-gpu-generations)

---

## Tutorial 1: Getting Started

### Prerequisites

- Python 3.9+
- numpy

### Installation

```bash
git clone https://github.com/daniel-kearney/firmus-ai-factory.git
cd firmus-ai-factory
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```python
import firmus_ai_factory
print(f"Version: {firmus_ai_factory.__version__}")
# Output: Version: 0.1.0
```

---

## Tutorial 2: GPU Power Profiling

This tutorial walks through creating GPU power models and simulating training workloads.

### Step 1: Create a GPU Model

```python
from firmus_ai_factory.computational.gpu_model import (
    GPUModel, H100_SXM_SPECS, H200_SPECS, B200_SPECS
)

# Create an H100 SXM GPU model
gpu = GPUModel(H100_SXM_SPECS)
print(f"GPU: {gpu.specs.name}")
print(f"TDP: {gpu.specs.tdp_watts} W")
print(f"Peak FP16: {gpu.specs.peak_flops_fp16} TFLOPS")
```

### Step 2: Calculate Power from Utilization

```python
# Power at different utilization levels
for util in [0.0, 0.25, 0.5, 0.75, 1.0]:
    power = gpu.power_from_utilization(util)
    print(f"  Utilization {util:.0%}: {power:.0f} W")
```

### Step 3: Simulate a Training Workload

```python
# Simulate 70B parameter model training for 10 seconds
profile = gpu.simulate_training_workload(
    model_params=70e9,   # 70 billion parameters
    batch_size=32,       # batch size
    duration=10.0,       # 10 second simulation
    dt=0.01              # 10ms time step
)

print(f"Mean power: {profile.mean_power:.1f} W")
print(f"Peak power: {profile.peak_power:.1f} W")
print(f"Energy consumed: {profile.energy_kwh:.4f} kWh")
```

### Step 4: Analyze Power Components

```python
import numpy as np

print(f"Compute power (mean): {np.mean(profile.compute_power):.1f} W")
print(f"Memory power (mean): {np.mean(profile.memory_power):.1f} W")
print(f"Transfer power (mean): {np.mean(profile.transfer_power):.1f} W")
```

### Step 5: Compute Model Efficiency

```python
# Calculate MFU for a given throughput
mfu = gpu.compute_mfu(
    tokens_per_second=5000,
    num_params=70e9,
    num_gpus=8
)
print(f"Model FLOPS Utilization: {mfu:.1%}")

# FLOPS per token
flops = gpu.compute_flops_per_token(num_params=70e9)
print(f"FLOPS per token: {flops:.2e}")
```

---

## Tutorial 3: Thermal Analysis with Immersion Cooling

This tutorial demonstrates thermal modeling of immersion cooling systems.

### Step 1: Create a Cooling System

```python
from firmus_ai_factory.thermal.immersion_cooling import (
    ImmersionCoolingSystem,
    EC100_PROPERTIES,
    NOVEC_7100_PROPERTIES
)

cooling = ImmersionCoolingSystem(
    coolant=EC100_PROPERTIES,  # 3M EC-100 dielectric fluid
    flow_rate=2.5,              # 2.5 L/min per GPU
    inlet_temp=35.0             # 35C inlet temperature
)
```

### Step 2: Inspect Coolant Properties

```python
coolant = EC100_PROPERTIES
print(f"Coolant: {coolant.name}")
print(f"Density: {coolant.density} kg/m3")
print(f"Specific heat: {coolant.specific_heat} J/kg*K")
print(f"Thermal conductivity: {coolant.thermal_conductivity} W/m*K")
print(f"Boiling point: {coolant.boiling_point} C")
print(f"Thermal diffusivity: {coolant.thermal_diffusivity:.2e} m2/s")
```

### Step 3: Analyze with a Constant Power Load

```python
import numpy as np

# Constant 650W per GPU, 8 GPUs
power = np.full(1000, 650.0)
result = cooling.analyze(power, num_gpus=8)

print(f"Max junction temp: {result.T_junction_max:.1f} C")
print(f"Mean junction temp: {result.T_junction_mean:.1f} C")
print(f"Coolant outlet temp: {result.T_coolant_out:.1f} C")
print(f"Cooling power: {result.P_cooling:.1f} W")
print(f"pPUE: {result.pPUE:.3f}")
```

### Step 4: Examine Thermal Resistance

```python
network = cooling.thermal_network
print(f"R_jc (junction-to-case): {network.R_jc:.3f} K/W")
print(f"R_ch (case-to-heatsink): {network.R_ch:.3f} K/W")
print(f"R_ha (heatsink-to-ambient): {network.R_ha:.3f} K/W")
print(f"Total resistance: {network.total_resistance:.3f} K/W")
```

---

## Tutorial 4: Full HGX System Analysis

Combine GPU power modeling with thermal analysis for an end-to-end simulation.

### Step 1: Set Up the System

```python
from firmus_ai_factory.computational.gpu_model import GPUModel, H100_SXM_SPECS
from firmus_ai_factory.thermal.immersion_cooling import (
    ImmersionCoolingSystem, EC100_PROPERTIES
)

gpu = GPUModel(H100_SXM_SPECS)
cooling = ImmersionCoolingSystem(
    coolant=EC100_PROPERTIES,
    flow_rate=2.5,
    inlet_temp=35.0
)
```

### Step 2: Simulate Training and Analyze Thermals

```python
# Simulate 70B model training
profile = gpu.simulate_training_workload(
    model_params=70e9,
    batch_size=32,
    duration=10.0,
    dt=0.01
)

# Thermal analysis for 8-GPU system
thermal = cooling.analyze(profile.total_power, num_gpus=8)

print(f"GPU Power:")
print(f"  Mean: {profile.mean_power:.1f} W")
print(f"  Peak: {profile.peak_power:.1f} W")
print(f"  Utilization: {profile.mean_power/gpu.specs.tdp_watts*100:.1f}%")
print(f"")
print(f"Thermal:")
print(f"  T_junction max: {thermal.T_junction_max:.1f} C")
print(f"  T_coolant out: {thermal.T_coolant_out:.1f} C")
print(f"  HTC: {thermal.heat_transfer_coeff:.0f} W/m2/K")
print(f"")
print(f"Efficiency:")
total_it = profile.mean_power * 8
print(f"  IT Power: {total_it/1000:.2f} kW")
print(f"  Cooling Power: {thermal.P_cooling:.1f} W")
print(f"  pPUE: {thermal.pPUE:.3f}")
```

### Step 3: Check Thermal Limits

```python
T_LIMIT = 83.0  # Typical GPU thermal throttle point

if thermal.T_junction_max < T_LIMIT:
    headroom = T_LIMIT - thermal.T_junction_max
    print(f"PASS: {headroom:.1f} C headroom to thermal limit")
else:
    print(f"WARNING: Exceeds thermal limit by "
          f"{thermal.T_junction_max - T_LIMIT:.1f} C")
```

---

## Tutorial 5: Comparing GPU Generations

Compare power and thermal characteristics across H100, H200, and B200.

```python
from firmus_ai_factory.computational.gpu_model import (
    GPUModel, H100_SXM_SPECS, H200_SPECS, B200_SPECS
)
from firmus_ai_factory.thermal.immersion_cooling import (
    ImmersionCoolingSystem, EC100_PROPERTIES
)

gpu_configs = [
    ("H100 SXM", H100_SXM_SPECS),
    ("H200", H200_SPECS),
    ("B200", B200_SPECS),
]

print(f"{'GPU':<12} {'TDP':>6} {'Mean P':>8} {'T_j':>6} {'pPUE':>6}")
print("-" * 42)

for name, specs in gpu_configs:
    gpu = GPUModel(specs)
    profile = gpu.simulate_training_workload(
        model_params=70e9,
        batch_size=32,
        duration=5.0
    )
    cooling = ImmersionCoolingSystem(
        coolant=EC100_PROPERTIES,
        flow_rate=2.5,
        inlet_temp=35.0
    )
    thermal = cooling.analyze(profile.total_power, num_gpus=8)

    print(f"{name:<12} {specs.tdp_watts:>5.0f}W "
          f"{profile.mean_power:>7.1f}W "
          f"{thermal.T_junction_max:>5.1f}C "
          f"{thermal.pPUE:>5.3f}")
```

---

## Next Steps

- Explore the [API Reference](../api/) for detailed class and method documentation
- Read the [Mathematical Theory](../theory/) for the physics behind the models
- Check the [examples/](../../examples/) directory for runnable scripts

---

*Document Version: 1.0*
*Last Updated: February 2026*
*Author: Firmus Engineering Team*
