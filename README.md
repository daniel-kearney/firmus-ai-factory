# Firmus AI Factory Digital Twin

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive multi-physics simulation framework for modeling AI data center infrastructure from GPU to grid. This Digital Twin enables design optimization, performance prediction, and real-time operational control of AI Factory systems.

## Overview

The Firmus AI Factory Digital Twin provides mathematical models and simulation tools for:

- **Computational Layer**: GPU power modeling, workload dynamics, cluster synchronization
- **Power Electronics Layer**: Power delivery networks, converter dynamics, UPS systems
- **Thermal Management Layer**: Immersion cooling, conjugate heat transfer, thermal networks
- **Energy Storage Layer**: Battery systems, supercapacitors, hybrid storage
- **Grid Interface Layer**: Utility interconnection, demand response, frequency regulation

## Key Features

- Multi-physics coupled simulation
- Real-time digital twin capabilities
- Design optimization with multi-objective algorithms
- Total Cost of Ownership (TCO) analysis
- Reduced-order modeling for fast simulation
- GPU-specific power models (H100, H200, B200)
- Immersion cooling thermal analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/daniel-kearney/firmus-ai-factory.git
cd firmus-ai-factory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Project Structure

```
firmus-ai-factory/
|-- src/
|   |-- firmus_ai_factory/
|       |-- core/                 # Base classes and system integration
|       |-- computational/        # GPU and workload modeling
|       |-- power/               # Power electronics and grid
|       |-- thermal/             # Heat transfer and cooling
|       |-- optimization/        # Multi-objective optimization
|       |-- utils/               # Constants and utilities
|-- examples/                    # Usage examples
|-- tests/                       # Unit tests
|-- docs/                        # Documentation
```

## Quick Start

```python
from firmus_ai_factory.computational import GPUModel, H100_SPECS
from firmus_ai_factory.thermal import ImmersionCoolingSystem
from firmus_ai_factory.core import AIFactorySystem

# Create GPU model
gpu = GPUModel(specs=H100_SPECS)

# Simulate training workload
power_profile = gpu.simulate_training_workload(
    model_params=70e9,  # 70B parameters
    batch_size=32,
    duration=3600  # 1 hour
)

# Create cooling system
cooling = ImmersionCoolingSystem(
    coolant='EC-100',
    flow_rate=2.5,  # L/min
    inlet_temp=35   # Celsius
)

# Analyze thermal performance
thermal_result = cooling.analyze(power_profile)
print(f"Max junction temp: {thermal_result.T_junction_max:.1f} C")
print(f"Cooling power: {thermal_result.P_cooling:.1f} W")
```

## Mathematical Foundation

### GPU Power Model

Instantaneous GPU power decomposes into:

```
P_GPU(t) = P_compute(t) + P_memory(t) + P_transfer(t)
```

For training workloads:
```
P_train(t) = P_base + sum(alpha_i * f_i(t)) + epsilon(t)
```

### Thermal Modeling

Conjugate heat transfer with Navier-Stokes coupling:

**Fluid Domain:**
```
rho * c_p * (dT/dt + u . nabla(T)) = k_f * nabla^2(T)
```

**Solid Domain:**
```
rho_s * c_ps * dT/dt = nabla . (k_s * nabla(T)) + q_gen
```

### Power Delivery Network

Multi-stage converter cascade:
```
G_system(s) = prod(G_i(s)) for i = 1 to N stages
```

Target impedance for voltage regulation:
```
Z_target = Delta_V_allowed / I_transient
```

## Documentation

For detailed documentation, see:
- [Mathematical Theory](docs/theory/)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Developed by Firmus for AI Factory infrastructure optimization.

## Contact

- Daniel Kearney - CTO, Firmus
- GitHub: [@daniel-kearney](https://github.com/daniel-kearney)
