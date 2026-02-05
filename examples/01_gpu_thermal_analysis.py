#!/usr/bin/env python3
"""Example: GPU Thermal Analysis for AI Factory.

This example demonstrates how to use the Firmus AI Factory Digital Twin
to analyze the thermal performance of an 8-GPU HGX system with
immersion cooling.

Key outputs:
- GPU power profiles during training
- Junction temperatures
- Cooling system power and pPUE
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from firmus_ai_factory.computational.gpu_model import (
    GPUModel,
    H100_SXM_SPECS,
    H200_SPECS,
    B200_SPECS
)
from firmus_ai_factory.thermal.immersion_cooling import (
    ImmersionCoolingSystem,
    EC100_PROPERTIES,
    NOVEC_7100_PROPERTIES
)


def analyze_hgx_system(
    gpu_specs,
    coolant_props,
    model_params: float,
    num_gpus: int = 8
):
    """Analyze an HGX system with immersion cooling.
    
    Args:
        gpu_specs: GPU specifications (H100, H200, or B200)
        coolant_props: Coolant properties
        model_params: Number of model parameters
        num_gpus: Number of GPUs in the system
    """
    print(f"\n{'='*60}")
    print(f"AI Factory Thermal Analysis")
    print(f"{'='*60}")
    print(f"GPU: {gpu_specs.name}")
    print(f"GPUs in system: {num_gpus}")
    print(f"Model size: {model_params/1e9:.0f}B parameters")
    print(f"Coolant: {coolant_props.name}")
    print(f"{'='*60}\n")
    
    # Create GPU model
    gpu = GPUModel(gpu_specs)
    
    # Simulate training workload (10 seconds)
    print("Simulating training workload...")
    power_profile = gpu.simulate_training_workload(
        model_params=model_params,
        batch_size=32,
        duration=10.0,
        dt=0.01
    )
    
    print(f"\nGPU Power Analysis:")
    print(f"  Mean power: {power_profile.mean_power:.1f} W")
    print(f"  Peak power: {power_profile.peak_power:.1f} W")
    print(f"  TDP: {gpu_specs.tdp_watts:.1f} W")
    print(f"  Utilization: {power_profile.mean_power/gpu_specs.tdp_watts*100:.1f}%")
    
    # Create cooling system
    cooling = ImmersionCoolingSystem(
        coolant=coolant_props,
        flow_rate=2.5,  # L/min per GPU
        inlet_temp=35.0
    )
    
    # Analyze thermal performance
    thermal_result = cooling.analyze(
        power_profile.total_power,
        num_gpus=num_gpus
    )
    
    print(f"\nThermal Analysis:")
    print(f"  Max junction temp: {thermal_result.T_junction_max:.1f} C")
    print(f"  Mean junction temp: {thermal_result.T_junction_mean:.1f} C")
    print(f"  Coolant outlet temp: {thermal_result.T_coolant_out:.1f} C")
    print(f"  Heat transfer coeff: {thermal_result.heat_transfer_coeff:.0f} W/m2/K")
    print(f"  Thermal resistance: {thermal_result.thermal_resistance:.3f} K/W")
    
    print(f"\nSystem Efficiency:")
    total_it_power = power_profile.mean_power * num_gpus
    print(f"  Total IT power: {total_it_power/1000:.2f} kW")
    print(f"  Cooling power: {thermal_result.P_cooling:.1f} W")
    print(f"  pPUE: {thermal_result.pPUE:.3f}")
    
    # Check thermal limits
    T_LIMIT = 83.0  # Typical GPU thermal limit
    if thermal_result.T_junction_max < T_LIMIT:
        print(f"\n[OK] Junction temperature within limits ({T_LIMIT}C)")
    else:
        print(f"\n[WARNING] Junction temperature exceeds limit ({T_LIMIT}C)!")
    
    return power_profile, thermal_result


def compare_gpu_generations():
    """Compare thermal performance across GPU generations."""
    print("\n" + "="*60)
    print("GPU Generation Comparison (70B Model, 8 GPUs)")
    print("="*60)
    
    results = {}
    for name, specs in [("H100 SXM", H100_SXM_SPECS), 
                        ("H200", H200_SPECS),
                        ("B200", B200_SPECS)]:
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
        
        results[name] = {
            "TDP": specs.tdp_watts,
            "Mean Power": profile.mean_power,
            "T_junction": thermal.T_junction_max,
            "pPUE": thermal.pPUE
        }
    
    # Print comparison table
    print(f"\n{'GPU':<12} {'TDP (W)':<10} {'Mean P (W)':<12} {'T_j (C)':<10} {'pPUE':<8}")
    print("-" * 52)
    for name, data in results.items():
        print(f"{name:<12} {data['TDP']:<10.0f} {data['Mean Power']:<12.1f} "
              f"{data['T_junction']:<10.1f} {data['pPUE']:<8.3f}")


if __name__ == "__main__":
    # Example 1: Analyze H100 HGX system with 70B model
    analyze_hgx_system(
        gpu_specs=H100_SXM_SPECS,
        coolant_props=EC100_PROPERTIES,
        model_params=70e9,
        num_gpus=8
    )
    
    # Example 2: Compare GPU generations
    compare_gpu_generations()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
