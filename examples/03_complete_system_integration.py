"""Complete system integration example.

This example demonstrates the integration of all Firmus AI Factory modules:
- Computational (GPU power modeling)
- Thermal (cooling systems)
- Power (PDN, converters, VRMs)
- Grid (interconnection, demand response)
- Storage (battery/UPS)
- Optimization (MPC)
- Economics (tariff, costs)
- Control (digital twin)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import all Firmus modules
from firmus_ai_factory.computational import GPUModel, GPUSpecs
from firmus_ai_factory.thermal import ImmersionCoolingSystem, CoolingSpecs
from firmus_ai_factory.power import (
    TransformerModel, TRANSFORMER_13_8KV_TO_480V,
    BuckConverterModel, BUCK_480V_TO_12V,
    MultiphaseVRM, VRM_H100_SXM
)
from firmus_ai_factory.grid import GridInterface, GRID_US_480V, DemandResponseManager
from firmus_ai_factory.storage import LithiumIonBattery, BATTERY_TESLA_MEGAPACK
from firmus_ai_factory.optimization import ModelPredictiveController
from firmus_ai_factory.economics import ElectricityTariff
from firmus_ai_factory.control import DigitalTwin


def main():
    """Run complete system integration example."""
    
    print("="*70)
    print("FIRMUS AI FACTORY - COMPLETE SYSTEM INTEGRATION")
    print("="*70)
    
    # ===================================================================
    # 1. GPU Computational Model
    # ===================================================================
    print("\n1. GPU Computational Model")
    print("-" * 70)
    
    gpu_specs = GPUSpecs(
        name="NVIDIA H100 SXM",
        TDP=700.0,
        P_idle=50.0,
        P_memory=100.0,
        P_tensor=450.0,
        P_cuda=100.0
    )
    
    gpu = GPUModel(gpu_specs)
    
    # Training workload
    utilization = {'tensor': 0.9, 'cuda': 0.3, 'memory': 0.8}
    P_gpu = gpu.calculate_power(utilization)
    
    print(f"GPU Model: {gpu_specs.name}")
    print(f"TDP: {gpu_specs.TDP} W")
    print(f"Training Power: {P_gpu:.1f} W")
    print(f"Utilization: Tensor={utilization['tensor']:.0%}, "
          f"CUDA={utilization['cuda']:.0%}, Memory={utilization['memory']:.0%}")
    
    # ===================================================================
    # 2. Thermal Management
    # ===================================================================
    print("\n2. Thermal Management System")
    print("-" * 70)
    
    cooling_specs = CoolingSpecs(
        name="Immersion Cooling",
        fluid_type="3M Novec 7100",
        T_fluid_in=35.0,
        flow_rate=100.0,
        pump_efficiency=0.85
    )
    
    cooling = ImmersionCoolingSystem(cooling_specs)
    
    # 8 GPUs in rack
    num_gpus = 8
    total_gpu_power = P_gpu * num_gpus
    
    thermal_result = cooling.analyze(total_gpu_power, num_gpus)
    
    print(f"Cooling System: {cooling_specs.name}")
    print(f"Total GPU Power: {total_gpu_power/1000:.1f} kW")
    print(f"Junction Temperature: {thermal_result.T_junction_max:.1f} °C")
    print(f"Cooling Power: {thermal_result.P_cooling:.1f} W")
    print(f"pPUE: {thermal_result.pPUE:.3f}")
    
    # ===================================================================
    # 3. Power Delivery Network
    # ===================================================================
    print("\n3. Power Delivery Network")
    print("-" * 70)
    
    # Transformer
    transformer = TransformerModel(TRANSFORMER_13_8KV_TO_480V)
    I_load = total_gpu_power / transformer.specs.V_secondary
    v_drop, efficiency = transformer.calculate_voltage_regulation(I_load, pf=0.95)
    
    print(f"Transformer: {transformer.specs.name}")
    print(f"Voltage Drop: {v_drop:.2f} V ({v_drop/transformer.specs.V_secondary*100:.2f}%)")
    print(f"Efficiency: {efficiency:.3%}")
    
    # DC-DC Converter
    converter = BuckConverterModel(BUCK_480V_TO_12V)
    I_out = total_gpu_power / converter.specs.V_out
    conv_efficiency = converter.calculate_efficiency(I_out)
    
    print(f"\nDC-DC Converter: {converter.specs.name}")
    print(f"Efficiency: {conv_efficiency:.3%}")
    
    # VRM
    vrm = MultiphaseVRM(VRM_H100_SXM)
    I_gpu = P_gpu / vrm.specs.V_out
    vrm_efficiency = vrm.calculate_efficiency(I_gpu * num_gpus)
    
    print(f"\nVRM: {vrm.specs.name}")
    print(f"Phases: {vrm.specs.num_phases}")
    print(f"Efficiency: {vrm_efficiency:.3%}")
    
    # Overall PDN efficiency
    pdn_efficiency = efficiency * conv_efficiency * vrm_efficiency
    print(f"\nOverall PDN Efficiency: {pdn_efficiency:.3%}")
    
    # ===================================================================
    # 4. Grid Interface
    # ===================================================================
    print("\n4. Grid Interface & Demand Response")
    print("-" * 70)
    
    grid = GridInterface(GRID_US_480V)
    
    # Frequency response
    f_grid = 59.95  # Grid frequency drops to 59.95 Hz
    P_current = total_gpu_power + thermal_result.P_cooling
    P_adjustment = grid.frequency_response(f_grid, P_current)
    
    print(f"Grid Connection: {grid.specs.name}")
    print(f"Nominal Frequency: {grid.specs.f_nominal} Hz")
    print(f"Measured Frequency: {f_grid} Hz")
    print(f"Frequency Response: {P_adjustment/1000:.1f} kW reduction")
    
    # Demand response
    dr_manager = DemandResponseManager(P_max=10e6, P_min=1e6)
    available_reduction = dr_manager.calculate_available_reduction(
        P_current=P_current,
        thermal_headroom=50000,  # 50 kW thermal headroom
        deferrable_load=200000   # 200 kW deferrable workload
    )
    
    print(f"\nDemand Response Capacity: {available_reduction/1000:.1f} kW")
    
    # ===================================================================
    # 5. Energy Storage
    # ===================================================================
    print("\n5. Energy Storage (Battery/UPS)")
    print("-" * 70)
    
    battery = LithiumIonBattery(BATTERY_TESLA_MEGAPACK, SOC_init=0.7)
    
    print(f"Battery: {battery.specs.name}")
    print(f"Capacity: {battery.specs.C_nom} Ah @ {battery.specs.V_nom} V")
    print(f"Energy: {battery.specs.C_nom * battery.specs.V_nom / 1000:.1f} kWh")
    print(f"State of Charge: {battery.SOC:.1%}")
    
    # Simulate discharge
    I_discharge = 100  # 100 A discharge
    dt = 3600  # 1 hour
    V_terminal, P_loss = battery.update_state(I_discharge, dt)
    
    print(f"\nAfter 1-hour discharge at {I_discharge} A:")
    print(f"Terminal Voltage: {V_terminal:.1f} V")
    print(f"SOC: {battery.SOC:.1%}")
    print(f"Cell Temperature: {battery.T_cell:.1f} °C")
    
    # ===================================================================
    # 6. Economic Analysis
    # ===================================================================
    print("\n6. Economic Analysis")
    print("-" * 70)
    
    tariff = ElectricityTariff(tariff_type="TOU")
    
    # Simulate 24-hour power profile
    hours = np.arange(24)
    timestamps = np.arange(0, 24*3600, 3600) + 1640000000  # Example timestamps
    power_profile = np.ones(24) * P_current
    
    total_cost, breakdown = tariff.calculate_cost(power_profile, timestamps)
    
    print(f"Electricity Tariff: {tariff.tariff_type}")
    print(f"\n24-Hour Cost Breakdown:")
    print(f"  Energy Cost: ${breakdown['energy_cost']:.2f}")
    print(f"  Demand Cost: ${breakdown['demand_cost']:.2f}")
    print(f"  Fixed Cost: ${breakdown['fixed_cost']:.2f}")
    print(f"  Total Cost: ${breakdown['total_cost']:.2f}")
    
    # ===================================================================
    # 7. Model Predictive Control
    # ===================================================================
    print("\n7. Model Predictive Control Optimization")
    print("-" * 70)
    
    try:
        mpc = ModelPredictiveController(horizon=24, dt=1.0)
        
        # Price forecast (sinusoidal pattern)
        price_forecast = 0.12 + 0.05 * np.sin(2 * np.pi * np.arange(24) / 24)
        
        current_state = {
            'T_junction': thermal_result.T_junction_max,
            'SOC': battery.SOC,
            'P_gpu': np.ones(8) * P_gpu
        }
        
        optimal_controls = mpc.optimize(
            current_state=current_state,
            price_forecast=price_forecast,
            workload_queue=[],
            grid_signals={'frequency': 60.0}
        )
        
        print(f"MPC Horizon: {mpc.horizon} hours")
        print(f"Optimal 24-hour Cost: ${optimal_controls['total_cost']:.2f}")
        print(f"Peak Junction Temperature: {np.max(optimal_controls['T_junction']):.1f} °C")
        
    except ImportError:
        print("MPC requires cvxpy. Install with: pip install cvxpy")
    
    # ===================================================================
    # 8. Digital Twin Integration
    # ===================================================================
    print("\n8. Digital Twin Integration")
    print("-" * 70)
    
    config = {
        'gpu': gpu_specs,
        'thermal': cooling_specs,
        'grid': GRID_US_480V
    }
    
    twin = DigitalTwin(config)
    
    # Run 24-hour scenario
    results = twin.run_scenario(duration_hours=24, dt=300)  # 5-minute intervals
    
    print(f"Scenario Duration: 24 hours")
    print(f"Time Steps: {len(results['time'])}")
    print(f"Average Power: {np.mean(results['P_total'])/1000:.1f} kW")
    print(f"Peak Power: {np.max(results['P_total'])/1000:.1f} kW")
    print(f"Total Energy: {np.trapz(results['P_total'], results['time']):.1f} kWh")
    print(f"Total Cost: ${results['cumulative_cost'][-1]:.2f}")
    
    # ===================================================================
    # 9. Visualization
    # ===================================================================
    print("\n9. Generating Visualizations...")
    print("-" * 70)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Power profile
    axes[0].plot(results['time'], results['P_total']/1000, 'b-', linewidth=2)
    axes[0].set_ylabel('Power (kW)', fontsize=12)
    axes[0].set_title('24-Hour System Simulation', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Temperature profile
    axes[1].plot(results['time'], results['T_junction'], 'r-', linewidth=2)
    axes[1].axhline(y=83, color='r', linestyle='--', label='Thermal Limit')
    axes[1].set_ylabel('Junction Temp (°C)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative cost
    axes[2].plot(results['time'], results['cumulative_cost'], 'g-', linewidth=2)
    axes[2].set_xlabel('Time (hours)', fontsize=12)
    axes[2].set_ylabel('Cumulative Cost ($)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'complete_system_integration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "="*70)
    print("SYSTEM INTEGRATION SUMMARY")
    print("="*70)
    print(f"Total GPU Power: {total_gpu_power/1000:.1f} kW")
    print(f"Cooling Power: {thermal_result.P_cooling/1000:.1f} kW")
    print(f"Total Facility Power: {P_current/1000:.1f} kW")
    print(f"pPUE: {thermal_result.pPUE:.3f}")
    print(f"PDN Efficiency: {pdn_efficiency:.1%}")
    print(f"DR Capacity: {available_reduction/1000:.1f} kW")
    print(f"24-Hour Energy: {np.trapz(results['P_total'], results['time']):.1f} kWh")
    print(f"24-Hour Cost: ${results['cumulative_cost'][-1]:.2f}")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
