"""GB300 NVL72 Rack Thermal Analysis Example.

This example demonstrates comprehensive thermal analysis of a
GB300 NVL72 rack (72 GPUs) using both direct-to-chip liquid cooling
and air cooling for peripheral components.

GB300 NVL72 Configuration:
- 72x GB300 GPUs @ 1400W TDP each (liquid cooled)
- 36x Grace CPUs @ 250W each (liquid cooled)
- 18x NVLink Switches @ ~3200W each (air cooled)
- Total rack power: ~150kW
"""

from firmus_ai_factory.thermal import (
    DirectToChipCooling,
    NVL72PeripheralAirCooling,
    ColdPlateSpec,
)
from firmus_ai_factory.computational.gpu_model import GB300_SPECS


def main():
    print("="*60)
    print("GB300 NVL72 Rack Thermal Analysis")
    print("="*60)
    
    # Print GPU specifications
    print(f"\nGPU Specifications:")
    print(f"  Model: {GB300_SPECS['name']}")
    print(f"  TDP: {GB300_SPECS['tdp']:.0f}W")
    print(f"  FP8 Performance: {GB300_SPECS['fp8_tflops']:.0f} TFLOPS")
    print(f"  HBM Capacity: {GB300_SPECS['hbm_capacity']:.0f} GB")
    
    # =========================================
    # Part 1: Direct-to-Chip Liquid Cooling
    # =========================================
    print(f"\n" + "-"*60)
    print("Part 1: Direct-to-Chip Liquid Cooling (GPUs + CPUs)")
    print("-"*60)
    
    # Initialize DLC system
    dlc = DirectToChipCooling(
        supply_temp=25.0,        # 25C supply water
        max_return_temp=45.0,    # 45C max return
        cdu_capacity_kw=250.0    # 250kW CDU
    )
    
    # Analyze full NVL72 rack
    result = dlc.analyze_nvl72_rack(
        gpu_power=GB300_SPECS['tdp'],  # 1400W per GPU
        num_gpus=72,
        num_cpus=36,
        cpu_power=250.0,
        switch_power=3200.0
    )
    
    print(f"\nThermal Results:")
    print(f"  GPU Junction Temp (max): {result.T_junction_max:.1f} C")
    print(f"  GPU Junction Temp (mean): {result.T_junction_mean:.1f} C")
    print(f"  Coolant Supply: {result.T_coolant_supply:.1f} C")
    print(f"  Coolant Return: {result.T_coolant_return:.1f} C")
    
    print(f"\nHydraulic Results:")
    print(f"  Flow Rate: {result.flow_rate_lpm:.1f} L/min")
    print(f"  Pump Power: {result.P_pump:.0f} W")
    
    print(f"\nCooling Power:")
    print(f"  CDU Power: {result.P_cdu:.0f} W")
    print(f"  Total Cooling: {result.P_cooling_total:.0f} W")
    print(f"  pPUE: {result.pPUE:.3f}")
    
    # =========================================
    # Part 2: Air Cooling for Peripherals
    # =========================================
    print(f"\n" + "-"*60)
    print("Part 2: Air Cooling (NVSwitches, NICs, Memory)")
    print("-"*60)
    
    # Initialize peripheral air cooling
    air_cooling = NVL72PeripheralAirCooling(
        num_switches=18,
        switch_power=3200.0,
        num_nics=36,
        nic_power=25.0,
        ambient_temp=25.0
    )
    
    air_result = air_cooling.analyze()
    
    print(f"\nHeat Loads:")
    print(f"  NVLink Switches: {air_result['switch_heat_load']/1000:.1f} kW")
    print(f"  NICs: {air_result['nic_heat_load']/1000:.2f} kW")
    print(f"  Other: {air_result['other_heat_load']/1000:.1f} kW")
    print(f"  Total Air-Cooled: {air_result['total_heat_load']/1000:.1f} kW")
    
    print(f"\nAir Cooling Requirements:")
    print(f"  Required Airflow: {air_result['airflow_cfm']:.0f} CFM")
    print(f"  Fan Power: {air_result['fan_power']:.0f} W")
    print(f"  Air Temperature Rise: {air_result['delta_t']:.1f} C")
    
    # =========================================
    # Part 3: Total Rack Summary
    # =========================================
    print(f"\n" + "="*60)
    print("Total NVL72 Rack Summary")
    print("="*60)
    
    # Calculate totals
    total_it_power = (72 * GB300_SPECS['tdp'] +  # GPUs
                      36 * 250.0 +                 # CPUs
                      air_result['total_heat_load'])  # Peripherals
    
    total_cooling_power = result.P_cooling_total + air_result['fan_power']
    total_power = total_it_power + total_cooling_power
    overall_pue = total_power / total_it_power
    
    print(f"\nIT Power Breakdown:")
    print(f"  GPUs (72x): {72 * GB300_SPECS['tdp'] / 1000:.1f} kW")
    print(f"  CPUs (36x): {36 * 250.0 / 1000:.1f} kW")
    print(f"  Peripherals: {air_result['total_heat_load'] / 1000:.1f} kW")
    print(f"  Total IT: {total_it_power / 1000:.1f} kW")
    
    print(f"\nCooling Power Breakdown:")
    print(f"  DLC (CDU + Pumps): {result.P_cooling_total / 1000:.2f} kW")
    print(f"  Air (Fans): {air_result['fan_power'] / 1000:.2f} kW")
    print(f"  Total Cooling: {total_cooling_power / 1000:.2f} kW")
    
    print(f"\nEfficiency Metrics:")
    print(f"  DLC pPUE: {result.pPUE:.3f}")
    print(f"  Overall PUE: {overall_pue:.3f}")
    print(f"  Cooling Overhead: {(overall_pue - 1) * 100:.1f}%")
    
    print(f"\n" + "="*60)
    print("Analysis Complete")
    print("="*60)


if __name__ == "__main__":
    main()
