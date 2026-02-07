"""Firmus AI Factory Configuration and Platform-Cooling-Region Mapping.

Defines the canonical mapping between GPU platforms, cooling systems,
and grid regions:

    HGX H100 / H200  →  Singapore  →  Immersion Cooling
    GB300 NVL72       →  Australia  →  Benmax HCU2500 Air-Liquid Cooling
    Vera Rubin NVL72  →  Australia  →  Benmax HCU2500 Air-Liquid Cooling

This module provides the top-level factory configuration that ties
together all subsystem models (GPU, thermal, power, grid, economics).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum

from firmus_ai_factory.computational.gpu_model import (
    GPUSpecifications, GPUModel, NVL72RackConfig, PowerProfile_Mode,
    H100_SXM_SPECS, H200_SPECS, GB300_SPECS,
    VERA_RUBIN_MAX_P_SPECS, VERA_RUBIN_MAX_Q_SPECS,
    GB300_NVL72_CONFIG, VR_NVL72_MAX_P_CONFIG, VR_NVL72_MAX_Q_CONFIG,
)
from firmus_ai_factory.thermal.vera_rubin_thermal import VRNvl72ThermalModel
from firmus_ai_factory.thermal.benmax_hcu2500 import (
    BenmaxHCU2500, BenmaxHypercube, HCURedundancyMode,
)
from firmus_ai_factory.grid.regional_grids import (
    GridRegion, RegionalGridModel,
    SINGAPORE_GRID, AUSTRALIA_NEM_GRID,
)


class GPUPlatform(Enum):
    """Supported GPU platforms."""
    HGX_H100 = "hgx_h100"
    HGX_H200 = "hgx_h200"
    GB300_NVL72 = "gb300_nvl72"
    VR_NVL72_MAX_P = "vr_nvl72_max_p"
    VR_NVL72_MAX_Q = "vr_nvl72_max_q"


class CoolingType(Enum):
    """Cooling system types."""
    IMMERSION = "immersion"
    BENMAX_HCU2500 = "benmax_hcu2500"


# =============================================================================
# Canonical Platform-Cooling-Region Mapping
# =============================================================================

PLATFORM_CONFIG = {
    GPUPlatform.HGX_H100: {
        'gpu_specs': H100_SXM_SPECS,
        'cooling_type': CoolingType.IMMERSION,
        'grid_region': GridRegion.SINGAPORE,
        'gpus_per_node': 8,
        'rack_power_kw': 10.2,  # Per HGX node
        'description': 'NVIDIA HGX H100 8-GPU node with immersion cooling in Singapore',
    },
    GPUPlatform.HGX_H200: {
        'gpu_specs': H200_SPECS,
        'cooling_type': CoolingType.IMMERSION,
        'grid_region': GridRegion.SINGAPORE,
        'gpus_per_node': 8,
        'rack_power_kw': 10.2,  # Per HGX node
        'description': 'NVIDIA HGX H200 8-GPU node with immersion cooling in Singapore',
    },
    GPUPlatform.GB300_NVL72: {
        'gpu_specs': GB300_SPECS,
        'rack_config': GB300_NVL72_CONFIG,
        'cooling_type': CoolingType.BENMAX_HCU2500,
        'grid_region': GridRegion.AUSTRALIA_NEM,
        'gpus_per_rack': 72,
        'rack_power_kw': 150.0,
        'description': 'NVIDIA GB300 NVL72 with Benmax HCU2500 cooling in Australia',
    },
    GPUPlatform.VR_NVL72_MAX_P: {
        'gpu_specs': VERA_RUBIN_MAX_P_SPECS,
        'rack_config': VR_NVL72_MAX_P_CONFIG,
        'cooling_type': CoolingType.BENMAX_HCU2500,
        'grid_region': GridRegion.AUSTRALIA_NEM,
        'gpus_per_rack': 72,
        'rack_power_kw': 227.0,
        'description': 'NVIDIA Vera Rubin NVL72 Max P with Benmax HCU2500 cooling in Australia',
    },
    GPUPlatform.VR_NVL72_MAX_Q: {
        'gpu_specs': VERA_RUBIN_MAX_Q_SPECS,
        'rack_config': VR_NVL72_MAX_Q_CONFIG,
        'cooling_type': CoolingType.BENMAX_HCU2500,
        'grid_region': GridRegion.AUSTRALIA_NEM,
        'gpus_per_rack': 72,
        'rack_power_kw': 187.0,
        'description': 'NVIDIA Vera Rubin NVL72 Max Q with Benmax HCU2500 cooling in Australia',
    },
}


# =============================================================================
# Factory Configuration
# =============================================================================

@dataclass
class FactoryConfig:
    """Complete AI Factory configuration.
    
    Ties together GPU platform, cooling system, grid region, and
    facility-level parameters.
    
    Attributes:
        name: Factory name/identifier
        platform: GPU platform
        num_racks: Number of GPU racks
        cooling_type: Cooling system type
        grid_region: Grid region
        coolant_inlet_temp_c: Coolant supply temperature (°C)
        ambient_temp_c: Ambient air temperature (°C)
    """
    name: str
    platform: GPUPlatform
    num_racks: int
    cooling_type: CoolingType
    grid_region: GridRegion
    coolant_inlet_temp_c: float = 35.0
    ambient_temp_c: float = 30.0
    
    @property
    def platform_info(self) -> Dict:
        return PLATFORM_CONFIG[self.platform]
    
    @property
    def rack_power_kw(self) -> float:
        return self.platform_info['rack_power_kw']
    
    @property
    def total_it_power_kw(self) -> float:
        return self.num_racks * self.rack_power_kw
    
    @property
    def total_it_power_mw(self) -> float:
        return self.total_it_power_kw / 1000.0
    
    @property
    def total_gpus(self) -> int:
        gpus_per = self.platform_info.get('gpus_per_rack', 
                   self.platform_info.get('gpus_per_node', 8))
        return self.num_racks * gpus_per


class FirmusAIFactory:
    """Top-level Firmus AI Factory model.
    
    Integrates all subsystem models based on the factory configuration:
    - GPU computational model
    - Thermal model (immersion or Benmax HCU2500)
    - Grid model (Singapore or Australia)
    - Power delivery model
    - Economics model
    
    Usage:
        config = FactoryConfig(
            name="Firmus SG-01",
            platform=GPUPlatform.HGX_H200,
            num_racks=100,
            cooling_type=CoolingType.IMMERSION,
            grid_region=GridRegion.SINGAPORE,
        )
        factory = FirmusAIFactory(config)
        report = factory.generate_full_report()
    """
    
    def __init__(self, config: FactoryConfig):
        """Initialize factory with all subsystem models.
        
        Args:
            config: Factory configuration
        """
        self.config = config
        
        # Validate platform-cooling-region mapping
        expected = PLATFORM_CONFIG[config.platform]
        if config.cooling_type != expected['cooling_type']:
            raise ValueError(
                f"Platform {config.platform.value} requires "
                f"{expected['cooling_type'].value} cooling, "
                f"got {config.cooling_type.value}"
            )
        if config.grid_region != expected['grid_region']:
            raise ValueError(
                f"Platform {config.platform.value} is deployed in "
                f"{expected['grid_region'].value}, "
                f"got {config.grid_region.value}"
            )
        
        # Initialize GPU model
        self.gpu_model = GPUModel(expected['gpu_specs'])
        
        # Initialize thermal model
        self._init_thermal_model()
        
        # Initialize grid model
        self.grid_model = RegionalGridModel(config.grid_region)
    
    def _init_thermal_model(self):
        """Initialize the appropriate thermal model."""
        if self.config.cooling_type == CoolingType.IMMERSION:
            # Immersion cooling for Singapore H100/H200
            self.thermal_model = None  # Use existing immersion_cooling.py
            self.cooling_model = None
        elif self.config.cooling_type == CoolingType.BENMAX_HCU2500:
            # Benmax HCU2500 for Australia GB300/VR
            platform = self.config.platform
            
            if platform in (GPUPlatform.VR_NVL72_MAX_P, GPUPlatform.VR_NVL72_MAX_Q):
                mode = "max_p" if platform == GPUPlatform.VR_NVL72_MAX_P else "max_q"
                rack_tdp = self.config.rack_power_kw
                self.thermal_model = VRNvl72ThermalModel(
                    rack_tdp_kw=rack_tdp, power_mode=mode)
            else:
                self.thermal_model = VRNvl72ThermalModel(
                    rack_tdp_kw=self.config.rack_power_kw, power_mode="max_p")
            
            # Hypercube: 32 racks per Hypercube
            num_hypercubes = max(1, self.config.num_racks // 32)
            racks_per_hypercube = min(32, self.config.num_racks)
            
            self.cooling_model = BenmaxHypercube(
                num_hcu=4,
                num_racks=racks_per_hypercube,
                rack_power_kw=self.config.rack_power_kw,
                primary_inlet_temp_c=self.config.coolant_inlet_temp_c,
            )
            self.num_hypercubes = num_hypercubes
    
    def compute_summary(self) -> Dict:
        """Generate compute capacity summary."""
        specs = self.config.platform_info['gpu_specs']
        return {
            'platform': self.config.platform.value,
            'total_gpus': self.config.total_gpus,
            'gpu_tdp_w': specs.tdp_watts,
            'peak_fp16_tflops_per_gpu': specs.peak_flops_fp16,
            'total_fp16_pflops': self.config.total_gpus * specs.peak_flops_fp16 / 1000,
            'total_hbm_tb': self.config.total_gpus * specs.hbm_capacity_gb / 1000,
            'nvlink_bw_gb_s_per_gpu': specs.nvlink_bandwidth_gb_s,
        }
    
    def thermal_summary(self) -> Dict:
        """Generate thermal analysis summary."""
        inlet_temp = self.config.coolant_inlet_temp_c
        
        if self.config.cooling_type == CoolingType.IMMERSION:
            return {
                'cooling_type': 'Immersion Cooling',
                'region': 'Singapore',
                'ambient_temp_c': self.config.ambient_temp_c,
                'total_it_power_kw': self.config.total_it_power_kw,
                'note': 'Immersion cooling model (see immersion_cooling.py)',
            }
        
        elif self.config.cooling_type == CoolingType.BENMAX_HCU2500:
            # Per-rack thermal analysis
            rack_report = self.thermal_model.generate_thermal_report(inlet_temp)
            
            # Hypercube cooling analysis
            hypercube_report = self.cooling_model.generate_report(
                HCURedundancyMode.FOUR_HCU)
            
            return {
                'cooling_type': 'Benmax HCU2500 (Air-Liquid)',
                'region': 'Australia',
                'num_hypercubes': getattr(self, 'num_hypercubes', 1),
                'per_rack': {
                    'power_kw': self.config.rack_power_kw,
                    'inlet_temp_c': inlet_temp,
                    'outlet_temp_c': rack_report['outlet_temp_c'],
                    'delta_t_c': rack_report['delta_T_c'],
                    'flowrate_lpm': rack_report['flowrate_lpm'],
                    'pressure_drop_psid': rack_report['pressure_drop_psid'],
                    'within_limits': rack_report['within_limits'],
                },
                'hypercube': {
                    'total_it_load_kw': hypercube_report['thermal']['total_it_load_kw'],
                    'cooling_capacity_kw': hypercube_report['thermal']['total_cooling_capacity_kw'],
                    'capacity_margin_pct': hypercube_report['thermal']['capacity_margin_pct'],
                    'pump_power_kw': hypercube_report['power']['total_pump_power_kw'],
                    'pPUE': hypercube_report['power']['pPUE'],
                },
                'nvidia_compliance': hypercube_report['nvidia_compliance'],
                'all_compliant': hypercube_report['all_compliant'],
            }
    
    def grid_summary(self) -> Dict:
        """Generate grid and economics summary."""
        return self.grid_model.generate_report(self.config.total_it_power_mw)
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive factory analysis report.
        
        Returns:
            Dict with complete factory analysis including compute,
            thermal, grid, and economic summaries.
        """
        compute = self.compute_summary()
        thermal = self.thermal_summary()
        grid = self.grid_summary()
        
        # Calculate total facility power including cooling
        it_power_kw = self.config.total_it_power_kw
        if (self.config.cooling_type == CoolingType.BENMAX_HCU2500 and 
            self.cooling_model is not None):
            cooling_power_kw = (self.cooling_model.total_pump_power_kw() * 
                               getattr(self, 'num_hypercubes', 1))
            total_power_kw = it_power_kw + cooling_power_kw
            pue = total_power_kw / it_power_kw
        else:
            cooling_power_kw = it_power_kw * 0.05  # Estimate 5% for immersion
            total_power_kw = it_power_kw + cooling_power_kw
            pue = total_power_kw / it_power_kw
        
        return {
            'factory': {
                'name': self.config.name,
                'platform': self.config.platform.value,
                'description': self.config.platform_info['description'],
                'num_racks': self.config.num_racks,
                'total_gpus': self.config.total_gpus,
            },
            'power': {
                'it_power_kw': it_power_kw,
                'it_power_mw': it_power_kw / 1000,
                'cooling_power_kw': cooling_power_kw,
                'total_facility_power_kw': total_power_kw,
                'total_facility_power_mw': total_power_kw / 1000,
                'pue': pue,
            },
            'compute': compute,
            'thermal': thermal,
            'grid': grid,
        }


# =============================================================================
# Pre-defined Factory Configurations
# =============================================================================

def singapore_h100_factory(num_nodes: int = 100) -> FirmusAIFactory:
    """Create Singapore HGX H100 factory with immersion cooling."""
    config = FactoryConfig(
        name="Firmus SG-H100",
        platform=GPUPlatform.HGX_H100,
        num_racks=num_nodes,
        cooling_type=CoolingType.IMMERSION,
        grid_region=GridRegion.SINGAPORE,
        coolant_inlet_temp_c=35.0,
        ambient_temp_c=32.0,
    )
    return FirmusAIFactory(config)


def singapore_h200_factory(num_nodes: int = 100) -> FirmusAIFactory:
    """Create Singapore HGX H200 factory with immersion cooling."""
    config = FactoryConfig(
        name="Firmus SG-H200",
        platform=GPUPlatform.HGX_H200,
        num_racks=num_nodes,
        cooling_type=CoolingType.IMMERSION,
        grid_region=GridRegion.SINGAPORE,
        coolant_inlet_temp_c=35.0,
        ambient_temp_c=32.0,
    )
    return FirmusAIFactory(config)


def australia_gb300_factory(num_racks: int = 32) -> FirmusAIFactory:
    """Create Australia GB300 NVL72 factory with Benmax HCU2500 cooling."""
    config = FactoryConfig(
        name="Firmus AU-GB300",
        platform=GPUPlatform.GB300_NVL72,
        num_racks=num_racks,
        cooling_type=CoolingType.BENMAX_HCU2500,
        grid_region=GridRegion.AUSTRALIA_NEM,
        coolant_inlet_temp_c=35.0,
        ambient_temp_c=25.0,
    )
    return FirmusAIFactory(config)


def australia_vera_rubin_factory(num_racks: int = 32,
                                  max_q: bool = False) -> FirmusAIFactory:
    """Create Australia Vera Rubin NVL72 factory with Benmax HCU2500 cooling.
    
    Args:
        num_racks: Number of VR NVL72 racks
        max_q: If True, use Max Q power mode; otherwise Max P
    """
    platform = GPUPlatform.VR_NVL72_MAX_Q if max_q else GPUPlatform.VR_NVL72_MAX_P
    config = FactoryConfig(
        name=f"Firmus AU-VR-{'MaxQ' if max_q else 'MaxP'}",
        platform=platform,
        num_racks=num_racks,
        cooling_type=CoolingType.BENMAX_HCU2500,
        grid_region=GridRegion.AUSTRALIA_NEM,
        coolant_inlet_temp_c=35.0,
        ambient_temp_c=25.0,
    )
    return FirmusAIFactory(config)


if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("  FIRMUS AI FACTORY - Platform Configuration Summary")
    print("=" * 70)
    
    factories = [
        ("Singapore H100", singapore_h100_factory(100)),
        ("Singapore H200", singapore_h200_factory(100)),
        ("Australia GB300", australia_gb300_factory(32)),
        ("Australia VR Max P", australia_vera_rubin_factory(32, max_q=False)),
        ("Australia VR Max Q", australia_vera_rubin_factory(32, max_q=True)),
    ]
    
    for name, factory in factories:
        report = factory.generate_full_report()
        print(f"\n--- {name} ---")
        print(f"  GPUs: {report['factory']['total_gpus']}")
        print(f"  IT Power: {report['power']['it_power_mw']:.1f} MW")
        print(f"  Total Power: {report['power']['total_facility_power_mw']:.1f} MW")
        print(f"  PUE: {report['power']['pue']:.4f}")
        print(f"  FP16: {report['compute']['total_fp16_pflops']:.1f} PFLOPS")
        print(f"  HBM: {report['compute']['total_hbm_tb']:.1f} TB")
        print(f"  Grid: {report['grid']['region']}")
        print(f"  Annual Energy Cost: {report['grid']['energy_cost']['currency']} "
              f"{report['grid']['energy_cost']['annual_cost']:,.0f}")
"""
