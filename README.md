# Polarizable-Ether Fluid Optics, Extended Longitudinal Electrodynamics, and Rank-2 Elastic Shear Modes

**Author:** Le Huynh Phuoc (*Independent Researcher*) · **Version:** `v1.1.1` (November 2026)

Consistency audit, algebraic corrections, and numerical benchmark suite for dynamical flat-space ether models in 3D Euclidean space and absolute time.

## Repository Structure

```text
.
├── paper/
│   ├── ether_monograph_4337101a.pdf   # Full English monograph (v1.1.1, 12 pages, MathML equations & figures)        
├── simulations/
│   ├── mmx_refutation_sim.py          # 1887 MMX vs. Lorentz-contracting ether & anisotropy bounds
│   ├── soliton_lorentz_sim.py         # 1D Sine-Gordon PDE: spontaneous contraction & breather time dilation
│   ├── longitudinal_S_wave_sim.py     # RK4 solver for Extended Maxwell longitudinal S-wave & Poynting balance
│   ├── ether_ray_lagally_sim.py       # Eikonal ray ODE & Lagally hydrodynamic sink surface integral
│   ├── solar_strongfield_tests.py     # 4 Solar System tests (incl. Mercury 42.98''/cy) & Shapiro delay
│   ├── kinematic_experiments_sim.py   # Kennedy-Thorndike (CMB) & Ives-Stilwell (ESR 2014) verification
│   └── tensor_gw_eht_sim.py           # EHT black-hole shadow bounds & LIGO Spin-0 vs. Spin-2 shear response
├── figures/                           # English publication figures embedded in the monograph
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Quick Start

```bash
pip install -r requirements.txt
python simulations/mmx_refutation_sim.py
python simulations/soliton_lorentz_sim.py
python simulations/longitudinal_S_wave_sim.py
python simulations/ether_ray_lagally_sim.py
python simulations/solar_strongfield_tests.py
python simulations/kinematic_experiments_sim.py
python simulations/tensor_gw_eht_sim.py
```
