#!/usr/bin/env python3

import numpy as np

import torch
from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel

from ase.md.langevin import Langevin
from ase.io import read, write
from ase.data import atomic_masses


# Parameters
num_steps = 1000
time_step = 0.01
temperature = 300.0  # Temperature in Kelvin
friction_coefficient = 1.0

atoms = read('test.xyz')
# Initialize particle positions (xyz) and velocities on GPU
xyz = torch.tensor(atoms.get_positions(), device='cuda', dtype=torch.float32, requires_grad=True)
v = torch.randn(xyz.shape, dtype=torch.float32, device='cuda')

# Load your QMLightning model
model = torch.jit.load('/home/heinen/QMLworkshop/model_sorf.pt')
#model.to('cuda')
# Calculate atomic masses from nuclear charges
masses = torch.tensor([atomic_masses[charge] for charge in nuclear_charges], device='cuda', dtype=torch.float32)

# Perform Langevin MD simulation
for step in range(num_steps):
    # Calculate the potential energy using the loaded model
    with torch.no_grad():
        input_data = (xyz[None], charges[None], atomIDs, molIDs, atom_counts)  # Adjust as needed
        potential_energy = model.forward(*input_data)

    # Calculate the force using gradient of the potential energy
    force = torch.autograd.grad(-potential_energy.sum(), xyz)[0]

    # Langevin thermostat - stochastic term
    eta = torch.randn(xyz.shape, dtype=torch.float32, device='cuda')
    stochastic_term = torch.sqrt(2 * temperature * friction_coefficient * time_step) * eta

    # Update velocity and position
    v = v + (force - friction_coefficient * v) * time_step + stochastic_term
    xyz = xyz + v * time_step

    if step % 100 == 0:
        print(f"Step {step}: Positions: {xyz}")
        print("Potential Energy:", potential_energy.item())


