#!/usr/bin/env python3

import numpy as np

import time

import torch
from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel

from ase.io import read, write
from ase.data import atomic_masses

def write_xyz(filename):
    # Write the XYZ coordinates to the file
    with open(file_path, 'w') as file:
        file.write(f'{xyz.size(0)}\n')  # Number of atoms
        file.write('\n')  # Optional comment line

        for atom in xyz:
            file.write(f'X {atom[0]} {atom[1]} {atom[2]}\n')

# Parameters
num_steps = torch.tensor(1000, dtype=torch.float32, device='cuda')
time_step = torch.tensor(0.5, dtype=torch.float32, device='cuda')
#temperature = 300.0  # Temperature in Kelvin
temperature = torch.tensor(300.0, dtype=torch.float32, device='cuda')
friction_coefficient = torch.tensor(1e-3, dtype=torch.float32, device='cuda')

atoms = read('test.xyz')

coords = atoms.get_positions()
nuclear_charges = atoms.get_atomic_numbers()
nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], coords.shape[0], axis=0)
unique_z = np.unique(np.concatenate(nuclear_charges)).astype(int)

test_coordinates = [atoms.get_positions()]
test_charges = [atoms.get_atomic_numbers()]

# Calculate atomic masses from nuclear charges
masses = torch.tensor([atomic_masses[charge] for charge in atoms.get_atomic_numbers()], device='cuda', dtype=torch.float32)

# Initialize particle positions (xyz) and velocities on GPU
xyz = torch.tensor(atoms.get_positions(), device='cuda', dtype=torch.float32, requires_grad=True)
v = torch.randn(xyz.shape, dtype=torch.float32, device='cuda')

charges = torch.tensor(test_charges[0], device='cuda', dtype=torch.float32)
atomIDs =torch.arange(xyz.shape[0],device='cuda', dtype=torch.int32)
molIDs = torch.zeros(xyz.shape[0], device='cuda',dtype=torch.int32)
atom_counts= torch.tensor([xyz.shape[0]], device='cuda', dtype=torch.int32)

# Load your QMLightning model
model = torch.jit.load('/home/heinen/QMLworkshop/task1/model_sorf.pt')
#model.to('cuda')

start = time.time()
# Perform Langevin MD simulation
for step in range(20):
    # Calculate the potential energy using the loaded model
    input_data = (xyz[None], charges[None], atomIDs, molIDs, atom_counts)  # Adjust as needed
    potential_energy = model.forward(*input_data)

    # Calculate the force using gradient of the potential energy
    force, = torch.autograd.grad(-potential_energy.sum(), xyz)

    # Langevin thermostat - stochastic term
    eta = torch.randn(xyz.shape, dtype=torch.float32, device='cuda')
    stochastic_term = torch.sqrt(2 * temperature * friction_coefficient * time_step) * eta

    # Update velocity
    v = v + (force / masses.view(-1, 1) - friction_coefficient * v) * time_step + stochastic_term

    # Update position
    xyz = xyz + v * time_step
    print(xyz)

#    if step % 10 == 0:
#        print(f"Step {step}: Positions: {xyz}")
    print("Potential Energy: {:6f}".format(potential_energy.item()))

end = time.time()

print("Run time: {:.4f}".format( (end-start)/60. ))

