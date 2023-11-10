#!/usr/bin/env python3


import numpy as np

from ase.md.langevin import Langevin
from ase.io import read, write

from ase.calculators.lj import LennardJones
from ase.md import MDLogger
from ase import units

from ase.calculators import ml
from ase.calculators import ff
from ase.calculators import gaussian
from ase.calculators.emt import EMT

import time

initial_atoms = read('test.xyz')

X      = np.load("X_mbdf.npy", allow_pickle=True)
Q      = np.load("Q_mbdf.npy", allow_pickle=True)
alphas = np.load("alphas_mbdf.npy", allow_pickle=True)
norms  = np.load("norms_1k.npy", allow_pickle=True)


initial_atoms.calc = ml.ML_calculator(initial_atoms, 0.64, alphas, X, Q, 42, [1, 6, 7, 8], 'tmp.out', norms)
#initial_atoms.calc = gaussian.Gaussian(method="PBE3", basis="sto-3g")

# Create a Lennard-Jones calculator (or use an appropriate calculator)
#initial_atoms.set_calculator(LennardJones())
#initial_atoms.set_calculator(EMT())

# Set up the Langevin dynamics
temperature = 300.0  # Kelvin
friction_coefficient = 1e-3  # Langevin thermostat friction coefficient
time_step = 0.5  # femtoseconds
total_steps = 1000  # Total number of MD steps
output_interval = 100  # Write output every 100 steps

dyn = Langevin(initial_atoms, timestep=time_step * units.fs, temperature_K=temperature, friction=friction_coefficient)

# Run the MD simulation
for step in range(total_steps):
    start = time.time()
    dyn.run(1)  # Run 1 MD step
    # Check if it's a step that you want to save (e.g., every 100 steps)
    #if step % 10 == 0:
    # Create a unique filename based on the step number
    filename = f'coords_ML_{step:04d}.xyz'

    # Write the coordinates to the XYZ file
    write(filename, initial_atoms, format='xyz')
    print("Step: {}, E: {}".format(step, initial_atoms.get_potential_energy()))
    end = time.time()
    fout = open("energies.out", 'a')
    fout.write("Step: {}, E: {}, time: {:.2f}\n".format(step, initial_atoms.get_potential_energy(), (end-start)/60. ))
    fout.close()

# Save the final structure to an XYZ file
write('final_structure.xyz', initial_atoms)
