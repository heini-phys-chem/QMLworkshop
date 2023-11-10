#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

# List of energy values
filename = sys.argv[1]

lines = open(filename, 'r').readlines()
energy_values = []

for line in lines:
    energy_values.append(float(line))


# Create a list of step values in steps of 10
num_steps = list(range(0, len(energy_values) * 10, 10))
print(len(num_steps))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(num_steps, energy_values, marker='o', linestyle='-')
plt.xlabel('Number of Steps (in steps of 100)')
plt.ylabel('Energy (eV)')
plt.title('Energy vs Number of Steps')

# Save the plot as an image file (e.g., PNG)
plt.grid(True)
plt.tight_layout()
plt.savefig('energy_vs_steps.png')

# Optionally, close the figure to release resources
plt.close()
