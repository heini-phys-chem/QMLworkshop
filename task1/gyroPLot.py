#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def compute_gyration_radius_no_hydrogen(coordinates, atom_types):
    # Exclude hydrogen atoms based on atom types (modify as needed).
    non_hydrogen_indices = [i for i, atom_type in enumerate(atom_types) if atom_type != 'H']

    # Filter atomic coordinates for non-hydrogen atoms.
    non_hydrogen_coords = coordinates[non_hydrogen_indices]
    center = non_hydrogen_coords.mean(0)

    squared_distances = np.sum((non_hydrogen_coords - center)**2, axis=1)
    gyration_radius = np.sqrt(np.mean(squared_distances))
    return gyration_radius




def extract_atom_labels_and_coordinates(xyz_file):
    atom_labels = []
    coordinates = []

    with open(xyz_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip().split()
        if len(line) == 4:  # Assumes atom lines have 4 columns (atom type and x, y, z coordinates).
            atom_labels.append(line[0])
            coordinates.append([float(line[1]), float(line[2]), float(line[3])])

    return np.asarray(atom_labels), np.array(coordinates)

# Example usage:
xyz_file = "xyz/save/out.xyz"  # Replace with your XYZ file path.
atom_labels, coordinates = extract_atom_labels_and_coordinates(xyz_file)

atom_labels = atom_labels.reshape(30,42)
coordinates = coordinates.reshape(30,42,3)



# Loop through frames and analyze each one.
gyrRad = np.array([])
for i, coords in enumerate(coordinates):
    gyration_radius = compute_gyration_radius_no_hydrogen(coords, atom_labels[i])
    gyrRad = np.append(gyrRad, gyration_radius)

    print(f"Frame - Gyration Radius: {gyration_radius}")

print(gyrRad.shape)
num_steps = list(range(0, len(gyrRad) * 100, 100))
# You can perform other analyses or save the results as needed.
plt.figure(figsize=(10, 6))
#plt.hist(gyrRad, bins=100, color='C0', alpha=0.7)
plt.plot(num_steps, gyrRad, marker='o', color='C0', linestyle='-')
plt.ylabel("Gyration Radius")
plt.xlabel("Step")
plt.grid(True)
plt.tight_layout()
plt.savefig('gyros.png')

