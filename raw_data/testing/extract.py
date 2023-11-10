#!/usr/bin/env python3

import numpy as np

# Load the .npz file
data = np.load('train.npz')

# List the keys in the .npz file
print("Keys in the .npz file:", data.files)

R = data['R']
F = data['F']
E = data['E']
z = data['z']

print(R[0])
print(F[0])
print(E[0])
print(z)
