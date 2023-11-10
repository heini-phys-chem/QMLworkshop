#!/usr/bin/env python3

import sys
import numpy as np

# Load the .npz file
data = np.load(sys.argv[1])

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


print(R.shape)
print(F.shape)
print(E.shape)
print(z.shape)
exit()

NAMES = {1:"H", 6:"C", 7:"N", 8:"O"}

for i, coords in enumerate(R[0]):
    print(NAMES[z[i]], coords[0], coords[1], coords[2])
