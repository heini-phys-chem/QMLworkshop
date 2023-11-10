import torch
import numpy as np

import argparse

from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel


model = torch.jit.load('model_sorf.pt')

print(model)


            total_energies = self.forward(coordinates, charges, atomIDs, molIDs, natom_counts, cells, cells)

            if (forces):
                forces_torch, = torch.autograd.grad(-total_energies.sum(), coordinates)
