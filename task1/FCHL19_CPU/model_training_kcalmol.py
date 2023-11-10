#!/usr/bin/env python3

import sys
#sys.path.insert(0, "/home/heinen/qml/build/lib.linux-x86_64-2.7")
#sys.path.insert(0, "/home/heinen/qml_develop/build/lib.linux-x86_64-2.7")
import time
from time import time
from random import shuffle

import scipy
import scipy.stats

import numpy as np
from numpy.linalg import norm, inv

#import cPickle
import _pickle as cPickle

import qml
from qml.math import cho_solve
from qml.math import svd_solve
from qml.math import qrlq_solve

#from qml.representations import generate_fchl_acsf, generate_bfcl_acsf, generate_vpbfcl_acsf
from qml.representations import generate_fchl_acsf

from qml.kernels import get_atomic_local_gradient_kernel
from qml.kernels import get_atomic_local_kernel

from MBDF_gradients import generate_mbdf_train, generate_mbdf_pred

# Function to parse datafile to a dictionary
def get_properties(filename):
    """ Returns a dictionary with energy and forces for each xyz-file.
    """
    # define dictionairies and constants
    properties = dict()

    data = np.load(filename)
    R = data['R']
    F = data['F']
    E = data['E']
    z = data['z']

    for i in range(E.shape[0]):
        properties[i] = [E[i], F[i], R[i], z]

    return properties

if __name__ == "__main__":
    mols = []
    mols_pred = []

    SIGMA   = 10
    llambda = 1e-15
    data = get_properties(sys.argv[1])

    for name in sorted(data.keys()):
      mol = qml.Compound()
      #mol.read_xyz("xyz/xyz_all/" + name + ".xyz")

      # Associate a property (heat of formation) with the object
      mol.energy = data[name][0]
      mol.forces = data[name][1]
      mol.coords = data[name][2]
      mol.z      = data[name][3]
      mols.append(mol)


#    print(mols[8].energy)
#    print(mols[8].forces)
#    print(mols[8].coords)
#    print(mols[8].z)
#    print(mols[8].energy.shape)
#    print(mols[8].forces.shape)
#    print(mols[8].coords.shape)
#    print(mols[8].z.shape)
#    exit()
    # REPRESENTATIONS
    print("\n [-] calculate representations")
    start = time()
    x = []
    disp_x = []
    f = []
    e = []
    q = []

    for mol in mols:
      (x1, dx1) = generate_fchl_acsf(mol.z, mol.coords, gradients=True, pad=42, elements=[1, 6, 7, 8])

      e.append(mol.energy)
      f.append(mol.forces)  # to match dict key
      x.append(x1)
      disp_x.append(dx1)
      q.append(mol.z)

    X_train = np.array(x)
    F_train = np.array(f, dtype=object)
    print(F_train.shape)
    F_train *= -1
    E_train = np.array(e)
    E_train = E_train.flatten()
    dX_train = np.array(disp_x)
    Q_train = q
    F_train = np.concatenate(F_train)

    end = time()
    print("X: ", X_train.shape)
    print("dX: ", dX_train.shape)
    print("E: ", E_train.shape)
    print("F: ", F_train.shape)
    print("F.f: ", F_train.flatten().shape)

    print(" [+] calculate representations ({:.2f} min)".format( (end-start)/60. ))
    print("")
    print(" [-] calculate Kernels")

    start = time()
    Kte = get_atomic_local_kernel(X_train,  X_train, Q_train,  Q_train,  SIGMA)
    Kt = get_atomic_local_gradient_kernel(X_train,  X_train, dX_train,  Q_train,  Q_train, SIGMA)
    #Kte = get_local_kernel(X_train,  X_train, Q_train,  Q_train,  SIGMA)
    #Kt = get_local_gradient_kernel(X_train,  X_train, dX_train,  Q_train,  Q_train, SIGMA)

    print("Ke shape: ", Kte.shape)
    print("Kt shape: ", Kt.shape)
    end = time()
    print(" [+] calculate Kernels ({:.2f} min)".format( (end-start)/60. ))
    print("")
    C = np.concatenate((Kte, Kt))
    Y = np.concatenate((E_train, F_train.flatten()))
    Y = Y.astype(float)
    print("C shape: ", C.shape)
    print("Y shape: ", Y.shape)
#    exit()

    print(" [-] calculate Alphas operator")
    start = time()
    alpha = svd_solve(C, Y, rcond=llambda)
    end = time()
    print(" [+] calculate Alphas operator ({:.2f} min)".format( (end-start)/60. ))
    print("")

    # Store alphas Q's and X's
    print(" [-] Xs, Qs, alphas")
    np.save('X_fchl.npy', X_train)
    np.save('alphas_fchl.npy', alpha)
    np.save('Q_fchl.npy', Q_train)
    print(" [+] Xs, Qs, alphas")
    print("")

    eYt = np.dot(Kte, alpha)
    fYt = np.dot(Kt, alpha)

    # print training error for energies and forces
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E_train, eYt)
    print("TRAINING ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(E_train - eYt)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F_train.flatten(), fYt.flatten())
    print("TRAINING FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
             (np.mean(np.abs(F_train.flatten() - fYt.flatten())), slope, intercept, r_value ))
