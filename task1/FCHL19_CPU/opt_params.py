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

from qml.kernels import get_local_kernels_gaussian
from qml.kernels import get_atomic_local_gradient_kernel
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_local_kernel
from qml.kernels import get_gdml_kernel
from qml.kernels import get_symmetric_gdml_kernel
from qml.kernels import get_local_gradient_kernel
from qml.kernels import get_gp_kernel
from qml.kernels import get_symmetric_gp_kernel

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

    #E -= np.min(E)

    for i in range(E.shape[0]):
        properties[i] = [E[i], F[i], R[i], z]

    return properties

if __name__ == "__main__":
    mols = []
    mols_val = []

    #SIGMA   = 1.0
    sigmas = [10., 15.0, 20.0, 25.0, 30.0]
#    llambda = 1e-12
    lls = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15]
    data = get_properties(sys.argv[1])
    data_val = get_properties(sys.argv[2])

    for name in sorted(data.keys()):
      mol = qml.Compound()

      # Associate a property (heat of formation) with the object
      mol.energy = data[name][0]
      mol.forces = data[name][1]
      mol.coords = data[name][2]
      mol.z      = data[name][3]
      mols.append(mol)

    for name in sorted(data_val.keys()):
      mol = qml.Compound()

      # Associate a property (heat of formation) with the object
      mol.energy = data_val[name][0]
      mol.forces = data_val[name][1]
      mol.coords = data_val[name][2]
      mol.z      = data_val[name][3]
      mols_val.append(mol)


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
    E_train = np.array(e)
    E_train = E_train.flatten()
    dX_train = np.array(disp_x)
    Q_train = q
    F_train = np.concatenate(F_train)

    x = []
    disp_x = []
    f = []
    e = []
    q = []

    for mol in mols_val:
      (x1, dx1) = generate_fchl_acsf(mol.z, mol.coords, gradients=True, pad=42, elements=[1, 6, 7, 8])

      e.append(mol.energy)
      f.append(mol.forces)  # to match dict key
      x.append(x1)
      disp_x.append(dx1)
      q.append(mol.z)

    X_val = np.array(x)
    F_val = np.array(f, dtype=object)
    E_val = np.array(e)
    E_val = E_val.flatten()
    dX_val = np.array(disp_x)
    Q_val = q
    F_val = np.concatenate(F_val)

    end = time()

    print(" [+] calculate representations ({:.2f} min)".format( (end-start)/60. ))
    print("")
    print(" [-] calculate Kernels")


    for SIGMA in sigmas:
        start = time()
        Kte = get_atomic_local_kernel(X_train,  X_train, Q_train,  Q_train,  SIGMA)
        Kt = get_atomic_local_gradient_kernel(X_train,  X_train, dX_train,  Q_train,  Q_train, SIGMA)

        Kte_val = get_atomic_local_kernel(X_train,  X_val, Q_train,  Q_val,  SIGMA)
        Kt_val  = get_atomic_local_gradient_kernel(X_train,  X_val, dX_val,  Q_train,  Q_val, SIGMA)
        end = time()
        print(" [+] calculate Kernels ({:.2f} min)".format( (end-start)/60. ))
        print("")

        for llambda in lls:
            C = np.concatenate((Kte, Kt))
            Y = np.concatenate((E_train, F_train.flatten()))
            Y = Y.astype(float)
            print(" [-] calculate Alphas operator")
            start = time()
            alpha = svd_solve(C, Y, rcond=llambda)
            end = time()
            print(" [+] calculate Alphas operator ({:.2f} min)".format( (end-start)/60. ))
            print("")

            eYt = np.dot(Kte_val, alpha)
            fYt = np.dot(Kt_val, alpha)

            MAE_e = np.mean(np.abs(eYt-E_val))
            MAE_f = np.mean(np.abs(fYt-F_val.flatten())/42.)
            print("Sigma: {}, lambda: {}, MAE energy: {:.4f} eV, MAE forces: {:.4f} eV/A/atom".format(SIGMA, llambda, MAE_e, MAE_f))
            f_out = open("find_params_200.txt", 'a')
            f_out.write("Sigma: {}, lambda: {},  MAE energy: {:.4f} eV, MAE forces: {:.4f} eV/A/atom\n".format(SIGMA, llambda, MAE_e, MAE_f))
            f_out.close()

