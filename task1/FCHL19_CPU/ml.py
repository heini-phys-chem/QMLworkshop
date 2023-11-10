import numpy as np
#sys.path.insert(0, "/home/heinen/qml_develop/build/lib.linux-x86_64-2.7")
import qml
from qml.math import cho_solve
#from qml.representations import generate_fchl_acsf, generate_bfcl_acsf, generate_vpbfcl_acsf
from qml.representations import generate_fchl_acsf
from qml.kernels import get_atomic_local_gradient_kernel
from qml.kernels import get_local_gradient_kernel
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_local_kernel

from MBDF_gradients import generate_mbdf_train, generate_mbdf_pred

from ase.calculators.general import Calculator
from ase.atoms import Atoms

from xtb.ase.calculator import XTB

convback   = -1/23.016# forces to gradients (kcal/mol/A eV/A)
convback_E = 1/23.016 # convert energies kcal/mol to eV

class ML_calculator(Calculator):
  name = 'ML_Calculator'
  implemented_properties = ['energy', 'forces']


  def __init__(self, atoms, sigma, alphas, X, Q, maxNumAtom, element_list, foutput, norms):
    self.alphas = alphas
    self.X     = X
    self.sigmas = sigma
    self.Q     = Q
    self.nAtoms = atoms.get_number_of_atoms()
    self.element_list = element_list
    self.maxNumAtom = maxNumAtom
    self.foutput = foutput
    self.norms = norms
    #self.nAtoms = atoms.get_global_number_of_atoms()

  def append_mol(self, atoms):
    fout = self.foutput
    f = open(fout, 'a')

    NAMES = {1:'H', 7:'N', 6:'C', 8:'O'}
    geoms = atoms.get_positions()
    labels = atoms.get_atomic_numbers()

    f.write("{}\n\n".format(self.nAtoms))
    for i in range(len(geoms)):
        f.write("{} {} {} {}\n".format(NAMES[labels[i]], geoms[i][0], geoms[i][1], geoms[i][2]))

    f.close()



  def get_potential_energy(self,atoms=None,force_consistent=False):
#    self.append_mol(atoms)
#    q = []
#    coords = []
    x = []
    disp_x = []
    q = []

    x1 = generate_fchl_acsf(atoms.get_atomic_numbers(), atoms.get_positions(), gradients=False, pad=self.maxNumAtom, elements=self.element_list)
    x.append(x1)
    q.append(atoms.get_atomic_numbers())

#    q.append(atoms.get_atomic_numbers())
#    coords.append(atoms.get_positions())
#    Qs = np.array(q)
#    coordinates = np.array(coords)
#    Xs, dXs = generate_mbdf_pred(Qs, coordinates, self.norms, n_jobs=1)

    Kse = get_atomic_local_kernel(self.X, Xs, self.Q, Qs, self.sigmas)
    energy = (float(np.dot(Kse, self.alphas)))*convback_E

    atoms.calc = XTB(method="GFN2-xTB")
    e_xtb = atoms.get_potential_energy()
    energy += e_xtb

    return energy

  def get_forces(self, atoms=None):
    x = []
    disp_x = []
#    q = []
#    coords = []

    (x1, dx1) = generate_fchl_acsf(atoms.get_atomic_numbers(), atoms.get_positions(), gradients=True, pad=self.maxNumAtom, elements=self.element_list)
    x.append(x1)
    disp_x.append(dx1)
    q.append(atoms.get_atomic_numbers())
#    coords.append(atoms.get_positions())
#
    Xs = np.array(x)
    dXs = np.array(disp_x)
#    Qs = np.array(q)
#    coordinates = np.array(coords)
#    Xs, dXs = generate_mbdf_pred(Qs, coordinates, self.norms, n_jobs=1)
    q = [atoms.get_atomic_numbers()]
    Qs = q

    Ks = get_atomic_local_gradient_kernel(self.X, Xs, dXs, self.Q, Qs, self.sigmas)
    self.fYs = np.dot(Ks, self.alphas)
    Fss = self.fYs.reshape((self.nAtoms,3))*convback

    atoms.calc = XTB(method="GFN2-xTB")
    f_xtb = atoms.get_forces()
    FSS += f_xtb

    return Fss
