import torch
import numpy as np

import argparse

from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument("-ntrain", type=str, default="1000")
    parser.add_argument("-ntest", type=str, default="1000")
    parser.add_argument("-nbatch_train", type=int, default=64)
    parser.add_argument("-nbatch_test", type=int, default=256)

    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=2.0)
    parser.add_argument("-llambda", type=float, default=1e-5)
    parser.add_argument("-npcas", type=int, default=128)
    parser.add_argument("-ntransforms", type=int, default=1)
    parser.add_argument("-nstacks", type=int, default=128)

    parser.add_argument('-rcut', type=float, default=6.0)
    parser.add_argument("-nRs2", type=int, default=24)
    parser.add_argument("-nRs3", type=int, default=22)
    parser.add_argument("-eta2", type=float, default=0.28)
    parser.add_argument("-eta3", type=float, default=3.2)
    parser.add_argument("-two_body_decay", type=float, default=2.3)
    parser.add_argument("-three_body_decay", type=float, default=0.65)
    parser.add_argument("-three_body_weight", type=float, default=18.8)

    parser.add_argument("-hyperparam_opt", type=int, choices=[0, 1], default=0)
    parser.add_argument("-forces", type=int, choices=[0, 1], default=1)

    parser.add_argument("-train_ids", type=str, default="splits/aspirin_train_ids.npy")
    parser.add_argument("-test_ids", type=str, default="splits/aspirin_test_ids.npy")
    parser.add_argument("-path", type=str, default="../data/aspirin_dft.npz")

    args = parser.parse_args()

    print ("---Argument Summary---")
    print (args)

    ntrain = args.ntrain
    ntest = args.ntest

    nbatch_train = args.nbatch_train
    nbatch_test = args.nbatch_test

    nstacks = args.nstacks
    ntransforms = args.ntransforms
    npcas = args.npcas

    rcut = args.rcut
    nRs2 = args.nRs2
    nRs3 = args.nRs3
    eta2 = args.eta2
    eta3 = args.eta3
    two_body_decay = args.two_body_decay
    three_body_decay = args.three_body_decay
    three_body_weight = args.three_body_weight

    sigma = args.sigma
    llambda = args.llambda

    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#    data = np.load(args.path, allow_pickle=True)
    data_train = np.load(ntrain, allow_pickle=True)
    data_val   = np.load(ntest, allow_pickle=True)

    if ('R' in data_train.keys()):
        coords_train = data_train['R']
        nuclear_charges_train = data_train['z']
        energies_train = data_train['E'].flatten()
        forces_train = data_train['F']
    else:
        coords = data['coords']
        nuclear_charges = data['nuclear_charges']
        energies = data['energies'].flatten()
        forces = data['forces']

    if ('R' in data_val.keys()):
        coords_val = data_val['R']
        nuclear_charges_val = data_val['z']
        energies_val = data_val['E'].flatten()
        forces_val = data_val['F']
    else:
        coords = data['coords']
        nuclear_charges = data['nuclear_charges']
        energies = data['energies'].flatten()
        forces = data['forces']


    nuclear_charges_train = np.repeat(nuclear_charges_train[np.newaxis,:], coords_train.shape[0], axis=0)
    nuclear_charges_val = np.repeat(nuclear_charges_val[np.newaxis,:], coords_val.shape[0], axis=0)

    #train_IDs = np.fromfile(args.train_ids, dtype=int)
    #test_indexes = np.fromfile(args.test_ids, dtype=int)[:args.ntest]

    unique_z = np.unique(np.concatenate(nuclear_charges_train)).astype(int)
#
#    ALL_IDX = np.arange(len(coords))
#
#    train_indexes = train_IDs[:ntrain]

    train_coordinates = coords_train[:3000]
    train_charges     = nuclear_charges_train[:3000]
    train_energies    = energies_train[:3000]
    train_forces      = forces_train[:3000]

    test_coordinates = coords_val
    test_charges     = nuclear_charges_val
    test_energies    = energies_val
    test_forces      = forces_val


    rep = FCHLCuda(species=unique_z, rcut=rcut, nRs2=nRs2, nRs3=nRs3, eta2=eta2, eta3=eta3,
                   two_body_decay=two_body_decay, three_body_decay=three_body_decay, three_body_weight=three_body_weight)

    model = HadamardFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nstacks=nstacks, ntransforms=ntransforms, npcas=npcas,
                                nbatch_train=nbatch_train, nbatch_test=nbatch_test)

    print ("Note: results are in kcal/mol (/A for forces)")

    print ("Calculating projection matrices...")
    model.get_reductors(coords_train, nuclear_charges_train, npcas=npcas)

    #model.set_subtract_self_energies(True)
    #model.calculate_self_energy(train_charges, train_energies)
    #model.calculate_self_energy(test_charges, test_energies)

    if (args.hyperparam_opt):
        model.hyperparam_opt_nested_cv(train_coordinates, train_charges, train_energies, sigmas=np.array([4.8, 9.6, 19.2]), lambdas=np.array([1e-5, 1e-7, 1e-9]), F=train_forces if args.forces else None)

    model.train(train_coordinates, train_charges, train_energies, train_forces if args.forces else None)

    data = model.format_data(test_coordinates, test_charges, test_energies, test_forces if args.forces else None)

    test_energies = data_val['E']#.flatten()
    max_natoms = 100 #data['natom_counts'].max().item()

    MAE_e = np.array([])
    MAE_f = np.array([])
    RMSE_e = np.array([])
    RMSE_f = np.array([])
    for i in range(test_energies.shape[0]):
    #    for i in range(test_energies.shape[0]):
        if (args.forces):
            #test_forces = data['forces']
            energy_predictions, force_predictions = model.predict(np.asarray([test_coordinates[i]]), np.asarray([test_charges[i]]), max_natoms, forces=True)
        else:
            energy_predictions = model.predict(np.asarray([test_coordinates[i]]), np.asarray([test_charges[i]]), max_natoms, forces=False)

        energy_predictions = energy_predictions.cpu().detach().numpy()
        force_predictions  = force_predictions.cpu().detach().numpy()
    #        print(energy_predictions)
    #        print(force_predictions)
    #    print("Energy MAE /w backwards:", torch.mean(torch.abs(energy_predictions - test_energies)))
        MAE_e = np.append(MAE_e, np.mean(np.abs( energy_predictions-np.asarray([test_energies[i]]) )))
        MAE_f = np.append(MAE_f, np.mean(np.abs( force_predictions-np.asarray([test_forces[i]]) )))

        #RMSE_e = np.append(RMSE_e, np.sqrt( (energy_predictions-np.asarray([test_energies[i]]))**2 ))
        #RMSE_f = np.append(RMSE_f, np.sqrt( (force_predictions-np.asarray([test_forces[i]]))**2 ))
        RMSE_e = np.append(RMSE_e, np.linalg.norm( (energy_predictions-np.asarray([test_energies[i]])) ) / np.sqrt(1))
        RMSE_f = np.append(RMSE_f, np.linalg.norm( (force_predictions-np.asarray([test_forces[i]])) ) / np.sqrt(126))
        #print("Energy MAE /w backwards:", np.mean(np.abs(energy_predictions - np.asarray([test_energies[i]]))))

        #if (args.forces):
        #    #print("Force MAE /w backwards:", torch.mean(torch.abs(force_predictions - test_forces)))
        #    print("Force MAE /w backwards:", np.mean(np.abs(force_predictions - np.asarray([test_forces[i]]))))

    print("MAE_e: ", np.mean(MAE_e))
    print("MAE_f: ", np.mean(MAE_f))
    print("RMSE_e: ", np.mean(RMSE_e))
    print("RMSE_f: ", np.mean(RMSE_f))

    model.save_jit_model()
