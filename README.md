# Scripts for QMLightning for the Workshop

## usage:

### Train
```
python predict.py -ntrain <path to train npz> -ntest <path to test/validation npz> -npcas 128 -nbatch_train 64 -nbatch_test 128
```

npcas should be kept at 128 to recover the entire representation. If not enough memory is available on the GPU, try to reduce nbatch_train and nbatch_test.
The hyperparameter scan can be activated with `-hyperparam_opt 1`.


### MD

```
python run_MD.py
```

Some key things: The filename of the model which is loaded is usually hardcoded!, Make sure you have a `xyz` directory to store the coordinates of every nth step.

### TODO
Upload the QMLightning calculator for ASE
