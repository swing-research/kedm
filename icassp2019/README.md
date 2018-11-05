# KEDM

This repository provides the source codes to regenerate the results provided in "On the move: Localization with Kinetic Euclidean Distance Matrices" paper, submitted to ICASSP2019.

## Summary
We regenerate the following simulations:
- Illustration of KEDM ambiguities: ```kedm_ambiguity.py```
- Noisy measurement experiment: ```sketch_experiment.py```
- Missing distance measurements experiment:```sparsity_experiment.py``` 

## Requirements
This python code has been proved to work with the following installed.
- cvxopt==1.2.2
- cvxpy==1.0.10
- numpy==1.15.4
- python==3.7.0

## Code

### Parameters class
The fundamental step in conducting all the expertiments is setting up the necessary parameters which is as follows:

```console
class parameters:
    def __init__(self):
        self.N = 6
        self.d = 2
        self.P = 3
        self.omega = 2*np.pi
        self.mode = 1
        self.T_trn = np.array([-1, 1])
        self.T_tst = np.array([-1, 1])
        self.N_trn = ktools.K_base(self)
        self.K = ktools.K_base(self)
        self.N_tst= 500
        self.Nr = 5
        self.n_del = 0
        self.sampling = 1
        self.delta = 0.99
        self.std = 1
        self.maxIter = 5
        self.n_del_init = 0
        self.bipartite = False
        self.N0 = 3
        self.Pr = 0.9
        self.path = '../../../results/kedm/python3/'
```
Let us now briefly explain each parameter (for more information, please read [KEDM-ICASSP19](https://github.com/swing-research/kedm-pubs/tree/master/icassp)).
- `self.N`: 

## SubNet, DirectNet

The subspace network (SubNet), takes the basis for the random projection as an input along with the measurements. One can use `subnet/subnet.py` to train the subnet. `subnet.py` allows for resuming training from a particular checkpoint and also, skipping training and moving directly to evaluation on required datasets. We have a common parser for `subnet.py`, `directnet.py` and `reconstruct_from_subnet.py` as they share many same arguments. The parser arguments are as below::

One must provide the `-dnpy` and `-mnpy` arguments which correspond to the data and the measurements numpy arrays. Along with that, a directory which has all the basis vectors must be provided via `-pdir` argument. Note that running the training does not require you to be in the subnet folder. 

```console
python3 subnet/subnet.py -niter 20000 -dnpy 'originals20k.npy' -mnpy 'custom25_10db.npy' -n test_subnet -e_orig [geo_originals.npy','geo_originals.npy'] -e_meas ['geo_pos_recon_10db.npy','geo_pos_recon_infdb.npy'] -e_name ['geo_tr0_t10','geo_tr0_tinf'] -pdir 'meshes/' -nproj 350 -ntri 50

```

To reconstruct from SubNet, one needs to run `reconstruct_from_subnet.py`. This file takes the coefficients calculated and the projections and runs an iterative projected least squares. Note that since we only train one network for all subspaces we need not use any extra regularization (like TV) to reconstruct the model. An example usage is given below:

```console
python3 subnet/subnet.py -lr 0.0005 -r_orig 'geo_originals.npy' -r_coef 'geo_tr10_tinf' -m 'mask.npy' -nproj 350 -ntri 50
```

## Direct inversion

The direct net uses the same parser as subnet. An example usage is given below for reference:

```console
python3 subnet/directnet.py -niter 20000 -dnpy 'originals20k.npy' -mnpy 'custom25_0db.npy' -n test_subnet -e_orig [geo_originals.npy','geo_originals.npy'] -e_meas ['geo_pos_recon_10db.npy','geo_pos_recon_infdb.npy'] -e_name ['geo_tr0_t10','geo_tr0_tinf']

```

