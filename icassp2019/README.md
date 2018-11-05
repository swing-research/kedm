# KEDM

This repository provides the source codes to regenerate the results provided in "On the move: Localization with Kinetic Euclidean Distance Matrices" paper, submitted to ICASSP2019.

## Summary
We regenerate the following simulations:
- Illustration of KEDM ambiguities: kedm_ambiguity.py
- Noisy measurement experiment: sketch_experiment.py
- Missingdistancemeasurements experiment: sparsity_experiment.py

## Requirements
This python code has been proved to work with the following installed.
- cvxopt==1.2.2
- cvxpy==1.0.10
- numpy==1.15.4
- python==3.7.0

## Code

### Training multiple ProjNets
To train multiple ProjNets, use ```projnet/train_projnets.py```.
The arguments are as follows:
```console
usage: train_projnets.py [-h] [--imgs IMGS] [--val VAL] [--orig ORIG]
[--input INPUT] [--nets NETS] [--path PATH]

optional arguments:
-h, --help            show this help message and exit
--imgs IMGS, --images IMGS
Number of images to load
--val VAL, --validation VAL
Number of images for validation
--orig ORIG, --originals ORIG
Path to original images .npy
--input INPUT, --input INPUT
Path to ProjNet input images .npy
--nets NETS, --networks NETS
Number of ProjNets to train
--path PATH, --projnetspath PATH
Directory to store ProjNets. Ensure this directory
exists.

```

For example, to train 50 ProjNets with 10,000 training images, 100 validation images and save them in ```my_nets``` (ensure this directory exists), we can run the following:
``` console
cd projnet/
python3 train_projnets.py --nets=50 --imgs=10000 --val=10 --path='my_nets' --orig=../originals20k.npy --input=../custom25_infdb.npy
```

## Reconstruct from trained ProjNets
To reconstruct from trained Projnets, use ```projnet/reconstruct_from_projnets.py```. Total variation regularization is used for the reconstruction.
The arguments are as follows:
```console
usage: reconstruct_from_projnets.py [-h] [--orig ORIG] [--input INPUT]
[--nets NETS] [--lam LAM] [--nc]
[--projnets PROJNETS] [--b B] [--c C]
[--path PATH]

optional arguments:
-h, --help            show this help message and exit
--orig ORIG, --originals ORIG
Path to original images .npy
--input INPUT, --input INPUT
Path to ProjNet input images .npy
--nets NETS, --networks NETS
Number of ProjNets to use
--lam LAM, --lambda LAM
TV regularization parameter
--nc, --nocoefs       Use if already calculated coefficients
--projnets PROJNETS, --projnetspath PROJNETS
ProjNets directory.
--b B, --basisstacked B
Path to save stacked basis functions .npy
--c C, --coefsstacked C
Path to save stacked coefficients .npy
--path PATH, --reconpath PATH
Reconstruction directory. Ensure this directory
exists.
```

For example, to reconstruct from 40 networks in ```my_nets``` with a regularization parameter of 0.003 and store the reconstructions in ```reconstructions```, we can run the following:
```console
cd projnet/
python3 reconstruct_from_projnets.py --nets=40 --projnets=my_nets --lam=0.003 --path=reconstructions --b=basis_40nets.npy --c=coefs_40nets.npy
```

The above command stores the stacked basis functions and stacked coefficients in ```basis.npy``` and ```coefs.npy```.
It is possible that you may wish to try a different regularization parameter for reconstruction. 
As you have saved the stacked basis functions and coefficients, you do not need to calculate these again. You can use the ```--nc``` option:
```console
python3 reconstruct_from_projnets.py --lam=0.002 --path=reconstructions_new_lam --b=basis_40nets.npy --c=coefs_40nets.npy --nc
```
## SubNet, DirectNet

The subspace network (SubNet), takes the basis for the random projection as an input along with the measurements. One can use `subnet/subnet.py` to train the subnet. `subnet.py` allows for resuming training from a particular checkpoint and also, skipping training and moving directly to evaluation on required datasets. We have a common parser for `subnet.py`, `directnet.py` and `reconstruct_from_subnet.py` as they share many same arguments. The parser arguments are as below::

```console
usage: subnet.py [-h] [-ni NITER] [-dnpy DATA_ARRAY] [-mnpy MEASUREMENT_ARRAY]
[-ntrain TRAINING_SAMPLES] [-t TRAIN] [-r_iter RESUME_FROM]
[-n NAME] [-lr LEARNING_RATE] [-bs BATCH_SIZE]
[-i_s IMG_SIZE] [-e EVAL] [-e_orig EVAL_ORIGINALS]
[-e_meas EVAL_MEASUREMENTS] [-e_name EVAL_NAME]
[-pdir PROJECTORS_DIR] [-nproj NUM_PROJECTORS_TO_USE]
[-ntri DIM_RAND_SUBSPACE] [-r_orig RECON_ORIGINALS]
[-r_coefs RECON_COEFFICIENTS] [-m MASK]

optional arguments:
-h, --help            show this help message and exit
-ni NITER, --niter NITER
Number of iterations fot the training to run
-dnpy DATA_ARRAY, --data_array DATA_ARRAY
Numpy array of the data
-mnpy MEASUREMENT_ARRAY, --measurement_array MEASUREMENT_ARRAY
Numpy array of the measurements
-ntrain TRAINING_SAMPLES, --training_samples TRAINING_SAMPLES
Number of training samples
-t TRAIN, --train TRAIN
Training flag
-r_iter RESUME_FROM, --resume_from RESUME_FROM
resume training from r_iter checkpoint, if 0 start
training afresh
-n NAME, --name NAME  Name of directory under results/ where the results of
the current run will be stored
-lr LEARNING_RATE, --learning_rate LEARNING_RATE
learning rate to be used for training
-bs BATCH_SIZE, --batch_size BATCH_SIZE
mini-batch size
-i_s IMG_SIZE, --img_size IMG_SIZE
image size
-e EVAL, --eval EVAL  Evaluation (on other dataset) flag
-e_orig EVAL_ORIGINALS, --eval_originals EVAL_ORIGINALS
list of strings with names of original .npy arrays
-e_meas EVAL_MEASUREMENTS, --eval_measurements EVAL_MEASUREMENTS
list of strings with names of measurement .npy arrays
-e_name EVAL_NAME, --eval_name EVAL_NAME
Name to be given to the evaluation experiments
-pdir PROJECTORS_DIR, --projectors_dir PROJECTORS_DIR
directory where all projector matrices are stored
-nproj NUM_PROJECTORS_TO_USE, --num_projectors_to_use NUM_PROJECTORS_TO_USE
Number of projector matrices to use
-ntri DIM_RAND_SUBSPACE, --dim_rand_subspace DIM_RAND_SUBSPACE
Number of triangles per mesh
-r_orig RECON_ORIGINALS, --recon_originals RECON_ORIGINALS
npy array of originals to compare against
-r_coefs RECON_COEFFICIENTS, --recon_coefficients RECON_COEFFICIENTS
npy array of coefficients to use
-m MASK, --mask MASK  Mask to apply for convex hull of sensors

```

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

