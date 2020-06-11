# Automatic Reparameterisation of Probabilistic Programs

This repository contains code associated with the paper:

M. I. Gorinova, D. Moore, and M. D. Hoffman. [*Automatic Reparameterisation of Probabilistic Programs*](https://arxiv.org/abs/1906.03028). 2019.


## Usage

The script `main.py` is the main entry point. For example, to evaluate the German credit model with four leapfrog steps per sample, you might run:

```shell
# Run variational inference to get step sizes and initialization.
python main.py --model=german_credit_lognormalcentered --inference=VI --method=CP --num_optimization_steps=3000 --results_dir=./results/
# Run HMC to sample from the posterior
python main.py --model=german_credit_lognormalcentered --inference=HMC --method=CP --num_leapfrog_steps=4 --num_samples=50000 --num_burnin_steps=10000 --num_adaptation_steps=6000 --results_dir=./results/
```

Available options are:

- `method`: `CP`, `NCP`, `cVIP`, `dVIP`, `i`. Note that `i` is only available when `inference` is set to `HMC`. 
- `inference`: `VI` or `HMC`. `VI` needs to be run first for every model in order for a log file to be created, which contains information such as initial step size to be adapted when running HMC.
- `model`: `radon_stddvs`, `radon`, `german_credit_lognormalcentered`,
  `german_credit_gammascale`, `8schools`, `electric`, `election` and `time_series`
- `dataset` (used only for radon models): `MA`, `IN`, `PA`, `MO`, `ND`, `MA`, or `AZ`

 To generate human-readable analysis, run

```shell
python analyze.py --elbos --all
python analyze.py --ess --all
python analyze.py --reparams --all
python analyze.py --elbos --model=8schools
```

The number of leapfrog steps will be automatically tuned if (1) no `num_leapfrog_steps` argument is supplied and (2) no entry `num_leapfrog_steps` exists in the respective `.json` file. 

When the number of leapfrog steps is tuned, the best number of leapfrog steps is recorded in a `.json` file, so that it can be reused accordingly.

This code has been tested with TensorFlow 1.14 and TensorFlow Probability 0.7.0.
