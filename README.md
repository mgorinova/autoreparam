# Automatic Reparameterisation in Probabilistic Programming



## Usage

The script `main.py` is the main entry point. For example, to evaluate the German credit model with four leapfrog steps per sample, you might run:

```shell
python main.py --model=german_credit_lognormalcentered --inference=HMC --method=CP --num_leapfrog_steps=4 --num_samples=50000 --burnin=10000 --num_adaptation_steps=6000 
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
python analyze.py --reparams --all
python analyze.py --elbos --model=8schools
```
