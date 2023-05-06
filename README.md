# A Detailed Data-Driven Network Model of Prefrontal cortex Reproduces Key Features of In Vivo Activity

> Hass J, Hert&auml;g L, Durstewitz D (2016) A Detailed Data-Driven Network Model of Prefrontal cortex Reproduces Key Features of <em>In Vivo</em> Activity.
PLoS Comput Biol 12(5): e1004930. doi:10.1371/journal.pcbi.1004930

This is a replication of the above paper. The original work was implemented in Matlab and C++. Here we used Python (3.6.8) with Brian 2 (2.3) for the simulation and data analysis.
For details on how to install Brian 2, follow steps in [The Brian Simulator](https://briansimulator.org/), and for other python packages see [Installing Packages](https://packaging.python.org/en/latest/tutorials/installing-packages/). If one needs to install a different version of Python (<em>i.e.</em> system Python version differs to that described below), one could use conda as described in [Installation -- conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
Below is a list of the packages and versions used in this work.

***

## System requirements

### Hardware requirements

- CPU/cores
- memory
- time to execute all codes or time to execute the most computational demanding code
- parallelization?

### Software requirements

0. Anaconda (2022.10)
0. PIP 22.3.1
1. Python (3.11.2)
2. Brian 2 (2.5.1)
3. Matploblib (3.7.1)
4. SciPy (1.10.1)
5. Numpy (1.24.2)
6. GSL (2.7)
7. Pandas (1.5.3)
8. Xarray (2023.3.0)
9. Seaborn (0.12.2)
10. TOML (0.10.2)
11. HMMLearn (0.3.0)

## Installing Python and required packages using Anaconda

Follow the steps below to create a conda environment (myenv, change if myenv already exists) with the necessary packages.

### Minimum requirements

```bash
$ conda create -n myenv python>=3.11.2 pip>=22.3.1
$ conda activate myenv
$ pip install git+https://github.com/marcelorempel/pfc_model
$ conda install -c gsl>=2.7
```

***

## Package structure

PCF_model package is organized as follows.










