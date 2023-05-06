# A Detailed Data-Driven Network Model of Prefrontal cortex Reproduces Key Features of In Vivo Activity

> Hass J, Hert&auml;g L, Durstewitz D (2016) A Detailed Data-Driven Network Model of Prefrontal cortex Reproduces Key Features of <em>In Vivo</em> Activity.
PLoS Comput Biol 12(5): e1004930. doi:10.1371/journal.pcbi.1004930

This is a reimplementation of the detailed prefrontal cortex model presented in the above paper. The original work was implemented in Matlab and C++. Here we developed a Python package named PFC_model, which  makes use of Python (3.11.2) and Brian 2 (2.5.1) for simulation and data analysis.

For details on how to install Brian 2, follow steps in [The Brian Simulator](https://briansimulator.org/), and for other python packages see [Installing Packages](https://packaging.python.org/en/latest/tutorials/installing-packages/). If one needs to install a different version of Python (<em>i.e.</em> system Python version differs to that described below), one could use conda as described in [Installation -- conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

***

## System requirements

### Hardware requirements

- CPU/cores
- memory
- time to execute all codes or time to execute the most computational demanding code
- parallelization?

### Software requirements

Before installing PFC_model from GitHub, one needs:

1. Anaconda (^2022.10)
2. PIP (^22.3.1)

The necessary Python version and dependencies are specified in PFC_model installation.

1. Python (^3.11.2)
2. Brian 2 (^2.5.1)
3. Matploblib (^3.7.1)
4. SciPy (^1.10.1)
5. Numpy (^1.24.2)
6. GSL (^2.7)
7. Pandas (^1.5.3)
8. Xarray (^2023.3.0)
9. Seaborn (^0.12.2)
10. TOML (^0.10.2)
11. HMMLearn (^0.3.0)
12. Notebook (^6.5.4)

## Installing PIP and PFC_model and GSL.

Follow the steps below to create a conda environment (myenv, change for the desired name) with PIP. GSL also needs to be installed in order to run some tests in replicating Hass et al's work.

```bash
$ conda create -n myenv python>=3.11.2 pip>=22.3.1
$ conda activate myenv
$ pip install git+https://github.com/marcelorempel/pfc_model
$ conda install -c gsl>=2.7
```

***

## Package structure

PCF_model package is organized as follows.

```bash
pfc_model
│   LICENSE.txt
│   pyproject.toml
│   README.md
│
├───dist
│       pfc_model-0.1.0-py3-none-any.whl
│       pfc_model-0.1.0.tar.gz
│
├───example
│       example1.py
│
├───results
│       spikingcells_fraction.txt
│       spikingcells_membparams.txt
│       spikingcells_pcon.txt
│       spikingcells_synparams.txt
│
├───src
│   └───pfc_model
│       │   cortex_setup.py
│       │   _auxiliary.py
│       │   _basics_setup.py
│       │   _network_auxiliary.py
│       │   _network_setup.py
│       │   __init__.py
│       │
│       ├───analysis
│       │       synchrony.py
│       │       tools.py
│       │       up_down.py
│       │       __init__.py
│       │    
│       └───replication
│               description_membparams.txt
│               Original_I.npy
│               Original_spiketime.txt
│               Original_t.npy
│               task1.py
│               task2.py
│               task3.py
│               task4.py
│               task5.py
│               task6.py
│               task7.py
│               task8.py
│               taskset.py
│               __init__.py
│
└─── tutorial
        Tutorial-1.ipynb
        Tutorial-2.ipynb
        Tutorial-3.ipynb
```

Scripts are placed inside src folder.

All the n໎ecessary items to perform a simulation must be imported from pfc_model

```bash
from pfc_model import *
```

One can check an example of a simple simulation in pfc_model/example/example1. Other explanations can be found on the notebook tutorials in pfc_model/tutorial.

The scripts which implement analyses on network results are located in the subpackage analysis. One can import it as follows.

```bash
from pfc_model.analysis import *
```

The scripts that replicate Hass et al's experiments and generate the results in our work can be found in the subpackage replication, which can be imported as follows.

```bash
from pfc_model.replication import *
```

## Generating figures and data
















