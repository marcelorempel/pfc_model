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

***

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

Scripts are placed inside src folder. All the n໎ecessary items to perform a simulation must be imported directly from pfc_model

```bash
from pfc_model import *
```

One can check an example of a simple simulation in "pfc_model/example/example1.py". If one download example1.py to the current working directory, one can run it as follows.

```bash
$ python -m example1.py
```

Other explanations can be found on the notebook tutorials in pfc_model/tutorial. After downloading them to the current workinf directory, one can run one of them, e.g. Tutorial-1.ipynb, as follows (notebook is listed in pfc_model dependencies).

```bash
$  jupyter notebook Tutorial-1.ipynb 
```

The scripts which implement analyses on network results are located in the subpackage analysis. One can import it as follows.

```bash
from pfc_model.analysis import *
```

The scripts that replicate Hass et al's experiments and generate the results in our work can be found in the subpackage replication, which can be imported as follows.

```bash
from pfc_model.replication import *
```

Using wildcard import of analysis or replication automatically includes wildcard import of pfc_model.

***

## Replicating the results

### Setting directory to save results

After importing pfc_model, one can set a new directory to save simulation and analyses results using the function set_simulation_dir.

```bash
from pfc_model import *
simulation_dir = set_simulation_dir(name=name_of_dir)
```

This code snippet creates a new directory called name_of_dir in the current working directory. If name_of_dir already exists, "_X" is appended to the end of the name, where X is the lowest positive integer so that the resulting name corresponds to a nonexistent directory (e.g. if name_of_dir already exists, name_of_dir_1 is created; if name_of_dir and name_of_dir_1 already exist, name_of_dir_2 is created, and so on). If no name is provided, name_of_dir is set to "Simulation_X", where X is the lowest positive integer so that the resulting name corresponds to a nonexistent directory (e.g. the first call of the function creates "Simulation_1", the second one creates "Simulation_2" and so on).

### Generating figures and data

We used the scripts in replication subpackage to generate figures and other results in our work. Note that the random seed are already set in these scripts so as to enable reproducibility. One can change the seeds in order do test the model with other random configurations.

The experiments and analyses were coded as functions in the scripts of replication subpackage. One can import replication and then call one or all of these functions according to their documentation.

In order perform task1, one can run:

```bash
from pfc_model.replication import *
import numpy as np

simulation_dir = set_simulation_dir('Results_task1')
seed = 0

task1(simulation_dir, seed)
```
Here, the results will be saved in a new directory called 'Results_task1' (or 'Results_task1_X', where X is the lowest positive integer so that the resulting name corresponds to a nonexistent directory, if 'Results_task1' already exists).

In order to perform all taks, one can run:
```bash
from pfc_model.replication import *
import numpy as np

simulation_dir = set_simulation_dir(Results_taskset)
seed = 0
Iexc_arr = np.arange(0, 600, 100)
Iinh_arr = np.arange(0, 600, 100)
duration=13000
Ntrials=100

task1(simulation_dir, seed)
task2(simulation_dir)
task3(simulation_dir=simulation_dir, Ntrials=Ntrials)
task4(simulation_dir=simulation_dir, seed=seed)
task5(simulation_dir=simulation_dir, seed=seed)
task6(simulation_dir=simulation_dir)
task7(simulation_dir=simulation_dir, seed=seed)
task8(simulation_dir=simulation_dir, Iexc_arr=Iexc_arr, 
    Iinh_arr=Iinh_arr, seed=seed, duration=duration)
```
Here, the results will be saved in a new directory called 'Results_taskset' (or 'Results_taskset_X', where X is the lowest positive integer so that the resulting name corresponds to a nonexistent directory, if 'Results_taskset' already exists).

The function taskset performs all the tasks. It can be set as:

```bash
from pfc_model.replication import *
import numpy as np

simulation_dir = set_simulation_dir('Results_taskset')
seed = 0
Iexc_arr = np.arange(0, 600, 100)
Iinh_arr = np.arange(0, 600, 100)
duration=13000
Ntrials=100

taskset(simulation_dir=simulation_dir, seed=seed, Ntrials=Ntrials,
          Iexc_arr=Iexc_arr, Iinh_arr=Iinh_arr, duration=duration)
```
Here, the results will be saved in a new directory called 'Results_task1' (or 'Results_task1_X', where X is the lowest positive integer so that the resulting name corresponds to a nonexistent directory, if 'Results_task1' already exists).


Alternatively, one can performing the tasks running the corresponding script, as follows. The function arguments are set in the scripts as presented in the above code snippets. 

- For taskY (where Y is 1, 2, 3, 4, 5, 6, 7 or 8):

```bash
$ python -m pfc_model.replication.taskY
```
Here, the directory name for saving results is set to 'Results_taskY'.

- For all tasks:

```bash
$ python -m pfc_model.replication.taskset
```

If one calls separetely the task scripts, one different directory, "Results_taskY" (or else "Results_taskY_X", if the same script is run more than once) will be created for each one, and the results will be saved in different directories.

If one manually sets simulation_dir once and calls all the task functions, or else if one runs pfc_model.replication.taskset, all results will be saved in the same directory.

Notice that running task3 and task8 (as well as taskset) scripts will take a larger amount of time as they perform multiple simulations (task3 is set to perform 100 simulations, and task8 is set to perform 24 x 24 = 576 simulations). Alternatively, one can manually call the functions task3 or task8 and pass other arguments, according to their documentation, to perform a faster study.

### Location of figures

We will generically call the directory where results are saved as "Results_dir" (the directory name generated by the scripts are explained in the section above).

#### Location of figures and data

We will generically call the directory where results are saved as "Results_dir" (the directory name generated by the scripts are explained above).

- Figure 1

It is generated by task1 (and taskset) and saved as "Results_dir/Figures/Fig01.png".

- Figure 2

It is generated by task2 (and taskset) and saved as "Results_dir/Figures/Fig02.png".

- Figure 3

It is generated by task2 (and taskset) and saved as "Results_dir/Figures/Fig03.png".

- Figure 4

It is generated by task4 (and taskset) and saved as "Results_dir/Figures/Fig04.png".

- Figure 5

It is generated by task4 (and taskset) and saved as "Results_dir/Figures/Fig05.png".

- Figure 6

It is generated by task4 (and taskset) and saved as "Results_dir/Figures/Fig06.png".

- Figure 7

It is generated by task4 (and taskset) and saved as "Results_dir/Figures/Fig07.png".

- Figure 8

It is generated by task4 (and taskset) and saved as "Results_dir/Figures/Fig08.png".

- Figure 9

It is generated by task5 (and taskset) and saved as "Results_dir/Figures/Fig09.png".

- Figure 10

It is generated by task6 (and taskset) and saved as "Results_dir/Figures/Fig10.png".

- Figure 11

It is generated by task7 (and taskset) and saved as "Results_dir/Figures/Fig11.png".

- Figure 12

It is generated by task8 (and taskset) and saved as "Results_dir/Figures/Fig12.png".


#### Location of data

We will generically call the directory where results are saved as "Results_dir" (the directory name generated by the scripts are explained in the section above).

- Mean and standard deviation of the generated membrane parameters

It is generated by task1 and saved in "Results_dir/Reports/Param_description/description_membparams.txt".

The result generated using seed 0 is provided in PFC_model package in "pfc_model/results/description_membparams.txt".

- Fraction of spiking cells

We calculated it using 100 independent baseline simulations with seeds from 0 to 99. It is generated by task3 and saved in "Results_dir/Reports/Spikingcells/spikingcells_fraction.txt". 

The results from individuals simulations are saved in "Results_dir/Reports/Spikingcells/individuals/spikingPC_seed_Y.txt", "Results_dir/Reports/Spikingcells/individuals/spikingIN_seed_Y.txt" and "Results_dir/Reports/Spikingcells/individuals/spikingALL_seed_Y.txt" where Y stands for the seed (from 0 to 99).

The result we obtained is provided in PFC_model package in "pfc_model/results/spikingcells_fraction.txt".

- Comparison of membrane parameters between spiking and not-spiking cells

We calculated it using 100 independent baseline simulations with seeds from 0 to 99. It is generated by task3 and saved in "Results_dir/Reports/Param_comparisons/spikingcells_membparams.txt". 

The results from individuals simulations are saved in "Results_dir/Reports/Param_comparisons/individuals/membr_params_comparison_seed_Y.txt" where Y stands for the seed (from 0 to 99).

The result we obtained is provided in PFC_model package in "pfc_model/results/spikingcells_membparams.txt".

- Comparison of synaptic parameters between synapses directed to spiking and not-spiking cells

We calculated it using 100 independent baseline simulations with seeds from 0 to 99. It is generated by task3 and saved in "Results_dir/Reports/Param_comparisons/spikingcells_synparams.txt". 

The results from individuals simulations are saved in "Results_dir/Reports/Param_comparisons/individuals/syn_params_comparison_AMPA_seed_Y.txt", "Results_dir/Reports/Param_comparisons/individuals/syn_params_comparison_GABA_seed_Y.txt" and "Results_dir/Reports/Param_comparisons/individuals/syn_params_comparison_NMDA_seed_Y.txt" where Y stands for the seed (from 0 to 99).

The result we obtained is provided in PFC_model package in "pfc_model/results/spikingcells_synparams.txt".


- Comparison of probability of connection directed to spiking and not-spiking cells

We calculated it using 100 independent baseline simulations with seeds from 0 to 99. It is generated by task3 and saved in "Results_dir/Reports/Param_comparisons/spikingcells_pcon.txt". 

The results from individuals simulations are saved in "Results_dir/Reports/Param_comparisons/individuals/pcon_contingency_AMPA_seed_Y.txt", "Results_dir/Reports/Param_comparisons/individuals/pcon_contingency_GABA_seed_Y.txt" and "Results_dir/Reports/Param_comparisons/individuals/pcon_contingency_NMDA_seed_Y.txt" where Y stands for the seed (from 0 to 99).

The result we obtained is provided in PFC_model package in "pfc_model/results/spikingcells_pcon.txt".


- ISI analysis

We calculated it using baseline simulations with algorithms rk4 and gsl_rk2 (with seed 0) and with results from Hass et al's original code (provided in "pfc_model/src/pfc_model/replication/Original_t.npy", "pfc_model/src/pfc_model/replication/Original_spiketime.txt" and "pfc_model/src/pfc_model/replication/Original_I.npy"). 

It is generated by task4 and saved in "Results_dir/Reports/ISI_analysis/ISIstats_rk4.txt", "Results_dir/Reports/ISI_analysis/Correlations_rk4.txt",
"Results_dir/Reports/ISI_analysis/ISIstats_rk2.txt", "Results_dir/Reports/ISI_analysis/Correlations_rk2.txt", "Results_dir/Reports/ISI_analysis/ISIstats_original.txt" and "Results_dir/Reports/ISI_analysis/Correlations_original.txt".

- V analysis

We calculated it using a baseline simulation with seed 0. It is generated by task4 and saved in "Results_dir/Reports/Vstats/VALLstats_rk4.txt" and "Results_dir/Reports/Vstats/VPCstats_rk4.txt".

- Spiking counts of stimulation protocols

We performed stimulation with regular pulses using seed 0 in task5. Spiking counts are saved in "Results_dir/Reports/Stimulation/RegPulses.txt" and "Results_dir/Reports/Stimulation/RegPulses_RedSTD.txt".

We analyzed the impact of reducing variability of membrane parameters in different networks (generated by seeds 0, 1, 2, 3 and 4). The results are saved in "Results_dir/Reports/Stimulation/Param_std_variation.txt".

***

## Reference:

1. J. Hass, L. Hertäg, and D. Durstewitz. “A Detailed Data-Driven Network Model of Prefrontal Cortex Reproduces
Key Features of In Vivo Activity.” In: PLOS Computational Biology 12.5 (May 2016), pp. 1–29. DOI: 10.1371/jour-
nal.pcbi.1004930. URL: https://doi.org/10.1371/journal.pcbi.1004930.

2. C. S. Quiroga-Lombard, J. Hass, and D. Durstewitz. “Method for stationarity-segmentation of spike train
data with application to the Pearson cross-correlation.” In: Journal of Neurophysiology 110.2 (2013). PMID:
23636729, pp. 562–572. DOI: 10.1152/jn.00186.2013. 

3. P. Dayan and L. F. Abbott. Theoretical Neuroscience: Computational and Mathematical Modeling of Neural
Systems. Cambridge, MA: The MIT Press, 2005.

----------------------------------------------------------------------------------------------

For questions of comments, please contact:
Marcelo Rafael Silva Rempel (marcelorempel@usp.br / marcelo.rafael@hotmail.com / marcelorafaelsilvarempel@gmail.com)

Department of Physics
Faculty of Philosophy, Sciences and Letters at Ribeirão Preto
University of Sao Paulo



















