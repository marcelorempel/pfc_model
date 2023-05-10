"""This script performs all the tasks from task1 through task8 in sequence.
"""


import numpy as np
from pfc_model.replication import *
from .._auxiliary import *

__all__ = ['taskset']

@time_report()
def taskset(simulation_dir, seed, Ntrials_task3, Ntrials_task6,
            Iexc_arr, Iinh_arr, duration):
    """Perform all the tasks from task1 through task8 in sequence.
    
    Parameters
    ----------
    simulation_dir: str
        Path to directory where results are to be saved.
    seed: int, optional
        Random seed. If not given, no seed is set.
    Ntrials_task3: int
        Number of simulations to be performed and included in the analyses
        in task3.
    Ntrials_task6: int
        Number of simulations to be performed and included in the analyses
        in task6.
    Iexc_arr: array_like
        Array of values of background current to PCs in L2/3.
    Iinh_arr: array_like
        Array of values of background current to PCs in L5.
    duration: int or float
        Duration of simulation in ms. If not given, it defaults to 13000.
    """
    task1(simulation_dir, seed)
    task2(simulation_dir)
    task3(simulation_dir=simulation_dir, Ntrials=Ntrials_task3)
    task4(simulation_dir=simulation_dir, seed=seed)
    task5(simulation_dir=simulation_dir, seed=seed)
    task6(simulation_dir=simulation_dir, Ntrials=Ntrials_task6)
    task7(simulation_dir=simulation_dir, seed=seed)
    task8(simulation_dir=simulation_dir, Iexc_arr=Iexc_arr, 
        Iinh_arr=Iinh_arr, seed=seed, duration=duration)

if __name__ == '__main__':
    simulation_dir = set_simulation_dir('Results_'+os.path.basename(__file__)[:-3])
    seed = 0

    Iexc_arr = np.arange(0, 600, 25)
    Iinh_arr = np.arange(0, 600, 25)
    duration=13000
    Ntrials_task3=100
    Ntrials_task6=5

    taskset(simulation_dir=simulation_dir, seed=seed, 
            Ntrials_task3=Ntrials_task3, Ntrials_task6=Ntrials_task6,
            Iexc_arr=Iexc_arr, Iinh_arr=Iinh_arr, duration=duration)
	