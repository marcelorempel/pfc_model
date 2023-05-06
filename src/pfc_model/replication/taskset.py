import numpy as np
from pfc_model.replication import *
from .._auxiliary import *

@time_report()
def tasks(simulation_dir, seed, Ntrials, Iexc_arr, Iinh_arr, duration):
    task1(simulation_dir, seed)
    task2(simulation_dir)
    task3(simulation_dir=simulation_dir, Ntrials=Ntrials)
    task4(simulation_dir=simulation_dir, seed=seed)
    task5(simulation_dir=simulation_dir, seed=seed)
    task6(simulation_dir=simulation_dir)
    task7(simulation_dir=simulation_dir, seed=seed)
    task8(simulation_dir=simulation_dir, Iexc_arr=Iexc_arr, 
        Iinh_arr=Iinh_arr, seed=seed, duration=duration)

if __name__ == '__main__':
    simulation_dir = set_simulation_dir()
    seed = 0

    Iexc_arr = np.arange(0, 600, 100)
    Iinh_arr = np.arange(0, 600, 100)
    duration=13000
    Ntrials=30

    tasks(simulation_dir=simulation_dir, seed=seed, Ntrials=Ntrials,
          Iexc_arr=Iexc_arr, Iinh_arr=Iinh_arr, duration=duration)
	