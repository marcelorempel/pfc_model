import numpy as np
from pfc_model.replication import *
from .._auxiliary import *

__all__ = ['taskset']

@time_report()
def taskset(simulation_dir, seed, Ntrials, Iexc_arr, Iinh_arr, duration):
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
    simulation_dir = set_simulation_dir('Results_'+os.path.basename(__file__)[:-3])
    seed = 0

    Iexc_arr = np.arange(0, 600, 25)
    Iinh_arr = np.arange(0, 600, 25)
    duration=13000
    Ntrials=100

    taskset(simulation_dir=simulation_dir, seed=seed, Ntrials=Ntrials,
          Iexc_arr=Iexc_arr, Iinh_arr=Iinh_arr, duration=duration)
	