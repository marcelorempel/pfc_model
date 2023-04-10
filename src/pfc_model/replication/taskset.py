import numpy as np
from pfc_model.replication import *
from .._auxiliary import set_simulation_dir

simulation_dir = set_simulation_dir()
seed = 0

Iexc_arr = np.arange(0, 600, 100)
Iinh_arr = np.arange(0, 600, 100)

task1(simulation_dir=simulation_dir, seed=seed)
task2(simulation_dir=simulation_dir, seed=seed)
task3(simulation_dir=simulation_dir, seed=seed)
task4(simulation_dir=simulation_dir)
task5(simulation_dir=simulation_dir, seed=seed)
task6(simulation_dir=simulation_dir, Iexc_arr=Iexc_arr, 
      Iinh_arr=Iinh_arr, seed=seed, duration=13000)
