import numpy as np
import os
from pfc_model import *


__all__ = ['task1']

@time_report()
def task1(simulation_dir, Ntrials=30, seed_list=None):
    
    if seed_list is None:
        seed_list = [i for i in range(Ntrials)]
    else:
        if len(seed_list) != Ntrials:
            raise ValueError('Length of seed_list must be equal to Ntrials.')
    
    n_cells=1000
    n_stripes=1
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    method = 'rk4'
    dt = 0.05
    transient = 1000   
    
    duration =16000
    
    spiking_threshold = 0.33
    PC_result = []
    
    for seed in seed_list:
        cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                              constant_stimuli=constant_stimuli, method=method,
                              dt=dt, transient=transient, seed=seed)    
        cortex.run(duration)
    

        spikingPC = cortex.spiking_idcs((np.greater_equal, spiking_threshold), 
                                        ('PC', 0))
        PC_len = len(cortex.neuron_idcs(('PC', 0)))
        PC_result.append(len(spikingPC)*100/PC_len)
       
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports',
                                      'SpikingPC')):
        os.mkdir(os.path.join(simulation_dir, 'Reports',
                              'SpikingPC'))

    with open(os.path.join(simulation_dir, 'Reports', 'SpikingPC',
                           'SpikingPC_stats.txt'), 'w') as f:
         print('Spiking PC statistics', file=f)
         print('Spiking threshold: {} Hz'.format(spiking_threshold), file=f)
         print('Number of trials: {}'.format(Ntrials), file=f)
         print('List of seeds:', seed_list, file=f)
         print(file=f)
         print('Mean: {:.2f} %'.format(np.mean(PC_result)), file=f)
         print('Standard deviation: {:.2f} %'.format(np.std(PC_result)), 
               file=f)
         print('Max: {:.2f} %'.format(np.max(PC_result)), file=f)
         print('Min: {:.2f} %'.format(np.min(PC_result)), file=f)
         print('-'*40, file=f)
         print(file=f)
         print('All results:', PC_result, file=f)
         
    