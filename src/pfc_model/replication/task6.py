""" This script defines task6.

task6 analyzes the effect of reducing variability of membrane parameters
on network response to stimuli.
"""

import numpy as np
import os
from matplotlib import pyplot as plt
from pfc_model import *
from pfc_model.analysis import*

__all__ = ['task6']

@time_report()
def task6(simulation_dir, Ntrials=5, seed_list=None):
    """Analyze the effect of reducing variability of membrane parameters
    on network response to stimuli.
    
    The figure representing the comparisons of network response is saved to
    "simulation_dir/Figures/Fig10.png"
        
    The measures are saved to "simulation_dir/Reports/Stimulation
    /Param_std_variation.txt".
    
    Here, "simulation_dir" is to be replaced by the actual argument. 
    
    Parameters
    ----------
    simulation_dir: str
        Path to directory where results are to be saved.
    Ntrials: int, optional
        Number of simulations to be performed and included in the analyses.
        If not given, it defaults to 5.
    seed_list: list[int]
        List of integers with seed numbers. Its length must be equal to 
        Ntrials. If not given, it default to the list from 0 to Ntrials-1.
    """

    Membr_std_list = np.asarray([0,20,40,60,80,100])/100
    
    if seed_list is None:
        seed_list=[seed for seed in range(Ntrials)]
    
    PC_L23_results = [[] for i in range(len(Membr_std_list))]
    PC_L5_results = [[] for i in range(len(Membr_std_list))]
    
 
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    
    method = 'rk4'
    dt = 0.01
    transient= 1000
    duration = 1200
    
    

    n_cells=1000
    n_stripes=1
    constant_stimuli = [
                        [('PC', 0), 250],
                        [('IN', 0), 200]
                        ]

    for trial in range(Ntrials):
        seed = seed_list[trial]
        for std in range(len(Membr_std_list)):
            print_out1a = '||   REPORT: Trial ' + str(trial)
            print_out2a = '||   REPORT: std_scale ' + str(Membr_std_list[std])
            print_out1 = (print_out1a 
                          + ' '*(max(len(print_out1a), len(print_out2a)) 
                                 - len(print_out1a)) 
                          + '   ||')
            print_out2 = (print_out2a 
                          + ' '*(max(len(print_out1a), len(print_out2a)) 
                                 - len(print_out2a)) 
                          + '   ||')
            print("="*len(print_out1))
            print(print_out1)
            print(print_out2)
            print("="*len(print_out1))
            
            basics_scales = {'membr_param_std': [
                (dict(par=list(membranetuple._fields),  group=group_sets['ALL']),
                 Membr_std_list[std])]}

            cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                                  constant_stimuli=constant_stimuli, 
                                  method=method, dt=dt, 
                                  basics_scales=basics_scales, 
                                  transient=transient, seed=seed)
            
            
            
            PC_L23 = cortex.neuron_idcs(('PC_L23',0))
            PC_L5 = cortex.neuron_idcs(('PC_L5',0))
            
            pCon_reg = 0.2
            pulse=(1100, 1105)
            rate=100000
            gmax_reg = 0.1
            pfail_reg=0
            
            cortex.set_regular_stimuli(
                'regular', 1, ['AMPA', 'NMDA'], PC_L23, pcon=pCon_reg, 
                rate=rate, start=pulse[0], stop=pulse[1], gmax=gmax_reg, 
                pfail=pfail_reg)
 
            cortex.run(duration)        
 
            PC_L23_results[std].append(
                np.sum(
                    cortex.spiking_count(neuron_idcs=PC_L23, 
                                         tlim=(pulse[0], pulse[0]+50))))
            PC_L5_results[std].append(
                np.sum(
                    cortex.spiking_count(neuron_idcs=PC_L5, 
                                         tlim=(pulse[0], pulse[0]+50))))
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
        
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports', 'Stimulation')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'Stimulation'))
    
        
    with open(os.path.join(
            simulation_dir, 'Reports', 'Stimulation', 
            'Param_std_variation.txt'), 'w') as f:
        print('N trials:', Ntrials, file=f)
        print('Seed list:', seed_list, file=f)
        print('Std list:', Membr_std_list, file=f)
        print(file=f)
        print('L2/3', file=f)
        for std in range(len(Membr_std_list)):        
            print('Std:', Membr_std_list[std], file=f)
            print(PC_L23_results[std], file=f)
            print(file=f)
        print('-'*20, file=f)
        print('L5', file=f)
        for std in range(len(Membr_std_list)):        
            print('Std:', Membr_std_list[std], file=f)
            print(PC_L5_results[std], file=f)
            print(file=f)
       
    _fig10(Membr_std_list, PC_L23_results, PC_L5_results, simulation_dir)

    
def _fig10(Membr_std_list, PC_L23, PC_L5, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    
    PC_L23_mean = []
    PC_L23_std = []
    PC_L5_mean = []
    PC_L5_std = []

    for scale in range(len(Membr_std_list)):
        PC_L23_mean.append(np.mean(PC_L23[scale]))
        PC_L23_std.append(np.std(PC_L23[scale]))
        PC_L5_mean.append(np.mean(PC_L5[scale]))
        PC_L5_std.append(np.std(PC_L5[scale]))
       
    
    variability =  Membr_std_list*100 
    PC_L23_mean = 100*np.asarray(PC_L23_mean)/PC_L23_mean[-1]
    PC_L5_mean = 100*np.asarray(PC_L5_mean)/PC_L5_mean[-1]

    PC_L23_SEM = (100*np.asarray(PC_L23_std)/
                  np.sqrt(len(PC_L23[0]))/PC_L23_mean[-1])
    PC_L5_SEM = 100*np.asarray(PC_L5_std)/np.sqrt(len(PC_L5[0]))/PC_L5_mean[-1]

    
    plt.figure(figsize=(18, 12))
    plt.ylabel('relative spiking activity', fontsize=26)
    plt.xlabel('fraction of original membrane parameter SD (%)', fontsize=26)
    plt.tick_params(labelsize=26)
    plt.plot(variability, PC_L23_mean, label='PC L23', color='blue')
    plt.errorbar(variability, PC_L23_mean, yerr = PC_L23_SEM, fmt = 'o', 
                 color = 'blue', ecolor = 'blue', elinewidth = 2, capsize=3)
    plt.plot(variability, PC_L5_mean, label='PC 5', color='orange')
    plt.errorbar(variability, PC_L5_mean, yerr = PC_L5_SEM, fmt = 'o', 
                 color = 'orange', ecolor = 'orange', elinewidth = 2, 
                 capsize=3)
    plt.xlim(-10, 120)
    plt.xticks([0, 50, 100])
    props23 = dict(boxstyle='round', facecolor='blue', alpha=0.2)
    props5 = dict(boxstyle='round', facecolor='orange', alpha=0.2)
    plt.text(45, 110, 'PC L23 100%: {:.1f} spikes'.format(np.mean(PC_L23[-1])), 
             fontsize=26, bbox=props23)
    plt.text(45, 15, 'PC L5   100%: {:.1f} spikes'.format(np.mean(PC_L5[-1])),
             fontsize=26, bbox=props5)
    plt.legend(prop={'size': 26})
    plt.savefig(os.path.join(path,'Figures','Fig10.png'))
    
if __name__ == '__main__':
    simulation_dir = set_simulation_dir('Results_'+os.path.basename(__file__)[:-3])
    Ntrials=5
    task6(simulation_dir=simulation_dir, Ntrials=5)