import numpy as np
import os
from matplotlib import pyplot as plt
from pfc_model import *
from pfc_model.analysis import*

__all__ = ['task3']     


@time_report()
def task3(simulation_dir, seed=None):
    basics_scales = {
        'membr_param_std': [(dict(group=group_sets['ALL'],
                             par=list(membranetuple._fields)), 0.2)]}

    n_cells=1000
    n_stripes=1
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    method='rk4'
    dt=0.01
    transient=1000   
    seed = seed
                          
    cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                          constant_stimuli=constant_stimuli, method=method, 
                          dt=dt, basics_scales=basics_scales, 
                          transient=transient, seed=seed)
    
    
    duration=1200 
    pCon_reg = 0.2
    pulse=(1100, 1105)
    rate=100000
    gmax_reg = 0.1
    pfail_reg=0
    
    PC_L23 = cortex.neuron_idcs(('PC_L23',0))
    PC_L5 = cortex.neuron_idcs(('PC_L5',0))
    
    cortex.set_regular_stimuli('regular1', 1, ['AMPA', 'NMDA'], PC_L23, 
                               pcon=pCon_reg, rate=rate, start=pulse[0], 
                               stop=pulse[1], gmax=gmax_reg, pfail=pfail_reg)
    
    cortex.run(duration)
    
    _fig07(cortex, pulse, simulation_dir)
    
    delta_t=1
    
    tpulse = np.arange(pulse[0], pulse[0]+50, delta_t)
    PCL23spikes_pulse = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L23,tlim=(pulse[0], pulse[0]+50), 
            delta_t=delta_t), 
        axis=0)
    PCL5spikes_pulse = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L5,tlim=(pulse[0], pulse[0]+50), 
            delta_t=delta_t),
        axis=0)
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
        
    with open(os.path.join(
            simulation_dir, 'Reports',
            'Regularpulses_reducedstd__spikingcounts.txt'), 'w') as f:
        print('Regular pulses on PC_L23', end='\n\n', file=f)
        print('gmax:', gmax_reg, file=f)
        print('pCon:', pCon_reg, file=f)
        print('pfail', pfail_reg, file=f, end='\n\n')
        
        print('Pulse', file=f)
        print('Start: {} ms, stop: {} ms'.format(*pulse), file=f)
        print('Rate: {} Hz'.format(rate),file=f, end='\n\n')
        
        print('Spike count on PC_L23', file=f)
        time_spikes = list(zip(tpulse, PCL23spikes_pulse))
        for i in range(len(time_spikes)):
            print('t: {} ms, spikes: {}'.format(*time_spikes[i]), file=f)
        print('Total spikes on PC_L23: {}'.format(np.sum(PCL23spikes_pulse)),
              end='\n\n', file=f)
        
        print('Spike count on PC_L5', file=f)
        time_spikes = list(zip(tpulse, PCL5spikes_pulse))
        for i in range(len(time_spikes)):
            print('t: {} ms, spikes: {}'.format(*time_spikes[i]), file=f)
        print('Total spikes on PC_L5: {}'.format(np.sum(PCL5spikes_pulse)),
              end='\n\n', file=f)

    
def _fig07(cortex, pulse, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    p_t0, p_t1 = pulse
    
    raster_plot(cortex, tlim=(p_t0-25, p_t1+60), show=False)
    plt.vlines(p_t0, 0, cortex.neurons.N+15, color='black', 
               linestyle='dotted', linewidth=2)
    plt.vlines(p_t0, min(cortex.neuron_idcs(('PC_L23', 0))),
               max(cortex.neuron_idcs(('PC_L23', 0))), 
               color='purple', linestyle='--', linewidth=3)
    plt.savefig(os.path.join(path,'Figures','Fig14.png'))
    
