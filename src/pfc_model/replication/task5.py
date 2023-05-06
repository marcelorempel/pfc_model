import numpy as np
import os
import matplotlib.transforms as mtransforms
from matplotlib import pyplot as plt
from pfc_model import *
from pfc_model.analysis import*
  
__all__ = ['task5']


@time_report()
def task5(simulation_dir, seed=None):
    
    n_cells=1000
    n_stripes=1
    constant_stimuli = [
                        [('PC', 0), 250],
                        [('IN', 0), 200]
                        ]
    method='rk4'
    dt=0.01
    transient=1000   
    seed = seed
    
    duration=2500
                          
    cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                          constant_stimuli=constant_stimuli, method=method, 
                          dt=dt, transient=transient, seed=seed)
     
    PC_L23 = cortex.neuron_idcs(('PC_L23',0))
    
    pCon_reg = 0.2
    pulse0, rate0 = (1100, 1105), 50000
    pulse1, rate1 = (2100, 2105), 100000
    gmax_reg = 0.1
    pfail_reg=0
    cortex.set_regular_stimuli('regular1', 1, ['AMPA', 'NMDA'], PC_L23, 
                               pcon=pCon_reg, rate=rate0, start=pulse0[0], 
                               stop=pulse0[1], gmax=gmax_reg, pfail=pfail_reg)
    cortex.set_regular_stimuli('regular2', 1, ['AMPA', 'NMDA'], PC_L23, 
                               pcon=pCon_reg, rate=rate1, start=pulse1[0], 
                               stop=pulse1[1], gmax=gmax_reg, pfail=pfail_reg)
    cortex.run(duration)
    
    PC_L23 = cortex.neuron_idcs(('PC_L23', 0))
    PC_L5 = cortex.neuron_idcs(('PC_L5', 0))
    
    delta_t=1
    
    tpulse0 = np.arange(pulse0[0], pulse0[1]+50, delta_t)
    PCL23spikes_pulse0 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L23,tlim=(pulse0[0], pulse0[0]+50), 
            delta_t=delta_t), 
        axis=0)
    PCL5spikes_pulse0 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L5,tlim=(pulse0[0], pulse0[0]+50), 
            delta_t=delta_t), 
        axis=0)
    
    tpulse1 = np.arange(pulse1[0], pulse1[1]+50, delta_t)
    PCL23spikes_pulse1 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L23, tlim=(pulse1[0], pulse1[0]+50),
            delta_t=delta_t), 
        axis=0)
    PCL5spikes_pulse1 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L5, tlim=(pulse1[0], pulse1[0]+50), 
            delta_t=delta_t), 
        axis=0)
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
        
    with open(os.path.join(simulation_dir, 'Reports', 
                           'RegPulses__spikingcounts.txt'), 'w') as f:
        print('Regular pulses on PC_L23', end='\n\n', file=f)
        print('gmax:', gmax_reg, file=f)
        print('pCon:', pCon_reg, file=f)
        print('pfail', pfail_reg, file=f, end='\n\n')
        
        print('Pulse0', file=f)
        print('Start: {} ms, stop: {} ms'.format(*pulse0), file=f)
        print('Rate: {} Hz'.format(rate0),file=f, end='\n\n')
        
        print('Spike count on PC_L23', file=f)
        time_spikes = list(zip(tpulse0, PCL23spikes_pulse0))
        for i in range(len(time_spikes)):
            print('t: {} ms, spikes: {}'.format(*time_spikes[i]), file=f)
        print('Total spikes on PC_L23: {}'.format(np.sum(PCL23spikes_pulse0)), 
              end='\n\n', file=f)
        
        print('Spike count on PC_L5', file=f)
        time_spikes = list(zip(tpulse0, PCL5spikes_pulse0))
        for i in range(len(time_spikes)):
            print('t: {} ms, spikes: {}'.format(*time_spikes[i]), file=f)
        print('Total spikes on PC_L5: {}'.format(np.sum(PCL5spikes_pulse0)), 
              end='\n\n', file=f)
        
        
        print('Pulse1', file=f)
        print('Start: {} ms, stop: {} ms'.format(*pulse1), file=f)
        print('Rate: {} Hz'.format(rate1),file=f, end='\n\n')
        
        print('Spike count on PC_L23', file=f)
        time_spikes = list(zip(tpulse1, PCL23spikes_pulse1))
        for i in range(len(time_spikes)):
            print('t: {} ms, spikes: {}'.format(*time_spikes[i]), file=f)
        print('Total spikes on PC_L23: {}'.format(np.sum(PCL23spikes_pulse1)), 
              end='\n\n', file=f)
        
        print('Spike count on PC_L5', file=f)
        time_spikes = list(zip(tpulse1, PCL5spikes_pulse1))
        for i in range(len(time_spikes)):
            print('t: {} ms, spikes: {}'.format(*time_spikes[i]), file=f)
        print('Total spikes on PC_L5: {}'.format(np.sum(PCL5spikes_pulse1)), 
              end='\n\n', file=f)

    basics_scales = {
        'membr_param_std': [(dict(group=group_sets['ALL'],
                             par=list(membranetuple._fields)), 0.2)]}
                          
    cortex_red = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                          constant_stimuli=constant_stimuli, method=method, 
                          dt=dt, basics_scales=basics_scales, 
                          transient=transient, seed=seed)
    
    
    PC_L23_red = cortex_red.neuron_idcs(('PC_L23',0))
    PC_L5_red = cortex_red.neuron_idcs(('PC_L5',0))
    
    cortex_red.set_regular_stimuli('regular1', 1, ['AMPA', 'NMDA'], PC_L23_red, 
                               pcon=pCon_reg, rate=rate0, start=pulse0[0], 
                               stop=pulse0[1], gmax=gmax_reg, pfail=pfail_reg)
    cortex_red.set_regular_stimuli('regular2', 1, ['AMPA', 'NMDA'], PC_L23_red, 
                               pcon=pCon_reg, rate=rate1, start=pulse1[0], 
                               stop=pulse1[1], gmax=gmax_reg, pfail=pfail_reg)
    
    cortex_red.run(duration)
    
    
    _fig09(cortex, cortex_red, pulse0, pulse1, simulation_dir)
    
    
    delta_t=1
    
    tpulse0 = np.arange(pulse0[0], pulse0[1]+50, delta_t)
    PCL23spikes_pulse0 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L23,tlim=(pulse0[0], pulse0[0]+50), 
            delta_t=delta_t), 
        axis=0)
    PCL5spikes_pulse0 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L5,tlim=(pulse0[0], pulse0[0]+50), 
            delta_t=delta_t), 
        axis=0)
    
    tpulse1 = np.arange(pulse1[0], pulse1[1]+50, delta_t)
    PCL23spikes_pulse1 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L23, tlim=(pulse1[0], pulse1[0]+50),
            delta_t=delta_t), 
        axis=0)
    PCL5spikes_pulse1 = np.sum(
        cortex.spiking_count(
            neuron_idcs=PC_L5, tlim=(pulse1[0], pulse1[0]+50), 
            delta_t=delta_t), 
        axis=0)
    
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
        
    with open(os.path.join(
            simulation_dir, 'Reports',
            'RegPulses_RedSTD__spikingcounts.txt'), 'w') as f:
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

            
def _fig09(cortex, cortex_red, pulse0, pulse1, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    
    p0_t0, p0_t1 = pulse0
    p1_t0, p1_t1 = pulse1
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    
    ax = axs[0, 0]
    plt.sca(ax)
    raster_plot(cortex, tlim=(p0_t0-25, p0_t0+60), show=False,
                s=12, newfigure=False)
    ax.text(-0.12, 0.88, '(A)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p0_t0, 0, cortex.neurons.N+15, 
               color='black', linestyle='dotted', linewidth=2)
    plt.vlines(p0_t0, min(cortex.neuron_idcs(('PC_L23', 0))),
               max(cortex.neuron_idcs(('PC_L23', 0))), 
               color='purple', linestyle='--', linewidth=3)
    plt.xticks([p0_t0-20, p0_t0, p0_t0+20, p0_t0+40, p0_t0+60])
    
    
    ax = axs[0, 1]
    plt.sca(ax)
    raster_plot(cortex, tlim=(p1_t0-25, p1_t0+60), show=False,
                s=12, newfigure=False)
    ax.text(-0.12, 0.88, '(B)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p1_t0, 0, cortex.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    plt.vlines(p1_t0, min(cortex.neuron_idcs(('PC_L23', 0))), 
               max(cortex.neuron_idcs(('PC_L23', 0))), 
               color='purple', linestyle='--', linewidth=3)
    plt.xticks([p1_t0-20, p1_t0, p1_t0+20, p1_t0+40, p1_t0+60])
    
    ax = axs[1, 0]
    plt.sca(ax)
    raster_plot(cortex_red, tlim=(p0_t0-25, p0_t0+60), show=False,
                s=12, newfigure=False)
    ax.text(-0.12, 0.88, '(C)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p0_t0, 0, cortex_red.neurons.N+15, 
               color='black', linestyle='dotted', linewidth=2)
    plt.vlines(p0_t0, min(cortex_red.neuron_idcs(('PC_L23', 0))),
               max(cortex_red.neuron_idcs(('PC_L23', 0))), 
               color='purple', linestyle='--', linewidth=3)
    plt.xticks([p0_t0-20, p0_t0, p0_t0+20, p0_t0+40, p0_t0+60])
    
    
    ax = axs[1, 1]
    plt.sca(ax)
    raster_plot(cortex_red, tlim=(p1_t0-25, p1_t0+60), show=False,
                s=12, newfigure=False)
    ax.text(-0.12, 0.88, '(D)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p1_t0, 0, cortex_red.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    plt.vlines(p1_t0, min(cortex_red.neuron_idcs(('PC_L23', 0))), 
               max(cortex_red.neuron_idcs(('PC_L23', 0))), 
               color='purple', linestyle='--', linewidth=3)
    plt.xticks([p1_t0-20, p1_t0, p1_t0+20, p1_t0+40, p1_t0+60])
    
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(path,'Figures','Fig09.png'))
    

if __name__ == '__main__':
    seed = 0
    simulation_dir = set_simulation_dir()
    task5(simulation_dir, seed)