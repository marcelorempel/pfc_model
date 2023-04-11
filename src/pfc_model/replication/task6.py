import os
from matplotlib import pyplot as plt
from pfc_model import *
from pfc_model.analysis import*

__all__ = ['task6']

@time_report()
def task6(simulation_dir, seed=None):

    # Network setup
    n_cells=1000
    n_stripes=1
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    method='rk4'
    dt=0.05
    transient=1000   
    seed = seed
    
    # Poisson stimuli
    pCon_reg = 0.05
    pulse0=(1100, 1200)
    pulse1=(2100, 2200)
    rate=10
    gmax_reg = 2
    pfail_reg=0
    
    # Simulation duration
    duration=3000
    
    # Cortex without gmax decrease
    cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                          constant_stimuli=constant_stimuli,method=method, 
                          dt=dt, transient=transient, seed=seed)
     
    
    PC_L23 = cortex.neuron_idcs(('PC_L23',0))
    PC_L5 = cortex.neuron_idcs(('PC_L5',0))
    
    cortex.set_regular_stimuli('poisson1', 100, ['AMPA', 'NMDA'], PC_L23,
                               pcon=pCon_reg, rate=rate, start=pulse0[0], 
                               stop=pulse0[1], gmax=gmax_reg, pfail=pfail_reg)
    cortex.set_regular_stimuli('poisson1', 100, ['AMPA', 'NMDA'], PC_L5, 
                               pcon=pCon_reg, rate=rate, start=pulse1[0], 
                               stop=pulse1[1], gmax=gmax_reg, pfail=pfail_reg)
    
    cortex.run(duration)
    
    _fig09(cortex, pulse0, pulse1, simulation_dir)
    
    ## Decreased gmax_mean
    basics_scales = {'gmax_mean': [(dict(target=group_sets['ALL'], 
                                         source=group_sets['IN']), 0.7)]}
    
    cortex_dec = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                          constant_stimuli=constant_stimuli,method=method, 
                          dt=dt, basics_scales=basics_scales, 
                          transient=transient, seed=seed)
       
    PC_L23 = cortex_dec.neuron_idcs(('PC_L23',0))
    PC_L5 = cortex_dec.neuron_idcs(('PC_L5',0))
    
    cortex_dec.set_regular_stimuli('poisson1', 100, ['AMPA', 'NMDA'], PC_L23, 
                               pcon=pCon_reg, rate=rate, start=pulse0[0], 
                               stop=pulse0[1], gmax=gmax_reg, pfail=pfail_reg)
    cortex_dec.set_regular_stimuli('poisson1', 100, ['AMPA', 'NMDA'], PC_L5, 
                               pcon=pCon_reg, rate=rate, start=pulse1[0], 
                               stop=pulse1[1], gmax=gmax_reg, pfail=pfail_reg)

    cortex_dec.run(duration)
    
    _fig10(cortex_dec, pulse0, pulse1, simulation_dir)


def _fig09(cortex, pulse0, pulse1, path):    
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    _poissonfigures(cortex, pulse0, pulse1, 
                    os.path.join(path,'Figures','Fig09a.png'), 
                    os.path.join(path,'Figures','Fig09b.png'))
    
def _fig10(cortex, pulse0, pulse1, path):    
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    _poissonfigures(cortex, pulse0, pulse1, 
                    os.path.join(path,'Figures','Fig10a.png'), 
                    os.path.join(path,'Figures','Fig10b.png'))


def _poissonfigures(cortex, pulse0, pulse1, file0, file1):
    PC_L23 = cortex.neuron_idcs(('PC_L23',0))
    PC_L5 = cortex.neuron_idcs(('PC_L5',0))
    
    p0_t0, p0_t1 = pulse0
    p1_t0, p1_t1 = pulse1
    
    raster_plot(cortex, tlim=(p0_t0-100, p0_t0+400), show=False)
    
    plt.vlines(p0_t0, 0, cortex.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    plt.vlines(p0_t1, 0, cortex.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    
    plt.vlines(p0_t0, min(PC_L23), max(PC_L23), color='purple', 
               linestyle='--',  linewidth=3)
    plt.vlines(p0_t1, min(PC_L23), max(PC_L23), color='green', 
               linestyle='--',  linewidth=3)
    
    plt.savefig(file0)
    
    raster_plot(cortex, tlim=(p1_t0-100, p1_t1+400), show=False)
    
    plt.vlines(p1_t0, 0, cortex.neurons.N+15, color='black', 
               linestyle='dotted', linewidth=2)
    plt.vlines(p1_t1, 0, cortex.neurons.N+15, color='black', 
               linestyle='dotted', linewidth=2)
    
    plt.vlines(p1_t0, min(PC_L5), max(PC_L5), color='purple',
               linestyle='--' , linewidth=3)
    plt.vlines(p1_t1, min(PC_L5), max(PC_L5), color='green', 
               linestyle='--' , linewidth=3)
    
    plt.savefig(file1)

