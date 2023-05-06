import os
import matplotlib.transforms as mtransforms
from matplotlib import pyplot as plt
from pfc_model import *
from pfc_model.analysis import*

__all__ = ['task7']

@time_report()
def task7(simulation_dir, seed=None):

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
    
    # _fig11(cortex, pulse0, pulse1, simulation_dir)
    
    ## Decreased gmax_mean
    basics_scales = {'gmax_mean': [(dict(target=group_sets['ALL'], 
                                         source=group_sets['IN']), 0.5)]}
    
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
    
    _fig11(cortex, cortex_dec, pulse0, pulse1, simulation_dir)


def _fig11(cortex, cortex_dec, pulse0, pulse1, path):    
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    _poissonfigures(cortex, cortex_dec, pulse0, pulse1, 
                    os.path.join(path,'Figures','Fig11.png')
                    )
    
# def _fig10(cortex, pulse0, pulse1, path):    
#     if not os.path.isdir(os.path.join(path, 'Figures')):
#         os.mkdir(os.path.join(path, 'Figures'))
#     _poissonfigures(cortex, pulse0, pulse1, 
#                     os.path.join(path,'Figures','Fig10.png') 
#                     )


def _poissonfigures(cortex, cortex_dec, pulse0, pulse1, file):

    p0_t0, p0_t1 = pulse0
    p1_t0, p1_t1 = pulse1
    
    PC_L23 = cortex.neuron_idcs(('PC_L23',0))
    PC_L5 = cortex.neuron_idcs(('PC_L5',0))
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    
    ax = axs[0, 0]
    plt.sca(ax)
    raster_plot(cortex, tlim=(p0_t0-100, p0_t1+200), show=False, 
                newfigure=False)
    ax.text(-0.12, 0.88, '(A)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p0_t0, 0, cortex.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    plt.vlines(p0_t1, 0, cortex.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    
    plt.vlines(p0_t0, min(PC_L23), max(PC_L23), color='purple', 
               linestyle='--',  linewidth=3)
    plt.vlines(p0_t1, min(PC_L23), max(PC_L23), color='green', 
               linestyle='--',  linewidth=3)

    plt.locator_params(axis='x', nbins=4)
    
    ax = axs[0, 1]
    plt.sca(ax)
    raster_plot(cortex, tlim=(p1_t0-100, p1_t1+200), show=False,
                newfigure=False)
    ax.text(-0.12, 0.88, '(B)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p1_t0, 0, cortex.neurons.N+15, color='black', 
               linestyle='dotted', linewidth=2)
    plt.vlines(p1_t1, 0, cortex.neurons.N+15, color='black', 
               linestyle='dotted', linewidth=2)
    
    plt.vlines(p1_t0, min(PC_L5), max(PC_L5), color='purple',
               linestyle='--' , linewidth=3)
    plt.vlines(p1_t1, min(PC_L5), max(PC_L5), color='green', 
               linestyle='--' , linewidth=3)
    
    
    plt.locator_params(axis='x', nbins=4)
    
    PC_L23_dec = cortex_dec.neuron_idcs(('PC_L23',0))
    PC_L5_dec = cortex_dec.neuron_idcs(('PC_L5',0))
    
    
    ax = axs[1, 0]
    plt.sca(ax)
    raster_plot(cortex_dec, tlim=(p0_t0-100, p0_t1+200), show=False, 
                newfigure=False)
    ax.text(-0.12, 0.88, '(C)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p0_t0, 0, cortex_dec.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    plt.vlines(p0_t1, 0, cortex_dec.neurons.N+15, color='black',
               linestyle='dotted', linewidth=2)
    
    plt.vlines(p0_t0, min(PC_L23_dec), max(PC_L23_dec), color='purple', 
               linestyle='--',  linewidth=3)
    plt.vlines(p0_t1, min(PC_L23_dec), max(PC_L23_dec), color='green', 
               linestyle='--',  linewidth=3)
    plt.locator_params(axis='x', nbins=4)
    
    ax = axs[1,1]
    plt.sca(ax)
    raster_plot(cortex_dec, tlim=(p1_t0-100, p1_t1+200), show=False,
                newfigure=False)
    ax.text(-0.12, 0.88, '(D)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    plt.vlines(p1_t0, 0, cortex_dec.neurons.N+15, color='black', 
               linestyle='dotted', linewidth=2)
    plt.vlines(p1_t1, 0, cortex_dec.neurons.N+15, color='black', 
               linestyle='dotted', linewidth=2)
    
    plt.vlines(p1_t0, min(PC_L5_dec), max(PC_L5_dec), color='purple',
               linestyle='--' , linewidth=3)
    plt.vlines(p1_t1, min(PC_L5_dec), max(PC_L5_dec), color='green', 
               linestyle='--' , linewidth=3)
    
    plt.locator_params(axis='x', nbins=4)
    
    plt.tight_layout()
    plt.savefig(file)

if __name__ == '__main__':
    seed = 0
    simulation_dir = set_simulation_dir()
    task7(simulation_dir=simulation_dir, seed=seed)