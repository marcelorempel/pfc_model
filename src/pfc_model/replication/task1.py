import numpy as np
import os
from pfc_model import *

__all__ = ['task1']

@time_report()
def task1(simulation_dir, seed=0):
    n_cells=5000
    n_stripes=1
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    method = 'rk4'
    dt = 0.05
    alternative_pcells = np.asarray([10, 5, 5, 5, 5, 10, 10,
                                     10, 5, 5, 5, 5, 10, 10])
    
    cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                          constant_stimuli=constant_stimuli, method=method,
                          dt=dt, alternative_pcells=alternative_pcells, 
                          seed=seed)    
    
    aliases = ['PC and IN_CC L2/3', 'PC and IN_CC L5','IN_L', 'IN_CL', 
               'IN_F']
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
        
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports', 'Param_description')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'Param_description'))
        
        
    get_membr_params(cortex,
                      [[('PC_L23', 0), ('IN_CC_L23', 0)], 
                       [('PC_L5', 0), ('IN_CC_L5', 0)],
                       ('IN_L_both',0), 
                       ('IN_CL_both',0), 
                      ('IN_F',0)], 
                      alias_list = aliases,
                      file=os.path.join(simulation_dir, 'Reports', 
                                        'Param_description',
                                        'description_membparams.txt'))
    _fig01(cortex, simulation_dir)

def _fig01(cortex, path):  
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
        
    distribution_sets = [
        [('PC_L23',0), ('IN_CC_L23',0)],
        ]

    xlabel_list = ['C (pF)', 'g$_L$ (nS)', 'E$_L$ (mV)', '$\Delta_T$ (mV)', 
                   'V$_T$ (mV)', 'V$_{up}$ (mV)', 'V$_r$ (mV)', 'b (pA)',
                   '$\\tau_w$ (ms)', '$\\tau_m$ (ms)']

    letter_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)',
                   '(I)', '(J)']

    clip_list = [[0, np.NaN], [0, np.NaN], [np.NaN, np.NaN],  [0, np.NaN],
                 [np.NaN, np.NaN], [np.NaN, np.NaN], [np.NaN, np.NaN], [0, np.NaN],
                 [0, np.NaN], [0, np.NaN]
                 ]

    for i in range(len(distribution_sets)):
        sets_ = distribution_sets[i]
        idcs = cortex.neuron_idcs(sets_)
        params = cortex.get_memb_params(idcs)
        params_list = [params.C, params.g_L, params.E_L, params.delta_T,
                       params.V_T, params.V_up, params.V_r, params.b,
                       params.tau_w, params.C/params.g_L]
        
        fig, axs = plt.subplots(5, 2, figsize=(12, 20))
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)

        for i in range(10):
            ax = axs[i//2,i%2]
            letter = letter_list[i]
            plt.sca(ax)
            ax.text(0.8, 1, letter, transform=ax.transAxes + trans,
                fontsize=24, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
            par = params_list[i]
            clip = clip_list[i]
            xlabel = xlabel_list[i]
            sns.kdeplot(par, clip=clip, lw=2)
            plt.xlabel(xlabel, fontsize=20)
            ax.yaxis.label.set_size(20)
            plt.tick_params(labelsize=20)
            plt.locator_params(nbins=4)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'Figures', 'Fig01.png'))

         
if __name__ == '__main__':
    simulation_dir = set_simulation_dir('Results_'+os.path.basename(__file__)[:-3])
    seed=0
    task1(simulation_dir, seed)
    