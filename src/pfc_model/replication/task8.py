import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from .._basics_setup import *
from pfc_model import *
from pfc_model.analysis import*

__all__ = ['task8']

@time_report()
def task8(simulation_dir, Iexc_arr, Iinh_arr, seed=None, transient=1000, 
          duration=13000):
    
    Iexc_applied = []
    Iinh_applied = []
    
    PCL23_result = []
    PCL5_result = []
    PC_result = []
    IN_result = []
    ALL_result = []
    
    ISImeanmean_result = []
    ISImeanstd_result = []
    ISICVmean_result = []
    ISICVstd_result = []
    
    meanCCmean_result = []
    stdCCmean_result = []
    meanCCstd_result = []
    stdCCstd_result = []
    
    results_list = [PCL23_result, PCL5_result, PC_result, IN_result, 
                    ALL_result, ISImeanmean_result, ISImeanstd_result, 
                    ISICVmean_result, ISICVstd_result, meanCCmean_result, 
                    stdCCmean_result]
        
    name_list = ['PCL23_result', 'PCL5_result', 'PC_result',
                 'IN_result', 'ALL_result', 'ISImeanmean_result',
                 'ISImeanstd_result','ISICVmean_result', 'ISICVstd_result', 
                 'meanCCmean_result', 'stdCCmean_result']
    
    
    label_list = ['fraction of spiking PC L23 (%)',
                  'fraction of spiking PC L5 (%)',
                  'fraction of spiking PC (%)',
                  'fraction of spiking IN (%)',
                  'fraction of spiking cells (%)',
                  'mean of mean ISI (ms)',
                  'standard deviation of mean ISI (ms)',
                  'mean of ISI CV', 'standard deviation of ISI CV',
                  'mean of zero-lag cross-correlations',
                  'standard deviation of zero-lag cross-correlations']
    
    
    for trial in range(len(Iexc_arr)*len(Iinh_arr)):
        
        print()
        print('='*20)
        print('Starting {} out of {}'.format(trial+1, 
                                             len(Iexc_arr)*len(Iinh_arr)))
        print('Iexc_arr = {} pA'.format(Iexc_arr[trial%len(Iexc_arr)]))
        print('Iinh_arr = {} pA'.format(Iinh_arr[trial//len(Iexc_arr)]))
        print('='*20)
        print()
        
        constant_stimuli = [
                            [('PC_L23', 0), Iexc_arr[trial%len(Iexc_arr)]],
                            [('IN_L23', 0), Iinh_arr[trial//len(Iexc_arr)]],
                            [('PC_L5', 0), 250],
                            [('IN_L5', 0), 200],
                            ]
        method='rk4'
        dt=0.05
        
        Iexc_applied.append(Iexc_arr[trial%len(Iexc_arr)])
        Iinh_applied.append(Iinh_arr[trial//len(Iexc_arr)])
        
        if trial == 0 and not os.path.isdir(os.path.join(simulation_dir, 'Task9_seed{}'.format(seed))):
            n_cells=1000
            n_stripes=1
           
            cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                                  constant_stimuli=constant_stimuli,
                                  method=method, dt=dt, transient=transient, 
                                  seed=seed)
            cortex.save(os.path.join(simulation_dir, 'Task9_seed{}'.format(seed)))
            
        else:
            cortex = Cortex.load(os.path.join(simulation_dir, 'Task9_seed{}'.format(seed)), constant_stimuli=constant_stimuli, 
                                 method=method, dt=dt, transient=transient)

        cortex.run(duration)
        
        spikingPCL23 = cortex.spiking_idcs((np.greater_equal, 0.33), 
                                           ('PC_L23', 0))
        spikingPCL5 = cortex.spiking_idcs((np.greater_equal, 0.33), 
                                          ('PC_L5', 0))
        spikingPC = cortex.spiking_idcs((np.greater_equal, 0.33), ('PC', 0))
        spikingIN= cortex.spiking_idcs((np.greater_equal, 0.33), ('IN', 0))
        spikingALL= cortex.spiking_idcs((np.greater_equal, 0.33), ('ALL', 0))
        
        ISImean, ISICV = get_ISI_stats(cortex, neuron_idcs=spikingALL)       
        _, _, _, meanCC, stdCC = get_correlations(cortex, idcs=spikingALL, 
                                                  display=True)
        

        PCL23_len = len(cortex.neuron_idcs(('PC_L23', 0)))
        PCL5_len = len(cortex.neuron_idcs(('PC_L5', 0)))
        PC_len = len(cortex.neuron_idcs(('PC', 0)))
        IN_len = len(cortex.neuron_idcs(('IN', 0)))
        ALL_len = len(cortex.neuron_idcs(('ALL', 0)))
        
        PCL23_result.append(len(spikingPCL23)*100/PCL23_len)
        PCL5_result.append(len(spikingPCL5)*100/PCL5_len)
        PC_result.append(len(spikingPC)*100/PC_len)
        IN_result.append(len(spikingIN)*100/IN_len)
        ALL_result.append(len(spikingALL)*100/ALL_len)
        ISImeanmean_result.append(np.mean(ISImean))
        ISImeanstd_result.append(np.std(ISImean))
        ISICVmean_result.append(np.mean(ISICV))
        ISICVstd_result.append(np.std(ISICV))      
        meanCCmean_result.append(np.mean(meanCC))
        stdCCmean_result.append(np.mean(stdCC))
        

    PCL23_result = np.asarray(PCL23_result)
    PCL5_result = np.asarray(PCL5_result)
    PC_result = np.asarray(PC_result)
    IN_result = np.asarray(IN_result)
    ALL_result = np.asarray(ALL_result)
    ISImeanmean_result = np.asarray(ISImeanmean_result)
    ISImeanstd_result = np.asarray(ISImeanstd_result)
    ISICVmean_result = np.asarray(ISICVmean_result)
    ISICVstd_result = np.asarray(ISICVstd_result)
    meanCCmean_result = np.asarray(meanCCmean_result)
    stdCCmean_result = np.asarray(stdCCmean_result)
    
    results_list = [PCL23_result, PCL5_result, PC_result, IN_result, 
                    ALL_result, ISImeanmean_result, ISImeanstd_result, 
                    ISICVmean_result, ISICVstd_result, 
                    meanCCmean_result, stdCCmean_result]
    

    order = [4, 5, 7, 9]
    
    results_list2 = [results_list[j] for j in order]
    name_list2 = [name_list[j] for j in order]
    label_list2 = [label_list[j] for j in order]
        
    _fig12(Iexc_arr, Iinh_arr, results_list2, name_list2,
             label_list2, simulation_dir)

    
              
def _fig12(Iexc_arr, Iinh_arr, results_list, name_list,
             label_list, save_dir):
    
    if not os.path.isdir(os.path.join(save_dir, 'Figures')):
        os.mkdir(os.path.join(save_dir, 'Figures'))
        
    fig, axs = plt.subplots(2, 2, figsize=(20,16))
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    
    fig_label = ['(A)', '(B)','(C)', '(D)']
    
    for i in range(len(results_list)):
        ax = axs[i//2, i%2]
        plt.sca(ax)
        label=fig_label[i]
        
        ax.text(-0.2, 0.95, label, transform=ax.transAxes + trans,
            fontsize=24, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
        result = results_list[i]
        name = name_list[i]
        
        label = label_list[i]
        
        z = result.reshape((len(Iinh_arr),len(Iexc_arr))).transpose().flatten()
        X, Y = np.meshgrid(Iinh_arr, Iexc_arr)
        Z = z.reshape(len(Iinh_arr), len(Iexc_arr))

        extent = [Iinh_arr[0], 2*Iinh_arr[-1] - Iinh_arr[-2],
                  Iexc_arr[0], 2*Iexc_arr[-1] - Iexc_arr[-2]]
        
        plt.imshow(Z,origin='lower',interpolation='bilinear', 
                   cmap='Spectral_r', extent=extent)
        cbar = plt.colorbar(fraction=0.046)
        cbar.ax.tick_params(labelsize=20)
        circ = plt.Circle((200,250), radius=10, facecolor=None, 
                          edgecolor='black', linewidth=3, fill=False)
        ax=plt.gca()
        ax.add_patch(circ)
        cbar.set_label(label, fontsize=22, labelpad=20)
        plt.tick_params(labelsize=22)
        plt.xlabel('I$_{inh} (pA)$', fontsize=26)
        plt.ylabel('I$_{exc} (pA)$', fontsize=26)
        
    plt.tight_layout(w_pad=10)
    plt.savefig(os.path.join(save_dir, 'Figures', 'Fig12.png'))
    plt.show()
    
        
if __name__ == '__main__':
    simulation_dir = set_simulation_dir('Results_'+os.path.basename(__file__)[:-3])
    seed = 0
    Iexc_arr = np.arange(0, 600, 25)
    Iinh_arr = np.arange(0, 600, 25)
    duration=13000

    task8(simulation_dir=simulation_dir, Iexc_arr=Iexc_arr, 
        Iinh_arr=Iinh_arr, seed=seed, duration=duration)
    