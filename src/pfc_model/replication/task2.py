import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.ndimage import gaussian_filter1d as gf1d
import os
from pathlib import Path
import seaborn as sns
from pfc_model import *
from pfc_model.analysis import*

__all__ = ['task2']

@time_report()
def task2(simulation_dir, seed=None):
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
    
    scr_path = Path(__file__).parent

    # run rk4
    n_cells=1000
    n_stripes=1
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    method_rk4 = 'rk4'
    dt = 0.05
    transient = 1000   
    
    duration_rk4 =16000
    cortex_rk4 = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                          constant_stimuli=constant_stimuli, method=method_rk4,
                          dt=dt, transient=transient, seed=seed)
    cortex_rk4.save(os.path.join(simulation_dir, 'task1'))

    ALL_rk4 = cortex_rk4.neuron_idcs(['ALL',0])
    cortex_rk4.set_longrun_neuron_monitors('I_tot', 'I_tot', ALL_rk4,  5000, 
                                       start=1000, stop=31000, 
                                       population_agroupate='sum')
    cortex_rk4.set_longrun_neuron_monitors('V', 'V', ALL_rk4,  5000, start=1000,
                                       stop=31000)   
    cortex_rk4.run(duration_rk4)
    # run rk4
    
    # run rk2
    n_cells=1000
    n_stripes=1
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    
    method_rk2 = 'gsl_rk2'
    dt=0.05
    transient=1000   
    
    duration_rk2=16000
                      
    cortex_rk2 = Cortex.load(os.path.join(simulation_dir, 'task1'), 
                             constant_stimuli, method_rk2, dt, 
                             transient=transient)    
    
    ALL_rk2 = cortex_rk2.neuron_idcs(['ALL',0])
    cortex_rk2.set_longrun_neuron_monitors('I_tot', 'I_tot', ALL_rk2,  5000, 
                                       start=1000, stop=31000, 
                                       population_agroupate='sum')
    cortex_rk2.set_longrun_neuron_monitors('V', 'V', ALL_rk2,  5000, 
                                       start=1000, stop=31000)

    cortex_rk2.run(duration_rk2)
    # run rk2
    
    
    # original
    with open(os.path.join(scr_path, 'Original_spiketime.txt'), 'r') as f:
        text = f.read()
    
    spike_trains_rk2 = [np.asarray(train.split(',')).astype(float) 
                    for train in text.split(';')]
    # original
    
    if not os.path.isdir(os.path.join(simulation_dir, 
                                      'Reports', 'Param_comparisons')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'Param_comparisons')) 
    
    # text output#
    aliases = ['PC', 'fast-spiking cells','bitufted cells', 'basket cells', 
               'Matinotti cells']
    get_membr_params(cortex_rk4,
                     [('PC',0), ('IN_L_both',0), ('IN_CL_both',0), 
                      ('IN_CC',0), ('IN_F',0)], 
                     alias_list = aliases, 
                     file=os.path.join(simulation_dir, 'Reports', 
                                       'Param_comparisons',
                                       'membr_params.txt'))
    
    get_spiking(cortex_rk4, 0.33, ('PC', 0), 
                file=os.path.join(simulation_dir, 'Reports', 
                                  'Param_comparisons', 'spiking.txt'))
    
    comp_membrparam_rategroup(
        cortex_rk4, 0.33, [('PC_L23',0), ('PC_L5', 0), ('PC', 0), ('ALL',0)], 
        file=os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                          'membr_params_comparison.txt'))
    
    for channel in cortex_rk4.network.basics.syn.channels.names:
        contingency(
            cortex_rk4, 0.33, 
            [('PC_L23',0), ('PC_L5', 0), ('PC', 0), ('ALL',0)], ('ALL', 0), 
            channel=channel,
            file=os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                              'pcon_contingency_{}.txt'.format(channel)))
        
        comp_synparam_rategroup(
            cortex_rk4, 0.33, [('PC_L23',0), ('PC_L5', 0), ('PC', 0), ('ALL',0)], 
            ('ALL', 0), channel=channel, 
            file=os.path.join(simulation_dir, 'Reports', 'Param_comparisons'
                              'syn_params_comparison_{}.txt'.format(channel)))
    
    # text output\\
    
    
    # -> raster plots
    _fig01(cortex_rk4, cortex_rk2, (duration_rk4-6000, duration_rk4), simulation_dir)
    
    # raster plot \\

    if not os.path.isdir(os.path.join(simulation_dir, 
                                      'Reports', 'ISI_analysis')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'ISI_analysis'))       

    #  -> ISI mean, CV and crossC
    spikingPC_rk4 = cortex_rk4.spiking_idcs((np.greater_equal, 0.33), ('PC', 0))
    spikingALL_rk4= cortex_rk4.spiking_idcs((np.greater_equal, 0.33), ('ALL', 0))
    
    lags_rk4 = np.arange(50)
    tlim=(transient, transient+15000)
    
    (C_lags_rk4, autoC_mean_rk4, autoC_std_rk4, crossC_mean_rk4,
     crossC_std_rk4) = get_correlations(
         cortex_rk4, idcs=spikingALL_rk4, tlim=tlim, delta_t=2, lags=lags_rk4,
         file=os.path.join(simulation_dir, 'Reports', 'ISI_analysis', 
                           'Correlations_rk4.txt'), 
         display=True, display_interval=5)
         
    ISImean_rk4, ISICV_rk4 = get_ISI_stats(
        cortex_rk4, neuron_idcs=spikingALL_rk4, tlim=tlim, 
        file=os.path.join(simulation_dir, 'Reports', 'ISI_analysis', 
                          'ISIstats_rk4.txt'))
    
    
    spikingALL_rk2= cortex_rk2.spiking_idcs((np.greater_equal, 0.33), ('ALL', 0))
    
    lags_rk2 = np.arange(50)
    tlim=(transient, transient+15000)

    (C_lags_rk2, autoC_mean_rk2, autoC_std_rk2, crossC_mean_rk2, 
     crossC_std) = get_correlations(
         cortex_rk2, idcs=spikingALL_rk2, tlim=tlim, delta_t=2, lags=lags_rk2, 
         file=os.path.join(simulation_dir, 'Reports', 'ISI_analysis', 
                           'Correlations_rk2.txt'), 
         display=True, 
         display_interval=5)
    ISImean_rk2, ISICV_rk2 = get_ISI_stats(
        cortex_rk2, neuron_idcs=spikingALL_rk2, tlim=tlim, 
        file=os.path.join(simulation_dir, 'Reports', 'ISI_analysis', 
                          'ISIstats_rk2.txt'))
    Correlation_sigma_rk2 = 1  
    
    sp_train =[]
    tlim=(1000,16000)
    t0, t1 = tlim
    minfq = 0.33
    mincount = (t1-t0)/1000 * minfq
    for train in spike_trains_rk2:   
        if len(train[(train>=t0) & (train<t1)])>=mincount:
            sp_train.append(train)
    spike_trains_rk2 = sp_train   
  
    tlim = (1000, 16000)    
        
    ISImean_orig, ISICV_orig = get_ISI_stats_from_spike_trains(
        spike_trains_rk2, tlim=tlim, 
        file=os.path.join(simulation_dir, 'Reports', 'ISI_analysis', 
                          'ISIstats_original.txt'))
    
    lags = np.arange(50)
    
    (C_lags_orig, autoC_mean_orig, autoC_std_orig,crossC_mean_orig, 
     crossC_std_orig) = get_correlations_from_spike_trains(
         spike_trains_rk2, tlim=tlim, delta_t=2, lags=lags, 
         file=os.path.join(simulation_dir, 'Reports', 'ISI_analysis', 
                           'Correlations_original.txt'), 
         display=True, display_interval=5)  
    
    Correlation_sigma_orig = 1

    ISImean_list = [ISImean_rk4, ISImean_rk2, ISImean_orig]
    ISICV_list = [ISICV_rk4, ISICV_rk2, ISICV_orig]
    crossC_mean_list = [crossC_mean_rk4, crossC_mean_rk2, crossC_mean_orig]
    C_lags = np.arange(50)
    correlation_sigma = 1
    
    _fig02(ISImean_list, ISICV_list, crossC_mean_list, C_lags,
          correlation_sigma, simulation_dir)
    
    # ISI mean, CV and crossC \\
    
        
    # -> autoC
    
    autoC_mean_list = [autoC_mean_rk4, autoC_mean_rk2, autoC_mean_orig]
    
    _fig03(autoC_mean_list, C_lags, simulation_dir)
    
    # autoC \\
    
    # -> LFP
    
    fq_rk4, pwr_rk4 = get_LFP_SPD(cortex_rk4, log=True, sigma=2)
    
    fq_rk2, pwr_rk2 = get_LFP_SPD(cortex_rk2, log=True, sigma=2)
    
    time_orig = np.load(os.path.join(scr_path, 'Original_t.npy'))
    Itot_orig = np.load(os.path.join(scr_path, 'Original_I.npy'))
    
    time_bins_orig = np.floor(time_orig*20).astype(int)
    time_bins_orig, unique_idc_orig = np.unique(time_bins_orig, return_index=True)
    
    Itot_orig = Itot_orig[unique_idc_orig]
    frequency_orig = 1000/0.05
    fq_orig, pwr_orig = get_LFP_SPD_from_Itotarray(Itot_orig, frequency_orig, log=True, sigma=2)
    
    
    fq_list = [fq_rk4, fq_rk2, fq_orig]
    pwr_list = [pwr_rk4, pwr_rk2, pwr_orig]
    
    _fig04(fq_list, pwr_list, simulation_dir)
    

    
    # LFP \\    
    
    
    # -> V
    
    ALL_rk4 = cortex_rk4.neuron_idcs(('ALL', 0))
    PC_rk4 = cortex_rk4.neuron_idcs(('PC', 0))
    
    PCnonzero_rk4 = cortex_rk4.spiking_idcs((np.greater, 0), ('PC', 0))
    ALLnonzero_rk4= cortex_rk4.spiking_idcs((np.greater, 0), ('ALL', 0))
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports', 'Vstats')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'Vstats'))
    
    
    VstdALL_rk4 = get_V_stats(
        cortex_rk4, ALLnonzero_rk4, 
        file=os.path.join(simulation_dir, 'Reports', 'Vstats', 
                          'VALLstats_rk4.txt'))[1]
    VstdPC_rk4 =get_V_stats(
        cortex_rk4, PCnonzero_rk4, 
        file=os.path.join(simulation_dir, 'Reports', 'Vstats',
                          'VPCstats_rk4.txt'))[1]
    _fig05(VstdALL_rk4, VstdPC_rk4, simulation_dir)
    # V \\
    


def _fig01(cortex_rk4, cortex_rk2, tlim, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    fig, axs = plt.subplots(2, 1, figsize=(14, 20))
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax=axs[0]
    ax.text(-0.12, 0.88, '(A)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    plt.sca(axs[0])
    raster_plot(cortex_rk4, tlim=tlim, show=False, newfigure=False)
    plt.sca(axs[1])
    ax=axs[1]
    ax.text(-0.12, 0.88, '(B)', transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))   
    
    raster_plot(cortex_rk2, tlim=tlim, show=False, newfigure=False)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'Figures', 'Fig01.png'))   
    
    
def _fig02(ISImean_list, ISICV_list, crossC_mean_list, C_lags,
            correlation_sigma, path):
    
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

  
    labels = ['(A)', '(B)', '(C)']

    
    _ISIfigures(ISImean_list, ISICV_list, crossC_mean_list, C_lags,
                 correlation_sigma, axs, fig, labels)

    
    plt.tight_layout()
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path,'Figures'))
    plt.savefig(os.path.join(path, 'Figures', 'Fig02.png'))
    
    
def _fig03(autoC_mean_list, C_lags, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    
    labels = ['(A)', '(B)']
   
    fig, axs = plt.subplots(1, 2, figsize=(18, 14))
    _autoCfigures(autoC_mean_list, C_lags,  axs, fig, labels)
   
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'Figures', 'Fig03.png'))
    
    
def _fig04(frequency_list, power_list, path):
    
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    
    plt.figure(figsize=(18,10))
        
    _SPDfigures(frequency_list, power_list)
   
    plt.savefig(os.path.join(path, 'Figures', 'Fig04.png'))
    
 
def _fig05(VstdALL, VstdPC, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
     
    plt.figure(figsize=(12,10))
    plt.xlabel('V standard deviation (mV)', fontsize=26)
    plt.ylabel('density', fontsize=26)
    # h = ax.hist(VstdPC, bins=int(round(np.sqrt(len(VstdPC)), 0)))
    sns.kdeplot(VstdPC, clip=[0, np.NaN])
    plt.tick_params(labelsize=26)
    plt.locator_params(axis='both', nbins=4)
    plt.xlim(0, 8)

    plt.tight_layout()
    plt.savefig(os.path.join(path,'Figures','Fig05.png'))

    
def _ISIfigures(ISImean_list, ISICV_list, crossC_mean_list, C_lags, 
                 correlation_sigma, axs, fig, labels):
        
    ax0, ax1, ax2 = axs
    lab0, lab1, lab2 = labels
    
    fig.subplots_adjust(left=0.08,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.35, 
                        hspace=0.1)
    
    plt.sca(ax0)
    
    leg = ['rk4', 'gsl_rk2', 'original']
    for i in range(len(ISImean_list)):
        ISImean = ISImean_list[i]
        sns.kdeplot(np.asarray(ISImean)/1000, clip=[0, 3])
    plt.xlim(-0.05, 2.5)

    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax=ax0
    ax.text(0.8, 1, lab0, transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
    
    plt.xticks([0, 1, 2], fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel('t (ms)', fontsize=26, labelpad=15)
    plt.ylabel('density', fontsize=26, labelpad=15)
    plt.legend(labels=leg, fontsize=18, loc=(0.55, 0.75))
    
    plt.sca(ax1)
    for ISICV in ISICV_list:
        sns.kdeplot(ISICV, clip=[0, np.NaN])
    plt.xticks([0, 2, 4], fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel('ISI CV', fontsize=26, labelpad=15)
    plt.ylabel('density', fontsize=26, labelpad=15)
    
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax=ax1
    ax.text(0.8, 1, lab1, transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
    plt.legend(labels=leg, fontsize=18, loc=(0.55, 0.75))
    
    plt.sca(ax2)
    
    for crossC_mean in crossC_mean_list:
        C_lags0 = C_lags[C_lags>=0]
        crossC_mean0 = crossC_mean[C_lags>=0]
        
        C_lags1=C_lags0[-1:0:-1]
        C_lags2 = np.concatenate(( -C_lags1, C_lags0))
        
        crossC_mean1 = crossC_mean0[-1:0:-1]
        crossC_mean2 = np.concatenate((crossC_mean1, crossC_mean0))
        
        C_lags2 = C_lags2/1000
        
        if correlation_sigma>0:
            ax2.plot(C_lags2, gf1d(crossC_mean2,correlation_sigma))
        else:
            ax2.plot(C_lags2, crossC_mean2)
            
    ax2.set_xlim(-0.030, 0.030)
    
    plt.xticks([-0.015, 0, 0.015], [-0.015, '0', 0.015], fontsize=26)
    yt1 = np.asarray([0, 0.001])
    plt.yticks(yt1,['0', '10$^{-3}$'], fontsize=26)
    plt.xlabel('lag (s)', fontsize=26, labelpad=15)
    
    ax2.text(-0.04, 0.5*0.001, 'cross-correlation', va='center', 
             rotation='vertical', fontsize=26)
    plt.legend(labels=leg, fontsize=18, loc=(0.05, 0.83))
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax=ax2
    ax.text(0.8, 1, lab2, transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
   
    
def _autoCfigures(autoC_mean_list, C_lags, axs, fig, labels, xmax=50):
    
    ax0, ax1 = axs
    lab0, lab1 = labels
    
    
    plt.sca(ax0)
    for autoC_mean in autoC_mean_list:
        plt.plot(C_lags, autoC_mean)
    # ax0.set_xlim(-10, xmax)
    plt.xlabel('lag (ms)', fontsize=26)
    plt.ylabel('autocorrelation', fontsize=26)
    plt.tick_params(labelsize=26)
    plt.xticks([0, 25, 50], fontsize=26)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax=ax0
    ax.text(0.8, 1, lab0, transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
    plt.legend(fontsize=24, loc=(0.4, 0.85), labels=['rk4', 'gsl_rk2', 'original'])
    
    
    plt.sca(ax1)
    for autoC_mean in autoC_mean_list:
        plt.plot(C_lags, autoC_mean)
    
    plt.ylim(-0.1, 0.1)
    plt.xlabel('lag (ms)', fontsize=26)
    plt.ylabel('autocorrelation', fontsize=26)
    plt.xticks([0, 25, 50], fontsize=26)
    plt.tick_params(labelsize=26)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax=ax1
    ax.text(0.8, 1, lab1, transform=ax.transAxes + trans,
        fontsize=24, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
    
    plt.legend(fontsize=24, loc=(0.4, 0.85), labels=['rk4', 'gsl_rk2', 'original'])
    
    
def _SPDfigures(frequency_list, power_list):
    
    for i in range(len(frequency_list)):    
        plt.plot(frequency_list[i], power_list[i])
        
    plt.xlim(2.5, 7)
    plt.ylim(-2, 18)
    plt.vlines(np.log(60), 6, 15, linestyle='--', color='black')
    plt.plot([2, np.log(60)], [13.5 + (np.log(60)-2), 13.5], 
            linestyle='--', color='blue')
    plt.plot([np.log(60), 7], [12.5, 12.5 - 2*(7-np.log(60))],
            linestyle='--', color='blue')
    plt.plot([np.log(60), 7], [8, 8 - 3*(7-np.log(60))],
            linestyle='--', color='blue')
    plt.xlabel('log(Frequency) (log[Hz])', fontsize=26)
    plt.ylabel('log(Power) (arbitrary unit)', fontsize=26)
    ax = plt.gca()
    plt.yticks([0, 4, 8, 12, 16], fontsize=26)
    plt.xticks([3, 4, 5, 6, 7], fontsize=26)
    ax.text( 3, 7, '60 Hz', fontsize=26)
    ax.arrow(3.6, 7.5, 0.4, 1, head_width=0.1)
    ax.text(3.25, 15.5, '1/f', fontsize=26)
    ax.text(5.5, 11, '$1/f^2$', fontsize=26)
    ax.text(5.5, 1, '$1/f^3$', fontsize=26)
    plt.legend(fontsize=20, labels=['rk4', 'gsl_rk2', 'original'])

if __name__ == '__main__':
    simulation_dir = set_simulation_dir()
    seed=0
    task2(simulation_dir=simulation_dir, seed=seed)