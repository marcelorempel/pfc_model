"""This script contains functions for network analysis."""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import brian2 as br2
from scipy.stats import ttest_ind as ttest
from scipy.stats import mannwhitneyu as mwtest
from scipy.stats import chi2_contingency as chi2
from collections import namedtuple
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.signal import periodogram
# from .._auxiliary import time_report
from _auxiliary import time_report

__all__= [
    'raster_plot_simple', 'raster_plot', 'get_V_stats', 'get_correlations', 
    'get_correlations_from_spike_trains', 'ISIcorrelations', 
    'show_correlations', 'get_LFP', 'get_LFP_SPD',
    'get_LFP_SPD_from_Itotarray', 'get_ISI_stats',
    'get_ISI_stats_from_spike_trains', 'ISIstats', 'comp_membrparam_rategroup', 
    'contingency', 'comp_synparam_rategroup', 'get_spiking', 
    'get_membr_params',  'binned_spiking', 'spike_stats', 'w_null', 
    'w_null_boundaries',
    ]

def raster_plot_simple(cortex, tlim=None, figsize=(18,10), s=3,
                       fontsize=24, labelsize=20, savefig=None, show=True):
    """Get a raster plot from a PFC model after simulation.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.
    tlim: tuple, optional
        2-tuple (initial t, final t) of inital and final values (in ms)
        of the horizontal axis. If not given, no limit is set.
    figsize: tuple, optional
        2-tuple defining figure size. If not given, it defaults to
        (18, 10).
    s: int or float, optional
        Size of dots. If not given, it defaults to 3.
    fontsize: int or float, optional
        Font-size. If not given, it defaults to 24.
    labelsize: int or float, optional
        Font-size in ticks. If not given, it defaults to 20.
    savefig: str, optional
        Name of file where the raster plot is to be saved. If not
        given, the raster plot is not saved.
    show: bool, optional
        Whether the raster plot is to be shown and closed. If false,
        it allows the figure to be further edited using pyplot. If
        not given, it defaults to True.
    """
    
    plt.figure(figsize=figsize)
    plt.scatter(cortex.spikemonitor.t_*1000, cortex.spikemonitor.i, s=s)
    
    plt.xlabel('time (ms)', fontsize=fontsize)
    plt.ylabel('neuron index', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    if tlim is not None:
        plt.xlim(tlim)
    
    if savefig is not None:
        plt.savefig(savefig)
     
    if show:
        plt.show()
        plt.close()

    
def raster_plot(cortex, tlim=None, figsize=(18,10), s=5, layerpadding=15,
                fontsize=24, labelsize=20, lw=3, savefig=None, show=True,
                newfigure=True):
    """Get a customized raster plot from a PFC model after simulation.
    The model must contain only 1 stripe. Dots representing PC are blue 
    and dots representing IN are red. Neurons in layer L2/3 and L5 are 
    separated by a horizontal line.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.
    tlim: tuple, optional
        2-tuple (initial t, final t) of inital and final values (in ms)
        of the horizontal axis. If not given, the limit is set from
        transient to the end of simulation.
    figsize: tuple, optional
        2-tuple defining figure size. If not given, it defaults to
        (18, 10).
    s: int or float, optional
        Size of dots. If not given, it defaults to 5.
    layerpadding: int or float, optional
        Space between L2/3 and L5. If not given, it default to 15.
    fontsize: int or float, optional
        Font-size. If not given, it defaults to 24.
    labelsize: int or float, optional
        Font-size in ticks. If not given, it defaults to 20.
    lw: int or float, optional
        Width of the line separating L2/3 and L5.
    savefig: str, optional
        Name of file where the raster plot is to be saved. If not
        given, the raster plot is not saved.
    show: bool, optional
        Whether the raster plot is to be shown and closed. If false,
        it allows the figure to be further edited using pyplot. If
        not given, it defaults to True.
    """
    
    sp_t = cortex.spikemonitor.t_*1000
    sp_i = cortex.spikemonitor.i
    
    PC_L23idc = cortex.neuron_idcs(('PC_L23',0))
    PC_L23_isin = np.isin(sp_i, PC_L23idc)
    PC_L23_i = sp_i[PC_L23_isin]
    PC_L23_t = sp_t[PC_L23_isin]
    
    PC_L5idc = cortex.neuron_idcs(('PC_L5',0))
    PC_L5_isin = np.isin(sp_i, PC_L5idc)
    PC_L5_i = sp_i[PC_L5_isin]
    PC_L5_t = sp_t[PC_L5_isin]
    
    IN_L23idc = cortex.neuron_idcs(('IN_L23',0))
    IN_L23_isin = np.isin(sp_i, IN_L23idc)
    IN_L23_i = sp_i[IN_L23_isin]
    IN_L23_t = sp_t[IN_L23_isin]
    
    IN_L5idc = cortex.neuron_idcs(('IN_L5',0))
    IN_L5_isin = np.isin(sp_i, IN_L5idc)
    IN_L5_i = sp_i[IN_L5_isin]
    IN_L5_t = sp_t[IN_L5_isin]
      
    if newfigure:
        plt.figure(figsize=figsize)
    plt.scatter(PC_L23_t, PC_L23_i, s=s, color='blue')
    plt.scatter(IN_L23_t, IN_L23_i, s=s, color='red')
    plt.scatter(PC_L5_t, PC_L5_i+layerpadding, s=s, color='blue')
    plt.scatter(IN_L5_t, IN_L5_i+layerpadding, s=s, color='red')
 
    if tlim is None:
        tlim = cortex.transient, cortex.neurons.t/br2.ms
    plt.xlim(tlim)

    x0, x1 = plt.gca().get_xlim()
    y0, y1 = plt.gca().get_ylim()
    division_idc = cortex.network.group_distr[0][7][0] + layerpadding/2
    plt.hlines(division_idc, x0, x1, color='black', lw=lw)

    plt.xlabel('time (ms)', fontsize=fontsize)
    plt.ylabel('neuron index', fontsize=fontsize)
    
    plt.tick_params(labelsize=labelsize)

    yticks_space = cortex.network.basics.struct.n_cells//100 * 20
    yticks = np.arange(5*yticks_space + 1)
    ytickspadding = yticks.copy()
    ytickspadding[yticks > division_idc] = (
        ytickspadding[yticks > division_idc] + layerpadding
        )
    if savefig is not None:
        plt.savefig(savefig)
    
    if show:
        plt.show()
        plt.close()


@time_report()
def get_V_stats(cortex, neuron_idcs=None, tlim=None, file=None):
    """Get statistical measures of V. 
    
    This function also analysis V after extracting pre-spikes periods.
    Pre-spike period is defined here as the period starting when V
    crosses V_T immediately before a spike and ending in the spike
    instant.
    
    V recording must be defined in neuron_monitors, otherwise this 
    function raises an exception.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.   
    neuron_idcs: array_like, optional
        Indices of neurons that are to be analysed. If not given, all
        neurons are analysed.
    tlim: tuple, optional
        2-tuple (initial t, final t in ms) indicating window of
        analysis. If not given, the whole recorded period will be
        analysed.
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
        
    Returns
    -------
    This function returns a 6-tuple (out1, ..., out6).
    out1: list
        Mean V. Each value is the mean value of V of a neuron along 
        the requested window of analysis.
    out2: list
        Standard deviation of V. Each value is the standard deviation
        of V of a neuron along the requested window of analysis.
    out3: list
        Mean difference between V and V_T. Each value is the mean value
        of V - V_T of a neuron along the requested window of analysis.
    out4: list
        Mean V after extracting pre-spike periods. Each value is the 
        mean value of V of a neuron along the requested window of 
        analysis.
    out5: list
        Standard deviation of V after extracting pre-spike periods. 
        Each value is the standard deviation of V of a neuron along the 
        requested window of analysis.
    out6: list
        Mean difference between V and V_T after extracting pre-spike 
        periods. Each value is the mean value of V - V_T of a neuron 
        along the requested window of analysis.
        
    Raises
    ------
    AttributeError: when V is not recorded.
    """
    def extract_spikes(Varr, V_T, spike_jump=1):
        
        def get_postspike_positions(Varr, spike_jump=1): 
    
            Varr_diff = np.diff(np.abs(Varr))
            sppos = np.where(Varr_diff >= spike_jump)[0]
            
            return sppos+1
    
        def get_interspikes(Varr, post_spikes):
            interspikes = []
            i0 = 0
            for i1 in post_spikes:
                interspikes.append(Varr[i0:i1])
                i0=i1
            else:
                interspikes.append(Varr[i0:])
            return interspikes
    
        def extract_prespike(interspikes, V_T):
            for i in range(len(interspikes)-1):
                try:
                    V_T_crossing = np.max(np.where(interspikes[i] < V_T)[0])
                    interspikes[i] = (interspikes[i][:V_T_crossing + 1])
                except ValueError:
                    interspikes[i] = np.asarray([])
    
        post_spikes = get_postspike_positions(Varr, spike_jump=1)
        interspikes = get_interspikes(Varr, post_spikes)
        extract_prespike(interspikes, V_T)
        
        return np.concatenate(interspikes)
    
    if 'V' not in cortex.recorded._keys():
        raise AttributeError('V was not recorded.')
        
    V = cortex.recorded.V.V/br2.mV
    if neuron_idcs is not None:
        V = V[neuron_idcs]
    else:
        neuron_idcs = np.arange(cortex.neurons.N)
    
    if tlim is not None:
        t0,t1=tlim
        t = cortex.recorded.V.t/br2.ms
        V = V[:, (t>=t0) & (t<t1)]
    
    Vmean = []
    Vstd = []
    V_minus_V_T_mean = []
    Vmean_extracted = []
    Vstd_extracted = []
    V_minus_V_T_mean_extracted = []
     
    for i in range(V.shape[0]):
        Varr = V[i]
        V_T = cortex.network.membr_params.loc[
            dict(par='V_T', cell_index=neuron_idcs[i])
            ].values
        V_extracted = extract_spikes(Varr, V_T)
        
        Vmean.append(np.mean(Varr))
        Vstd.append(np.std(Varr))
        V_minus_V_T_mean.append(np.mean(Varr-V_T))
        Vmean_extracted.append(np.mean(V_extracted))   
        Vstd_extracted.append(np.std(V_extracted))
        V_minus_V_T_mean_extracted.append(np.mean(V_extracted - V_T))
                
    if file is not None:
        with open(file, 'w') as f:
            print('V correlations', file=f)
            print('{} cells'.format(len(neuron_idcs)), end='\n\n', file=f)
            print('Full V arrays', file=f)
            print('Vmean --> mean: {:.2f} mV,'
                  ' std: {:.2f} ''mV'.format(
                      np.mean(Vmean), np.std(Vmean)
                      ), file=f)
            print('Vstd --> mean: {:.2f} mV,'
                  ' std: {:.2f} mV'.format(
                      np.mean(Vstd), np.std(Vstd)
                      ), file=f)
            print('V minus V_T mean --> mean: {:.2f} mV,'
                  ' std: {:.2f} mV'.format(
                      np.mean(V_minus_V_T_mean), np.std(V_minus_V_T_mean)
                      ), file=f, end='\n\n')
            print('V arrays after spike extraction', file=f)
            print('Vmean_extracted --> mean: {:.2f} mV,'
                  ' std: {:.2f} mV'.format(
                      np.mean(Vmean_extracted), np.std(Vmean_extracted)
                      ), file=f)
            print('Vstd_extracted --> mean: {:.2f} mV,'
                  ' std: {:.2f} mV'.format(
                      np.mean(Vstd_extracted), np.std(Vstd_extracted)
                      ), file=f)
            print('V minus V_T mean extracted --> mean: {:.2f} mV,'
                  ' std: {:.2f} mV'.format(
                      np.mean(V_minus_V_T_mean_extracted), 
                      np.std(V_minus_V_T_mean_extracted)
                      ), file=f, end='\n\n')
    
    return (Vmean, Vstd, V_minus_V_T_mean, Vmean_extracted, Vstd_extracted,
            V_minus_V_T_mean_extracted)
  

@time_report('Correlations')
def get_correlations(cortex, tlim=None,  idcs=None, delta_t=2, lags=0, 
                     file=None, display=False, display_interval=10):
    """Get populational mean of Pearson ISI correlations.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.   
    tlim: tuple, optional
        2-tuple (initial t, final t) of inital and final values (in ms)
        of the horizontal axis. If not given, the limit is set from
        transient to the end of simulation.
    idcs: array_like, optional
        Indices of neurons that are to be analysed. If not given, all
        neurons are analysed.
    delta_t: int or float, optional
        Length of bins (in ms) for correlation analysis. If not given, 
        it defaults to 2.
    lags: int, float or list[int, float], optional
        Value or list of values of requested lags (in ms). If not given,
        it defaults to 0.
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    display: bool, optional
        Whether progress reports are tobe displayed. If not given, it
        defaults to False.
    display_interval: int or float, optional
        Interval (in percentage of progress) between progress reports
        (if display=False, it has no effect). If not given. it defaults
        to 10.
    
    Returns
    -------
    This function returns a 5-tuple:
    out1: array
        Array with lag values (in ms).
    out2: array
        mean values of autocorrelation for each lag in out1.
    out3: array 
        standard deviation of autocorrelation values for each lag
        in out1.
    out4: array 
        mean values of cross-correlation for each lag in out1.
    out5: array 
        standard deviation of cross-correlation values for each lag
        in out1.
    """
    
    def get_binned_spiking_trains(cortex, idcs, tlim, delta_t):
        if idcs is None:
            idcs = np.arange(cortex.neurons.N)
        elif isinstance(idcs, int):
            idcs = [idcs]
        
        if tlim is None:
            tlim = (cortex.transient, cortex.neurons.t/br2.ms)
        
        t0, t1 = tlim
        t1 = np.ceil(t1).astype(int)
        bin_arr = np.zeros((len(idcs), (t1-t0)//delta_t))
        
        spike_trains = cortex.spikemonitor.spike_trains()
        for i in range(len(idcs)):
            train = spike_trains[idcs[i]]/br2.ms
            train = train[(train>=t0)&(train<t1)]
            idc_arr =((train-t0)//delta_t).astype(int)
            bin_arr[i, idc_arr] = 1
            
        return bin_arr.astype(int)
        
    
    bin_arr = get_binned_spiking_trains(cortex, idcs, tlim, delta_t)
                       
    return ISIcorrelations(
        bin_arr, delta_t, lags, file, display, display_interval
        )

@time_report('original spike trains correlation')
def get_correlations_from_spike_trains(spike_trains, tlim, idcs=None, 
                                       delta_t=2, lags=0, file=None, 
                                       display=False, display_interval=10):
    """Get populational mean of Pearson ISI correlations.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    spike_trains: list[list]
       A list of spike trains. Each spike train represents a different
       neuron and consists of a list with spike instants (in ms).
    tlim: tuple
        2-tuple (initial t, final t) of inital and final values (in ms)
        of the horizontal axis. If not given, the limit is set from
        transient to the end of simulation.
    idcs: array_like, optional
        Indices of neurons that are to be analysed. If not given, all
        neurons are analysed.
    delta_t: int or float, optional
        Length of bins (in ms) for correlation analysis. If not given, 
        it defaults to 2.
    lags: int, float or list[int, float], optional
        Value or list of values of requested lags (in ms).
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    display: bool, optional
        Whether progress reports are tobe displayed. If not given, it
        defaults to False.
    display_interval: int or float, optional
        Interval (in percentage of progress) between progress reports
        (if display=False, it has no effect). If not given. it defaults
        to 10.
    
    Returns
    -------
    This function returns a 5-tuple:
    out1: array
        Array with lag values (in ms).
    out2: array
        mean values of autocorrelation for each lag in out1.
    out3: array 
        standard deviation of autocorrelation values for each lag
        in out1.
    out4: array 
        mean values of cross-correlation for each lag in out1.
    out5: array 
        standard deviation of cross-correlation values for each lag
        in out1.
    """
    def get_binned_spiking_trains(spike_trains, idcs, tlim, delta_t):
        if idcs is None:
            idcs = np.arange(len(spike_trains))
        elif isinstance(idcs, int):
            idcs = [idcs]
             
        t0,t1= tlim
        t1 = np.ceil(t1).astype(int)
        bin_arr = np.zeros((len(idcs), (t1-t0)//delta_t))
        
        for i in range(len(idcs)):
            train = spike_trains[idcs[i]]
            train = train[(train>=t0)&(train<t1)]
            idc_arr =((train-t0)//delta_t).astype(int)
            bin_arr[i, idc_arr] = 1
            
        return bin_arr.astype(int)
        
    bin_arr = get_binned_spiking_trains(spike_trains, idcs, tlim, delta_t)
                       
    return ISIcorrelations(bin_arr, delta_t, lags, file, display, 
                           display_interval)

def ISIcorrelations(bin_arr, delta_t, lags, file=None, display=False, 
                    display_interval=10):
    """Get populational mean of Pearson ISI correlations.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    bin_arr: array
       A 2d-array representing the binned spike trains. Axis 0 must
       specify neurons and axis 1 must specify time bins; 0 represents
       no spike and 1 represents spike in the bin.
    delta_t: int or float, optional
        Length of bins (in ms) for correlation analysis. If not given, 
        it defaults to 2.
    lags: int, float or list[int, float], optional
        Value or list of values of requested lags (in ms).
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    display: bool, optional
        Whether progress reports are tobe displayed. If not given, it
        defaults to False.
    display_interval: int or float, optional
        Interval (in percentage of progress) between progress reports
        (if display=False, it has no effect). If not given. it defaults
        to 10.
    
    Returns
    -------
    This function returns a 5-tuple:
    out1: array
        Array with lag values (in ms).
    out2: array
        mean values of autocorrelation for each lag in out1.
    out3: array 
        standard deviation of autocorrelation values for each lag
        in out1.
    out4: array 
        mean values of cross-correlation for each lag in out1.
    out5: array 
        standard deviation of cross-correlation values for each lag
        in out1.
    """
    
    def group_correlations(bin_df, lag=0):
        
        def df_shifted(df, target, lag=0):
            cross = []
            auto = []
            for c in df.columns:       
                if c==target:
                    auto.append(df[target].corr(df[c].shift(periods=lag)))
                else:
                    cross.append(df[target].corr(df[c].shift(periods=lag)))

            return auto, cross
        
        auto = []
        cross = []
        for idc in range(bin_df.shape[1]):
            auto_idc, cross_idc = df_shifted(bin_df, idc, lag)
            auto.extend(auto_idc)
            cross.extend(cross_idc)
        return auto, cross

    if isinstance(lags, int):
        lags = [lags]
        

    bin_df = pd.DataFrame(bin_arr.transpose())
    auto_mean = []
    auto_std = []
    cross_mean = []
    cross_std = []
    curr = 0
    last = 0
    for l in lags:
        if display and curr==0 and last==0:
            print('Calculating correlations ...')
        auto, cross = group_correlations(bin_df, l)
        auto_mean.append(np.mean(auto))
        auto_std.append(np.std(auto))
        cross_mean.append(np.mean(cross))
        cross_std.append(np.std(cross))
        if display:
            curr += 100/len(lags)
            if curr>=display_interval:
                last += curr//display_interval
                curr=curr%display_interval
                print('{:.1f} % done'.format(last*display_interval+curr))
    
    lags = np.asarray(lags)*delta_t
    
    auto_mean = np.asarray(auto_mean)
    auto_std = np.asarray(auto_std)
    cross_mean = np.asarray(cross_mean)
    cross_std = np.asarray(cross_std)
    
    if file is not None:
        auto_zip = list(zip(lags, auto_mean, auto_std))
        cross_zip = list(zip(lags, cross_mean, cross_std))
        with open(file, 'w') as f:
            print('Auto-correlations:', file=f, end='\n\n')
            for lag in auto_zip:
                print('lag {} ms --> mean: {:.4f},'
                      ' std: {:.4f}'.format(*lag), file=f)
            print(file=f)
            print('Auto-correlations:', file=f, end='\n\n')
            for lag in cross_zip:
                print('lag {} ms --> mean: {:.4f},'
                      ' std: {:.4f}'.format(*lag), file=f)
                       
    return lags, auto_mean, auto_std, cross_mean, cross_std

def show_correlations(cortex, lags, savefig=None, savedata=None):
    """A customized ISI correlation analysis.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.  
    lags: int, float or list[int, float]
        Value or list of values of requested lags (in ms).
    savefig: str, optional
        File name where figures are to be saved. If not given, the
        figures are not saved.
    savedata: str, optional
        File name where results dataframe is to be saved. If not given,
        the dataframe is not saved.
    
    Returns
    -------
    out: dataframe
        Dataframe containing correlation results.
    """
    
    if savefig is not None:
        namesplit = savefig.split('.')
        autofig = namesplit.copy()
        autofig[-2] = autofig[-2]  + '_auto'
        autofig = '.'.join(autofig)
        crossfig = namesplit.copy()
        crossfig[-2] = crossfig[-2] + '_cross'
        crossfig='.'.join(crossfig)
         
    lags, auto_mean, auto_std, cross_mean, cross_std = get_correlations(cortex, lags)

    lag_ms = lags*5
    
    plt.figure(figsize=(18,10))
    plt.plot(lag_ms, auto_mean, lw=3)
    plt.xlabel('lag (ms)', fontsize=24)
    plt.ylabel('autocorrelation', fontsize=24)
    plt.tick_params(labelsize=20)
    if savefig is not None:
        plt.savefig(autofig)
    plt.show()
    plt.close()


    plt.figure(figsize=(18,10))
    plt.plot(lag_ms, cross_mean, lw=3)
    plt.xlabel('lag (ms)', fontsize=24)
    plt.ylabel('crosscorrelation', fontsize=24)
    plt.tick_params(labelsize=20)
    if savefig is not None:
        plt.savefig(crossfig)
    plt.show()
    plt.close()
    
    corr_data = np.array(
        [auto_mean, auto_std, cross_mean, cross_std]
        ).transpose()
    corr_df = pd.DataFrame(
        corr_data, index=lag_ms,
        columns=['auto_mean', 'auto_std', 'cross_mean', 'cross_std']
        )
    corr_df.index.name = 'lag (ms)'
    
    if savedata is not None:
        with open(savedata, 'w') as f:
            print(corr_df, file=f)
            
    return corr_df
                  
def get_LFP(cortex, invert_Itot=True, population_agroupate=None):
    """
    Get LFP estimate.
    
    LFP is estimated here as the sum of all synaptic currents in the
    network.
    
    I_tot recording must be defined in neuron_monitors, otherwise this 
    function raises an exception.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.   
    invert_Itot: bool, optional
        If I_tot values is to be inverted so that LFP descent
        represents positive charge influx and populational 
        excitation.
    population_agroupate: str, optional.
        It specifies how the values should be represented. If 'mean',
        LFP is populational I_tot mean. If 'sum', LFP is populational
        I_tot sum. If not given, the row I_tot array is retrieved.
    
    Returns
    -------
    This function returns a 2-tuple:
    out1: array
        Frequency (in Hz).
    out2: array
        LFP SPD.
    
    Raises
    ------
    AttributeError: when I_tot is not recorded.
    ValueError: when population_agroupate is str but is neither 'sum' nor 'mean'.
    """
    
    if 'I_tot' not in cortex.recorded._keys():
        raise AttributeError('I_tot was not recorded.')
    if (isinstance(population_agroupate, str) 
            and population_agroupate not in ['mean', 'sum']):
        raise ValueError("population_agroupate must be 'sum' or 'mean'")
    
    LFP = cortex.recorded.var('I_tot')/br2.pA
    t_LFP = cortex.recorded.t('I_tot')/br2.ms
    
    if population_agroupate=='mean':
        LFP = np.mean(LFP, axis=0)
    elif population_agroupate=='sum':
        LFP = np.sum(LFP, axis=0)
        
    if invert_Itot:
        LFP = -LFP
        
    return t_LFP, LFP

def get_LFP_SPD(cortex, log=True, sigma=None, population_agroupate=None):
    """ Get LFP spectral power density.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.   
    log: bool, optional
        Whether SPD is to be represented as log-log graph. If not
        given, it defaults to True.
    sigma: int or float, optional
        Standard deviation of 1d-gaussian filter to be applied on SPD. 
        If not given, gaussian filter is not applied.        
    population_agroupate: str, optional.
        It specifies how LFP values should be represented. If 'mean',
        LFP is populational I_tot mean. If 'sum', LFP is populational
        I_tot sum. If not given, the row I_tot array is retrieved.
    
    Returns
    -------
    This function returns a 2-tuple:
    out1: array
        Frequency (in Hz).
    out2: array
        LFP SPD.
    
    Raises
    ------
    AttributeError: when I_tot is not recorded.
    ValueError: when population_agroupate is str but is neither 'sum' 
    nor 'mean'; when sigma <= 0.
    """
    
    if 'I_tot' not in cortex.recorded._keys():
        raise AttributeError('I_tot was not recorded.')
    if (isinstance(population_agroupate, str) 
            and population_agroupate not in ['mean', 'sum']):
        raise ValueError("population_agroupate must be 'sum' or 'mean'")
    if isinstance(sigma, (int, float)) and sigma <= 0:
        raise ValueError('sigma must be greater than 0')
    
    t, LFP = get_LFP(
        cortex, invert_Itot=False, population_agroupate=population_agroupate
        )
    fq = 1000/(t[1]-t[0])
    
    frequency, power = periodogram(LFP, fq)
    
    if log:
        power = np.log(power[frequency>0])
        frequency = np.log(frequency[frequency>0])
       
    if sigma is not None:
        power = gf1d(power, sigma)
    
    return frequency, power
    
def get_LFP_SPD_from_Itotarray(Itotarray, fs, log=True, sigma=None):
    """ Get LFP spectral power density.
    
    Parameters
    ----------
    Itotarray: array_like   
        Array containing LFP estimate.
    fs: int or float
        Frequency of sampling (in Hz).
    log: bool, optional
        Whether SPD is to be represented as log-log graph. If not
        given, it defaults to True.
    sigma: int or float, optional
        Standard deviation of 1d-gaussian filter to be applied on SPD. 
        If not given, gaussian filter is not applied.        
    
    Returns
    -------
    This function returns a 2-tuple:
    out1: array
        Frequency (in Hz).
    out2: array
        LFP SPD.
    
    Raises
    ------
    ValueError: when sigma <= 0.
    """
    
    if isinstance(sigma, (int, float)) and sigma <= 0:
        raise ValueError('sigma must be greater than 0')
        
    Itotarray = np.asarray(Itotarray)
    frequency, power = periodogram(Itotarray, fs)
    
    if log:
        power = np.log(power[frequency>0])
        frequency = np.log(frequency[frequency>0])    
    if sigma is not None:
        power = gf1d(power, sigma)
    
    return frequency, power
    

def get_ISI_stats(cortex, neuron_idcs=None, tlim=None, file=None):
    """Get statistical measures of ISIs.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.   
    neuron_idcs: array_like, optional
        Indices of neurons that are to be analysed. If not given, all
        neurons are analysed.
    tlim: tuple, optional
        2-tuple (initial t, final t in ms) indicating window of
        analysis. If not given, the whole recorded period will be
        analysed.
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    
    Returns
    -------
    This function returns a 2-tuple
    out1: list
        Mean ISI for each requested neuron.
    out2: list
        Coefficient of variation for each requested neuron.
    """
    
    spike_trains = [*cortex.spikemonitor.spike_trains().values()]
    
    if neuron_idcs is None:
        neuron_idcs = np.arange(spike_trains)
    spike_trains = [spike_trains[idc]/br2.ms for idc in neuron_idcs]
    
    if tlim is not None:
        t0, t1 = tlim
        spike_trains = [train[(train>=t0)&(train<t1)] 
                        for train in spike_trains]
     
    return ISIstats(spike_trains, file)
   
 
def get_ISI_stats_from_spike_trains(spike_trains, neuron_idcs=None, tlim=None, 
                                    file=None):
    """Get statistical measures of ISIs.
    
    A summary of results can be saved to a specified file.
    
    Returns
    -------
    spike_trains: list[list]
       A list of spike trains. Each spike train represents a different
       neuron and consists of a list with spike instants (in ms).  
    neuron_idcs: array_like, optional
        Indices of neurons that are to be analysed. If not given, all
        neurons are analysed.
    tlim: tuple, optional
        2-tuple (initial t, final t in ms) indicating window of
        analysis. If not given, the whole recorded period will be
        analysed.
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
        
    Returns
    -------
    This function returns a 2-tuple
    out1: list
        Mean ISI for each requested neuron.
    out2: list
        Coefficient of variation for each requested neuron.
    """
    
    if neuron_idcs is None:
        neuron_idcs = np.arange(len(spike_trains))
    spike_trains = [spike_trains[idc] for idc in neuron_idcs]
    
    
    if tlim is not None:
        t0, t1 = tlim
        spike_trains = [train[(train>=t0)&(train<t1)] 
                        for train in spike_trains]
        
    return ISIstats(spike_trains, file)


def ISIstats(spike_trains, file=None):
    """Get statistical measures of ISIs.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    spike_trains: list[list]
       A list of spike trains. Each spike train represents a different
       neuron and consists of a list with spike instants (in ms).  
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    
    Returns
    -------
    This function returns a 2-tuple
    out1: list
        Mean ISI for each requested neuron.
    out2: list
        Coefficient of variation for each requested neuron.
    """
    
    ISImean = []
    ISICV = []
    
    for train in spike_trains:
        ISI = np.diff(train)
        ISImean.append(np.mean(ISI))
        ISICV.append(np.std(ISI)/np.mean(ISI))
    
    if file is not None:
        with open(file,'w') as f:
            print('ISI stats\n', file=f)
            print('Cells:', len(ISImean), file=f)
            print('ISImean mean: {:.3f} ms'.format(np.mean(ISImean)), file=f)
            print('ISImean std: {:.3f}ms'.format(np.std(ISImean)), file=f)
            print('ISICV mean: {:.3f}'.format(np.mean(ISICV)), file=f)
            print('ISICV std: {:.3f}'.format(np.std(ISICV)), file=f)
        
    return ISImean, ISICV



def comp_membrparam_rategroup(cortex, rate, groupstripe_list, file=None):
    """Get comparisons between membrane parameters of neurons with 
    spiking rate less than and greater or equal to a requested rate 
    value. The comparisons are carried through Mann-Whitney's U and 
    Student's t tests.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance. 
    rate: int or float
        Separating rate value (in Hz).
    groupstripe_list: tuple or list
        Information on neurons to be compared (as in cortex.neuron_idcs).
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    
    Returns
    -------
    out: dict
        Dictionary holding comparison results.
    """
    
    def save_membrparam_rategroup(group_dict, rate, file):
        
        with open(file, 'w') as f:
            for groupstripe in group_dict:
                print(groupstripe, file=f, end='\n\n')

                for par in group_dict[groupstripe]:
                    less = group_dict[groupstripe][par]['less']
                    geq = group_dict[groupstripe][par]['greater_equal']
                    mw = group_dict[groupstripe][par]['mwtest']
                    tt = group_dict[groupstripe][par]['ttest']
                    
                    print('{}:'.format(par), file=f)
                    print('rate <  {} Hz --> mean: {:.2f}, std: {:.2f} '
                          '({} cells)'.format(
                              rate, less.mean, less.std, less.n
                              ), file=f)
                    print('rate >= {} Hz --> mean: {:.2f}, std: {:.2f} '
                          '({} cells)'.format(
                              rate, geq.mean, geq.std, geq.n
                              ), file=f, end='\n\n')
                    
                    if mw.pvalue>=0.05:
                        mw_star = ''
                    elif mw.pvalue>=0.01:
                        mw_star = '*'
                    elif mw.pvalue >= 0.001:
                        mw_star = '**'
                    else:
                        mw_star = '***'
                    
                    if tt.pvalue>=0.05:
                        tt_star = ''
                    elif tt.pvalue>=0.01:
                        tt_star = '*'
                    elif tt.pvalue >= 0.001:
                        tt_star = '**'
                    else:
                        tt_star = '***'
                    
                    print('Mann-Whitney U: {:.2f}, p-value: {:.3f}'
                          .format(mw.statistic, mw.pvalue) + mw_star, file=f)
                    print("Student's t: {:.2f}, p-value: {:.3f}"
                          .format(tt.statistic, tt.pvalue) + tt_star, file=f)
                    print('-'*40, file=f)
                print('='*40, file=f)
                print('='*40, file=f)
             
    
    Stats = namedtuple('Stats', ['mean', 'std', 'n'])
    
    if isinstance(groupstripe_list[0], (int, str)):
        groupstripe_list = [groupstripe_list]    
    group_dict = {}
    
    for groupstripe in groupstripe_list:
        group, stripe = groupstripe
        gsname = '{}_stripe_{}'.format(group, stripe)
        
        cell_less = cortex.spiking_idcs((np.less, rate), groupstripe)
        cell_geq = cortex.spiking_idcs((np.greater_equal, rate), groupstripe)
        
        param_dict = {}
        
        for par in cortex.network.membr_params.coords['par'].values:
            less_params = cortex.network.membr_params.loc[
                dict(par=par, cell_index=cell_less)
                ].values
            geq_params = cortex.network.membr_params.loc[
                dict(par=par, cell_index=cell_geq)
                ].values

            param_dict[par] = {}
            param_dict[par]['less'] = (
                Stats(np.mean(less_params), np.std(less_params), 
                      len(less_params))
                )
            param_dict[par]['greater_equal'] = Stats(
                np.mean(geq_params), np.std(geq_params), len(geq_params)
                )
            param_dict[par]['mwtest'] = mwtest(
                less_params, geq_params, alternative='two-sided'
                )
            param_dict[par]['ttest'] = ttest(less_params, geq_params)
   
        rheo_less = cortex.rheobase(cell_less) 
        rheo_geq = cortex.rheobase(cell_geq)
        
        param_dict['Rheobase'] = {}
        param_dict['Rheobase']['less'] = Stats(
            np.mean(rheo_less), np.std(rheo_less), len(rheo_less)
            )
        param_dict['Rheobase']['greater_equal'] = Stats(
            np.mean(rheo_geq), np.std(rheo_geq), len(rheo_geq)
            )
        param_dict['Rheobase']['mwtest'] = mwtest(
            rheo_less, rheo_geq, alternative='two-sided'
            )
        param_dict['Rheobase']['ttest'] = ttest(rheo_less, rheo_geq)
    
        group_dict[gsname] = param_dict
     
    if file is not None:
        save_membrparam_rategroup(group_dict, rate, file)
        
    return group_dict

def contingency(cortex, rate, target_groupstripe, source_groupstripe, 
                channel=None, file=None):
    
    """Get comparisons between connection probabilities based on 
    spiking post-synaptic neuron rate less than and greater or equal 
    to a requested rate value. The comparisons are carried through a
    contingency table.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance. 
    rate: int or float
        Separating rate value (in Hz).
    target_groupstripe: tuple or list
        Information on post-synaptic (target) neurons (as in 
        cortex.neuron_idcs).
    source_groupstripe: tuple or list
        Information on pre-synaptic (source) neurons (as in 
        cortex.neuron_idcs).
    channels: str, optional
        Name of requested synaptic channel. If not given,
        default to all synapses.
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    
    Returns
    -------
    out: dict
        Dictionary holding comparison results.
    """
    
    def save_contingency(group_dict, rate, file, channel):
        
        with open(file, 'w') as f:
            for grouptarget in group_dict:
                (cont_tab, N_less, N_geq, pCon_less, pCon_geq, 
                 chi2_res, pvalue) = group_dict[grouptarget]
                
                if channel is not None:
                    print('Channel:', channel, file=f, end='\n\n')
                print(grouptarget, file=f, end='\n\n')
                print('rate <  {} Hz --> pCon = {:.2f}% ({} cells)'
                      .format(rate, pCon_less*100, N_less), file=f)
                print('rate >= {} Hz --> pCon = {:.2f}% ({} cells)'
                      .format(rate, pCon_geq*100, N_geq), file=f, end='\n\n')
                
                print('Contingency table', file=f)
                print('[{}]'.format(cont_tab[0]), file=f)
                print('[{}]'.format(cont_tab[1]), file=f, end='\n\n')
                if pvalue >= 0.05:
                    chistar = ''
                elif pvalue >= 0.01:
                    chistar = '*'
                elif pvalue >= 0.001:
                    chistar = '**'
                else:
                    chistar = '***'
                print('chi2 = {:.2f}, p-value: {:.3f}'
                      .format(chi2_res, pvalue)+chistar, file=f)
                print('-'*40, file=f, end='\n\n')
                
    if isinstance(target_groupstripe[0], (int, str)):
        target_groupstripe = [target_groupstripe]
         
    if isinstance(source_groupstripe[0], (int, str)):
        source_groupstripe = [source_groupstripe]
    
    group_dict = {}
    
    for source in source_groupstripe:
        source_group, source_stripe = source
        source_cell = cortex.neuron_idcs(source_groupstripe)
    
        for target in target_groupstripe:
            target_group, target_stripe = target  
            target_cell_less = cortex.spiking_idcs((np.less, rate), target)
            target_cell_geq = cortex.spiking_idcs((np.greater_equal, rate), 
                                                  target)
           
            gsname = ('to_{}_stripe_{}_from_{}_stripe_{}'
                      .format(target_group, target_stripe, 
                              source_group, source_stripe))

            syn_less = cortex.syn_idcs_from_neurons(target_cell_less, 
                                                    source_cell, channel)
            syn_geq = cortex.syn_idcs_from_neurons(target_cell_geq, 
                                                   source_cell, channel)
            
            Ncon_less = len(syn_less)
            Ntot_less = len(target_cell_less) * len(source_cell)
            Ndis_less = Ntot_less - Ncon_less
  
            Ncon_geq = len(syn_geq)
            Ntot_geq = len(target_cell_geq) * len(source_cell)
            Ndis_geq = Ntot_geq - Ncon_geq
            
            contingency_tab = [[Ncon_less, Ndis_less], [Ncon_geq, Ndis_geq]]
            chi2_result, pvalue = chi2(contingency_tab)[:2]
            
            group_dict[gsname] = (contingency_tab, len(target_cell_less), 
                                  len(target_cell_geq), Ncon_less/Ntot_less, 
                                  Ncon_geq/Ntot_geq, chi2_result, pvalue)
     
    if file is not None:
        save_contingency(group_dict, rate, file, channel)
        
    return group_dict
    
def comp_synparam_rategroup(cortex, rate, target_groupstripe, 
                            source_groupstripe, channel=None, file=None):
    """Get comparisons between synaptic parameters of neurons with 
    post-synaptic spiking rate less than and greater or equal to a 
    requested rate value. The comparisons are carried through 
    Mann-Whitney's U and Student's t tests.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance. 
    rate: int or float
        Separating rate value (in Hz).
    target_groupstripe: tuple or list
        Information on post-synaptic (target) neurons (as in 
        cortex.neuron_idcs).
    source_groupstripe: tuple or list
        Information on pre-synaptic (source) neurons (as in 
        cortex.neuron_idcs).
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    
    Returns
    -------
    out: dict
        Dictionary holding comparison results.
    """
    
    def save_synparam_rategroup(group_dict, rate, file, channel):
        with open(file, 'w') as f:
        
            for grouptarget in group_dict:
                if channel is not None:
                    print('Channel:', channel, file=f, end='\n\n')
                
                print(grouptarget, file=f, end='\n\n')
                # print(grouptarget)
                # print(group_dict[grouptarget])
                for par in group_dict[grouptarget]:
                    less = group_dict[grouptarget][par]['less']
                    geq = group_dict[grouptarget][par]['greater_equal']
                    mw = group_dict[grouptarget][par]['mwtest']
                    tt = group_dict[grouptarget][par]['ttest']
                    
                    print('{}:'.format(par), file=f)
                    print('rate <  {} Hz --> mean: {:.2f}, std: {:.2f} '
                          '({} synapses)'.format(
                              rate, less.mean, less.std, less.n
                              ), file=f)
                    print('rate >= {} Hz --> mean: {:.2f}, std: {:.2f} '
                          '({} synapses)'.format(
                              rate, geq.mean, geq.std, geq.n
                              ), file=f, end='\n\n')
                    
                    if mw.pvalue>=0.05:
                        mw_star = ''
                    elif mw.pvalue>=0.01:
                        mw_star = '*'
                    elif mw.pvalue >= 0.001:
                        mw_star = '**'
                    else:
                        mw_star = '***'
                    
                    if tt.pvalue>=0.05:
                        tt_star = ''
                    elif tt.pvalue>=0.01:
                        tt_star = '*'
                    elif tt.pvalue >= 0.001:
                        tt_star = '**'
                    else:
                        tt_star = '***'
                    
                    print('Mann-Whitney U: {:.2f}, p-value: {:.3f}'
                          .format(mw.statistic, mw.pvalue)+mw_star, file=f)
                    print("Student's t: {:.2f}, p-value: {:.3f}"
                          .format(tt.statistic, tt.pvalue)+tt_star, file=f)
                    print('-'*40, file=f)
                print('='*40, file=f)
                print('='*40, file=f)

    if isinstance(target_groupstripe[0], (int, str)):
        target_groupstripe = [target_groupstripe]         
    if isinstance(source_groupstripe[0], (int, str)):
        source_groupstripe = [source_groupstripe]
    
    Stats = namedtuple('Stats', ['mean', 'std', 'n'])
    group_dict = {}
    
    for source in source_groupstripe:
        source_group, source_stripe = source
        source_cell = cortex.neuron_idcs(source_groupstripe)
    
        for target in target_groupstripe:
            target_group, target_stripe = target  
            target_cell_less = cortex.spiking_idcs((np.less, rate), target)
            target_cell_geq = cortex.spiking_idcs(
                (np.greater_equal, rate), target
                )
           
            gsname = ('to_{}_stripe_{}_from_{}_stripe_{}'
                      .format(target_group, target_stripe, 
                              source_group, source_stripe))
            
            syn_less = cortex.syn_idcs_from_neurons(target_cell_less, 
                                                    source_cell, channel)
            syn_geq = cortex.syn_idcs_from_neurons(target_cell_geq, 
                                                   source_cell, channel)
            param_dict={}
            for k in cortex.network.syn_params:
                for par in (cortex.network.syn_params[k]
                              .coords['par'].values):
                    less_params = (cortex.network.syn_params[k]
                                   .loc[dict(par=par, syn_index=syn_less)]
                                   .values)
                   
                    geq_params = (cortex.network.syn_params[k]
                                  .loc[dict(par=par, syn_index=syn_geq)]
                                  .values)

                    param_dict[par] = {}
                    param_dict[par]['less'] = Stats(
                        np.mean(less_params), np.std(less_params), 
                        len(less_params)
                        )
                    param_dict[par]['greater_equal'] = Stats(
                        np.mean(geq_params), np.std(geq_params), 
                        len(geq_params)
                        )
                    param_dict[par]['mwtest'] = mwtest(
                        less_params, geq_params, alternative='two-sided'
                        )
                    param_dict[par]['ttest'] = ttest(less_params, geq_params)
                      
            
            group_dict[gsname] = param_dict

    if file is not None:
        save_synparam_rategroup(group_dict, rate, file, channel)
        
    return group_dict
    
def get_spiking(cortex, rate, groupstripe, file=None):
    """Get fraction of spiking neurons based on a threshold rate
    value.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance. 
    rate: int or float
        Threshold rate value (in Hz).
    groupstripe_list: tuple or list
        Information on neurons to be compared (as in cortex.neuron_idcs).
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    
    Returns
    -------
    This function returns a 2-tuple.
    out1: float
        Fraction of spiking neurons.
    out2: float
        Fraction of not-spiking neurons.
    """
    Ntotal = cortex.network.basics.struct.n_cells_total
    Nspiking = len(cortex.spiking_idcs((np.greater_equal, rate), groupstripe))
    Nnotspiking = len(cortex.spiking_idcs((np.less, rate), groupstripe))
    
    Pspiking = Nspiking/Ntotal
    Pnotspiking = Nnotspiking/Ntotal
    
    if file is not None:
        with open(file, 'w') as f:
            print('Group_stripe:', groupstripe, file=f)
            print('Spiking rate: >= {} Hz'.format(rate), file=f)
            print('Total N: {} cells'.format(Ntotal), file=f, end='\n\n')
            print('Spiking: {:.2f} % ({} cells)'
                  .format(Pspiking*100, Nspiking), file=f)
            print('Not spiking: {:.2f} % ({} cells)'
                  .format(Pnotspiking*100, Nnotspiking), file=f)
              
    return Pspiking, Pnotspiking


def get_membr_params(cortex, groupstripe_list, alias_list=None, file=None):
    """Get mean and standard deviation of membrane parameters of
    requestes groups.
    
    A summary of results can be saved to a specified file.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance. 
    groupstripe_list: tuple or list
        Information on requested groups of neurons (as in 
        cortex.neuron_idcs).
    alias_list: list, optional
        Aliases for the groups specified in groupstripe_list.
    file: str, optional
        Name of file where the summary of results is to be saved. If
        not given, no summary is created.
    
    Returns
    -------
    out1: dict
        Dictionary with requested statistical measures.
    
    Raises
    ------
    ValueError: if the length of alias_list is not he same as 
    groupstripe_list's.  
    """
    
    if isinstance(groupstripe_list[0], (str, int)):
        groupstripe_list = [groupstripe_list]
    
    if alias_list is not None:
        if len(alias_list) != len(groupstripe_list):
            raise ValueError("Length of alias_list has to be the same"
                             " as groupstripe_list's.")
                            
    group_dict = {}
    for groupstripe in groupstripe_list:
        gsname = 'group_{}_stripe_{}'.format(*groupstripe)
        neuron_idcs = cortex.neuron_idcs(groupstripe)
        
        param_dict = {}
        for par in cortex.network.membr_params.coords['par'].values:
            param_values = cortex.network.membr_params.loc[
                dict(par=par, cell_index=neuron_idcs)
                ].values
            param_dict[par] = np.mean(param_values), np.std(param_values)
        
        group_dict[gsname] = param_dict
     
    if alias_list is not None:
        alias_dict={}
        for gs in range(len(group_dict)):
            alias_dict[list(group_dict.keys())[gs]] = alias_list[gs] 
     
    if file is not None:
        with open(file, 'w') as f:
            for gsname in group_dict:
                if alias_list is not None:
                    print(alias_dict[gsname], '({})'.format(gsname),
                          file=f, end='\n\n')
                else:
                    print(gsname, end='\n\n', file=f)
                for par in group_dict[gsname]:
                    print(par, '--> mean: {:.2f}, std: {:.2f}'
                          .format(*group_dict[gsname][par]), file=f)
               
                print('-'*30, file=f, end='\n\n')
            print('='*40, file=f, end='\n\n')
    
    return group_dict


def binned_spiking(cortex, idcs=None, tlim=None, delta_t=5):
    """Get binned spiking arrays. Axis 0 defines neurons and axis 1
    defines time.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance.
    idcs: array_like, optional
        Indices of neurons that are to be analysed. If not given, all
        neurons are analysed.
    tlim: tuple, optional
        2-tuple (initial t, final t in ms) indicating window of
        analysis. If not given, the whole recorded period will be
        analysed.
    delta_t: int or float, optional
        Time bin (in ms). If not given, it defaults to 5.
    
    Returns
    -------
    out: array
        2d-array representing binned spiking array. Axis 0 defines 
        neurons and axis 1 defines time.
    """
    
    if idcs is None:
        idcs = np.arange(cortex.neurons.N)
    elif isinstance(idcs, int):
        idcs = [idcs]
    
    if tlim is None:
        tlim = (cortex.transient, cortex.neurons.t/br2.ms)
    
    t0, t1 = tlim
    t1 = np.ceil(t1).astype(int)
    bin_arr = np.zeros((len(idcs), (t1-t0)//delta_t))
    
    spike_trains = cortex.spikemonitor.spike_trains()
    for i in range(len(idcs)):
        train = spike_trains[idcs[i]]/br2.ms
        train = train[(train>=t0)&(train<t1)]
        idc_arr =((train-t0)//delta_t).astype(int)
        bin_arr[i, idc_arr] = 1
        
    return bin_arr.astype(int)
   
def spike_stats(bins, delta_t=5):
    """Statistical measures of binned spiking arrays.
    
    Parameters
    ----------
    bin: array
        2d-array representing binned spiking array. Axis 0 defines 
        neurons and axis 1 defines time.
    delta_t: int or float, optional
        Time bin (in ms). If not given, it defaults to 5.
    
    Returns
    -------
    This function returns a 4-tuple.
    out1: float
        Mean of populational mean array.
    out2: float
        Standard deviation of populational mean array.
    out3: list
        List of mean values of individual arrays.
    out4: list
        List of standard deviations of individual arrays.
    """
    bins = bins*1000/delta_t
    meanspiking = np.mean(bins, axis=0)
    spike_mean = []
    spike_std = []
    for i in range(bins.shape[0]):
        spike_mean.append(np.mean(bins[i]))
        spike_std.append(np.std(bins[i]))
        
    return np.mean(meanspiking), np.std(meanspiking), spike_mean, spike_std


def w_null(V, memb, I):    
    """w-nullcline of a neuron. It is defined as values of w as
    a function of V.
    
    Parameters
    ----------
    V: int orfloat
        Value of V (in mV).
    memb: membranetuple
        A membranetuple holindg membrane parameters of a neuron.
    I: int or float
        Value of current (in pA).
    
    Returns
    -------
    out: float
        Value of w on w-nullcline.
    """
    return (- memb.g_L * (V - memb.E_L) 
            + memb.g_L * memb.delta_T * np.exp((V - memb.V_T)/memb.delta_T) 
            + I)

def w_null_boundaries(self, cortex, idc, I=0, Vlim=None):
    """Get w-nullcline and the lower and upper boundaries of the
    second regime of simpAdEx dynamics.
    
    Paramters
    ---------
    cortex: Cortex
        A PFC model instance.
    idc: int
        Index of the requested neuron.
    I: int or float, optional
        Current value (in pA). If not given, it default to 0.
    vlim: tuple, optional
        2-tuple (start, stop) indicating the interval of values of V  
        (in mV) where w-nullcline and simpAdEx boundaries are to be
        calculated. If not given, the interval between -200 and 0 mV
        is used.

    Returns
    -------
    This function returns a 4-tuple.
    out1: array     
        Array of V values (in mV).
    out2: array        
        Array of w values (in pA) on w-nullcline.
    out3: array        
        Array of w values (in pA) on lower (left) boundary of the
        second simpAdEx dynamical regime.
    out4: array        
        Array of w values (in pA) on upper (right) boundary of the
        second simpAdEx dynamical regime.
    """
    
    memb = cortex.get_memb_params(idc)

    if Vlim is None:
        Vlim = (-200, 0)
    
    Varr = np.arange(Vlim[0], Vlim[1], 0.1)
    w_null_arr = np.asarray([w_null(V, memb, I) for V in Varr])
    e_l = w_null_arr * (1- memb.C/(memb.tau_w*memb.g_L))
    e_r = w_null_arr* (1 + memb.C/(memb.tau_w*memb.g_L))
    
    return Varr, w_null, e_l, e_r

    
if __name__ == '__main__':    
    pass