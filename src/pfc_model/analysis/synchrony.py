import numpy as np
from scipy.signal import hilbert, butter, lfilter
# from .tools import get_LFP_SPD
from tools import get_LFP_SPD

__all__ = ['SE_signal', 'PLV_signal_filtered', 'chi2_synchrony']

def chi2_synchrony(std_of_mean, std_list):
    """Get Golomb chi^2 synchrony measure.
    
    Parameters
    ----------
    std_of_mean: float
        Standard deviation of populational mean array.
    std_list: list
        List of standard deviations of individual arrays.
    
    Results
    -------
    out: float
        chi^2 measure.
    """
    return std_of_mean**2/np.mean(np.square(std_list))


def SE_signal(cortex, flim=None, return_periodogram=False):
    """Get spectral entropy from a LFP SPD.
    
    Parameters
    ----------
    cortex: Cortex
        A PFC model instance. 
    flim: tuple, optional
        Frequency band where analysis is to be restricted. If not
        given, all frequencies are analysed.
    return_periodogram: bool, optional
        Whether periodogram results should be returned.
    
    Returns
    -------
    This function can return a single value or a 3-tuple.
    out1: float
        Spectral entropy value.
    out2: array (if return_periodogram)
        Frequency values.
    out3: array (if return_periodogram)
        Power values
    """
    
    def xlogx(x):
        x = np.asarray(x)
        xlogx = np.zeros(x.shape)
        xlogx[x < 0] = np.nan
        valid = x > 0
        xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(2)
        return xlogx
    
    frequency, power = get_LFP_SPD(cortex, log=False)
    
    if flim is not None:
        f0, f1 = flim
        f_bool = np.where((frequency>= f0)&(frequency<f1))[0]
        frequency = frequency[f_bool]
        power = power[f_bool]
    
    power_normal = power / power.sum()
    se = -xlogx(power_normal).sum()

    se /= np.log2(len(power_normal))
    if return_periodogram:
        return se, frequency, power
    else:
        return se
  
def PLV_signal_filtered(sgn_arr, time_arr, t0, t1, lowcut, highcut, fs=20000,
                        order=1):
    """Filter and get PLV (phase lock value) from signals in a
    group of neurons. A Butterworth filter is applied.
    
    Parameters
    ---------
    sgn_arr: array
        A 2d-array containing a signal array for each requested neuron.       
        Axis 0 must define neurons and axis 1 must define time.
    time_arr: array_like
        Array with time of observations (in ms).
    t0: int or float
        Start time (in ms).
    t1: int or float
        Stop time (in ms).
    lowcut: int or float
        Lower cut value for filtering (in Hz).
    highcut: int or float
        Upper cut value for filtering (in Hz).
    fs: int or float, optional
        Frequency of sampling (in Hz). If not given, it default to
        20000.
    order: int, optional
        Order of filter. If not given, it defaults to 1.
        
    Returns
    -------
    out: float
        PLV (phase lock value).
    """
    
    def PLV_signal(sgn_arr, time_arr, t0, t1):
        
        def phase_signal(sign_arr, time_arr, t0, t1):
            
            bool_arr = np.where((time_arr>=t0) & (time_arr<t1))[0]
            t_arr = time_arr[bool_arr] - t0
            signal = sign_arr[bool_arr]
            
            analytic_signal = hilbert(signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            
            return t_arr, instantaneous_phase  
        
        def PLV_from_phase_diff(theta1, theta2):
            complex_phase_diff = np.exp(complex(0,1)*(theta1 - theta2))
            plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
            return plv           

        phase_list = []
        for i in range(len(sgn_arr)):
            phase_list.append(phase_signal(sgn_arr[i], time_arr, t0, t1)[1])
        
        pop = []
        Ntot = len(phase_list) * (len(phase_list)-1)/2

        for i in range(len(phase_list)):
            for k in range(0, i):
                
                pop.append(PLV_from_phase_diff(phase_list[i],phase_list[k]))

        return np.mean(pop)
    
    filtered_sgn_arr = []
    
    for i in range(len(sgn_arr)):
        filtered_sgn_arr.append(_butter_bandpass_filter(sgn_arr[i], lowcut, 
                                                        highcut, fs, order))
    
    return PLV_signal(filtered_sgn_arr, time_arr, t0, t1)

def _butter_bandpass(lowcut,  highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def _butter_bandpass_filter(data, lowcut, highcut, fs=20000, order=1):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y