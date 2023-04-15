"""
This module sets the PFC network model inside Brian2 and allows the
user to scale parameters, to set monitors and stimuti and to run
simulations. 

This file contain the following class:
    Cortex - contains all the network, monitors and stimuli setup and
    allow simulation runs.
"""

import brian2 as br2
import numpy as np
from numpy.matlib import repmat
from dataclasses import dataclass
from importlib import import_module
import os
import shutil
import toml
from ._network_setup import *
from ._auxiliary import *
from ._basics_setup import membranetuple

br2.BrianLogger.suppress_name('resolution_conflict', 'device')

__all__ = ['Cortex']

class Cortex(BaseClass):
    
    """A class that contains all the network, monitor and stimuli setup
    and allow simulation runs.
    
    Parameters
    ----------
    n_cells: int
        Number of cells per stripe.
    n_stripes: int
        Number of stripes.
    constant_stimuli: array_like
        Constant background stimuli applied on each group (its length
        must be the number of different groups in each stripe).
    method: str
        Numerical integration method (as in Brian 2).
    dt: float
        Simulation step (in ms).
    transient: float or int, optional
        Transient initial period (in ms) that is not to be recorded 
        (due to artifact in the beginning of simulations). If not 
        given, it defaults to 0.
    basics_scales: dict, optional
        Dictionary with basics scales. If not given, no basic scaling 
        is applied.
    fluctuant_stimuli: array, optional
        Non-constant current array for each group. If not given, no
        fluctuant stimuli are applied.    
    seed: int, optional
        Random seed. If not given, no random seed is set.
    alternative_pcells: array_like, optional
        Alternative cell distribution. If not given, it defaults to
        _pcells_per_group in _basics_setup.
    basics_disp: bool, optional
        Set display of basic setup. If not given, warning message
        on basic setup may be displayed.
    cortex_neuron_scales: dict, optional
        Dictionary with neuron parameter scales. If not given, no
        scaling in membrane parameters is set.
    cortex_syn_scales: dict, optional
        Dictionary with synaptic parameter scales. If not given, no
        scaling in membrane parameters is set.
    """
    
    @staticmethod
    @time_report('Cortex setup')
    def setup(n_cells, n_stripes, constant_stimuli, method, dt, transient=0, 
              basics_scales=None, fluctuant_stimuli=None, seed=None,
              alternative_pcells=None, basics_disp=True,
              cortex_neuron_scales=None, cortex_syn_scales=None):
        """Set a new model instance.
        
        Parameters
        ----------
        n_cells: int
            Number of cells per stripe.
        n_stripes: int
            Number of stripes.
        constant_stimuli: array_like
            Constant background stimuli applied on each group (its length
            must be the number of different groups in each stripe).
        method: str
            Numerical integration method (as in Brian 2).
        dt: float
            Simulation step (in ms).
        transient: float or int, optional
            Transient initial period (in ms) that is not to be recorded 
            (due to artifact in the beginning of simulations). If not 
            given, it defaults to 0.
        basics_scales: dict, optional
            Dictionary with basics scales. If not given, no basic scaling 
            is applied.
        fluctuant_stimuli: list, optional
            Fluctuant stimuli applied on each group. Each element in the
            list is a tuple (target, function, start, stop), where target
            is group identification as in neuron_idcs (group, stripe);
            function defines the stimulus value in each instant (in ms);
            start and stop are the instants when stimuli begin and end
            (in ms). If not given, no fluctuant stimuli are applied.   
        seed: int, optional
            Random seed. If not given, no random seed is set.
        alternative_pcells: array_like, optional
            Alternative cell distribution. If not given, it defaults to
            _pcells_per_group in _basics_setup.
        basics_disp: bool, optional
            Set display of basic setup. If not given, warning message
            on basic setup may be displayed.
        cortex_neuron_scales: dict, optional
            Dictionary with neuron parameter scales. If not given, no
            scaling in membrane parameters is set.
        cortex_syn_scales: dict, optional
            Dictionary with synaptic parameter scales. If not given, no
            scaling in membrane parameters is set.
        
        Returns
        -------
        Out: Cortex
            Prefrontal cortex model instance.
        """
        
        
        network = network_setup(n_cells, n_stripes, basics_scales, seed,
                                alternative_pcells, basics_disp)
        
        cortex = __class__(network, constant_stimuli, method, dt, transient, 
                         fluctuant_stimuli, seed, cortex_neuron_scales, 
                         cortex_syn_scales)
        
        cortex.basics_scales = basics_scales
        cortex.alternative_pcells = alternative_pcells
        cortex.basics_disp = basics_disp
        cortex.cortex_neuron_scales = cortex_neuron_scales
        cortex.cortex_syn_scales = cortex_syn_scales
        
        return cortex
    
    @staticmethod
    @time_report('Cortex setup (with network loading)')
    def load(path, constant_stimuli, method, dt, transient=0, 
             fluctuant_stimuli=None, cortex_neuron_scales=None,
             cortex_syn_scales=None):
        """Set a model instance from previously saved data.
        
        Parameters
        ----------
        path: str
            Path where saved data are to be retrieved from.
        constant_stimuli: array_like
            Constant background stimuli applied on each group (its length
            must be the number of different groups in each stripe).
        method: str
            Numerical integration method (as in Brian 2).
        dt: float
            Simulation step (in ms).
        transient: float or int, optional
            Transient initial period (in ms) that is not to be recorded 
            (due to artifact in the beginning of simulations). If not 
            given, it defaults to 0.
        fluctuant_stimuli: array, optional
            Non-constant current array for each group. If not given, no
            fluctuant stimuli are applied.  
        cortex_neuron_scales: dict, optional
            Dictionary with neuron parameter scales. If not given, no
            scaling in membrane parameters is set.
        cortex_syn_scales: dict, optional
            Dictionary with synaptic parameter scales. If not given, no
            scaling in membrane parameters is set.
         alternative_basics_setup: function
             Alternative basics setup function. If not given, basics_setup
             from _basics_setup is used.
        
        Returns
        -------
        Out: Cortex
            Prefrontal cortex model instance.
        """
      
        network = Network.load(path)
        seed = network.seed
        
        cortex = __class__(network, constant_stimuli, method, dt, transient, 
                         fluctuant_stimuli, seed, cortex_neuron_scales, 
                         cortex_syn_scales)
        
        if not os.path.isabs(path):
            cortex._loadfile = os.path.join(os.getcwd(), path)
        else:
            cortex._loadfile = path
        

        cortex.cortex_neuron_scales = cortex_neuron_scales
        cortex.cortex_syn_scales = cortex_syn_scales
        
        return cortex
    
    @staticmethod
    @time_report()
    def repeat_experiment(path, file_name):
        
        if '.toml' not in file_name:
            file_name = file_name + '.toml'
        toml_dict = toml.load(os.path.join(path, file_name))
        
        dt = toml_dict['setup']['dt']
        method = toml_dict['setup']['method']
        n_stripes = toml_dict['setup']['n_stripes']
        n_cells = toml_dict['setup']['n_cells']
        transient = float(toml_dict['simulation']['transient'])
        simulated_time = float(toml_dict['simulation']['simulated_time'])
        basics_disp = toml_dict['setup']['basics_disp']
        seed = (toml_dict['setup']['seed'] if 'seed' in toml_dict['setup'] 
                else None)
        alternative_pcells = (toml_dict['setup']['alternative_pcells'] 
                              if 'alternative_pcells' in toml_dict['setup']
                              else None)
        basics_scales = None
        if 'basics' in  toml_dict['scales']:
            basics_scales = {}
            for scl in toml_dict['scales']['basics']:
                basics_scales[scl] = []
                for i in toml_dict['scales']['basics'][scl]:
                    scl_dict = toml_dict['scales']['basics'][scl][i]
                    basics_scales[scl].append(
                        (scl_dict['info'], scl_dict['scale']))
                
        cortex_neuron_scales=None
        if 'neuron' in toml_dict['scales']:
            cortex_neuron_scales = {}
            for scl in toml_dict['scales']['neuron']:
                cortex_neuron_scales[scl] = []
                for i in toml_dict['scales']['neuron'][scl]:
                    scl_dict = toml_dict['scales']['neuron'][scl][i]
                    cortex_neuron_scales[scl].append(
                        [(scl_dict['group'], scl_dict['stripe']), 
                         scl_dict['factor']])
        
        cortex_syn_scales=None
        if 'syn' in toml_dict['scales']:
            cortex_syn_scales = {}
            for scl in toml_dict['scales']['syn']:
                cortex_syn_scales[scl] = []
                for i in toml_dict['scales']['syn'][scl]:
                    scl_dict = toml_dict['scales']['syn'][scl][i]
                    cortex_syn_scales[scl].append(
                        [(scl_dict['tgt_group'], scl_dict['tgt_stripe']),
                         (scl_dict['src_group'], scl_dict['src_stripe']),
                         scl_dict['channels'], scl_dict['factor']])
        
        constant_stimuli = []
        for stim in list(toml_dict['constant_stimuli'].values()):
            constant_stimuli.append([(stim['group'], stim['stripe']), 
                                     stim['current']])
            
        
        fluctuant_stimuli = ([] if len(toml_dict['fluctuant_stimuli']) > 0 
                             else None)
        fluct_copy = []
        for stim in list(toml_dict['fluctuant_stimuli'].values()):
            group = stim['group']
            stripe = stim['stripe']
            start = stim['start']
            stop = stim['stop']
            script = stim['script']
            func = stim['function']
            mod_path, mod_file = os.path.split(script)
            if not os.path.samefile(mod_path, os.getcwd()):
                shutil.copyfile(script, os.path.join(os.getcwd(), mod_file))
                fluct_copy.append(os.path.join(os.getcwd(), mod_file))

            mod = mod_file[:-3]         
            fluctuant_stimuli.append([(group, stripe), mod, func, start, stop])
        
        if 'save_file' in toml_dict['setup']:
            file = toml_dict['setup']['save_file']
            cortex = Cortex.load(file, constant_stimuli, method, dt, transient,
                                 fluctuant_stimuli, cortex_neuron_scales,
                                 cortex_syn_scales)
        elif 'load_file' in toml_dict['setup']:
            file = toml_dict['setup']['load_file']
            cortex = Cortex.load(file, constant_stimuli, method, dt, transient,
                                 fluctuant_stimuli, cortex_neuron_scales,
                                 cortex_syn_scales)
        else:
            cortex = Cortex.setup(n_cells, n_stripes, constant_stimuli,
                                  method, dt, transient, basics_scale,
                                  fluctuant_stimuli, seed, alternative_pcells,
                                  basics_disp, cortex_neuron_scales,
                                  cortex_syn_scales)
        alternative_pcells
        for monitor in toml_dict['monitors']:
            mon = toml_dict['monitors'][monitor]
            type_ = mon['type']
            variables =mon['variables']
            idcs = mon['idcs']
            start = mon['start'] if 'start' in mon else None 
            stop = mon['stop'] if 'stop' in mon else None 
            interval = mon['interval'] if 'interval' in mon else None
            population_agroupate = (mon['population_agroupate'] 
                                    if 'population_agroupate' in mon else None)
                
            if type_ == 'neuron':
                print()
                cortex.set_neuron_monitors(monitor, variables, idcs, start, stop)
            elif type_ == 'synapse':
                cortex.set_synapse_monitors(monitor, variables, idcs, start, stop)
            elif type_ == 'neuron_longrun':
                cortex.set_longrun_neuron_monitors(monitor, variables, idcs,
                                                 interval, start, stop, 
                                                 population_agroupate)
            elif type_ == 'synapse_longrun':
                cortex.set_longrun_synapse_monitors(monitor, variables, idcs,
                                                 interval, start, stop, 
                                                 population_agroupate)
                
        for stimulus in toml_dict['external_stimuli']:
            stim = toml_dict['external_stimuli'][stimulus]
            type_ = stim['stimulator_type']
            n_source = stim['n_source']
            channels = stim['channels']
            target_idc = stim['target_idc']
            pcon = stim['pcon']
            rate = stim['rate']
            start = stim['start']
            stop = stim['stop']
            gmax = stim['gmax']
            pfail = stim['pfail']
           
            if type_ == 'poisson':
                cortex.set_poisson_stimuli(stimulus, n_source, channels, 
                                           target_idc, pcon, rate, start, stop, 
                                           gmax, pfail)
            elif type_ == 'regular':
                cortex.set_regular_stimuli(stimulus, n_source, channels, 
                                           target_idc, pcon, rate, start, stop, 
                                           gmax, pfail)
        
        if simulated_time > 0:
            cortex.run(simulated_time)
            
        for cp in fluct_copy:
            os.remove(cp)
        
        return cortex
    

    def __init__(self, network, constant_stimuli, method, dt, transient=0, 
                 fluctuant_stimuli=None, seed=None, cortex_neuron_scales=None, 
                 cortex_syn_scales=None):
        """Initiate model instance."""
    
        np.random.seed(seed)
        br2.seed(seed)
        self.network = network
        self.method = method
        self.dt = dt * br2.ms 
        
        if (constant_stimuli is not None 
            and isinstance(constant_stimuli[-1], (int,float))):
            constant_stimuli = [constant_stimuli]
        self.constant_stimuli = constant_stimuli
        
        if (fluctuant_stimuli is not None 
            and isinstance(fluctuant_stimuli[-1], (int,float))):
            fluctuant_stimuli = [fluctuant_stimuli]
        self.fluctuant_stimuli = fluctuant_stimuli
        
        self.transient = transient
        self.simulated_time = 0
        self.seed = seed
        self.cortex_neuron_scales = cortex_neuron_scales
        self.cortex_syn_scales = cortex_syn_scales
        self._loadfile = None
        self._savefile = None
        
        self.net = br2.Network()      
        self._set_fluctuant_stimuli()
        
        membrane_events = self._get_membrane_events_dict()     
        self.membrane_events = membrane_events
        membr_model = (
            network.basics.equations.membr_model.format(self.fluctuant_str))
        
        self.neurons = br2.NeuronGroup(
            
            N=network.basics.struct.n_cells_total, model=membr_model,
            threshold=network.basics.equations.membr_threshold, 
            reset=network.basics.equations.membr_reset, events=membrane_events,
            method=method, refractory=5*br2.ms, dt=self.dt)
       
        self._set_constant_stimuli()
        self._set_membrane_events()
        self._set_neuron_params()
        self._set_channels()
        self._set_auxiliar_vars()
        self._set_initial_state()
           
        syn_dict_pathway = self._get_syn_pathway_dict()
        self.synapses = br2.Synapses(
            self.neurons, self.neurons, 
            model=network.basics.equations.syn_model, on_pre=syn_dict_pathway,
            method=method, dt=self.dt)
        
        if self.network.syn_pairs.shape[1]>0:
            self.synapses.connect(
                i=self.network.syn_pairs[1,:], j=self.network.syn_pairs[0,:])
        else:
            self.synapses.active = False
        
        self._set_synapses()            
        self.spikemonitor = br2.SpikeMonitor(self.neurons)
        if self.transient>0:
            self.spikemonitor.active=False
            
        self._set_syn_scales()            
        self._set_neuron_scales()   
         
        self.net.add(self.neurons, self.synapses, self.spikemonitor,
                     *self.event_monitors._values())
                    
        self.external_stimuli=_NetworkHolder()
        
        self.neuron_monitors=_NetworkHolder()
        self.synapse_monitors=_NetworkHolder()
        
        self.longrun_neuron_monitors=_NetworkHolder()
        self.longrun_synapse_monitors=_NetworkHolder()
        self.recorded=_VarHolder()
        
        self._monitor_schedule={}
        self._longrun_monitor_control=[]
        
        self.external_stimuli_info = {}
        self.monitors_info = {}
        
        self.basics_scales = None
        self.alternative_pcells = None
        self.basics_disp = True
        self.cortex_neuron_scales = None
        self.cortex_syn_scales = None
        
        
    
    def save(self, path):
        """Save cortex data (structure, connectivity and parameters).
        
        Parameter
        ---------
        path: str
            Path where data will be saved.
        """
        
        self.network.save(path)
        if not os.path.isabs(path):
            self._savefile = os.path.join(os.getcwd(), path)
        else:
            self._savefile = path
    
    @time_report('Cortex simulation')
    def run(self, t, erase_longrun=True):
        """
        Run a simulation.
        
        Parameter
        ---------
        t: int or float
            Duration of simulation (in ms).
        erase_longrun: bool, optional
            Whether directory created for long run monitors are to be
            deleted. If not given, it defaults to True.
        """
        print_out = "::   REPORT: Preparing simulation   ::"
        
        print('.'*len(print_out))
        print(print_out)
        print('.'*len(print_out), end='\n\n')
        
        print('Time step: {} ms'.format(self.dt/br2.ms))
        print('Integration method:', self.method)
        print('Seed:', self.seed, end='\n\n')
            
        ### fluctuant_array needs to be declared here again
        ### fluctuant_array appears inside the model code of NeuronGroup
        if self.fluctuant_array is not None:
            fluctuant_array = br2.TimedArray(self.fluctuant_array, dt=self.dt)
            
        schedule_times = np.asarray(list(self._monitor_schedule.keys()))
        t0 = self.neurons.t/br2.ms
             
        longrun_times = []
        longrun_all_times = []
        for longrun_control in self._longrun_monitor_control:
            longrun_stop = (longrun_control['stop'] if longrun_control['stop'] 
                            is not None and longrun_control['stop'] <=t else t)    
            times = np.arange(
                longrun_control['start'], longrun_stop,
                longrun_control['interval'])
            times = np.append(times, longrun_stop)
            longrun_times.append(times)
            longrun_all_times.extend(times)
    
        current_schedule_times = np.concatenate(
            (schedule_times, longrun_all_times, [self.transient]))
        current_schedule_times = current_schedule_times[np.where(
            (current_schedule_times >=t0 ) & (current_schedule_times < t0+t))]
        current_schedule_times = np.unique(current_schedule_times)
        
        if t0 not in current_schedule_times:
            current_schedule_times = np.append(current_schedule_times, t0)
        if t0+t not in current_schedule_times:
            current_schedule_times = np.append(current_schedule_times, t0+t)
        
        current_schedule_times = np.sort(current_schedule_times)
        current_schedule_intervals = np.diff(current_schedule_times)
        
        print('Start time: {} ms'.format(t0))
        print('Stop time: {} ms'.format(t0+t))
        print('Total duration : {:.2f} ms (from {:.2f}'
              ' to {:.2f} ms)'.format(t, t0, t0+t), end='\n\n')
        print('Simulation points:', ', '.join(['{} ms'.format(i) for i 
                                               in current_schedule_times]))
        print(len(current_schedule_intervals), 
              'simulation segment(s) of duration:', 
              ', '.join(np.round(current_schedule_intervals,2).astype(str)), 
              'ms', end='\n\n')
        
        if t0 in schedule_times:
            for monitor in self._monitor_schedule[t0]['start']:
                monitor.active=True
            for monitor in self._monitor_schedule[t0]['stop']:
                monitor.active=False
        
        for i in range(len(current_schedule_intervals)):
            print_out1a = ('::   Preparing segment {} out of {}'
                           .format(i+1, len(current_schedule_intervals)))
            print_out2a = ('::   Duration {:.2f} ms (from {:.2f} to {:.2f} ms)'
                           .format(current_schedule_intervals[i], 
                                   current_schedule_times[i],
                                   current_schedule_times[i+1]))
            print_out1 = (print_out1a 
                          + ' '*(max(len(print_out1a), len(print_out2a))
                                 -len(print_out1a)) 
                          + '   ::')
            print_out2 = (print_out2a 
                          + ' '*(max(len(print_out1a), len(print_out2a))
                                 -len(print_out2a)) 
                          + '   ::')

            print('.'*len(print_out1))
            print(print_out1)
            print(print_out2)
            print('.'*len(print_out1), end='\n\n')
            self.net.run(current_schedule_intervals[i]*br2.ms, report='text',
                         report_period=10*br2.second)
            self.simulated_time += current_schedule_intervals[i]
            
            t1 = current_schedule_times[i+1]
            if t1 >= self.transient:
                self.spikemonitor.active=True
                
            if (t1 in longrun_all_times 
                    or t1 == current_schedule_times[-1]):
                for l in range(len(longrun_times)):
                    if (t1 in longrun_times[l] 
                            or t1 == current_schedule_times[-1]):
                        self._process_longrun(l, t1)
            
            if t1 in schedule_times:
                for monitor in self._monitor_schedule[t1]['start']:
                    monitor.active=True
                for monitor in self._monitor_schedule[t1]['stop']:
                    monitor.active=False
        print()
        self._restore_longrun(erase_longrun)
        
        
    def set_poisson_stimuli(self, stimulator_name, n_source, channels, 
                            target_idc, pcon, rate, start, stop, gmax, pfail):
        """Set external stimuli generated by Poisson processes.
        
        Parameters
        ----------
        stimulator_name: str
            Name of the stimulator (the stimulator can be acessed in 
            external_stimuli).
        n_source: int
            Number of external stimuli sources.
        channels: str or list[str]
            Channels that will be stimulated.
        target_idc: array_like
            Indices of post-synaptic (target) neurons.
        pcon: float
            Probability of connection between external sources and
            target neurons.
        rate: float
            Mean spiking rate (in Hz) of external sources.
        start: float
            Instant of stimuli start (in ms).
        stop: float
            Instant of stimuli stop (in ms).
        gmax: float
            Synaptic strength (in nS).
        pfail: float
            Probability of failure.
        """
        
        spike_times = []
        spike_idcs = []
     
        source_idc = np.arange(n_source)
        target_idc = np.asarray(target_idc)
        Ntarget = len(target_idc)
        for src in range(n_source):
            lambd = 1000/rate
            upper_est = int(round(2*(stop-start)/lambd, 0))
            spike_array =  start + np.cumsum(np.clip(
                -lambd *np.log(1-np.random.rand(upper_est)), 
                self.dt/br2.ms, np.NaN))
            
            spikeact = spike_array[spike_array < stop]
            spike_times.extend(spikeact)
            spike_idcs.extend([src]*len(spikeact))
            
        spike_idcs = np.asarray(spike_idcs)
        Nconnections = int(round(pcon*Ntarget*n_source, 0))
        local_pairs = np.arange(n_source*Ntarget)
        np.random.shuffle(local_pairs)
        local_pairs = local_pairs[:Nconnections]
        local_target_idc = local_pairs//n_source
        local_source_idc = local_pairs % n_source
        
        pairs_connected = np.zeros((2,Nconnections))
        pairs_connected[0,:] = target_idc[local_target_idc]
        pairs_connected[1,:] = source_idc[local_source_idc]
        pairs_connected=pairs_connected.astype(int)
        
        self.external_stimuli_info[stimulator_name] = {
            'stimulator_type': 'poisson',
            'n_source': n_source,
            'channels': channels,
            'target_idc': target_idc,
            'pcon': pcon,
            'rate': rate,
            'start': start,
            'stop': stop,
            'gmax': gmax,
            'pfail': pfail}
        
        
        self._set_custom_stimuli(stimulator_name, n_source, channels, spike_idcs,
                                spike_times, pairs_connected, gmax, pfail)
    
    
    def set_regular_stimuli(self, stimulator_name, n_source, channels, 
                            target_idc, pcon, rate, start, stop, gmax, pfail):
        """Set regular external stimuli.
        
        Parameters
        ----------
        stimulator_name: str
            Name of the stimulator (the stimulator can be acessed in 
            external_stimuli).
        n_source: int
            Number of external stimuli sources.
        channels: str or list[str]
            Channels that will be stimulated.
        target_idc: array_like
            Indices of post-synaptic (target) neurons.
        pcon: float
            Probability of connection between external sources and
            target neurons.
        rate: float
           Spiking rate (in Hz) of external sources.
        start: float
            Instant of stimuli start (in ms).
        stop: float
            Instant of stimuli stop (in ms).
        gmax: float
            Synaptic strength (in nS).
        pfail: float
            Probability of failure.
        """
        source_idc = np.arange(n_source)
        n_spikes=round((stop-start)*rate/1000)

        if 1000/rate < self.dt/br2.ms:
            raise ValueError('ERROR: time step is larger than spikes'
                             ' intervals')

        spike_times = list(
            np.linspace(start, stop, n_spikes, endpoint=False))*n_source
        spike_idcs=np.concatenate(
            ([[src]*n_spikes for src in source_idc])).astype(int)

        Nconnections = int(round(pcon*len(target_idc)*n_source, 0))

        local_pairs = np.arange(n_source*len(target_idc))
        np.random.shuffle(local_pairs)
        local_pairs = local_pairs[:Nconnections]
        local_target_idc = local_pairs//n_source
        local_source_idc = local_pairs % n_source

        pairs_connected = np.zeros((2,Nconnections))
        pairs_connected[0,:] = target_idc[local_target_idc]
        pairs_connected[1,:] = source_idc[local_source_idc]
        pairs_connected=pairs_connected.astype(int)
        
        self.external_stimuli_info[stimulator_name] = {
            'stimulator_type': 'poisson',
            'n_source': n_source,
            'channels': channels,
            'target_idc': target_idc,
            'pcon': pcon,
            'rate': rate,
            'start': start,
            'stop': stop,
            'gmax': gmax,
            'pfail': pfail}
        
        

        self._set_custom_stimuli(stimulator_name, n_source, channels, 
                                 spike_idcs, spike_times, pairs_connected,
                                 gmax, pfail)
    
    
    def set_neuron_monitors(self, name, variables, neuron_idcs, 
                            start=None, stop=None):
        """Set monitor of neuron variables. The monitor can be accessed
        in neuron_monitors.
        
        Recorded data can be retrieved in neuron_monitors (through
        monitor name) and recorded (through variable name) attributes.
        
        Parameters
        ----------
        name: str
            Name of monitor.
        variables:
            Neuron variables that are to be monitored.
        neuron_idcs: array_like
            Indices from neurons to be recorded.
        start: float, optional
            Start instant (in ms). If not given, it defaults to
            transient (instance attribute).
        stop: float, optional
            Stop instant (in ms). If not given, monitoring is carried
            until the end of simulation.
        """
        
        if isinstance(variables, str):
            variables = [variables]
              
        self.neuron_monitors[name] =  br2.StateMonitor(
            self.neurons, variables, neuron_idcs, dt=self.dt)
        self._set_monitors(self.neuron_monitors[name], variables, start, stop)
        self.monitors_info[name] = {
            'type': 'neuron',
            'variables': variables,
            'idcs': neuron_idcs,
            'interval': None,
            'start': start,
            'stop': stop,
            'population_agroupate': None}
        
        
    def set_synapse_monitors(self, name, variables, syn_idcs, 
                             start=None, stop=None):
        """Set monitor of synaptic variables. The monitor can be accessed
        in neuron_monitors.
        
        Recorded data can be retrieved in synapse_monitors (through
        monitor name) and recorded (through variable name) attributes.
        
        Parameters
        ----------
        name: str
            Name of monitor.
        variables:
            Synaptic variables that are to be monitored.
        syn_idcs: array_like
            Indices from synapses to be recorded.
        start: float, optional
            Start instant (in ms). If not given, it defaults to
            transient (instance attribute).
        stop: float, optional
            Stop instant (in ms). If not given, monitoring is carried
            until the end of simulation.
        """
        if isinstance(variables, str):
            variables = [variables]
          
        self.synapse_monitors[name] =  br2.StateMonitor(
            self.synapses, variables, syn_idcs, dt=self.dt)
        self._set_monitors(self.synapse_monitors[name], variables, start, stop)      
        self.monitors_info[name] = {
            'type': 'synapse',
            'variables': variables,
            'idcs': syn_idcs,
            'interval': None,
            'start': start,
            'stop': stop,
            'population_agroupate': None}

    def set_longrun_neuron_monitors(self, name, variables, neuron_idcs,
                                    interval, start=None, stop=None, 
                                    population_agroupate=None):
        """Set monitor of neuron variables for long run simulations.
        The simulation is divided in segments, and monitor data are
        saved to disk and cleaned between them. It is intended to avoid
        memory overflow during long simulations.
        
        Recorded data can be automatically saved as sum or mean of
        population values (if this is the desired information, this
        synthetized storing spares memory.)
        
        Recorded data can be retrieved in longrun_neuron_monitors (through
        monitor name) and recorded (through variable name) attributes.
        
        Parameters
        ----------
        name: str
            Name of monitor.
        variables:
            Neuron variables that are to be monitored.
        neuron_idcs: array_like
            Indices from neurons to be recorded.
        start: float, optional
            Start instant (in ms). If not given, it defaults to
            transient (instance attribute).
        stop: float, optional
            Stop instant (in ms). If not given, monitoring is carried
            until the end of simulation.       
        """
        if isinstance(variables, str):
            variables = [variables]
        
        self.longrun_neuron_monitors[name] =  br2.StateMonitor(
            self.neurons, variables, neuron_idcs, dt=self.dt)
        self.longrun_neuron_monitors[name].active = False
        self.neuron_monitors[name] = _NetworkHolder()
        self.neuron_monitors[name].t = None
        for var in variables:
            self.neuron_monitors[name][var] = None
            
        self._set_longrun_monitors(
            self.neuron_monitors[name], self.longrun_neuron_monitors[name],
            variables, interval, start, stop, population_agroupate)
        self.monitors_info[name] = {
            'type': 'neuron_longrun',
            'variables': variables,
            'idcs': neuron_idcs,
            'interval': interval,
            'start': start,
            'stop': stop,
            'population_agroupate': population_agroupate}
       
    def set_longrun_synapse_monitors(
            self, name, variables, syn_idcs, 
            interval, start=None, stop=None,
            population_agroupate=None):
        """Set monitor of synaptic  variables for long run simulations.
        The simulation is divided in segments, and monitor data are
        saved to disk and cleaned between them. It is intended to avoid
        memory overflow during long simulations.
        
        Recorded data can be automatically saved as sum or mean of
        population values (if this is the desired information, this
        synthetized storing spares memory.)
        
        Recorded data can be retrieved in longrun_synapse_monitors (through
        monitor name) and recorded (through variable name) attributes.
        
        Parameters
        ----------
        name: str
            Name of monitor.
        variables:
            Neuron variables that are to be monitored.
        syn_idcs: array_like
            Indices from synapses to be recorded.
        start: float, optional
            Start instant (in ms). If not given, it defaults to
            transient (instance attribute).
        stop: float, optional
            Stop instant (in ms). If not given, monitoring is carried
            until the end of simulation.       
        """
        
        if isinstance(variables, str):
            variables = [variables]
        
          
        self.longrun_synapse_monitors[name] =  br2.StateMonitor(
            self.synapses, variables, syn_idcs, dt=self.dt)
        self.longrun_synapse_monitors[name].active = False
        self.synapse_monitors[name] = _NetworkHolder()
        self.synapse_monitors[name].t = None
        for var in variables:
            self.synapse_monitors[name][var] = None
            
        self._set_longrun_monitors(
            self.synapse_monitors[name], self.longrun_synapse_monitors[name], 
            variables, interval, start, stop, population_agroupate)
        self.monitors_info[name] = {
            'type': 'synapse_longrun',
            'variables': variables,
            'idcs': syn_idcs,
            'interval': interval,
            'start': start,
            'stop': stop,
            'population_agroupate': population_agroupate}
        
    def get_memb_params(self, idc):
        """Retrieve membrane parameters as membranetuple.
        
        Parameter
        ---------
        idc: int
            Index of requested neuron.
        
        Returns
        -------
        Out: membranetuple
            membranetuple holding the requestes parameters.
        """
        C = self.neurons.C[idc]/br2.pF
        g_L = self.neurons.g_L[idc]/br2.nS
        E_L = self.neurons.E_L[idc]/br2.mV
        delta_T = self.neurons.delta_T[idc]/br2.mV
        V_up = self.neurons.V_up[idc]/br2.mV
        tau_w = self.neurons.tau_w[idc]/br2.ms
        b = self.neurons.V_T[idc]/br2.pA
        V_r = self.neurons.V_r[idc]/br2.mV
        V_T = self.neurons.V_T[idc]/br2.mV
        
        return membranetuple(C, g_L, E_L, delta_T, V_up, tau_w, b, V_r, V_T)
          
    def group_idcs(self, group):
        """Get group index from group name.
        
        Parameters
        ----------
        group:str or int
            Index or name of requested group or set of groups (as in 
            group_sets in _basics_setup).
        
        Returns
        -------
        Out: list
            List of indices from requested groups. 
        """

        return self.network.group_idcs(group)

    def neuron_idcs(self, groupstripe_list):    
        return self.network.neuron_idcs(groupstripe_list)
    """Retrieve indices from neurons corresponding to the
    given group and stripe.
    
    Parameters
    ----------
    groupstripe_list: list
        List of requested groups and stripes.
    
    Returns
    -------
    Out: np.array
        Array of neuron indices.
    """
    
    def syn_idcs_from_neurons(self, target, source, channel=None):
        """Synaptic indices from neuron information.
        
        Parameters
        ----------
        target: array_like
            Indices of post-synaptic neurons.
        source: array_like
            Indices of pre-synaptic neurons.
        channels: str, optional
            Name of requested synaptic channel. If not given,
            default to all synapses.
            
        Returns
        -------
        Out: array
            Requested synaptic indices.
        """
        
        return self.network.syn_idcs_from_neurons(target, source, channel)
       
    def syn_idcs_from_groups(self, target_groupstripe_list, 
                             source_groupstripe_list, channel=None):
        """Synaptic indices from group information.
        
        Parameters
        ----------
        tgt_groupstripe_list: list
            Group and stripe of post-synaptic neurons.
        source: array_like
            Group and stripe of pre-synaptic neurons.
        channels: str, optional
            Name of requested synaptic channel. If not given,
            default to all synapses.
            
        Returns
        -------
        Out: array
            Requested synaptic indices.
        """
        
        return self.network.syn_idcs_from_groups(
            target_groupstripe_list, source_groupstripe_list, channel)
        
    def spiking_idcs(self, comparisons, groupstripe=None, tlim=None):
        """Get indices of neurons with the requested rate. The requested
        rate is defined by comparisons (>=, <= or == a certain rate).
        
        Parameters
        ----------
        comparisons: list or tuple
            A tuple of a list of tuples composed by a np.ufunc and the
            rate value (in Hz). The np.ufunc is intended to be
            np.greater, np.less, np.greater_equal or np.less_equal.
            E.g. (np.greater_equal, 0.33) retrieves neuron indices from
            neurons with spiking rare greater or equal to 0.33 Hz.
        
        groupstripe: list or tuple, optional
            Group information as in neuron_idcs. If not given, all
            the network neurons are analysed.
        
        tlim: tuple, optional
            A 2-tuple (start, stop), defining the beginning and end (in
            ms) of analysis window.
            
        Returns
        -------
        Out: array
            Array of requested neuron indices.
        """
        
        if isinstance(comparisons[0], np.ufunc):
            comparisons = [comparisons]
            
        rate = self.spiking_rate(tlim)
        
        idc_bool_list = []
        for relation, comparison_rate in comparisons: 
            idc_bool_list.append(relation(rate, comparison_rate))
        
        sp_idc = np.arange(self.neurons.N)
        
        if groupstripe is not None:
            neuronidc = self.neuron_idcs(groupstripe)
            sp_isin = np.isin(sp_idc, neuronidc)
            idc_bool_list.append(sp_isin)            
        
        idc_bool = np.isfinite(sp_idc)
        
        for item in idc_bool_list:
            idc_bool = idc_bool & item
            
        sp_idc= sp_idc[idc_bool]
        
        return sp_idc
   
    def spiking_count(self, neuron_idcs=None, tlim=None, delta_t=None):
        """Get neuron spiking count.
        
        Parameters
        ----------
        neuron_idcs: array_like, optional
            Indices of neurons that are to be analysed. If not given, 
            all neurons are analysed.
        
        tlim: tuple, optional
            A 2-tuple (start, stop), defining the beginning and end (in
            ms) of analysis window.
        
        delta_t: float, optional
            Define time bins for spiking count. If not given, the
            function retrieves the spiking count in the whole
            simulation period.
            
        Returns
        -------
        Out: array
            Array of requested count.
        """
        
        if tlim is None and delta_t is None:
            count = self.spikemonitor.count
            if neuron_idcs is not None:
                neuron_idcs = np.asarray(neuron_idcs)
                count = count[neuron_idcs]
            return count
        
        else:
            if tlim is None:
                t0, t1 = self.transient, self.neurons.t/br2.ms
            else:
                t0,t1 = tlim
            
            if neuron_idcs is None:
                neuron_idcs = np.arange(self.neurons.N)
            else:
                neuron_idcs = np.asarray(neuron_idcs)
                
            spike_trains = self.spikemonitor.spike_trains()
       
            if delta_t is None:
                count=[]
                for idc in neuron_idcs:
                    spiking = spike_trains[idc]/br2.ms
                    count.append(
                        len(spiking[(spiking >= t0) & (spiking < t1)]))
                count = np.asarray(count)
                
            else:
                N_intervals = np.ceil((t1-t0) / delta_t).astype(int)
                count = [[] for i in  range(len(neuron_idcs))]
                
                for i in range(len(neuron_idcs)):
                    spiking = spike_trains[neuron_idcs[i]]/br2.ms                
                    for n in range(N_intervals):
                        count[i].append(len(spiking[
                            (spiking >= t0 + n*delta_t) 
                            & (spiking < t0 + (n+1) * delta_t)]))
                
                count = np.asarray(count)
                
            return count
         
    def spiking_rate(self, neuron_idcs=None, tlim=None, delta_t=None):
        """Get neuron spiking rate.
        
        Parameters
        ----------
        neuron_idcs: array_like, optional
            Indices of neurons that are to be analysed. If not given, 
            all neurons are analysed.
        
        tlim: tuple, optional
            A 2-tuple (start, stop), defining the beginning and end (in
            ms) of analysis window.
        
        delta_t: float, optional
            Define time bins for spiking count. If not given, the
            function retrieves the spiking rate in the whole
            simulation period.
            
        Returns
        -------
        Out: array
            Array of requested rates.
        """
        count = self.spiking_count(neuron_idcs, tlim, delta_t)
        
        if delta_t is None:
            if tlim is None:
                return count/(self.simulated_time/1000 - self.transient/1000)
            else:
                t0,t1 = tlim
                return count/((t1-t0)/1000)
        else:
            return count/(delta_t/1000)
       
    def rheobase(self, neuron_idcs=None):   
        """Retrieve rheobase current from neurons.
        
        Parameter
        ---------
        neuron_idcs: array_like, optional
            Indices of neurons whose rheobase is requested. If not
            given, the rheobase value of all neurons is retrived.
        
        Returns
        -------
        Out: nd.array
            Values of rheobase.
        """
        return self.network.rheobase(neuron_idcs)
    
    def save_experiment(self, path, file_name):
        toml_dict = {'setup': {}, 'scales': {}, 'constant_stimuli': {}, 
                     'fluctuant_stimuli': {}, 'external_stimuli': {}, 
                     'monitors': {}, 'simulation':{}}
        
        if '.toml' not in file_name:
            file_name = file_name+'.toml'
        
        if os.path.isabs(path):
            abs_path = path
        else:
            abs_path = os.path.join(os.getcwd(), path)
        
        toml_dict['setup']['load_file'] = self._loadfile
        toml_dict['setup']['save_file'] = self._savefile
        toml_dict['setup']['method'] = self.method
        toml_dict['setup']['dt'] = self.dt/br2.ms
        toml_dict['setup']['seed'] = self.seed
        toml_dict['setup']['n_stripes'] = self.network.basics.struct.stripes.n
        toml_dict['setup']['n_cells'] = self.network.basics.struct.n_cells
        
        default_pcells = np.asarray(([47. ,  1.55,  1.55,  1.3 ,  1.3 ,  2.6 , 
                                      2.1 , 38. , 0.25, 0.25, 0.25, 0.25, 1.8 , 
                                      1.8 ]))
        
        # actual_pcells = self.network.basics.struct.p_cells_per_group
        # if np.max(np.abs(actual_pcells - default_pcells)) > 0:
        #     toml_dict['extra_setup']['alternative_pcells'] = actual_pcells
        
        toml_dict['setup']['basics_disp'] = self.basics_disp
        toml_dict['setup']['alternative_pcells'] = (
            self.alternative_pcells)
        
        if self.basics_scales is not None:
            toml_dict['scales']['basics'] = {}
            for par in self.basics_scales:
                toml_dict['scales']['basics'][par] = {}
                for i in range(len(self.basics_scales[par])):
                    info_dict, scale = self.basics_scales[par][i]
                    toml_dict['scales']['basics'][par][str(i)] = {}
                    scl = toml_dict['scales']['basics'][par][str(i)]
                    scl['info'] = info_dict
                    scl['scale'] = scale
        
        
        if self.cortex_neuron_scales is not None:
            toml_dict['scales']['neuron'] = {}
            for par in self.cortex_neuron_scales:
                toml_dict['scales']['neuron'][par] = {}
                for i in range(len(self.cortex_neuron_scales[par])):
                    gr_stripe, factor = self.cortex_neuron_scales[par][i]
                    toml_dict['scales']['neuron'][par][str(i)] = {}
                    scl = toml_dict['scales']['neuron'][par][str(i)]
                    scl['group'] = gr_stripe[0]
                    scl['stripe'] = gr_stripe[1]
                    scl['factor'] = factor
                
        if self.cortex_syn_scales is not None:
            toml_dict['scales']['syn'] = {}
            for par in self.cortex_syn_scales:
                toml_dict['scales']['syn'][par] = {}
                for i in range(len(self.cortex_syn_scales[par])):
                    tgt_groups, src_groups, channels, factor = (
                        self.cortex_syn_scales[par][i])
                    
                    toml_dict['scales']['syn'][par][str(i)] = {}
                    scl = toml_dict['scales']['syn'][par][str(i)]
                    
                    scl['tgt_group'] = tgt_groups[0]
                    scl['tgt_stripe'] = tgt_groups[1]
                    scl['src_group'] = src_groups[0]
                    scl['src_stripe'] = src_groups[1]
                    scl['channels'] = channels
                    scl['factor'] = factor
                

        for i in range(len(self.constant_stimuli)):
            toml_dict['constant_stimuli'][str(i)] = {}
            toml_dict['constant_stimuli'][str(i)]['group'] = self.constant_stimuli[i][0][0]
            toml_dict['constant_stimuli'][str(i)]['stripe'] = self.constant_stimuli[i][0][1]
            toml_dict['constant_stimuli'][str(i)]['current'] = self.constant_stimuli[i][1]
            
        
        toml_dict['simulation']['transient'] = self.transient
        toml_dict['simulation']['simulated_time'] = self.simulated_time
        
        if self.fluctuant_stimuli is not None:
            for i in range(len(self.fluctuant_stimuli)):
                mod_src = os.path.join(*self.fluctuant_stimuli[i][1].split('.'))+'.py'
                mod_name = os.path.split(mod_src)[-1]
                print(abs_path)
                shutil.copyfile(mod_src, os.path.join(abs_path, mod_name))
                
                
                toml_dict['fluctuant_stimuli'][str(i)] = {}
                stim = toml_dict['fluctuant_stimuli'][str(i)] 
                stim['group'] = self.fluctuant_stimuli[i][0][0]
                stim['stripe'] = self.fluctuant_stimuli[i][0][1]
                stim['script'] = os.path.join(abs_path, mod_name)
                stim['function'] = self.fluctuant_stimuli[i][2]
                stim['start'] = self.fluctuant_stimuli[i][3]
                stim['stop'] = self.fluctuant_stimuli[i][4]
                
            
        for stimulus in cortex.external_stimuli_info:
            info = cortex.external_stimuli_info[stimulus]
            toml_dict['external_stimuli'][stimulus] = {}
            stim = toml_dict['external_stimuli'][stimulus]
            stim['stimulator_type'] = info['stimulator_type']
            stim['n_source'] = info['n_source']
            stim['channels'] = info['channels']
            stim['target_idc'] = info['target_idc']
            stim['target_idc'] = info['target_idc']
            stim['pcon'] = info['pcon']
            stim['rate'] = info['rate']
            stim['start'] = info['start']
            stim['stop'] = info['stop']
            stim['gmax'] = info['gmax']
            stim['pfail'] = info['pfail']
            
    
        for mon in cortex.monitors_info:
            info = cortex.monitors_info[mon]
            toml_dict['monitors'][mon] = {}
            toml_dict['monitors'][mon]['type'] = info['type']
            toml_dict['monitors'][mon]['variables'] = info['variables']
            toml_dict['monitors'][mon]['idcs'] = info['idcs']
            toml_dict['monitors'][mon]['interval'] = info['interval']
            toml_dict['monitors'][mon]['start'] = info['start']
            toml_dict['monitors'][mon]['stop'] = info['stop']
            toml_dict['monitors'][mon]['population_agroupate'] = info['population_agroupate']
            
        
        
        
        
        with open(os.path.join(path, file_name), 'w') as f:
            toml.dump(toml_dict, f)
            
        
    
    def _get_membrane_events_dict(self):
        membrane_events = {}
        ctx_events = self.network.basics.equations.membr_events
        for event in ctx_events:        
            membrane_events[event] = (ctx_events[event]['condition'])
        return membrane_events

    def _set_membrane_events(self):
        self.event_monitors=_NetworkHolder()
        ctx_events = self.network.basics.equations.membr_events
        for event in ctx_events:
            self.event_monitors[event] = br2.EventMonitor(
                self.neurons, event, variables=ctx_events[event]['vars'])
            self.neurons.run_on_event(event, ctx_events[event]['reset'])    

    def _set_neuron_params(self):
        ctx_membr = self.network.basics.membr
        for par in ctx_membr.name_units:
            unit = ctx_membr.unitbr2_dict[ctx_membr.name_units[par]['unit']]
            value = ctx_membr.name_units[par]['value']
            if isinstance(value, str):  
               setattr(self.neurons, par, self.network.membr_params.loc[
                   dict(par=value)].values.astype(float)*unit)
            elif isinstance(value, int) or isinstance(value, float):
                 setattr(self.neurons, par, value*unit)
        
    def _set_auxiliar_vars(self):
        self.neurons.I_ref = self.network.refractory_current.values*br2.pA     
        self.neurons.last_spike = -1000*br2.ms
        
    def _set_initial_state(self):
        self.neurons.V = self.neurons.E_L
        self.neurons.w = 0 * br2.pA
    
    def _set_channels(self):
        ctx_channels = self.network.basics.syn.channels
        param_dict = ctx_channels.unitvalue_dict
        for channel in ctx_channels.names:
            for par in ctx_channels.params.coords['par'].values:
                paramchannel = '{}_{}'.format(par, channel)
                unit = self.network.basics.membr.unitbr2_dict[
                    param_dict[paramchannel]['unit']]
                value = float(ctx_channels.params.loc[
                    dict(par=par, channel=channel)].values)  
                setattr(
                    self.neurons, '{}_{}'.format(par, channel), value*unit)
                             
    def _set_stsp_vars(self):
        for par in list(self.network.basics.syn.stsp.decl_vars.keys()):
            unit = self.network.basics.membr.unitbr2_dict[
                self.network.basics.syn.stsp.decl_vars[par]['unit']]
            value = self.network.basics.syn.stsp.decl_vars[par]['value']
            if isinstance(value, str):  
               setattr(self.synapses, par, 
                       (self.network.syn_params['STSP_params']
                        .loc[dict(par=value)].values.astype(float)*unit))
            elif isinstance(value, int) or isinstance(value, float):
                 setattr(self.synapses, par, value*unit)
    
    def _set_syn_spike_params(self):    
        for par in list(self.network.basics.syn.spiking.names.keys()):          
            unit = self.network.basics.membr.unitbr2_dict[
                self.network.basics.syn.spiking.names[par]['unit']
                ]
            value = self.network.basics.syn.spiking.names[par]['value']
            if isinstance(value, str):  
               setattr(self.synapses, par, 
                       (self.network.syn_params['spiking']
                        .loc[dict(par=value)].values.astype(float)*unit)
                       )
            elif isinstance(value, int) or isinstance(value, float):
                 setattr(self.synapses, par, value*unit)
    
    def _get_gsyn_amp(self):
        gsyn_amp={}
        for name in self.network.basics.syn.channels.names:
            factor = (self.network.basics.syn.channels.gsyn_factor
                      .loc[dict(channel=name, par=['factor'])].values)
            tau_on, tau_off = (
                self.network.basics.syn.channels.params
                .loc[dict(channel=name, par=['tau_on', 'tau_off'])].values
                )
            gsyn_amp[name] =  factor * tau_off * tau_on/(tau_off-tau_on)
        
        return gsyn_amp
    
    
    def _set_syn_channels(self):
        for name in self.network.basics.syn.channels.names:
            setattr(
                self.synapses, name, 
                (self.network.syn_params['channel'].loc[dict(par='channel')]
                 .values==self.network.basics.syn.channels.names.index(name))
                .astype(int)
                )
            self.synapses.gsyn_amp = (
                self.synapses.gsyn_amp 
                + getattr(self.synapses, name) * self.gsyn_amp[name]
                )
            
    def _set_delay(self):
        delay = (self.network.syn_params['delay'].loc[dict(par='delay')]
                 .values.astype(float) * br2.ms)
        
        for p in range(len(self.network.basics.equations.syn_pathway)):
            getattr(
                self.synapses, 
                'p{}'.format(p)
                ).order = (self.network.basics.equations
                           .syn_pathway[p]['order'])
                           
            if self.network.basics.equations.syn_pathway[p]['delay']:
                getattr(self.synapses, 'p{}'.format(p)).delay = delay    
        
    def _get_syn_pathway_dict(self):
        syn_dict_pathway = {}
        for p in range(len(self.network.basics.equations.syn_pathway)):
            syn_dict_pathway['p{}'.format(p)] = (self.network.basics.equations
                                                 .syn_pathway[p]['eq'])
            
        return syn_dict_pathway 
    
    def _get_ext_syn_pathway_dict(self):
        syn_dict_pathway = {}
        for p in range(len(self.network.basics.equations.ext_syn_pathway)):
            syn_dict_pathway['p{}'.format(p)] = (self.network.basics.equations
                                                 .ext_syn_pathway[p]['eq'])
        return syn_dict_pathway 
    
    def _set_synapses(self):
        
        self.gsyn_amp=self._get_gsyn_amp()   
        
        if self.network.syn_pairs.shape[1] > 0:
            self._set_syn_channels()         
            self._set_stsp_vars()
            self._set_syn_spike_params()
            self._set_delay()    
    
    def _process_longrun(self, l, t1):  
        if not os.path.isdir('longrun'):
            os.mkdir('longrun')
        name_units = self.network.basics.equations.var_units
        unitbr2_dict = self.network.basics.membr.unitbr2_dict
        
        longrun = self._longrun_monitor_control[l]
        variables = longrun['monitor']._keys()
        
        if (((longrun['stop'] is not None 
                  and longrun['start']<=t1<longrun['stop']) 
                 or (longrun['stop'] is None and longrun['start']<=t1)) 
                and not longrun['longrun_monitor'].active): 
            longrun['longrun_monitor'].active = True
            for var in variables:
                longrun['files'][var] = []
        elif longrun['longrun_monitor'].active:
            i = round((t1 - longrun['start'])/longrun['interval'])
            for var in variables:
                mon_var = (getattr(longrun['longrun_monitor'], var)
                           /unitbr2_dict[name_units[var]])
                if longrun['population_agroupate'] is not None and var!='t':
                    if longrun['population_agroupate'] == 'mean':
                        mon_var = np.mean(mon_var, axis=0)
                    elif longrun['population_agroupate'] == 'sum':
                        mon_var = np.sum(mon_var, axis=0)
          
            
                with open('longrun/{}_{}_{}.npy'.format(var, i, l),'wb') as f:
                    np.save(f, mon_var)
                    longrun['files'][var].append(
                        'longrun/{}_{}_{}.npy'.format(var, i, l))
            idc = longrun['longrun_monitor'].record
            v = longrun['longrun_monitor'].record_variables
            source = longrun['longrun_monitor'].source
            
            self.net.remove(longrun['longrun_monitor'])
            longrun['longrun_monitor'] = br2.StateMonitor(source, v, idc, 
                                                          dt=self.dt)
            self.net.add(longrun['longrun_monitor'])
        
        if (longrun['stop'] is not None and t1 >= longrun['stop'] 
                and  longrun['longrun_monitor'].active):     
            longrun['longrun_monitor'].active=False
            self.net.remove(longrun['longrun_monitor'])
    
    def _restore_longrun(self, erase=True):
        
        name_units = self.network.basics.equations.var_units
        unitbr2_dict = self.network.basics.membr.unitbr2_dict
        for longrun in self._longrun_monitor_control:
            for var in list(longrun['files'].keys()):
                longvar = []
                for file in longrun['files'][var]:                                   
                    longvar.append(np.load(file))  
                
                if longrun['population_agroupate'] is not None  or var=='t':
                    longrun['monitor'][var] = np.concatenate(longvar)
                else:
                    longrun['monitor'][var] = np.concatenate(longvar, axis=1)
                    
                longrun['monitor'][var] = (longrun['monitor'][var]
                                           *unitbr2_dict[name_units[var]])
                
        if len(self._longrun_monitor_control) and erase:
            shutil.rmtree('longrun', onerror=remove_read_only)
    
    def _set_fluctuant_stimuli(self):
        if self.fluctuant_stimuli is not None:
            for target, mod, func, start, stop in self.fluctuant_stimuli:
                   
                function = getattr(import_module(mod), func)
                neuron_idcs = self.neuron_idcs(target)
                
                Nsteps_total = round(stop/(self.dt/br2.ms))
                Nsteps_start = round(start/(self.dt/br2.ms))     
                
                fluctuant_array = np.zeros(
                    (Nsteps_total, self.network.basics.struct.n_cells))
                I_arr = np.asarray(
                    [0]*Nsteps_start 
                    + [function(t) for t in np.linspace(
                        start, stop, Nsteps_total-Nsteps_start, 
                        endpoint=False)]
                    )
                
                if len(neuron_idcs)>0:
                    fluctuant_array[:, neuron_idcs] = repmat(
                        I_arr, len(neuron_idcs), 1
                        ).transpose()
                        
            fluctuant_str = 'fluctuant_array(t, i)'
            
        else:
            fluctuant_array = None
            fluctuant_str = '0'
        
        self.fluctuant_array = fluctuant_array
        self.fluctuant_str = fluctuant_str
    
    def _set_constant_stimuli(self):
        I_DC = np.zeros(self.network.basics.struct.n_cells_total)
        if self.constant_stimuli is not None:
            for target, value in self.constant_stimuli:
                if len(self.neuron_idcs(target)) > 0:
                    I_DC[self.neuron_idcs(target)] = value
         
        self.neurons.I_DC = I_DC*br2.pA
    
    def _set_custom_stimuli(self, name, n_source, channel, spike_idcs, 
                            spike_times, pairs_connected, gmax, pfail):
        
        if isinstance(channel, str):
            channel = [channel]        
        
        n_connected = pairs_connected.shape[1]
        
        channel_arr = []
        for ch in channel:
            channel_arr.append([ch]*n_connected)
        channel_arr = np.concatenate(channel_arr)
        pairs_connected = repmat(pairs_connected, 1, len(channel))
        
        if isinstance(gmax, (int, float)):
            gmax = [gmax]
        
        if len(gmax)==1:
            gmax_arr = gmax*n_connected*len(channel)
        elif len(gmax)==n_connected:   
            gmax_arr = gmax*len(channel)
            
        if isinstance(pfail, (int, float)):
            pfail = [pfail]
            
        if len(pfail)==1:
            pfail_arr = pfail*n_connected*len(channel)
        elif len(pfail)==n_connected:   
            pfail_arr = pfail*len(channel)
        
        self.external_stimuli[name] = _NetworkHolder()
        self.external_stimuli[name].generator = br2.SpikeGeneratorGroup(
            n_source, spike_idcs, spike_times*br2.ms, dt=self.dt)
        ext_syn_model = self.network.basics.equations.ext_syn_model
        ext_syn_dict = self._get_ext_syn_pathway_dict()
        
        self.external_stimuli[name].synapses = br2.Synapses(
            self.external_stimuli[name].generator, self.neurons, 
            model=ext_syn_model, on_pre=ext_syn_dict, dt=self.dt)
        self.external_stimuli[name].synapses.connect(
            i=pairs_connected[1,:], j=pairs_connected[0,:])

        for ch_name in self.network.basics.syn.channels.names:
            setattr(self.external_stimuli[name].synapses, ch_name, 
                    (channel_arr==ch_name).astype(int))
            self.external_stimuli[name].synapses.gsyn_amp = (
                self.external_stimuli[name].synapses.gsyn_amp 
                + getattr(self.external_stimuli[name].synapses, ch_name) 
                * self.gsyn_amp[ch_name])
        
        self.external_stimuli[name].synapses.pfail = pfail_arr
        self.external_stimuli[name].synapses.gmax = gmax_arr *br2.nS 
        self.external_stimuli[name].spikemonitor = br2.SpikeMonitor(
            self.external_stimuli[name].generator)
        
        self.net.add(self.external_stimuli[name].generator, 
                     self.external_stimuli[name].synapses, 
                     self.external_stimuli[name].spikemonitor)
 
    
    def _set_syn_scales(self):
        if self.cortex_syn_scales is not None:
            for par in self.cortex_syn_scales:
                if isinstance(self.cortex_syn_scales[par][-1],
                              (int, float)):
                    self.cortex_syn_scales[par] = [
                        self.cortex_syn_scales[par]]
                for scale in self.cortex_syn_scales[par]:
                    target_groups, source_groups, channels, factor = scale
                    idc = self.syn_idcs_from_groups(target_groups, 
                                                    source_groups, channels)
                    getattr(self.synapses, par)[idc] =  (
                        getattr(self.synapses, par)[idc] * factor)
      
    def _set_neuron_scales(self):
        if self.cortex_neuron_scales is not None:
            for par in self.cortex_neuron_scales:
                if isinstance(self.cortex_neuron_scales[par][-1], 
                              (int, float)):
                    self.cortex_neuron_scales[par] = (
                        [self.cortex_neuron_scales[par]])
                for scale in self.cortex_neuron_scales[par]:
                    groups, factor = scale
                    idc = self.neuron_idcs(groups)
                    getattr(self.neurons, par)[idc] = (
                        getattr(self.neurons, par)[idc] * factor)
                    
    def _set_monitors(self, monitor, variables, start, stop):
        
        if self.transient>0:
            monitor.active = False
        for var in variables:
            self.recorded[var] = monitor
            
        if start is None:
            start = self.transient
        
        if start in self._monitor_schedule:
            self._monitor_schedule[start]['start'].append(monitor)
        else:
            self._monitor_schedule[start] = dict(start=[monitor], stop=[])
            
        if stop is not None:
            if stop in self._monitor_schedule:
                self._monitor_schedule[stop]['stop'].append(monitor)
            else:
                self._monitor_schedule[stop] = dict(start=[], stop=[monitor])
                
        self.net.add(monitor)

    def _set_longrun_monitors(self, monitor, longrun_monitor, variables, 
                             interval, start, stop, population_agroupate):
        
        if self.transient>0:
            longrun_monitor.active = False
        for var in variables:
            self.recorded[var] = monitor
            
        if start is None:
            start = self.transient
        
        self._longrun_monitor_control.append(
            dict(start=start, stop=stop, interval=interval, 
                 longrun_monitor=longrun_monitor, monitor=monitor, 
                 population_agroupate=population_agroupate, files={}))
    
        self.net.add(longrun_monitor)    
    

@dataclass
class _NetworkHolder(BaseClass):
    pass
    
    
@dataclass
class _VarHolder(_NetworkHolder):
    pass
    
    def var(self, var):
        return getattr(self[var], var)
    
    def t(self, var):
        return self[var].t
    
        
@dataclass
class _StimulatorSetup(BaseClass):
    n_source: int
    spike_idcs: np.array
    spike_times: np.array
    pairs: np.array
    channel:np.array
    gmax: np.array
    pfail: np.array
    
    def __post_init__(self):
        self.channel = np.asarray(self.channel)
        self.spike_idcs = np.asarray(self.spike_idcs)
        self.spike_times = np.asarray(self.spike_times)
        self.pfail= np.asarray(self.pfail)
        
    def __add__(self, instance2):
        n_source = self.n_source + instance2.n_source
        spike_idcs = np.concatenate((self.spike_idcs, instance2.spike_idcs))
        spike_times = np.concatenate((self.spike_times, instance2.spike_times))
        pairs = np.concatenate((self.pairs, instance2.pairs))
        channel = np.concatenate((self.channel, instance2.channel))
        gmax=np.concatenate((self.gmax, instance2.gmax))
        pfail=np.concatenate((self.pfail, instance2.pfail))
        
        return _StimulatorSetup(
            n_source, spike_idcs, spike_times, pairs, channel, gmax, pfail)
    