""" This script sets the network structure and parameters based on
basics data set in _basics_setup.

This module contains:
    
    network_setup: a function that sets network structure and
    parameters.
    
    Network: a class that holds network data.
"""

import numpy as np
import xarray as xr
from dataclasses import dataclass
from warnings import filterwarnings
import os
import json
from ._auxiliary import time_report, BaseClass
from ._network_auxiliary import *
from ._basics_setup import *

__all__ = ['network_setup', 'Network']

filterwarnings("ignore", category=RuntimeWarning) 

@time_report('Network setup')
def network_setup(n_cells_prompted, n_stripes=1, basics_scales=None, seed=None,
                  alternative_pcells=None, basics_disp=True):   
    """Set network structure and information. Return Network object.
    
    Parameters
    ----------
    n_cells_prompted: int
        Number of requestes cells in stripe.
    n_stripes: int, optional
        Number of stripes. If not given, defaults to 1.
    basics_scale: dict, optional
        Parameter scales. If not given, no scale is applied.
    seed: int, optional
        Random seed. If not given, no random seed is set.
    alternative_pcells: array_like, optional
        Alternative cell distribution. If not given, the same default
        of basics_setup is used.
    basics_disp: bool, optional
        Set basics display. If not given, warning display may be
        displayed.
    """
    
    np.random.seed(seed)

    basics=basics_setup(n_cells_prompted, n_stripes, basics_scales, 
                        alternative_pcells, basics_disp)
    
    index_stripes = np.arange(n_stripes).astype(int)
    index_cells_total = np.arange(basics.struct.n_cells_total).astype(int)
    
    syn_pairs = xr.DataArray(
        np.asarray([[],[]]), coords=[['target', 'source'],[]], 
        dims=['cell', 'syn_index'])
    
    syn_channel = ['channel']
    syn_spike_params = ['gmax', 'pfail']
    syn_STSP_params = ['U', 'tau_rec', 'tau_fac']
    syn_delay = ['delay']
    
    syn_params = {}
    syn_params['channel'] = np.asarray([[] for i in range(len(syn_channel))])
    syn_params['spiking'] =np.asarray(
        [[] for i in range(len(syn_spike_params))])
    
    syn_params['STSP_params'] = np.asarray(
        [[] for i in range(len(syn_STSP_params))])
    
    syn_params['delay'] = np.asarray([[] for i in range(len(syn_delay))])
    
    
    membr_params = xr.DataArray(
        np.zeros((len(basics.membr.names), basics.struct.n_cells_total)), 
        coords=(basics.membr.names, index_cells_total), 
        dims=['par', 'cell_index'])
    
    i_refractory = xr.DataArray(
        np.zeros(len(index_cells_total)), 
        coords=[index_cells_total,], 
        dims='cell_index')
    
    # ------- Set neuron parameters and most synaptic connections -------------
 
    group_distr = [[[] for j in range(basics.struct.groups.n)] 
                   for i in range (n_stripes)]
    group_names = basics.struct.groups.names
    
    n_syn_current = 0
    n_cell_current = 0
    for stripe in index_stripes:   
        for group in group_names:
            Ncell_new = int(basics.struct.n_cells_per_group.loc[group].values)
            set_current = np.arange(Ncell_new).astype(int)+n_cell_current               
            group_distr[stripe][group_names.index(group)].extend(set_current)
            membr_params = set_membr_params(membr_params, set_current,
                                            group, basics)  
            n_cell_current += Ncell_new
                 
            for cell in set_current:
                memb = membranetuple(
                    *membr_params.loc[dict(cell_index=cell)].to_numpy())                          
                i_refractory[cell] = get_iref(memb)
                
        group_distr = redistribute(group_distr, stripe, membr_params, basics)
               
        for group in group_names:
            set_current = group_distr[stripe][group_names.index(group)]  
            pcon = float(basics.struct.conn.pcon.loc[group, group].values)
            
            if len(set_current) > 0:
                if basics.struct.conn.cluster.loc[group, group] == 1:
                    target_source, syn_idc = setcon_commneigh(
                        n_syn_current, set_current, pcon, 0.47)                 
                else: 
                    target_source, syn_idc = setcon_standard(
                        n_syn_current, set_current, set_current, pcon)         
                    
                if len(syn_idc) > 0:
                    syn_params = set_syn_params(
                        syn_params, syn_idc, group, group, basics)
                    
                    channels_curr = basics.syn.channels.kinds_to_names[
                        str(basics.syn.kinds.loc[
                            group, group].values)]
                    
                    for channel in channels_curr:
                        syn_pairs = np.concatenate(
                            (syn_pairs, target_source), axis=1)
                        n_syn_current += len(syn_idc) 
                    
        for group_tgt in group_names:
            set_current_target = (group_distr[stripe]
                                  [group_names.index(group_tgt)])
            
            for group_src in (
                    [name for name in group_names if name != group_tgt]):
                set_current_source = (group_distr[stripe]
                                      [group_names.index(group_src)])
                pcon = float(basics.struct.conn.pcon
                             .loc[group_tgt, group_src].values)
                
                if len(set_current_target)*len(set_current_source) > 0:
                    target_source, syn_idc = setcon_standard(
                        n_syn_current, set_current_target, set_current_source,
                        pcon)
                    
                    if target_source.shape[1] > 0:
                        syn_params = set_syn_params(
                             syn_params, syn_idc, group_tgt, group_src, basics)
                        
                        channels_curr = basics.syn.channels.kinds_to_names[
                            str(basics.syn.kinds.loc[
                                group_tgt, group_src].values)]
          
                        for channel in channels_curr:
                            syn_pairs = np.concatenate(
                                (syn_pairs, target_source), axis=1)
                            n_syn_current += len(syn_idc) 
                        
    # # # ------------------ Define inter-stripe connections ------------------
    
    if n_stripes > 1:
       
        for inter_set in basics.struct.stripes.inter:
            inter_dict = basics.struct.stripes.inter[inter_set]
            
            group_tgt, group_src = inter_dict['pair']
            pcon = float(
                basics.struct.conn.pcon.loc[group_tgt, group_src].values)
            
            for stripe in range(n_stripes):             
                for conn  in range(len(inter_dict['connections'])):
                    target_stripe = stripe+inter_dict['connections'][conn]
                    
                    while target_stripe < 0:
                        target_stripe += n_stripes
                    
                    while target_stripe >= n_stripes:
                        target_stripe -= n_stripes
                    
                    dist_act = abs(inter_dict['connections'][conn])
                    pcon_curr = pcon*np.exp(- dist_act
                                            /inter_dict['coefficient_0'])
                    n_target = len(group_distr[target_stripe]
                                  [group_names.index(group_tgt)]) 
                    n_source = len(group_distr[target_stripe]
                                  [group_names.index(group_src)]) 
                    
                    target_source,syn_idc = setcon_standard(
                        n_syn_current, n_target, n_source, pcon_curr)                                                                                               
                    
                    if len(syn_idc) > 0:
                        gmax_fac = np.exp(-dist_act
                                          /inter_dict['coefficient_0'])
                        delay_fac = inter_dict['coefficient_1']*dist_act
        
                        syn_params = set_syn_params(
                            syn_params, syn_idc, group_tgt, group_src,
                            basics, gmax_fac, delay_fac)
                        
                        channels_curr = basics.syn.channels.kinds_to_names[
                            str(basics.syn.kinds.loc[
                                group_tgt, group_src].values)]
                        
                        for channel in channels_curr:
                            syn_pairs = np.concatenate(
                                (syn_pairs, target_source), axis=1)
                            n_syn_current += len(syn_idc) 
                        
    syn_params['channel'] = xr.DataArray(
        syn_params['channel'], 
        coords=[syn_channel, np.arange(syn_pairs.shape[1])], 
        dims=['par', 'syn_index'])
    
    syn_params['spiking'] = xr.DataArray(
        syn_params['spiking'], 
        coords=[syn_spike_params,  np.arange(syn_pairs.shape[1])], 
        dims=['par', 'syn_index'])
    
    syn_params['STSP_params'] = xr.DataArray(
        syn_params['STSP_params'], 
        coords=[syn_STSP_params,  np.arange(syn_pairs.shape[1])], 
        dims=['par', 'syn_index'])
    
    syn_params['delay'] = xr.DataArray(
        syn_params['delay'], 
        coords=[syn_delay, np.arange(syn_pairs.shape[1])], 
        dims=['par', 'syn_index'])
    
    network = Network(membr_params, i_refractory, syn_params,
                      syn_pairs.astype(int), group_distr, seed, basics)
    
    return  network

@dataclass
class Network(BaseClass):
    """This class stores network data. It is a subclass of BaseClass.
    
    Attributes
    ----------
    membr_params: membrane parameters.
    
    refractory_current: refractory currents.
    
    syn_params: synaptic parameters.
    
    group_distr: distribution of neuron indices among groups.
    
    seed: random seed.
    
    basics: BaseClass instance.
    """
    
    membr_params: xr.DataArray
    refractory_current: xr.DataArray
    syn_params: dict
    syn_pairs: np.ndarray
    group_distr: list
    seed: int or None
    basics: BaseClass
    
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
        g_L, V_T, E_L, delta_T = self.membr_params.loc[
            dict(par=['g_L', 'V_T', 'E_L', 'delta_T'])
            ]
        i_rheo =  self.basics.equations.rheobase(g_L, V_T, E_L, delta_T)
        
        if neuron_idcs is not None:
            neuron_idcs = np.asarray(neuron_idcs)
            i_rheo = i_rheo.loc[dict(cell_index=neuron_idcs)]
        
        return i_rheo.values

    def save(self, path):
        """Save network data into files.
        
        Parameter
        ---------
        path: str
            Path where data will be stored.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        self.membr_params.to_netcdf(os.path.join(path,'membr_params.nc'))
        self.refractory_current.to_netcdf(
            os.path.join(path,'refractory_current.nc'))
        
        np.save(os.path.join(path,'syn_pairs.npy'), self.syn_pairs)
        os.mkdir(os.path.join(path,'syn_params'))
        for k in self.syn_params.keys():
            self.syn_params[k].to_netcdf(
                os.path.join(path,'syn_params','{}.nc'.format(k)))
        
       
        stripe_list = []
        for stripe in self.group_distr:
            group_list = []
            for group in stripe:
                group_list.append(','.join(np.array(group).astype(str)))
            stripe_list.append(';'.join(group_list))
        group_distr = '\n'.join(stripe_list)      
        
        with open(os.path.join(path,'group_distr.txt'), 'w') as f:
            f.write(group_distr)
            
        with open(os.path.join(path,"input.txt"), 'w') as f:
            print(self.basics.struct.n_cells_prompt, 
                  self.basics.struct.stripes.n, sep='\n', end='', file=f)
        # shutil.copyfile('_basics_setup.py', 
        #                 os.path.join(path,'_basics_setup.py'))
        
        if self.basics.scales is not None:
           with open(os.path.join(path,'basics_scales.json'), 'w') as f:
               json.dump(self.basics.scales, f)
        if self.seed is not None:
            with open(os.path.join(path, 'seed.txt'), 'w') as f:
                f.write(str(self.seed))
                
           
    def load(path, alternative_pcells=None, basics_disp=True,
             alternative_basics_setup=None):
        """Load data from previously generated network.
        
        Parameter
        ---------
        alternative_pcells: array_like, optional
            Alternative cell distribution. If not given, the same default
            of basics_setup is used.
        basics_disp: str
            Set basics display. If not given, warning display may be
            displayed.
        alternative_basics_setup: function
            Alternative basics setup function. If not given, basics_setup
            from _basics_setup is used.
            
        Return
        ------
        Out: Network
            Network object holding data from previsouly generated 
            network.
        """
        
        print('REPORT: Loading Network from {}'.format(path), end='\n\n')
        
        refractory_current = xr.open_dataarray(
            os.path.join(path, 'refractory_current.nc'))
        membr_params = xr.open_dataarray(
            os.path.join(path, 'membr_params.nc'))
        channel = xr.open_dataarray(
            os.path.join(path, 'syn_params', 'channel.nc'))
        delay = xr.open_dataarray(
            os.path.join(path, 'syn_params', 'delay.nc'))
        spiking = xr.open_dataarray(
            os.path.join(path, 'syn_params', 'spiking.nc'))
        STSP_params = xr.open_dataarray(
            os.path.join(path, 'syn_params', 'STSP_params.nc'))
        syn_pairs = np.load(os.path.join(path, 'syn_pairs.npy'))
            
        with open(os.path.join(path, 'group_distr.txt'), 'r') as f:
            group_distr_str = f.read()
        
        group_distr = []
        
        stripe_list = group_distr_str.split('\n')
        for stripe in stripe_list:
            group_distr.append([])
            group_list = stripe.split(';')
            for group in group_list:
                group_distr[-1].append([])
                cell_list = group.split(',')
                for cell in cell_list:
                    if len(cell_list) > 1 or '' not in cell_list:
                        group_distr[-1][-1].append(int(cell))
        
        syn_params={'channel': channel, 'spiking': spiking,
                    'STSP_params': STSP_params, 'delay': delay}
        
        
        # basics_script = import_module('{}.basics_setup'.format(path))
        
        with open(os.path.join(path, 'input.txt'), 'r') as f:
            prompt_str = f.read()
        n_cells_prompted, n_stripes = prompt_str.split('\n')
        n_cells_prompted = int(n_cells_prompted) 
        n_stripes = int(n_stripes)
        
        basics_scales = None
        if 'basics_scales.json' in os.listdir(path):
            with open(os.path.join(path, 'basics_scales.json'), 'r') as f:
                basics_scales = json.load(f)
    
        seed = None
        if 'seed.txt' in os.listdir(path):
            with open(os.path.join(path, 'seed.txt'), 'r') as f:
                seed = int(f.read())
        
        if alternative_basics_setup is not None:
            basics = alternative_basics_setup(
                n_cells_prompted, n_stripes, basics_scales, alternative_pcells,
                basics_disp)
        else:
            basics = basics_setup(
                n_cells_prompted, n_stripes, basics_scales, alternative_pcells,
                basics_disp)
        # basics = basics_script.basics_setup(
        #     n_cells_prompted, n_stripes, basics_scales, alternative_pcells,
        #     basics_disp)
         
        print('REPORT: Network loaded from {}'.format(path), end='\n\n')
        
        return Network(membr_params, refractory_current, syn_params,
                       syn_pairs, group_distr, seed, basics)
         
    def group_idcs(self, group):
        """Retrieve indices corresponding to the given groups.
        
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
        if isinstance(group, str):
            names = []
            for gr in self.basics.struct.groups.sets[group]:
                names.append(self.basics.struct.groups.idcs[gr])
            return names
        elif isinstance(group, int):
            return [group]
        
    def neuron_idcs(self, groupstripe_list):
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
        
        if isinstance(groupstripe_list[0], (str,int)):
            groupstripe_list = [groupstripe_list]
            
        neuron_idcs = []
        for groupstripe in groupstripe_list:
            group, stripe_idc = groupstripe
            group_idc = self.group_idcs(group)
            for idc in group_idc:
                neuron_idcs.extend(self.group_distr[stripe_idc][idc])
        
        return np.array(neuron_idcs)
    
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
        
        syn_idcs = np.arange(self.syn_pairs.shape[1])
        target_isin = np.isin(self.syn_pairs[0,:], np.array(target))
        source_isin = np.isin(self.syn_pairs[1,:], np.array(source))
        both_isin = target_isin & source_isin
        
        syn_idcs = syn_idcs[both_isin]
        
        if channel is not None:
            ch_isin = []
            if isinstance(channel, str):
                channel = [channel]
            for ch in channel:
                ch_idc = np.where(
                    self.syn_params['channel'].values[0] 
                    == self.basics.syn.channels.names.index(ch)
                    )
                ch_isin.append(syn_idcs[np.isin(syn_idcs, ch_idc)])
            ch_isin = np.concatenate(ch_isin) 
            syn_idcs = syn_idcs[np.isin(syn_idcs, ch_isin)]
            
        return syn_idcs
        
    
    def syn_idcs_from_groups(self, tgt_groupstripe_list, src_groupstripe_list,
                             channel=None):
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
        
        if isinstance(tgt_groupstripe_list[0], (str,int)):
            tgt_groupstripe_list = [tgt_groupstripe_list]
        
        if (isinstance(src_groupstripe_list[0], str) 
            or isinstance(src_groupstripe_list[0], int)):
            src_groupstripe_list = [src_groupstripe_list]
        
        syn_idcs = np.array([])
        all_syn = np.arange(self.syn_pairs.shape[1])
        for target_groupstripe in tgt_groupstripe_list:
            for source_groupstripe in src_groupstripe_list:
                targetgroup, targetstripe_idc = target_groupstripe
                sourcegroup, sourcestripe_idc = source_groupstripe
                
                targetgroup_idc = self.group_idcs(targetgroup)
                sourcegroup_idc = self.group_idcs(sourcegroup)
                
                targetneurons_idc = []
                for gr in targetgroup_idc:
                    targetneurons_idc.extend(
                        self.group_distr[targetstripe_idc][gr])
               
                sourceneurons_idc = []
                for gr in sourcegroup_idc:
                    sourceneurons_idc.extend(
                        self.group_distr[sourcestripe_idc][gr])
                
                target_isin = np.isin(
                    self.syn_pairs[0,:], np.array(targetneurons_idc))
                source_isin = np.isin(
                    self.syn_pairs[1,:], np.array(sourceneurons_idc))
                both_isin = target_isin & source_isin
                syn_idcs = np.concatenate((syn_idcs, all_syn[both_isin]))
        
        syn_idcs = syn_idcs.astype(int)   
        
        if channel is not None:
            ch_isin = []
            if isinstance(channel, str):
                channel = [channel]
            for ch in channel:
                ch_idc = np.where(
                    self.syn_params['channel'].values[0] 
                    == self.basics.syn.channels.names.index(ch))
                ch_isin.append(syn_idcs[np.isin(syn_idcs, ch_idc)])
            ch_isin = np.concatenate(ch_isin) 
            syn_idcs = syn_idcs[np.isin(syn_idcs, ch_isin)]
            
        return syn_idcs
    
    
    
if __name__ == '__main__':
    
    # basics_scales={}
    # gmax_scales = [(dict(target=group_sets['PC'], 
    #source=group_sets['PC']), 2)]    
    # basics_scales['gmax_mean'] = gmax_scales
    # network = network_setup(1000, 1)
    # network.save('New6')
    # network = Network.load('New5')

    # Ntrials = 25
    # pop1 = np.zeros((Ntrials,2,7))
    # for trial in range(Ntrials):
    # alternative_pcells = np.asarray([10, 5, 5, 5, 5, 10, 10,
    #                                  10, 5, 5, 5, 5, 10, 10])
    network = network_setup(1000, 1)#, alternative_pcells=alternative_pcells)
    
    # group_sets = [[('PC_L23',0), ('IN_CC_L23', 0)],
    #[('PC_L5',0), ('IN_CC_L5', 0)], [('IN_L_both', 0)], 
    #[('IN_CL_both')], [('IN_F',0)] ]
    # network.save('new_test')
    # network.save('new_5000')
    # for group in group_sets:
    pass
    