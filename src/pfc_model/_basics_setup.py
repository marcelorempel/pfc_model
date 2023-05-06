"""  
This script defines equations, network structure and membrane and synaptic
paramaters.

This module contains:
    basics_setup: a function that sets network basic data necessary to 
    build the model.
    
    membranetuple: a named tuple that holds the membrane parameter names.
    
    group_sets: a dictionary that defines aliases for group sets.  
"""

import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from numpy.matlib import repmat
from collections import namedtuple
import brian2 as br2
from ._auxiliary import *

__all__ = ['basics_setup', 'membranetuple', 'group_sets']

#################################################
########## Set network basic structure ##########
#################################################

# Define the name and the kind (here, excitatory or inhibitory)
# of each group; all groups have to be declared here. 
_group_kinds = {
    'PC_L23': 'exc', 'IN_L_L23': 'inh', 'IN_L_d_L23': 'inh', 
    'IN_CL_L23': 'inh', 'IN_CL_AC_L23': 'inh', 'IN_CC_L23': 'inh', 
    'IN_F_L23': 'inh', 
    'PC_L5': 'exc', 'IN_L_L5': 'inh',  'IN_L_d_L5': 'inh', 'IN_CL_L5': 'inh',
    'IN_CL_AC_L5': 'inh', 'IN_CC_L5': 'inh', 'IN_F_L5': 'inh'}

_group_names = list(_group_kinds.keys())

# Define membrane parameters, their units (as in Brian 2) and the value
# they will assume in the model equations; all membrane parameters have
# to be declared here.
# The values defined here seems redundant, but it is necessary if the value
# of one parameter depends on the other (as in STSP parameters), or if it is 
# constant.
_membpar_dict = dict(
    C=dict(unit='pF', value='C'), 
    g_L=dict(unit='nS', value='g_L'),
    E_L=dict(unit='mV', value='E_L'), 
    delta_T=dict(unit='mV', value='delta_T'),
    V_up=dict(unit='mV', value='V_up'),
    tau_w=dict(unit='ms', value='tau_w'),
    b=dict(unit='pA', value='b'), 
    V_r=dict(unit='mV', value='V_r'),
    V_T=dict(unit='mV', value='V_T'))

# Dictionary that connects the string unit names to Brian 2 corresponding
# object.
_unitbr2_dict = {'pF': br2.pF, 'nS': br2.nS, 'mV': br2.mV, 'ms': br2.ms, 
                'pA': br2.pA, 1: 1}
# Corresponding base units (they are necessary in Brian 2 model equations).
_baseunit_dict = {'pF': 'farad', 'nS': 'siemens', 'mV': 'volt', 'ms': 'second', 
                 'pA': 'amp', 1: 1}

_membpar_names = list(_membpar_dict.keys())

# Named tuple holding the names of membrane parameters
# This varible is used in other modules
membranetuple = namedtuple('membranetuple', _membpar_names)

# Define the synaptic channel names and kinds (here, excitatory or
# inhibitory). All channels have to be declared here.
_channel_kinds = {'AMPA': 'exc', 'GABA': 'inh', 'NMDA': 'exc'}

_channel_names = [name for name in list(_channel_kinds.keys())]

# Define channel parameters and its units (all paramaters have to be declared
# here); all parameters will be created for each type of channel
_channelpar_units = dict(
    tau_on='ms',
    tau_off='ms',
    E='mV',
    Mg_fac=1,
    Mg_slope=1,
    Mg_half=1)

_channelpar_names = list(_channelpar_units.keys())

# Dictionary connecting 
# For each channel "parameter" and each "type" of channel , this block creates
# "parameter_type" and connects it to the parameter unit and defines its value
# in the model equation
_channelnamepar_units = {}
for name in _channel_names:
    for par in _channelpar_names:
        _channelnamepar_units['{}_{}'.format(par, name)] = dict(
            unit=_channelpar_units[par], value='{}_{}'.format(par, name))
            
        
# Fraction of cells in each group defined in _group_kinds (in percentage).
# _pcells_per_group values must be in the same order as _group_kinds
# Here: 
# PC_L23, IN_L_L23, IN_L_d_L23, IN_CL_L23, IN_CL_AC_L23, IN_CC_L23, IN_F_L23
# PC_L5, IN_L_L5, IN_L_d_L5, IN_CL_L5, IN_CL_AC_L5, IN_CC_L5, IN_F_L5
_pcells_per_group = np.asarray([47, 1.55, 1.55, 1.3, 1.3, 2.6, 2.1,
                                38, 0.25, 0.25, 0.25, 0.25, 1.8, 1.8])

# Aliases for each set of groups
# This dictionary is exported to other modules
group_sets = {
    'ALL': ['PC_L23','IN_L_L23','IN_L_d_L23','IN_CL_L23','IN_CL_AC_L23', 
            'IN_CC_L23','IN_F_L23', 'PC_L5','IN_L_L5','IN_L_d_L5','IN_CL_L5',
            'IN_CL_AC_L5','IN_CC_L5','IN_F_L5'],   
    'L23': ['PC_L23','IN_L_L23','IN_L_d_L23','IN_CL_L23','IN_CL_AC_L23',
            'IN_CC_L23','IN_F_L23'],
    'L5': ['PC_L5', 'IN_L_L5','IN_L_d_L5','IN_CL_L5','IN_CL_AC_L5','IN_CC_L5', 
           'IN_F_L5'],    
    'PC': ['PC_L23', 'PC_L5'], 
    'PC_L23': ['PC_L23'],
    'PC_L5': ['PC_L5'],   
    'IN': ['IN_L_L23', 'IN_L_d_L23','IN_CL_L23','IN_CL_AC_L23','IN_CC_L23', 
           'IN_F_L23', 'IN_L_L5','IN_L_d_L5','IN_CL_L5','IN_CL_AC_L5', 
           'IN_CC_L5','IN_F_L5'],
    'IN_L23': ['IN_L_L23', 'IN_L_d_L23', 'IN_CL_L23', 'IN_CL_AC_L23', 
               'IN_CC_L23', 'IN_F_L23'],
    'IN_L5': ['IN_L_L5', 'IN_L_d_L5','IN_CL_L5','IN_CL_AC_L5','IN_CC_L5', 
              'IN_F_L5'],
    
    'IN_L_both': ['IN_L_L23', 'IN_L_d_L23', 'IN_L_L5', 'IN_L_d_L5'],
    'IN_CL_both': ['IN_CL_L23', 'IN_CL_AC_L23', 'IN_CL_L5', 'IN_CL_AC_L5'],
    'IN_L_both_L23': ['IN_L_L23', 'IN_L_d_L23'],
    'IN_L_both_L5': ['IN_L_L5', 'IN_L_d_L5'],
    'IN_CL_both_L23': ['IN_CL_L23', 'IN_CL_AC_L23'],
    'IN_CL_both_L5': ['IN_CL_L5', 'IN_CL_AC_L5'],
    
    'IN_L': ['IN_L_L23', 'IN_L_L5'],
    'IN_L_d': ['IN_L_d_L23', 'IN_L_d_L5'],
    'IN_CL': ['IN_CL_L23', 'IN_CL_L5'],
    'IN_CL_AC': ['IN_CL_AC_L23', 'IN_CL_AC_L5'],
    'IN_CC': ['IN_CC_L23', 'IN_CC_L5'],
    'IN_F': ['IN_F_L23', 'IN_F_L5'],
     
    'IN_L_L23': ['IN_L_L23'],
    'IN_L_d_L23': ['IN_L_d_L23'],
    'IN_CL_L23': ['IN_CL_L23'],
    'IN_CL_AC_L23': ['IN_CL_AC_L23'],
    'IN_CC_L23': ['IN_CC_L23'],
    'IN_F_L23': ['IN_F_L23'],
    
    'IN_L_L5': ['IN_L_L5'],
    'IN_L_d_L5': ['IN_L_d_L5'],
    'IN_CL_L5': ['IN_CL_L5'],
    'IN_CL_AC_L5': ['IN_CL_AC_L5'],
    'IN_CC_L5': ['IN_CC_L5'],
    'IN_F_L5': ['IN_F_L5']}

# Setup for inter-stripe connections (for simulations with more than 1 stripe)
_interstripe_sets ={}
_interstripe_sets['A']={
    'pair': ['PC_L23','PC_L23'],
    'coefficient_0': 0.909,
    'coefficient_1': 1.4587,
    'connections': [-4, -2, 2, 4]}

_interstripe_sets['B']={
    'pair': ['PC_L5','IN_CC_L5'],
    'coefficient_0': 0.909,
    'coefficient_1': 1.0375,
    'connections':[-1, 1]}

_interstripe_sets['C']={
    'pair':  ['PC_L5', 'IN_F_L5'],
    'coefficient_0': 0.909,
    'coefficient_1': 1.0375,
    'connections': [-1, 1]}

############################################
########## Set channel parameters ##########
############################################

# Define channel parameters
# in ms and mV
_ampa_par = np.array([1.4, 10, 0, 0, 0, 0], dtype='float64') 
_gaba_par = np.asarray([3, 40, -70, 0, 0, 0], dtype='float64') 
_nmda_par = np.array([4.3, 75, 0, 1, 0.0625, 0], dtype='float64')

_channel_par = xr.DataArray(
    data=np.asarray([_ampa_par, _gaba_par, _nmda_par]), 
    coords=[list(_channel_names), _channelpar_names], 
    dims=['channel', 'par'],
    name='channel par')


#############################################
########## Set membrane parameters ##########
#############################################

# Mean of parameters in the transformed space
_membpar_mean = xr.DataArray(
    data=np.zeros((len(_membpar_names), len(_group_names))),
    coords=[_membpar_names, _group_names], 
    dims=['par', 'group'],
    name='Memb_param mean')

_membpar_mean.loc[dict(par='C', group=['PC_L23', 'IN_CC_L23'])] = 3.0751
_membpar_mean.loc[dict(par='C', group=group_sets['IN_L_both'])] = 1.6902
_membpar_mean.loc[dict(par='C', group=group_sets['IN_CL_both'])] = 3.0014
_membpar_mean.loc[dict(par='C', group=group_sets['IN_F'])] = 3.3869
_membpar_mean.loc[dict(par='C', group=['PC_L5', 'IN_CC_L5'])] = 2.2513

_membpar_mean.loc[dict(par='g_L', group=['PC_L23', 'IN_CC_L23'])] = 1.9661
_membpar_mean.loc[dict(par='g_L', group=group_sets['IN_L_both'])] = 1.0353
_membpar_mean.loc[dict(par='g_L', group=group_sets['IN_CL_both'])] = 1.4581
_membpar_mean.loc[dict(par='g_L', group=group_sets['IN_F'])] = 1.0106
_membpar_mean.loc[dict(par='g_L', group=['PC_L5', 'IN_CC_L5'])] = 1.0196

_membpar_mean.loc[dict(par='E_L', group=['PC_L23', 'IN_CC_L23'])] = 3.5945
_membpar_mean.loc[dict(par='E_L', group=group_sets['IN_L_both'])] = 2.9528
_membpar_mean.loc[dict(par='E_L', group=group_sets['IN_CL_both'])] = 3.0991
_membpar_mean.loc[dict(par='E_L', group=group_sets['IN_F'])] = 3.8065
_membpar_mean.loc[dict(par='E_L', group=['PC_L5', 'IN_CC_L5'])] = 3.4415

_membpar_mean.loc[dict(par='delta_T', group=['PC_L23', 'IN_CC_L23'])] = 1.0309
_membpar_mean.loc[dict(par='delta_T', group=group_sets['IN_L_both'])] = 3.2163
_membpar_mean.loc[dict(par='delta_T', group=group_sets['IN_CL_both'])] = 3.1517
_membpar_mean.loc[dict(par='delta_T', group=group_sets['IN_F'])] = 3.0269
_membpar_mean.loc[dict(par='delta_T', group=['PC_L5', 'IN_CC_L5'])] = 1.5178

_membpar_mean.loc[dict(par='V_up', group=['PC_L23', 'IN_CC_L23'])] = 3.1428
_membpar_mean.loc[dict(par='V_up', group=group_sets['IN_L_both'])] = 2.8230
_membpar_mean.loc[dict(par='V_up', group=group_sets['IN_CL_both'])] = 2.9335
_membpar_mean.loc[dict(par='V_up', group=group_sets['IN_F'])] = 2.3911
_membpar_mean.loc[dict(par='V_up', group=['PC_L5', 'IN_CC_L5'])] = 1.0702

_membpar_mean.loc[dict(par='tau_w', group=['PC_L23', 'IN_CC_L23'])] = 4.4809
_membpar_mean.loc[dict(par='tau_w', group=group_sets['IN_L_both'])] = 1.0542
_membpar_mean.loc[dict(par='tau_w', group=group_sets['IN_CL_both'])] = 1.0730
_membpar_mean.loc[dict(par='tau_w', group=group_sets['IN_F'])] = 4.1986
_membpar_mean.loc[dict(par='tau_w', group=['PC_L5', 'IN_CC_L5'])] = 4.5650

_membpar_mean.loc[dict(par='b', group=['PC_L23', 'IN_CC_L23'])] = 1.0189
_membpar_mean.loc[dict(par='b', group=group_sets['IN_L_both'])] = 2.5959
_membpar_mean.loc[dict(par='b', group=group_sets['IN_CL_both'])] = 0.6931
_membpar_mean.loc[dict(par='b', group=group_sets['IN_F'])] = 0.8080
_membpar_mean.loc[dict(par='b', group=['PC_L5', 'IN_CC_L5'])] = 1.1154

_membpar_mean.loc[dict(par='V_r', group=['PC_L23', 'IN_CC_L23'])] = 5.0719
_membpar_mean.loc[dict(par='V_r', group=group_sets['IN_L_both'])] = 4.1321
_membpar_mean.loc[dict(par='V_r', group=group_sets['IN_CL_both'])] = 1.9059
_membpar_mean.loc[dict(par='V_r', group=group_sets['IN_F'])] = 3.0051
_membpar_mean.loc[dict(par='V_r', group=['PC_L5', 'IN_CC_L5'])] = 4.3414

_membpar_mean.loc[dict(par='V_T', group=['PC_L23', 'IN_CC_L23'])] = 2.9010
_membpar_mean.loc[dict(par='V_T', group=group_sets['IN_L_both'])] = 3.6925
_membpar_mean.loc[dict(par='V_T', group=group_sets['IN_CL_both'])] = 2.9462
_membpar_mean.loc[dict(par='V_T', group=group_sets['IN_F'])] = 3.0701
_membpar_mean.loc[dict(par='V_T', group=['PC_L5', 'IN_CC_L5'])] = 3.3302

# DataArray holding normalized matrices of membrane parameters  
# in the transformed space
_membcov_norm = xr.DataArray(
    np.zeros((len(_group_names), len(_membpar_names), len(_membpar_names))),
    coords=[_group_names, _membpar_names, _membpar_names], 
    dims=['group', 'memb_par0', 'memb_par1'],
    name='memb_par covariance normalized')    
                
_membcov_norm.loc[dict(group=['PC_L23', 'IN_CC_L23'])] = np.matrix([
    [1.0000, 0.1580, -0.5835, 0.4011, -0.0561, 0.0718, -0.2038, 0.2615,
     -0.2365],
    [0.1580, 1.0000, 0.0141, -0.1272, -0.4327, 0.1778, -0.0902, -0.0329, 
     -0.3778],
    [-0.5835, 0.0141, 1.0000, -0.6295, -0.2949, -0.2008, 0.3164, -0.2615, 
     -0.0536],
    [0.4011, -0.1272, -0.6295, 1.0000, 0.6960, -0.2587, -0.0988, 0.6113,
     0.5636],
    [-0.0561, -0.4327, -0.2949, 0.6960, 1.0000, -0.3370, 0.2042, 0.3959, 
     0.8581],
    [0.0718, 0.1778, -0.2008, -0.2587, -0.3370, 1.0000, -0.0634, -0.5202, 
     -0.3829],
    [-0.2038, -0.0902, 0.3164, -0.0988, 0.2042, -0.0634, 1.0000, 0.0559, 
     0.3322],
    [0.2615, -0.0329, -0.2615, 0.6113, 0.3959, -0.5202, 0.0559, 1.0000, 
     0.3210],
    [-0.2365, -0.3778, -0.0536, 0.5636, 0.8581, -0.3829, 0.3322, 0.3210, 
     1.0000]])
    
_membcov_norm.loc[dict(group=group_sets['IN_L_both'])] = np.matrix([
    [1.0000, -0.2894, 0.0381, 0.0664, -0.2418, 0.2253, 0.2822, -0.2919, 
     0.0581],
    [-0.2894, 1.0000, -0.2259, 0.4265, 0.1859, -0.6307, -0.0140, 0.4944, 
     0.2495],
    [0.0381, -0.2259, 1.0000, -0.2855, 0.0724, 0.1199, -0.1487, -0.3773, 
     0.1881],
    [0.0664, 0.4265, -0.2855, 1.0000, 0.2208, -0.3752, 0.0660, 0.3415, 
     0.7289],
    [-0.2418, 0.1859, 0.0724, 0.2208, 1.0000, 0.1412, -0.2931, 0.1993, 
     0.4609],
    [0.2253, -0.6307, 0.1199, -0.3752, 0.1412, 1.0000, -0.2855, -0.2046, 
     -0.1974],
    [0.2822, -0.0140, -0.1487, 0.0660, -0.2931, -0.2855, 1.0000, -0.1172, 
     -0.0851],
    [-0.2919, 0.4944, -0.3773, 0.3415, 0.1993, -0.2046, -0.1172, 1.0000, 
     0.0530],
    [0.0581, 0.2495, 0.1881, 0.7289, 0.4609, -0.1974, -0.0851, 0.0530, 
     1.0000]])

_membcov_norm.loc[dict(group=group_sets['IN_CL_both'])] = np.matrix([
    [1.0000, -0.2394, -0.6001, 0.3114, -0.2367, 0.5856, 0.2077, 0.0171,
     -0.4079],
    [-0.2394, 1.0000, -0.1764, 0.4675, 0.1810, -0.4942, -0.4389, 0.6950,
     0.0811],
    [-0.6001, -0.1764, 1.0000, -0.6002, 0.2170, -0.0922, 0.2129, -0.3566,
     0.4204],
    [0.3114, 0.4675, -0.6002, 1.0000, 0.2597, -0.1039, -0.5507, 0.7230,
     0.0775],
    [-0.2367, 0.1810, 0.2170, 0.2597, 1.0000, 0.2159, -0.7123, 0.0193,
     0.8494],
    [0.5856, -0.4942, -0.0922, -0.1039, 0.2159, 1.0000, 0.0587, -0.4724,
     0.0957],
    [0.2077, -0.4389, 0.2129, -0.5507, -0.7123, 0.0587, 1.0000, -0.3395,
     -0.5780],
    [0.0171, 0.6950, -0.3566, 0.7230, 0.0193, -0.4724, -0.3395, 1.0000,
     -0.1084],
    [-0.4079, 0.0811, 0.4204, 0.0775, 0.8494, 0.0957, -0.5780, -0.1084,
     1.0000]])

_membcov_norm.loc[dict(group=group_sets['IN_F'])] = np.matrix([
    [1.0000, -0.1586, 0.1817, -0.0195, -0.0884, 0.0282, 0.0560, -0.1369,
     0.0099],
    [-0.1586, 1.0000, 0.0440, 0.1013, -0.2510, -0.0046, -0.1105, 0.0738,
     -0.1152],
    [0.1817, 0.0440, 1.0000, -0.5118, 0.0414, 0.2570, 0.0932, 0.0961,
     0.4938],
    [-0.0195, 0.1013, -0.5118, 1.0000, 0.0480, -0.1155, -0.2463, -0.0754,
     0.0204],
    [-0.0884, -0.2510, 0.0414, 0.0480, 1.0000, 0.2577, -0.0581, 0.3152,
     0.3151],
    [0.0282, -0.0046, 0.2570, -0.1155, 0.2577, 1.0000, -0.1598, 0.4397,
     0.1107],
    [0.0560, -0.1105, 0.0932, -0.2463, -0.0581, -0.1598, 1.0000, -0.4617,
     0.1872],
    [-0.1369, 0.0738, 0.0961, -0.0754, 0.3152, 0.4397, -0.4617, 1.0000,
     -0.0114],
    [0.0099, -0.1152, 0.4938, 0.0204, 0.3151, 0.1107, 0.1872, -0.0114,
     1.0000]])

_membcov_norm.loc[dict(group=['PC_L5', 'IN_CC_L5'])] = np.matrix([
    [1.0000, -0.2440, -0.2729, 0.2863, -0.0329, 0.2925, -0.0588, 0.3377,
     -0.1914],
    [-0.2440, 1.0000, 0.0874, -0.1523, -0.2565, -0.1605, 0.0874, -0.2895,
     -0.2125],
    [-0.2729, 0.0874, 1.0000, -0.6332, 0.2012, -0.0578, 0.0283, -0.1100,
     0.3013],
    [0.2863, -0.1523, -0.6332, 1.0000, 0.3140, 0.2152, -0.1084, 0.4114,
     0.1732],
    [-0.0329, -0.2565, 0.2012, 0.3140, 1.0000, 0.3184, -0.1923, 0.3761,
     0.8433],
    [0.2925, -0.1605, -0.0578, 0.2152, 0.3184, 1.0000, 0.1246, 0.4736,
     0.2078],
    [-0.0588, 0.0874, 0.0283, -0.1084, -0.1923, 0.1246, 1.0000, 0.0752,
     -0.1578],
    [0.3377, -0.2895, -0.1100, 0.4114, 0.3761, 0.4736, 0.0752, 1.0000,
     0.2114],
    [-0.1914, -0.2125, 0.3013, 0.1732, 0.8433, 0.2078, -0.1578, 0.2114,
     1.0000]])

# Standard deviation for each membrane parameter in the transformed space
_membpar_std = xr.DataArray(
    data=np.zeros((len(_membpar_names), len(_group_names))),
    coords=[_membpar_names, _group_names],
    dims=['par', 'group'],
    name='memb_par std')

_membpar_std.loc[dict(par='C', group=['PC_L23', 'IN_CC_L23'])] = 0.4296
_membpar_std.loc[dict(par='C', group=group_sets['IN_L_both'])] = 0.0754
_membpar_std.loc[dict(par='C', group=group_sets['IN_CL_both'])] = 0.3283
_membpar_std.loc[dict(par='C', group=group_sets['IN_F'])] = 0.5029
_membpar_std.loc[dict(par='C', group=['PC_L5', 'IN_CC_L5'])] = 0.1472

_membpar_std.loc[dict(par='g_L', group=['PC_L23', 'IN_CC_L23'])] = 0.3558
_membpar_std.loc[dict(par='g_L', group=group_sets['IN_L_both'])] = 0.0046
_membpar_std.loc[dict(par='g_L', group=group_sets['IN_CL_both'])] = 0.1844
_membpar_std.loc[dict(par='g_L', group=group_sets['IN_F'])] = 0.0022
_membpar_std.loc[dict(par='g_L', group=['PC_L5', 'IN_CC_L5'])] = 0.0030

_membpar_std.loc[dict(par='E_L', group=['PC_L23', 'IN_CC_L23'])] = 0.3644
_membpar_std.loc[dict(par='E_L', group=group_sets['IN_L_both'])] = 0.3813
_membpar_std.loc[dict(par='E_L', group=group_sets['IN_CL_both'])] = 0.3630
_membpar_std.loc[dict(par='E_L', group=group_sets['IN_F'])] = 0.3359
_membpar_std.loc[dict(par='E_L', group=['PC_L5', 'IN_CC_L5'])] = 0.2846

_membpar_std.loc[dict(par='delta_T', group=['PC_L23', 'IN_CC_L23'])] = 0.0048
_membpar_std.loc[dict(par='delta_T', group=group_sets['IN_L_both'])] = 0.7107
_membpar_std.loc[dict(par='delta_T', group=group_sets['IN_CL_both'])] = 0.3568
_membpar_std.loc[dict(par='delta_T', group=group_sets['IN_F'])] =  0.7395
_membpar_std.loc[dict(par='delta_T', group=['PC_L5', 'IN_CC_L5'])] = 0.0554

_membpar_std.loc[dict(par='V_up', group=['PC_L23', 'IN_CC_L23'])] = 0.5259
_membpar_std.loc[dict(par='V_up', group=group_sets['IN_L_both'])] = 0.5033
_membpar_std.loc[dict(par='V_up', group=group_sets['IN_CL_both'])] = 0.4372
_membpar_std.loc[dict(par='V_up', group=group_sets['IN_F'])] = 0.3035
_membpar_std.loc[dict(par='V_up', group=['PC_L5', 'IN_CC_L5'])] = 0.0062

_membpar_std.loc[dict(par='tau_w', group=['PC_L23', 'IN_CC_L23'])] = 0.4947
_membpar_std.loc[dict(par='tau_w', group=group_sets['IN_L_both'])] = 0.0052
_membpar_std.loc[dict(par='tau_w', group=group_sets['IN_CL_both'])] = 0.0170
_membpar_std.loc[dict(par='tau_w', group=group_sets['IN_F'])] = 0.3186
_membpar_std.loc[dict(par='tau_w', group=['PC_L5', 'IN_CC_L5'])] = 0.6356

_membpar_std.loc[dict(par='b', group=['PC_L23', 'IN_CC_L23'])] = 0.0113
_membpar_std.loc[dict(par='b', group=group_sets['IN_L_both'])] = 1.9269
_membpar_std.loc[dict(par='b', group=group_sets['IN_CL_both'])] = 1.4550
_membpar_std.loc[dict(par='b', group=group_sets['IN_F'])] = 1.0353
_membpar_std.loc[dict(par='b', group=['PC_L5', 'IN_CC_L5'])] = 1.3712

_membpar_std.loc[dict(par='V_r', group=['PC_L23', 'IN_CC_L23'])] = 0.6104
_membpar_std.loc[dict(par='V_r', group=group_sets['IN_L_both'])] = 0.4817
_membpar_std.loc[dict(par='V_r', group=group_sets['IN_CL_both'])] = 0.1504
_membpar_std.loc[dict(par='V_r', group=group_sets['IN_F'])] = 0.1813
_membpar_std.loc[dict(par='V_r', group=['PC_L5', 'IN_CC_L5'])] = 0.3497

_membpar_std.loc[dict(par='V_T', group=['PC_L23', 'IN_CC_L23'])] = 0.4608
_membpar_std.loc[dict(par='V_T', group=group_sets['IN_L_both'])] = 0.4385
_membpar_std.loc[dict(par='V_T', group=group_sets['IN_CL_both'])] = 0.4311
_membpar_std.loc[dict(par='V_T', group=group_sets['IN_F'])] = 0.3632
_membpar_std.loc[dict(par='V_T', group=['PC_L5', 'IN_CC_L5'])] = 0.2857

# Lambdas of Tukey's "Ladder of Power" transformation
_membpar_lambd_transf = xr.DataArray(
    data=np.zeros((len(_membpar_names), len(_group_names))),
    coords=[_membpar_names, _group_names], 
    dims=['par', 'group'],
    name='memb_par lambda transf')

_membpar_lambd_transf.loc[dict(group=['PC_L23', 'IN_CC_L23'])] = np.asarray([[
    0.37, 0 , 0 , 0.01, 0 , 0 , 0.01, 0 , 0]]).T
_membpar_lambd_transf.loc[dict(group=group_sets['IN_L_both'])] = np.asarray([[
    0.22, 0.02, 0.36, 0, 0, 0.02, 0.36, 0, 0]]).T
_membpar_lambd_transf.loc[dict(group=group_sets['IN_CL_both'])] = np.asarray([[
    0, 0, 0, 0, 0, 0.02, 0, 0.12, 0]]).T
_membpar_lambd_transf.loc[dict(group=group_sets['IN_F'])] = np.asarray([[
    0, 0.01, 0, 0, 0, 0, 0, 0.26, 0]]).T
_membpar_lambd_transf.loc[dict(group=['PC_L5', 'IN_CC_L5'])] = np.asarray([[
    0.23, 0.01, 0, 0.13, 0.02, 0, 0, 0, 0]]).T

# Minimum of each membrane parameter in the inverted (original) space
_membpar_min = xr.DataArray(
    data=np.zeros((len(_membpar_names), len(_group_names))),
    coords=[_membpar_names, _group_names], 
    dims=['par', 'group'],
    name='Memb_param min')

_membpar_min.loc[dict(par='C', group=['PC_L23', 'IN_CC_L23'])] = 61.4187
_membpar_min.loc[dict(par='C', group=group_sets['IN_L_both'])] = 42.1156
_membpar_min.loc[dict(par='C', group=group_sets['IN_CL_both'])] = 51.8447
_membpar_min.loc[dict(par='C', group=group_sets['IN_F'])] = 32.3194
_membpar_min.loc[dict(par='C', group=['PC_L5', 'IN_CC_L5'])] = 110.7272

_membpar_min.loc[dict(par='g_L', group=['PC_L23', 'IN_CC_L23'])] = 3.2940
_membpar_min.loc[dict(par='g_L', group=group_sets['IN_L_both'])] = 3.6802
_membpar_min.loc[dict(par='g_L', group=group_sets['IN_CL_both'])] = 2.9852
_membpar_min.loc[dict(par='g_L', group=group_sets['IN_F'])] = 2.1462
_membpar_min.loc[dict(par='g_L', group=['PC_L5', 'IN_CC_L5'])] = 3.4510

_membpar_min.loc[dict(par='E_L', group=['PC_L23', 'IN_CC_L23'])] = -104.9627
_membpar_min.loc[dict(par='E_L', group=group_sets['IN_L_both'])] = -96.9345
_membpar_min.loc[dict(par='E_L', group=group_sets['IN_CL_both'])] = -98.8335
_membpar_min.loc[dict(par='E_L', group=group_sets['IN_F'])] = -102.3895
_membpar_min.loc[dict(par='E_L', group=['PC_L5', 'IN_CC_L5'])] = -101.5624

_membpar_min.loc[dict(par='delta_T', group=['PC_L23', 'IN_CC_L23'])] = 10.5568
_membpar_min.loc[dict(par='delta_T', group=group_sets['IN_L_both'])] = 2.1840
_membpar_min.loc[dict(par='delta_T', group=group_sets['IN_CL_both'])] = 11.0503
_membpar_min.loc[dict(par='delta_T', group=group_sets['IN_F'])] = 1.8285
_membpar_min.loc[dict(par='delta_T', group=['PC_L5', 'IN_CC_L5'])] = 12.7969

_membpar_min.loc[dict(par='V_up', group=['PC_L23', 'IN_CC_L23'])] = -62.5083
_membpar_min.loc[dict(par='V_up', group=group_sets['IN_L_both'])] = -60.6745
_membpar_min.loc[dict(par='V_up', group=group_sets['IN_CL_both'])] = -65.4193 
_membpar_min.loc[dict(par='V_up', group=group_sets['IN_F'])] = -42.8895
_membpar_min.loc[dict(par='V_up', group=['PC_L5', 'IN_CC_L5'])] = -66.1510

_membpar_min.loc[dict(par='tau_w', group=['PC_L23', 'IN_CC_L23'])] = 54.0018
_membpar_min.loc[dict(par='tau_w', group=group_sets['IN_L_both'])] = 10.2826
_membpar_min.loc[dict(par='tau_w', group=group_sets['IN_CL_both'])] = 12.2898
_membpar_min.loc[dict(par='tau_w', group=group_sets['IN_F'])] = 20.0311
_membpar_min.loc[dict(par='tau_w', group=['PC_L5', 'IN_CC_L5'])] = 33.1367

_membpar_min.loc[dict(par='b', group=['PC_L23', 'IN_CC_L23'])] = 1.2406
_membpar_min.loc[dict(par='b', group=group_sets['IN_L_both'])] = 1.0000
_membpar_min.loc[dict(par='b', group=group_sets['IN_CL_both'])] = 1.0000
_membpar_min.loc[dict(par='b', group=group_sets['IN_F'])] = 1.0000
_membpar_min.loc[dict(par='b', group=['PC_L5', 'IN_CC_L5'])] = 1.0000

_membpar_min.loc[dict(par='V_r', group=['PC_L23', 'IN_CC_L23'])] = -219.2039
_membpar_min.loc[dict(par='V_r', group=group_sets['IN_L_both'])] = -128.4559
_membpar_min.loc[dict(par='V_r', group=group_sets['IN_CL_both'])] = -271.9846
_membpar_min.loc[dict(par='V_r', group=group_sets['IN_F'])] = -105.1880
_membpar_min.loc[dict(par='V_r', group=['PC_L5', 'IN_CC_L5'])] = -124.5158

_membpar_min.loc[dict(par='V_T', group=['PC_L23', 'IN_CC_L23'])] = -63.2375
_membpar_min.loc[dict(par='V_T', group=group_sets['IN_L_both'])] = -85.2096
_membpar_min.loc[dict(par='V_T', group=group_sets['IN_CL_both'])] = -70.3537
_membpar_min.loc[dict(par='V_T', group=group_sets['IN_F'])] = -53.3897
_membpar_min.loc[dict(par='V_T', group=['PC_L5', 'IN_CC_L5'])] = -69.5922

# Maximum of each membrane parameter in the inverted (original) space
_membpar_max = xr.DataArray(
    np.zeros((len(_membpar_names), len(_group_names))),
    coords=[_membpar_names, _group_names],
    dims=['par', 'group'],
    name='Memb_param max')

_membpar_max.loc[dict(par='C', group=['PC_L23', 'IN_CC_L23'])] = 337.9765
_membpar_max.loc[dict(par='C', group=group_sets['IN_L_both'])] = 94.6939
_membpar_max.loc[dict(par='C', group=group_sets['IN_CL_both'])] = 126.2367
_membpar_max.loc[dict(par='C', group=group_sets['IN_F'])] = 201.3221
_membpar_max.loc[dict(par='C', group=['PC_L5', 'IN_CC_L5'])] = 617.2776

_membpar_max.loc[dict(par='g_L', group=['PC_L23', 'IN_CC_L23'])] = 10.8106
_membpar_max.loc[dict(par='g_L', group=group_sets['IN_L_both'])] = 8.6130
_membpar_max.loc[dict(par='g_L', group=group_sets['IN_CL_both'])] = 5.6192
_membpar_max.loc[dict(par='g_L', group=group_sets['IN_F'])] = 5.3460
_membpar_max.loc[dict(par='g_L', group=['PC_L5', 'IN_CC_L5'])] = 15.6329

_membpar_max.loc[dict(par='E_L', group=['PC_L23', 'IN_CC_L23'])] = -76.8526 
_membpar_max.loc[dict(par='E_L', group=group_sets['IN_L_both'])] = -71.7548
_membpar_max.loc[dict(par='E_L', group=group_sets['IN_CL_both'])] = -75.7868
_membpar_max.loc[dict(par='E_L', group=group_sets['IN_F'])] = -59.6898
_membpar_max.loc[dict(par='E_L', group=['PC_L5', 'IN_CC_L5'])] = -66.4770

_membpar_max.loc[dict(par='delta_T', group=['PC_L23', 'IN_CC_L23'])] = 45.3814
_membpar_max.loc[dict(par='delta_T', group=group_sets['IN_L_both'])] = 40.4333
_membpar_max.loc[dict(par='delta_T', group=group_sets['IN_CL_both'])] = 31.3533
_membpar_max.loc[dict(par='delta_T', group=group_sets['IN_F'])] =  47.6214
_membpar_max.loc[dict(par='delta_T', group=['PC_L5', 'IN_CC_L5'])] = 43.5882

_membpar_max.loc[dict(par='V_up', group=['PC_L23', 'IN_CC_L23'])] = -30.0577
_membpar_max.loc[dict(par='V_up', group=group_sets['IN_L_both'])] = -36.5929
_membpar_max.loc[dict(par='V_up', group=group_sets['IN_CL_both'])] = -45.6445
_membpar_max.loc[dict(par='V_up', group=group_sets['IN_F'])] = -30.7977
_membpar_max.loc[dict(par='V_up', group=['PC_L5', 'IN_CC_L5'])] = -25.2891

_membpar_max.loc[dict(par='tau_w', group=['PC_L23', 'IN_CC_L23'])] = 232.8699
_membpar_max.loc[dict(par='tau_w', group=group_sets['IN_L_both'])] = 21.9964
_membpar_max.loc[dict(par='tau_w', group=group_sets['IN_CL_both'])] = 120.5043
_membpar_max.loc[dict(par='tau_w', group=group_sets['IN_F'])] = 102.4180
_membpar_max.loc[dict(par='tau_w', group=['PC_L5', 'IN_CC_L5'])] = 909.5520

_membpar_max.loc[dict(par='b', group=['PC_L23', 'IN_CC_L23'])] = 40.2930
_membpar_max.loc[dict(par='b', group=group_sets['IN_L_both'])] = 196.7634
_membpar_max.loc[dict(par='b', group=group_sets['IN_CL_both'])] = 71.0958
_membpar_max.loc[dict(par='b', group=group_sets['IN_F'])] = 54.2781
_membpar_max.loc[dict(par='b', group=['PC_L5', 'IN_CC_L5'])] = 325.7906

_membpar_max.loc[dict(par='V_r', group=['PC_L23', 'IN_CC_L23'])] = -45.0393
_membpar_max.loc[dict(par='V_r', group=group_sets['IN_L_both'])] = -56.5047
_membpar_max.loc[dict(par='V_r', group=group_sets['IN_CL_both'])] = -56.8682
_membpar_max.loc[dict(par='V_r', group=group_sets['IN_F'])] = -35.7409
_membpar_max.loc[dict(par='V_r', group=['PC_L5', 'IN_CC_L5'])] = -35.1145

_membpar_max.loc[dict(par='V_T', group=['PC_L23', 'IN_CC_L23'])] = -36.8701
_membpar_max.loc[dict(par='V_T', group=group_sets['IN_L_both'])] = -39.1085
_membpar_max.loc[dict(par='V_T', group=group_sets['IN_CL_both'])] = -49.0974
_membpar_max.loc[dict(par='V_T', group=group_sets['IN_F'])] = -20.6720
_membpar_max.loc[dict(par='V_T', group=['PC_L5', 'IN_CC_L5'])] = -27.8669

# Minimum of tau_m in the inverted (original) space.
_membtau_min = xr.DataArray(
    data=np.zeros(len(_group_names)),
    coords=[_group_names], 
    dims='group',
    name='Memb_param tau_m min',)   

_membtau_min.loc[dict(group=['PC_L23', 'IN_CC_L23'])] = 10.3876
_membtau_min.loc[dict(group=group_sets['IN_L_both'])] = 7.3511
_membtau_min.loc[dict(group=group_sets['IN_CL_both'])] = 9.2264
_membtau_min.loc[dict(group=group_sets['IN_F'])] = 5.8527
_membtau_min.loc[dict(group=['PC_L5', 'IN_CC_L5'])] = 16.7015

# Maximum of tau_m in the inverted (original) space.
_membtau_max =  xr.DataArray(
    data=np.zeros(len(_group_names)),
    coords=[_group_names], 
    dims='group',
    name='Memb_param tau_m max')

_membtau_max.loc[dict(group=['PC_L23', 'IN_CC_L23'])] = 42.7304
_membtau_max.loc[dict(group=group_sets['IN_L_both'])] = 15.9128
_membtau_max.loc[dict(group=group_sets['IN_CL_both'])] = 25.9839
_membtau_max.loc[dict(group=group_sets['IN_F'])] = 48.7992
_membtau_max.loc[dict(group=['PC_L5', 'IN_CC_L5'])] = 67.7062

######################################
########## Set connectivity ##########
######################################

# Connection probabilities for each pair 
# pre- (source) and post- (target) synaptic
_syn_pcon = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))),
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='pCon')

_syn_pcon.loc[dict(target='PC_L23', source='PC_L23')] = 0.1393
_syn_pcon.loc[dict(target='PC_L23', source='PC_L5')] = 0.0449
_syn_pcon.loc[dict(target='PC_L5', source='PC_L23')] = 0.2333
_syn_pcon.loc[dict(target='PC_L5', source='PC_L5')] = 0.0806

_syn_pcon.loc[
    dict(target=group_sets['IN_L_both_L23'],  source='PC_L23')] = 0.3247
_syn_pcon.loc[
    dict(target=group_sets['IN_L_both_L23'], source='PC_L5')] = 0.1875
_syn_pcon.loc[
    dict(target=group_sets['IN_L_both_L5'], source='PC_L23')] = 0.0870
_syn_pcon.loc[
    dict(target=group_sets['IN_L_both_L5'], source='PC_L5')] = 0.3331  
_syn_pcon.loc[
    dict(target=group_sets['IN_CL_both_L23'], source='PC_L23')] = 0.1594
_syn_pcon.loc[
    dict(target=group_sets['IN_CL_both_L23'], source='PC_L5')] = 0.0920
_syn_pcon.loc[
    dict(target=group_sets['IN_CL_both_L5'], source='PC_L23')] = 0.0800
_syn_pcon.loc[
    dict(target=group_sets['IN_CL_both_L5'], source='PC_L5')] = 0.0800   

_syn_pcon.loc[dict(target='IN_CC_L23', source='PC_L23')] = 0.3247
_syn_pcon.loc[dict(target='IN_CC_L23', source='PC_L5')] = 0.1875
_syn_pcon.loc[dict(target='IN_CC_L5', source='PC_L23')] = 0.0870
_syn_pcon.loc[dict(target='IN_CC_L5', source='PC_L5')] = 0.3331
_syn_pcon.loc[dict(target='IN_F_L23', source='PC_L23')] = 0.2900
_syn_pcon.loc[dict(target='IN_F_L23', source='PC_L5')] = 0.1674
_syn_pcon.loc[dict(target='IN_F_L5', source='PC_L23')] = 0.1500
_syn_pcon.loc[dict(target='IN_F_L5', source='PC_L5')] = 0.3619

_syn_pcon.loc[
    dict(target='PC_L23', source=group_sets['IN_L_both_L23'])] = 0.4586
_syn_pcon.loc[
    dict(target='PC_L23', source=group_sets['IN_L_both_L5'])] = 0.0991
_syn_pcon.loc[
    dict(target='PC_L5', source=group_sets['IN_L_both_L23'])] = 0.2130
_syn_pcon.loc[
    dict(target='PC_L5', source=group_sets['IN_L_both_L5'])] = 0.7006
_syn_pcon.loc[
    dict(target='PC_L23', source=group_sets['IN_CL_both_L23'])] = 0.4164
_syn_pcon.loc[
    dict(target='PC_L23', source=group_sets['IN_CL_both_L5'])] = 0.0321
_syn_pcon.loc[
    dict(target='PC_L5', source=group_sets['IN_CL_both_L23'])] = 0.1934
_syn_pcon.loc[
    dict(target='PC_L5', source=group_sets['IN_CL_both_L5'])] = 0.2271
_syn_pcon.loc[dict(target='PC_L23', source='IN_CC_L23')] = 0.4586
_syn_pcon.loc[dict(target='PC_L23', source='IN_CC_L5')] = 0.0991
_syn_pcon.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.2130
_syn_pcon.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.7006
_syn_pcon.loc[dict(target='PC_L23', source='IN_F_L23')] = 0.6765
_syn_pcon.loc[dict(target='PC_L23', source='IN_F_L5')] = 0.1287
_syn_pcon.loc[dict(target='PC_L5', source='IN_F_L23')] = 0.3142
_syn_pcon.loc[dict(target='PC_L5', source='IN_F_L5')] = 0.9096

_syn_pcon.loc[
    dict(target=group_sets['IN_L23'], source=group_sets['IN_L23'])] = 0.25
_syn_pcon.loc[
    dict(target=group_sets['IN_L5'], source=group_sets['IN_L5'])] = 0.60

# Define groups following the commom neighbour rule
_syn_clusterflag = xr.DataArray(
    data=np.zeros((len(_group_names),len(_group_names))),
    coords=[_group_names,_group_names],
    dims=['target', 'source'],
    name='Clustering flag')                      

_syn_clusterflag.loc[dict(target='PC_L23', source='PC_L23')] = 1
_syn_clusterflag.loc[dict(target='PC_L5', source='PC_L5')] = 1

#############################################
########## Set synaptic parameters ##########
#############################################

_synapse_kind = xr.DataArray(
    data=np.array([[tp for tp in _group_kinds.values()] 
                   for i in range(len(_group_names))]),
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='synapse kinds')

# Define synaptic parameters directly associated to spiking activity and its 
# units and values; here, they are gmax (synaptic strength) and pfail.
_synspike_par = {'gmax': dict(unit='nS', value='gmax'), 
                'pfail': dict(unit=1, value='pfail')}

_synspike_par_names = list(_synspike_par.keys())

# Auxiliary synaptic parameters/variables and its units
_synaux_par = {'failure': dict(unit=1), 'gsyn_amp': dict(unit=1)}
_synaux_par_names = list(_synaux_par.keys())

### STSP parameters
### ---------------

# Define STSP parameters and its units and values
_stsp_par_dict = {'U': dict(unit=1, value='U'), 
                  'tau_rec': dict(unit='ms', value='tau_rec'), 
                  'tau_fac': dict(unit='ms', value='tau_rec')}
# Define STSP variables and its units and values
_stsp_var_dict = {'u_temp': dict(unit=1, value='U'), 
                  'u': dict(unit=1, value='U'), 
                  'R_temp': dict(unit=1, value=1),
                  'R': dict(unit=1, value=1), 
                  'a_syn':dict(unit=1, value='U')}

_stsp_all_dict = _stsp_par_dict | _stsp_var_dict

# Names of parameters and of parameters and variables together defined in
# _stsp_par_dict and _stsp_all_dict
_stsp_par_names = list(_stsp_par_dict.keys())
_stsp_all_names = list(_stsp_all_dict.keys())

# Types of STSP and corresponding mean and standard deviation of their
# parameters
_stsp_type_dict = {
    'E1': {'mean': np.asarray([0.28, 194, 507]), 
           'std': np.asarray([0.02, 18, 37])},              
    'E2': {'mean': np.asarray([0.25, 671, 17]),
           'std': np.asarray([0.02, 17, 5])},
    'E3': {'mean': np.asarray([0.29, 329, 326]),
           'std': np.asarray([0.03, 53, 66])},
    'I1': {'mean': np.asarray([0.16, 45, 376]),
           'std': np.asarray([0.10, 21, 253])},
    'I2': {'mean': np.asarray([0.25, 706, 21]),
           'std': np.asarray([0.13, 405, 9])},
    'I3': {'mean': np.asarray([0.32, 144, 62]),
           'std': np.asarray([0.14, 80, 31])}}

_stsp_types = {}
_stsp_types['mean'] = xr.DataArray(
    data=np.array([stsp['mean'] for stsp in list(_stsp_type_dict.values())]), 
    coords=[list(_stsp_type_dict.keys()), _stsp_par_names], 
    dims=['kind', 'par'],
    name='STSP means')
_stsp_types['std'] = xr.DataArray(
    data=np.array([stsp['std'] for stsp in list(_stsp_type_dict.values())]), 
    coords=[list(_stsp_type_dict.keys()), _stsp_par_names], 
    dims=['kind', 'par'],
    name='STSP std')

# Combination of STSP types; values in each array are the fraction of
# cells with each STSP type in the same order as in _stsp_type_dict
_stsp_set_dict = {'A': np.asarray([0.45, 0.38, 0.17, np.NaN, np.NaN, np.NaN]),
                  'B': np.asarray([1, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
                  'C': np.asarray([np.NaN, 1, np.NaN, np.NaN, np.NaN, np.NaN]),
                  'D': np.asarray([np.NaN, np.NaN, np.NaN, 0.25, 0.5, 0.25]),
                  'E': np.asarray([np.NaN, np.NaN, np.NaN, np.NaN, 1, np.NaN]),
                  'F': np.asarray([np.NaN, np.NaN, np.NaN, 0.29, 0.58, 0.13])}
_stsp_set_probs = xr.DataArray(
    data=np.array(list(_stsp_set_dict.values())), 
    coords=[list(_stsp_set_dict.keys()), list(_stsp_type_dict.keys())],
    dims=['set', 'kind'],
    name='STSP set distribution')

# Distribution of the kinds of combinations defined in _stsp_set_dict
_stsp_set_distrib = np.asarray(
    [['' for i in range(len(_group_names))] for j in range(len(_group_names))], 
    dtype='U16')
_stsp_set_distrib = xr.DataArray(
    data=_stsp_set_distrib,
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='STSP distribution groups')
_stsp_set_distrib.loc[
    dict(target=group_sets['PC'], source=group_sets['PC'])] = 'A'
_stsp_set_distrib.loc[
    dict(target=group_sets['IN_L'], source=group_sets['PC'])] = 'B'
_stsp_set_distrib.loc[
    dict(target=group_sets['IN_L_d'], source=group_sets['PC'])] = 'C'
_stsp_set_distrib.loc[
    dict(target=group_sets['IN_CL'], source=group_sets['PC'])] = 'C'
_stsp_set_distrib.loc[
    dict(target=group_sets['IN_CL_AC'], source=group_sets['PC'])] = 'B'
_stsp_set_distrib.loc[
    dict(target=group_sets['IN_CC'], source=group_sets['PC'])] = 'B'
_stsp_set_distrib.loc[
    dict(target=group_sets['IN_F'], source=group_sets['PC'])] = 'C'

_stsp_set_distrib.loc[
    dict(target=group_sets['PC'], source=group_sets['IN_L'])] = 'D'
_stsp_set_distrib.loc[
    dict(target=group_sets['PC'], source=group_sets['IN_L_d'])] = 'E'
_stsp_set_distrib.loc[
    dict(target=group_sets['PC'], source=group_sets['IN_CL'])] = 'E'
_stsp_set_distrib.loc[
    dict(target=group_sets['PC'], source=group_sets['IN_CL_AC'])] = 'E'
_stsp_set_distrib.loc[
    dict(target=group_sets['PC'], source=group_sets['IN_CC'])] = 'E'
_stsp_set_distrib.loc[
    dict(target=group_sets['PC'], source=group_sets['IN_F'])] = 'E'

_stsp_set_distrib.loc[
    dict(target=group_sets['IN_L23'], source=group_sets['IN_L23'])] = 'F'
_stsp_set_distrib.loc[
    dict(target=group_sets['IN_L5'], source=group_sets['IN_L5'])] = 'F'

### Delay
### -----

# Define synaptic delay mean
_syn_delay = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))), 
    coords=[_group_names, _group_names], 
    dims=['target', 'source'])

_syn_delay.loc[dict(target='PC_L23', source='PC_L23')] = 1.5465
_syn_delay.loc[dict(target='PC_L23', source='PC_L5')] = 2.7533
_syn_delay.loc[dict(target='PC_L5', source='PC_L23')] = 1.9085
_syn_delay.loc[dict(target='PC_L5', source='PC_L5')] = 1.5667

_syn_delay.loc[dict(target='PC_L23', source=group_sets['IN_L23'])] = 1.2491
_syn_delay.loc[dict(target='PC_L23', source=group_sets['IN_L5'])] = 1.4411
_syn_delay.loc[dict(target='PC_L5', source=group_sets['IN_L23'])] = 1.5415  
_syn_delay.loc[dict(target='PC_L5', source=group_sets['IN_L5'])]  = 0.82

_syn_delay.loc[dict(target=group_sets['IN_L23'], source='PC_L23')] = 0.9581
_syn_delay.loc[dict(target=group_sets['IN_L23'], source='PC_L5')]  = 1.0544
_syn_delay.loc[dict(target=group_sets['IN_L5'], source='PC_L23')]  = 1.1825
_syn_delay.loc[dict(target=group_sets['IN_L5'], source='PC_L5')] = 0.6
_syn_delay.loc[
    dict(target=group_sets['IN_L23'], source=group_sets['IN_L23'])] = 1.1
_syn_delay.loc[
    dict(target=group_sets['IN_L5'], source=group_sets['IN_L5'])] = 1.1

# Define delay standard deviation
_syn_delay_sigma = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))),
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='delay sigma')

_syn_delay_sigma.loc[dict(target='PC_L23', source='PC_L23')] = 0.3095
_syn_delay_sigma.loc[dict(target='PC_L23', source='PC_L5')] = 0.1825
_syn_delay_sigma.loc[dict(target='PC_L5', source='PC_L23')] = 0.1651
_syn_delay_sigma.loc[dict(target='PC_L5', source='PC_L5')] = 0.4350  
_syn_delay_sigma.loc[
    dict(target=group_sets['IN_L23'], source='PC_L23')] = 0.2489
_syn_delay_sigma.loc[
    dict(target=group_sets['IN_L23'], source='PC_L5')]  = 0.0839
_syn_delay_sigma.loc[
    dict(target=group_sets['IN_L5'], source='PC_L23')]  = 0.1327
_syn_delay_sigma.loc[
    dict(target=group_sets['IN_L5'], source='PC_L5')] = 0.2000
_syn_delay_sigma.loc[
    dict(target='PC_L23', source=group_sets['IN_L23'])] = 0.1786
_syn_delay_sigma.loc[
    dict(target='PC_L23', source=group_sets['IN_L5'])] = 0.0394
_syn_delay_sigma.loc[
    dict(target='PC_L5', source=group_sets['IN_L23'])] = 0.0940
_syn_delay_sigma.loc[
    dict(target='PC_L5', source=group_sets['IN_L5'])] = 0.0940
_syn_delay_sigma.loc[
    dict(target=group_sets['IN_L23'], source=group_sets['IN_L23'])] = 0.4
_syn_delay_sigma.loc[
    dict(target=group_sets['IN_L5'], source=group_sets['IN_L5'])] = 0.4

# Define delay minimum
_syn_delay_min = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))),
    coords=[_group_names, _group_names],
    dims=['target', 'source'],
    name='Delay min')
  
# Define delay maximum
_syn_delay_max = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))),
    coords=[_group_names, _group_names],
    dims=['target', 'source'],
    name='delay max')
_syn_delay_max[:,:] = 2    

### Spiking parameters
### ------------------

### gmax ###

# Define gmax (synaptic strength) mean
_syn_gmax = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))),
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='Syn gmax')

_syn_gmax.loc[dict(target='PC_L23', source=group_sets['ALL'])] = [
    0.8405, 2.2615, 2.2615, 0.18, 0.18, 2.2615, 1.8218,
    0.8378, 0.2497, 0.2497, 0.0556, 0.0556, 0.2497, 0.2285,
    ]
_syn_gmax.loc[dict(target='PC_L5', source=group_sets['ALL'])]  = [
    0.9533, 1.0503, 1.0503, 0.0836, 0.0836, 1.0503, 0.8461, 
    0.8818, 1.7644, 1.7644, 0.3932, 0.3932, 1.7644, 1.6146,
    ]
_syn_gmax.loc[dict(target=group_sets['IN_L23'], source='PC_L23')] = [
    1.3403, 1.3403, 0.4710, 0.4710, 1.3403, 0.2500,
    ]
_syn_gmax.loc[dict(target=group_sets['IN_L5'], source='PC_L23')]  = [
    1.5201, 1.5201, 0.5342, 0.5342, 1.5201, 0.2835,
    ]
_syn_gmax.loc[dict(target=group_sets['IN_L23'], source='PC_L5')]  = [
    0.7738, 0.7738, 0.2719, 0.2719, 0.7738, 0.1443,
    ]
_syn_gmax.loc[dict(target=group_sets['IN_L5'], source='PC_L5')] = [
    1.7431, 1.7431, 0.88, 0.88, 1.7431, 0.28,
    ]
_syn_gmax.loc[
    dict(target=group_sets['IN_L23'], source=group_sets['IN_L23'])] = 1.35    
_syn_gmax.loc[
    dict(target=group_sets['IN_L5'], source=group_sets['IN_L5'])] = 1.35

# Adjustment in gmax, mentioned in Hass et al.
_syn_gmax_adjustment = {
    'PC': np.array(
         [[1.0569, 0.5875, 0.6587, 0.7567, 0.6728, 0.9899, 0.6294,
           1.6596, 0.5941, 0.6661, 0.7647, 0.6799, 1.5818, 0.6360]]
         ).transpose(),
    'IN': np.array(
          [[2.3859, 1.6277, 1.6277, 1.6671, 1.6671, 2.3142, 1.4363, 
           3.5816, 1.6277, 1.6277, 1.6671, 1.6671, 3.4016, 1.4363]]
         ).transpose()
    }

for kind in ['PC', 'IN']: 
    gmax_fac = repmat(_syn_gmax_adjustment[kind], 1, len(group_sets[kind]))
    gmax_adj = _syn_gmax.loc[dict(target=group_sets['ALL'], 
                                 source=group_sets[kind])] * gmax_fac
    _syn_gmax.loc[dict(target=group_sets['ALL'], 
                      source=group_sets[kind])] = gmax_adj

# Correcting factor (Hass et al.)
_channel_syn_gmax_factor = xr.DataArray(
    data=np.asarray([[1],[1],[1.09]]), 
    coords=[list(_channel_names), ['factor']],
    dims=['channel', 'par'],
    name='gmax factor')

# Define gmax standard deviation
_syn_gmax_sigma = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))), 
    coords=[_group_names, _group_names],
    dims=['target', 'source'],
    name='gmax sigma')
      
_syn_gmax_sigma.loc[dict(target='PC_L23', source='PC_L23')] = 0.4695
_syn_gmax_sigma.loc[dict(target='PC_L23', source='PC_L5')] = 0.1375
_syn_gmax_sigma.loc[dict(target='PC_L5', source='PC_L23')] = 0.3530
_syn_gmax_sigma.loc[dict(target='PC_L5', source='PC_L5')] = 0.9653
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_L_both_L23'], source='PC_L23')] = 1.0855
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_L_both_L23'], source='PC_L5')] = 0.6267
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_L_both_L5'], source='PC_L23')] = 0.8588
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_L_both_L5'], source='PC_L5')] = 1.1194
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_CL_both_L23'], source='PC_L23')] = 0.1999
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_CL_both_L23'], source='PC_L5')] = 0.1154
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_CL_both_L5'], source='PC_L23')] = 0.1581
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_CL_both_L5'], source='PC_L5')] = 0.7033
_syn_gmax_sigma.loc[dict(target='IN_CC_L23', source='PC_L23')] = 1.0855
_syn_gmax_sigma.loc[dict(target='IN_CC_L23', source='PC_L5')] = 0.6267
_syn_gmax_sigma.loc[dict(target='IN_CC_L5', source='PC_L23')] = 0.8588
_syn_gmax_sigma.loc[dict(target='IN_CC_L5', source='PC_L5')] = 1.1194
_syn_gmax_sigma.loc[dict(target='IN_F_L23', source='PC_L23')] = 0.2000
_syn_gmax_sigma.loc[dict(target='IN_F_L23', source='PC_L5')] = 0.1155
_syn_gmax_sigma.loc[dict(target='IN_F_L5', source='PC_L23')] = 0.1582
_syn_gmax_sigma.loc[dict(target='IN_F_L5', source='PC_L5')] = 0.3000
_syn_gmax_sigma.loc[
    dict(target='PC_L23', source=group_sets['IN_L_both_L23'])] = 1.9462
_syn_gmax_sigma.loc[
    dict(target='PC_L23', source=group_sets['IN_L_both_L5'])] = 0.0362
_syn_gmax_sigma.loc[
    dict(target='PC_L5', source=group_sets['IN_L_both_L23'])] = 0.9038
_syn_gmax_sigma.loc[
    dict(target='PC_L5', source=group_sets['IN_L_both_L5'])] = 0.2557
_syn_gmax_sigma.loc[
    dict(target='PC_L23', source=group_sets['IN_CL_both_L23'])] = 0.6634
_syn_gmax_sigma.loc[
    dict(target='PC_L23', source=group_sets['IN_CL_both_L5'])] = 0.0093
_syn_gmax_sigma.loc[
    dict(target='PC_L5', source=group_sets['IN_CL_both_L23'])] = 0.3081
_syn_gmax_sigma.loc[
    dict(target='PC_L5', source=group_sets['IN_CL_both_L5'])] = 0.0655
_syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_CC_L23')] = 1.9462
_syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_CC_L5')] = 0.0362
_syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.9038
_syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.2557
_syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_F_L23')] = 3.6531
_syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_F_L5')] = 0.1828
_syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_F_L23')] = 1.6966
_syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_F_L5')] = 1.2919
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_L23'], source=group_sets['IN_L23'])] = 0.35
_syn_gmax_sigma.loc[
    dict(target=group_sets['IN_L5'], source=group_sets['IN_L5'])] = 0.35

# Define gmax minimum
_syn_gmax_min = xr.DataArray(
    data=np.zeros((len(_group_names), len(_group_names))), 
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='gmax min')

# Define gmax maximum
_syn_gmax_max = xr.DataArray(
    np.zeros((len(_group_names), len(_group_names))), 
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='gmax max')
_syn_gmax_max[:,:] = 100    

### pfail ###

# Define probability of failure
_syn_pfail = xr.DataArray(
    data= np.ones((len(_group_names), len(_group_names)))*0.3,
    coords=[_group_names, _group_names], 
    dims=['target', 'source'],
    name='syn pfail')

# -----

_spiking_params = {}
for par in _synspike_par_names:
    _spiking_params[par] = dict()
_spiking_params['pfail'] = _syn_pfail
_spiking_params['gmax'] = dict(mean=_syn_gmax, sigma=_syn_gmax_sigma, 
                               min=_syn_gmax_min, max=_syn_gmax_max)


###################################
########## Set equations ##########
###################################

_neuron_eq_memb_par = ['{}: {}'.format(
    name, _baseunit_dict[_membpar_dict[name]['unit']])
    for name in _membpar_names]
_neuron_eq_memb_par = '\n'.join(_neuron_eq_memb_par)

_neuron_eq_channel_par = [
    '{}: {}'.format(par, _baseunit_dict[_channelnamepar_units[par]['unit']]) 
    for par in _channelnamepar_units]

_neuron_eq_channel_par = '\n'.join(_neuron_eq_channel_par)

_neuron_eq_auxiliary_var = (
    'I_ref: amp\n'
    'last_spike: second'
    )

# Mg correcting factor
_neuron_eq_channel_Mgfactor = (
    'Mg_{0} = (1/(1 +  Mg_fac_{0} * 0.33 * exp(Mg_slope_{0} * '
    '(Mg_half_{0}*mV - V)/mV))):1'
    )
_neuron_eq_channel_Mgfactor = [_neuron_eq_channel_Mgfactor.format(name) 
                              for name in _channel_names]
_neuron_eq_channel_Mgfactor = '\n'.join(_neuron_eq_channel_Mgfactor)

# Channel ionic current
_neuron_eq_channel_I = (
    'I_{0} = g_{0} * (E_{0} - V) * Mg_{0}: amp'
    )
_neuron_eq_channel_I = [_neuron_eq_channel_I.format(name) 
                       for name in _channel_names]
_neuron_eq_channel_I = '\n'.join(_neuron_eq_channel_I)

# Conductance as difference between exponential curves
_neuron_eq_channel_g = (
    'g_{0} = g_{0}_off - g_{0}_on: siemens'
    )
_neuron_eq_channel_g = [_neuron_eq_channel_g.format(name) 
                       for name in _channel_names]
_neuron_eq_channel_g = '\n'.join(_neuron_eq_channel_g)

# Onset and offset expontential curves for conductnce
_neuron_eq_channel_dgdt  = (
    'dg_{0}_off/dt = - (1/tau_off_{0}) * g_{0}_off: siemens\n'
    'dg_{0}_on/dt = - (1/tau_on_{0}) * g_{0}_on: siemens'
    )

_neuron_eq_channel_dgdt = [_neuron_eq_channel_dgdt.format(name) 
                          for name in _channel_names]
_neuron_eq_channel_dgdt = '\n'.join(_neuron_eq_channel_dgdt)

# Membrane currents
_neuron_eq_memb_I = (
    'I_DC: amp\n'
    'I_AC = {}*pA: amp\n'
    'I_syn = ' + ' + '.join(['I_{0}'.format(name) 
                             for name in _channel_names]) + ': amp\n'
    'I_inj = I_DC + I_AC: amp\n'
    'I_tot =  I_syn + I_inj: amp'
    )

# Membrane state variables
_neuron_eq_membr_state = (
    'I_exp = g_L * delta_T * exp((clip(V, -500*mV, V_up) - V_T)/delta_T): amp\n'
    'w_V = I_tot + I_exp -g_L * (clip(V, -500*mV, V_up) - E_L): amp\n'
    
    'dV = int(I_tot >= I_ref) * int(t-last_spike < 5*ms) * (-g_L/C) * (clip(V, -500*mV, V_up)-V_r)'
    '+ (1 - int(I_tot >= I_ref) * int(t-last_spike < 5*ms))'
    '* (I_tot + I_exp - g_L * (clip(V, -500*mV, V_up)-E_L) - w)/C: volt/second\n'   
    'dV/dt = dV: volt\n'
    
    'D0 = (C/g_L) * w_V:  coulomb\n'
    'dD0 = C *(exp((clip(V, -500*mV, V_up) - V_T)/delta_T)-1): farad\n'
    'dw/dt = int(w > w_V-D0/tau_w) * int(w < w_V+D0/tau_w) * int(V <= V_T)'
    '* int(I_tot < I_ref) *'
    ' -(g_L * (1 - exp((clip(V, -500*mV, V_up)-V_T)/delta_T)) + dD0/tau_w)*dV: amp'
    )

_neuron_eq_model = '\n\n'.join(
    [_neuron_eq_memb_par, _neuron_eq_channel_par, _neuron_eq_channel_Mgfactor, 
     _neuron_eq_channel_I, _neuron_eq_channel_g, _neuron_eq_channel_dgdt, 
     _neuron_eq_auxiliary_var, _neuron_eq_memb_I, _neuron_eq_membr_state]
    )

# Reset equations
_neuron_eq_thres = "V > V_up"
_neuron_eq_reset = "V = V_r; w += b"

_neuron_eq_event = {}
_neuron_eq_event['w_cross'] = {}

# Correct trajectories that reach w-nullcline
_neuron_eq_event['w_cross']['condition'] = ('w > w_V - D0/tau_w and '
                                           'w < w_V + D0/tau_w and '
                                           'V <= V_T')

_neuron_eq_event['w_cross']['vars'] = ["V", "w"]
_neuron_eq_event['w_cross']['reset'] = "w=w_V - D0/tau_w"

def _neuron_rheobase(g_L, V_T, E_L, delta_T):
    return g_L*(V_T - E_L - delta_T)

_syn_eq_channel = '\n'.join(
    ['{0}: 1'.format(name) for name in _channel_names])
_syn_eq_stsp_var = '\n'.join(
    ['{}: {}'.format(var, _baseunit_dict[_stsp_all_dict[var]['unit']])
     for var in _stsp_all_names])
_syn_eq_spike_par = '\n'.join(
    ['{}: {}'.format(par, _baseunit_dict[_synspike_par[par]['unit']]) 
    for par in _synspike_par_names])
_syn_eq_aux_var = '\n'.join(
    ['{}: {}'.format(par, _baseunit_dict[_synaux_par[par]['unit']]) 
    for par in _synaux_par_names])

_syn_eq_model = '\n\n'.join(
    [_syn_eq_channel,_syn_eq_stsp_var, _syn_eq_spike_par, _syn_eq_aux_var]
    )

_extsyn_eq_model = '\n\n'.join(
    [_syn_eq_channel, _syn_eq_spike_par,_syn_eq_aux_var]
    )

# STSP equations
# This block accounts also for the falure probability
_syn_eq_pathway = []
_syn_eq_pathway.append(dict(
    eq='failure = int(rand()<pfail)',
    order=0, 
    delay=False))
_syn_eq_pathway.append(dict(
    eq="u_temp = U + u * (1 - U) * exp(-(t - last_spike_pre)/tau_fac)",
    order=1,
    delay=False))
_syn_eq_pathway.append(dict(
    eq="R_temp = 1 + (R - u * R - 1) * exp(- (t - last_spike_pre)/tau_rec)",
    order=2,
    delay=False))
_syn_eq_pathway.append(dict(
    eq='u = u_temp',
    order=3,
    delay=False))
_syn_eq_pathway.append(dict(
    eq='R = R_temp',
    order=4,
    delay=False))
_syn_eq_pathway.append(dict(
    eq='last_spike_pre = t',
    order=5,
    delay=False))
_syn_eq_pathway.append(dict(
    eq='a_syn = u * R',
    order=6,
    delay=False))

# Perturbation of conductance due to pre-synaptic spikes
# The amplitude is defined by gmax, a_syn (STSP) and by
# the faliure probability
# For each channel, gsyn_amp is a factor so that the peak of the perturbation
# is gmax * a_syn * (1-failure)
for name in _channel_names:    
    _syn_eq_pathway.append(dict(
        eq="g_{0}_on_post += {0} * gmax * "
        "a_syn * (1-failure) * gsyn_amp".format(name),
        order=7,
        delay=True))
    _syn_eq_pathway.append(dict(
        eq="g_{0}_off_post += {0} * gmax "
        "* a_syn * (1 - failure) * gsyn_amp".format(name),
        order=7,
        delay=True))

# Equations for external stimuli (without STSP)
_eq_extern_syn_pathway = []
_eq_extern_syn_pathway.append(dict(eq='failure = int(rand()<pfail)',
                                  order=0, 
                                  delay=False))
for name in _channel_names:    
    _eq_extern_syn_pathway.append(dict(
        eq="g_{0}_on_post += {0} * gmax * (1-failure) * gsyn_amp".format(name), 
        order=0, 
        delay=False))
    _eq_extern_syn_pathway.append(dict(
        eq="g_{0}_off_post += {0} * gmax * "
        "(1-failure) * gsyn_amp".format(name), 
        order=0,
        delay=False))

_eq_var_units = dict(V ='mV', w='pA', t='ms', I_tot='pA', I_syn='pA', 
                    I_inj='pA', I_DC='pA') 
for name in _channel_names:
    _eq_var_units['I_{0}'.format(name)]='pA'
for var_unit in [_membpar_dict, _stsp_all_dict, _synspike_par, _synaux_par]:
    for var in var_unit:
        _eq_var_units[var]=var_unit[var]['unit']

################################
########## Set basics ##########
################################

@time_report('Basics setup')
def basics_setup(n_cells_prompted, n_stripes, basics_scales=None, 
                 alternative_pcells=None, disp=True):
    """Set the basic network data and return cell distribution,
    membrane and synptic parameters and connectivity in a dataclass.
    
    Parameters
    ----------
    n_cells_prompted: int
        Requested number of cells (may differ from the effective number
        due to roundings).
    n_stripes: int
        Requested number of stripes.
    basics_scales: dict, optional
        Scaling of network parameters. If not given, no scales are 
        applied.
    alternative_pcells: array_like, optional.
        Alternative cell distribution between groups. If not given,
        _pcells_per_group is used.      
    disp: bool, optional
        Set display. If not given, warning message may be displayed.
        
    Returns
    -------
    output: _BasicsSetup
        Data class holding network data.
    """
    
    # The copy variables in this block are necessary when more than one 
    # simulation is carried out without re-importing this module; 
    # otherwise, the modifications by basics_scales would remain
    # in DataArrays between subsequent simulations
    pcon_copy = _syn_pcon.copy()
    cellstd_copy = _membpar_std.copy()
    syn_gmax_copy = _syn_gmax.copy()
    spiking_params_copy = _spiking_params.copy()
    spiking_params_copy['gmax'] = dict(mean=syn_gmax_copy, 
                                       sigma=_syn_gmax_sigma, 
                                       min=_syn_gmax_min,
                                       max=_syn_gmax_max)

    scalables = {'pCon': pcon_copy, 
                 'membr_param_std': cellstd_copy, 
                 'gmax_mean': syn_gmax_copy}
    
    
    if alternative_pcells is None:
        pcells = _pcells_per_group
    else:
        pcells = alternative_pcells
    
    
    if basics_scales is not None:  
        for par in basics_scales:              
            for TS_dict, scale in basics_scales[par]:
                
                
                new_param = scalables[par].loc[TS_dict].values * scale
                scalables[par].loc[TS_dict] = new_param
    
    # DataArray holding membrane parameters (non-normalized) covariance. They
    # are drawn from the normalized covariance and the standard deviations.
    membpar_covar=xr.DataArray(
        np.zeros((len(_group_names), len(_membpar_names), 
                  len(_membpar_names))),
        coords=[_group_names, _membpar_names, _membpar_names], 
        dims=['group', 'memb_par0', 'memb_par1'],
        name='memb_par covariance')  
    
    for group in _membpar_std.coords['group'].values:
        for param0 in _membpar_std.coords['par'].values:
            for param1 in _membpar_std.coords['par'].values:
                    cov = _membcov_norm.loc[
                        dict(group=group, memb_par0=param0, memb_par1=param1)
                        ].values
                    std0 = cellstd_copy.loc[
                        dict(group=group, par=param0)
                        ].values
                    std1 = cellstd_copy.loc[
                        dict(group=group, par=param1)
                        ].values
                    membpar_covar.loc[
                        dict(group=group, memb_par0=param0, memb_par1=param1)
                        ] = cov*std0*std1
            
    group_setup = _GroupSetup(_group_kinds, group_sets)          
    stripes_setup = _StripeSetup(n_stripes, _interstripe_sets)
    connection_setup = _ConnectionSetup(pcon_copy, _syn_clusterflag)          
    cellsetup = _StructureSetup(n_cells_prompted, pcells, group_setup, 
                               stripes_setup, connection_setup)
    
    membrane_setup = _MembraneSetup(_membpar_names, _membpar_dict, 
                                    _unitbr2_dict, _baseunit_dict, 
                                    _membpar_mean, membpar_covar, 
                                    _membpar_lambd_transf, cellstd_copy, 
                                    _membpar_min, _membpar_max, _membtau_min,
                                    _membtau_max) 
    
    STSP_setup=_STSPSetup(decl_vars=_stsp_all_dict,kinds=_stsp_types, 
                         sets=_stsp_set_probs, distr=_stsp_set_distrib)
    channels_setup = _ChannelSetup(_channel_kinds, _channel_par, 
                                  _channel_syn_gmax_factor,
                                  _channelnamepar_units)      
    
    spike_params_data = _ParamSetup()
    for par in _synspike_par_names:
        spike_params_data[par] = spiking_params_copy[par]
        
    spike_params_setup = _SpikeParamSetup(_synspike_par, spike_params_data)
    delay_setup = _DelaySetup(_syn_delay, _syn_delay_sigma, _syn_delay_min, 
                             _syn_delay_max)
    synapses_setup = _SynapseSetup(_synapse_kind, spike_params_setup, 
                                  delay_setup, STSP_setup, channels_setup)
    eqs_setup = _EquationsSetup(_neuron_eq_model, _neuron_eq_thres, 
                               _neuron_eq_reset, _neuron_eq_event, 
                               _neuron_rheobase, _syn_eq_model, 
                               _syn_eq_pathway, _extsyn_eq_model, 
                               _eq_extern_syn_pathway, _eq_var_units)
    
    if cellsetup.n_cells != n_cells_prompted and disp:
        print(('REPORT: The number of neurons was adjusted from {} to {} due '
               'to rounding.\n'.format(n_cells_prompted, cellsetup.n_cells)))
             
    return _BasicsSetup(cellsetup, membrane_setup, synapses_setup, 
                       eqs_setup, scalables, basics_scales)

######################################
########## Set data classes ##########
######################################
    
@dataclass
class _ChannelSetup(BaseClass):
    kinds: list[str]
    params: xr.DataArray
    gsyn_factor: xr.DataArray
    unitvalue_dict: dict
    names: list = field(default_factory=list)
    kinds_to_names: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.names = list(self.kinds.keys())
        self.kinds_to_names = {'exc':[], 'inh':[]}
        for name in self.names:
            self.kinds_to_names[self.kinds[name]].append(name)


@dataclass
class _SynapseSetup(BaseClass):
    kinds: any
    spiking: any
    delay: any
    stsp: any
    channels: any


@dataclass
class _SpikeParamSetup(BaseClass):
    names: any
    params: any

    
@dataclass
class _DelaySetup(BaseClass):
    delay_mean: any
    delay_sigma: any
    delay_min: any
    delay_max: any
    

@dataclass
class _StripeSetup(BaseClass):
    n: int
    inter: any


@dataclass
class _ConnectionSetup(BaseClass):
    pcon: any
    cluster: any
    

@dataclass
class _STSPSetup(BaseClass):
    decl_vars: any
    kinds: dict
    sets: any
    distr: dict


@dataclass
class _MembraneSetup(BaseClass):  
    names: any
    name_units: any
    unitbr2_dict: any
    unit_main_dict: any
    mean: float
    covariance: float
    lambd: float
    std: float
    min: float
    max:float
    tau_m_min: any
    tau_m_max: any

 
@dataclass
class _GroupSetup(BaseClass):   
    kinds: any
    sets:any
    names: list[str] = field(default_factory=list)
    n:int = 0
    
    def __post_init__(self):
        self.names = list(self.kinds.keys())
        self.n = len(self.names)
        self.idcs = {}
        for name_idc in range(len(self.names)):
            self.idcs[self.names[name_idc]] = name_idc


@dataclass
class _StructureSetup(BaseClass):      
    n_cells_prompt: int
    p_cells_per_group: xr.DataArray
    groups: _GroupSetup
    stripes: dict
    conn: any
    n_cells: int = 0
    n_cells_total: int = 0
    n_cells_per_group: xr.DataArray = field(default_factory=xr.DataArray)
        
    def __post_init__(self):
        n_cells_per_group = np.ceil(
            (self.n_cells_prompt*self.p_cells_per_group)/100
            ).astype(int)
        self.n_cells_per_group = xr.DataArray(n_cells_per_group, 
                                             coords=[self.groups.names], 
                                             dims='group')
        self.n_cells = int(sum(self.n_cells_per_group))
        self.n_cells_total = self.stripes.n*self.n_cells
        
    
@dataclass
class _ParamSetup(BaseClass):
    pass


@dataclass
class _EquationsSetup(BaseClass):
    membr_model: str
    membr_threshold: str
    membr_reset: str
    membr_events: dict
    rheobase: any
    syn_model: str
    syn_pathway: list
    ext_syn_model: str
    ext_syn_pathway: list
    var_units: dict
        
    
@dataclass
class _BasicsSetup(BaseClass):
    struct: any
    membr: any
    syn: any
    equations: any
    scalables: any
    scales: any

    
if __name__ == '__main__':
   
    n1=1000
    n_stripes=1

    membr_param_std = [(dict(group=group_sets['ALL'], 
                              par=_membpar_names), 0.9)]
    basics_scales = {'membr_param_std': membr_param_std}

    _basics = basics_setup(n1, n_stripes,
                            basics_scales=basics_scales, 
                           disp=False)