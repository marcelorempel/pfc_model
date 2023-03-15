"""  
This script defines equations,  network structure, membrane and synaptic
paramateres distributions.
"""

import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from numpy.matlib import repmat
from collections import namedtuple
import brian2 as br2
from _auxiliar import time_report, BaseClass

__all__ = ['basics_setup', 'membranetuple']


group_kinds = {
    'PC_L23': 'exc', 'IN_L_L23': 'inh', 'IN_L_d_L23': 'inh', 
    'IN_CL_L23': 'inh', 'IN_CL_AC_L23': 'inh', 'IN_CC_L23': 'inh', 
    'IN_F_L23': 'inh', 
    'PC_L5': 'exc', 'IN_L_L5': 'inh',  'IN_L_d_L5': 'inh', 'IN_CL_L5': 'inh',
    'IN_CL_AC_L5': 'inh', 'IN_CC_L5': 'inh', 'IN_F_L5': 'inh'
    }

group_names = list(group_kinds.keys())

memb_par_dict = dict(
    C=dict(unit='pF', value='C'), g_L=dict(unit='nS', value='g_L'),
    E_L=dict(unit='mV', value='E_L'), delta_T=dict(unit='mV', value='delta_T'),
    V_up=dict(unit='mV', value='V_up'), tau_w=dict(unit='ms', value='tau_w'),
    b=dict(unit='pA', value='b'), V_r=dict(unit='mV', value='V_r'),
    V_T=dict(unit='mV', value='V_T')
    )

unitbr2_dict = {'pF':br2.pF, 'nS': br2.nS, 'mV': br2.mV, 'ms': br2.ms, 
                'pA': br2.pA, 1:1}
baseunit_dict = {'pF': 'farad', 'nS': 'siemens', 'mV': 'volt', 'ms': 'second', 
                 'pA': 'amp', 1: 1}

memb_par_names = list(memb_par_dict.keys())
membranetuple = namedtuple('membranetuple', memb_par_names)

channel_kinds={'AMPA': 'exc', 'GABA':'inh', 'NMDA':'exc'}
channel_names = [name for name in list(channel_kinds.keys())]

channel_par_names= ['tau_on', 'tau_off', 'E', 'Mg_fac', 'Mg_slope', 'Mg_half']


channelpar_units = {}
for channel in channel_names:
    channelpar_units['tau_on_{}'.format(channel)] = dict(
        unit='ms', value='tau_on_{}'.format(channel)
        )
    channelpar_units['tau_off_{}'.format(channel)] = dict(
        unit='ms', value='tau_off_{}'.format(channel)
        )
    channelpar_units['E_{}'.format(channel)] = dict(
        unit='mV', value='E_{}'.format(channel)
        )
    channelpar_units['Mg_fac_{}'.format(channel)] = dict(
        unit=1, value='Mg_fac_{}'.format(channel)
        )
    channelpar_units['Mg_slope_{}'.format(channel)] = dict(
        unit=1, value='Mg_slope_{}'.format(channel)
        )
    channelpar_units['Mg_half_{}'.format(channel)] = dict(
        unit=1, value='Mg_half_{}'.format(channel)
        )

 
pcells_per_group = np.asarray([47, 1.55, 1.55, 1.3, 1.3, 2.6, 2.1,
                                38, 0.25, 0.25, 0.25, 0.25, 1.8, 1.8])


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

memb_par_mean = xr.DataArray(
    data=np.zeros((len(memb_par_names), len(group_names))),
    coords=[memb_par_names, group_names], 
    dims=['param', 'group'],
    name='Memb_param mean',
    )

memb_par_mean.loc[dict(param='C')] = [
    3.0751, 1.6902, 1.6902, 3.0014, 3.0014, 3.0751, 3.3869,
    2.2513, 1.6902, 1.6902, 3.0014, 3.0014, 2.2513, 3.3869,
    ]
memb_par_mean.loc[dict(param='g_L')] = [
    1.9661, 1.0353, 1.0353, 1.4581, 1.4581, 1.9661, 1.0106, 
    1.0196, 1.0353, 1.0353, 1.4581, 1.4581, 1.0196, 1.0106,
    ]
memb_par_mean.loc[dict(param='E_L')] = [
    3.5945, 2.9528, 2.9528, 3.0991, 3.0991, 3.5945, 3.8065,
    3.4415, 2.9528, 2.9528, 3.0991, 3.0991, 3.4415, 3.8065,
    ]
memb_par_mean.loc[dict(param='delta_T')] = [
    1.0309, 3.2163, 3.2163, 3.1517, 3.1517, 1.0309, 3.0269, 
    1.5178, 3.2163, 3.2163, 3.1517, 3.1517, 1.5178, 3.0269,
    ]
memb_par_mean.loc[dict(param='V_up')] = [
    3.1428, 2.8230, 2.8230, 2.9335, 2.9335, 3.1428, 2.3911, 
    1.0702, 2.8230, 2.8230, 2.9335, 2.9335, 1.0702, 2.3911,
    ]
memb_par_mean.loc[dict(param='tau_w')] = [
    4.4809, 1.0542, 1.0542, 1.0730, 1.0730, 4.4809, 4.1986, 
    4.5650, 1.0542, 1.0542, 1.0730, 1.0730, 4.5650, 4.1986,
    ]
memb_par_mean.loc[dict(param='b')] = [
    1.0189, 2.5959, 2.5959, 0.6931, 0.6931, 1.0189, 0.8080, 
    1.1154, 2.5959, 2.5959, 0.6931, 0.6931, 1.1154, 0.8080,
    ]
memb_par_mean.loc[dict(param='V_r')] = [    
    5.0719, 4.1321, 4.1321, 1.9059, 1.9059, 5.0719, 3.0051, 
    4.3414, 4.1321, 4.1321, 1.9059, 1.9059, 4.3414, 3.0051,
    ]
memb_par_mean.loc[dict(param='V_T')] = [
    2.9010, 3.6925, 3.6925, 2.9462, 2.9462, 2.9010, 3.0701,
    3.3302, 3.6925, 3.6925, 2.9462, 2.9462, 3.3302, 3.0701,
    ]

memb_par_covar_normalized=xr.DataArray(
    np.zeros((len(group_names), len(memb_par_names), len(memb_par_names))),
    coords=[group_names, memb_par_names, memb_par_names], 
    dims=['group', 'memb_par0', 'memb_par1'],
    name='memb_par covariance normalized',
    )                        
memb_par_covar_normalized.loc[dict(group='PC_L23')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_L_L23')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_L_d_L23')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_CL_L23')] = np.matrix([
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
     0.5780],
    [0.0171, 0.6950, -0.3566, 0.7230, 0.0193, -0.4724, -0.3395, 1.0000,
     -0.1084],
    [-0.4079, 0.0811, 0.4204, 0.0775, 0.8494, 0.0957, -0.5780, -0.1084,
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_CL_AC_L23')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_CC_L23')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_F_L23')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='PC_L5')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_L_L5')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_L_d_L5')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_CL_L5')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_CL_AC_L5')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_CC_L5')] = np.matrix([
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
     1.0000],
    ])
memb_par_covar_normalized.loc[dict(group='IN_F_L5')] = np.matrix([
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
     1.0000],
    ])                         
                  
memb_par_std = xr.DataArray(
    data=np.zeros((len(memb_par_names), len(group_names))),
    coords=[memb_par_names, group_names],
    dims=['param', 'group'],
    name='memb_par std',
    )
memb_par_std.loc[dict(param='C')] = [
    0.4296, 0.0754, 0.0754, 0.3283, 0.3283, 0.4296, 0.5029,
    0.1472, 0.0754, 0.0754, 0.3283, 0.3283, 0.1472, 0.5029,
    ] 
memb_par_std.loc[dict(param='g_L')] = [
    0.3558, 0.0046, 0.0046, 0.1844, 0.1844, 0.3558, 0.0022,
    0.0030, 0.0046, 0.0046, 0.1844, 0.1844, 0.0030, 0.0022,
    ]
memb_par_std.loc[dict(param='E_L')] = [
    0.3644, 0.3813, 0.3813, 0.3630, 0.3630, 0.3644, 0.3359,
    0.2846, 0.3813, 0.3813, 0.3630, 0.3630, 0.2846, 0.3359,
    ]
memb_par_std.loc[dict(param='delta_T')] = [
    0.0048, 0.7107, 0.7107, 0.3568, 0.3568, 0.0048, 0.7395,
    0.0554, 0.7107, 0.7107, 0.3568, 0.3568, 0.0554, 0.7395,
    ]
memb_par_std.loc[dict(param='V_up')] = [
    0.5259, 0.5033, 0.5033, 0.4372, 0.4372, 0.5259, 0.3035, 
    0.0062, 0.5033, 0.5033, 0.4372, 0.4372, 0.0062, 0.3035,
    ]
memb_par_std.loc[dict(param='tau_w')] = [
    0.4947, 0.0052, 0.0052, 0.0170, 0.0170, 0.4947, 0.3186, 
    0.6356, 0.0052, 0.0052, 0.0170, 0.0170, 0.6356, 0.3186,
    ]
memb_par_std.loc[dict(param='b')] = [
    0.0113, 1.9269, 1.9269, 1.4550, 1.4550, 0.0113, 1.0353, 
    1.3712, 1.9269, 1.9269, 1.4550, 1.4550, 1.3712, 1.0353,
    ]
memb_par_std.loc[dict(param='V_r')] = [
    0.6104, 0.4817, 0.4817, 0.1504, 0.1504, 0.6104, 0.1813, 
    0.3497, 0.4817, 0.4817, 0.1504, 0.1504, 0.3497, 0.1813,
    ]
memb_par_std.loc[dict(param='V_T')] = [
    0.4608, 0.4385, 0.4385, 0.4311, 0.4311, 0.4608, 0.3632, 
    0.2857, 0.4385, 0.4385, 0.4311, 0.4311, 0.2857, 0.3632,
    ]

memb_par_covar=xr.DataArray(
    np.zeros((len(group_names), len(memb_par_names), len(memb_par_names))),
    coords=[group_names, memb_par_names, memb_par_names], 
    dims=['group', 'memb_par0', 'memb_par1'],
    name='memb_par covariance',
    )     

for group in memb_par_std.coords['group'].values:
    for param0 in memb_par_std.coords['param'].values:
        for param1 in memb_par_std.coords['param'].values:
            dict_covar = dict(group=group, memb_par0=param0, memb_par1=param1)
            cov = memb_par_covar_normalized.loc[dict_covar].values
            std0 = memb_par_std.loc[dict(group=group, param=param0)].values
            std1 = memb_par_std.loc[dict(group=group, param=param1)].values
            memb_par_covar.loc[dict_covar] = cov*std0*std1
        
       
memb_par_ktransf = xr.DataArray(
    data=np.zeros((len(memb_par_names), len(group_names))),
    coords=[memb_par_names,group_names], 
    dims=['param', 'group'],
    name='memb_par ktransf',
    )
memb_par_ktransf[:,:]= np.matrix([
    [0.37, 0.22, 0.22, 0.00, 0.00, 0.37, 0.00, 0.23, 0.22, 0.22, 0.00, 0.00, 
     0.23, 0.00],
    [0.00, 0.02, 0.02, 0.00, 0.00, 0.00, 0.01, 0.01, 0.02, 0.02, 0.00, 0.00, 
     0.01, 0.01],
    [0.00, 0.36, 0.36, 0.00, 0.00, 0.00, 0.00, 0.00, 0.36, 0.36, 0.00, 0.00, 
     0.00, 0.00],
    [0.01, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.13, 0.00, 0.00, 0.00, 0.00, 
     0.13, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 
     0.02, 0.00],
    [0.00, 0.02, 0.02, 0.02, 0.02, 0.00, 0.00, 0.00, 0.02, 0.02, 0.02, 0.02, 
     0.00, 0.00],
    [0.01, 0.36, 0.36, 0.00, 0.00, 0.01, 0.00, 0.00, 0.36, 0.36, 0.00, 0.00, 
     0.00, 0.00],
    [0.00, 0.00, 0.00, 0.12, 0.12, 0.00, 0.26, 0.00, 0.00, 0.00, 0.12, 0.12,
     0.00, 0.26],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 
     0.00, 0.00],
    ])




memb_par_min = xr.DataArray(
    data=np.zeros((len(memb_par_names), len(group_names))),
    coords=[memb_par_names, group_names], 
    dims=['param', 'group'],
    name='Memb_param min',
    )

memb_par_min.loc[dict(param='C')] = [
    61.4187, 42.1156, 42.1156, 51.8447, 51.8447, 61.4187, 32.3194, 
    110.7272, 42.1156, 42.1156, 51.8447, 51.8447, 110.7272, 32.3194,
    ]
memb_par_min.loc[dict(param='g_L')] = [
    3.2940, 3.6802, 3.6802, 2.9852, 2.9852, 3.2940, 2.1462, 
    3.4510, 3.6802, 3.6802, 2.9852, 2.9852, 3.4510, 2.1462,
    ]
memb_par_min.loc[dict(param='E_L')] = [
    -104.9627, -96.9345, -96.9345, -98.8335, -98.8335, -104.9627, -102.3895, 
    -101.5624, -96.9345, -96.9345, -98.8335, -98.8335, -101.5624, -102.3895,
    ]
memb_par_min.loc[dict(param='delta_T')] = [
    10.5568, 2.1840, 2.1840, 11.0503, 11.0503, 10.5568, 1.8285, 
    12.7969, 2.1840, 2.1840, 11.0503, 11.0503, 12.7969, 1.8285,
    ]
memb_par_min.loc[dict(param='V_up')] = [
    -62.5083, -60.6745, -60.6745, -65.4193, -65.4193, -62.5083, -42.8895, 
    -66.1510, -60.6745, -60.6745, -65.4193, -65.4193, -66.1510, -42.8895,
    ]
memb_par_min.loc[dict(param='tau_w')] = [
    54.0018, 10.2826, 10.2826, 12.2898, 12.2898, 54.0018, 20.0311, 
    33.1367, 10.2826, 10.2826, 12.2898, 12.2898, 33.1367, 20.0311,
    ]
memb_par_min.loc[dict(param='b')] = [
    1.2406, 1.0000, 1.0000, 1.0000, 1.0000, 1.2406, 1.0000, 
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    ]
memb_par_min.loc[dict(param='V_r')] = [
    -219.2039, -128.4559, -128.4559, -271.9846, -271.9846, -219.2039, 
    -105.1880, 
    -124.5158, -128.4559, -128.4559, -271.9846, -271.9846, -124.5158, 
    -105.1880,
    ]
memb_par_min.loc[dict(param='V_T')] = [
    -63.2375, -85.2096, -85.2096, -70.3537, -70.3537, -63.2375, -53.3897, 
    -69.5922, -85.2096, -85.2096, -70.3537, -70.3537, -69.5922, -53.3897,
    ]

memb_par_max = xr.DataArray(
    np.zeros((len(memb_par_names), len(group_names))),
    coords=[memb_par_names, group_names],
    dims=['param', 'group'],
    name='Memb_param max',
    )

memb_par_max.loc[dict(param='C')] = [
    337.9765, 94.6939, 94.6939, 126.2367, 126.2367, 337.9765, 201.3221, 
    617.2776, 94.6939, 94.6939, 126.2367, 126.2367, 617.2776, 201.3221,
    ]
memb_par_max.loc[dict(param='g_L')] = [
    10.8106, 8.6130, 8.6130, 5.6192, 5.6192, 10.8106, 5.3460,  
    15.6329, 8.6130, 8.6130, 5.6192, 5.6192, 15.6329, 5.3460,
    ]
memb_par_max.loc[dict(param='E_L')] = [
    -76.8526, -71.7548, -71.7548, -75.7868, -75.7868, -76.8526, -59.6898,  
    -66.4770, -71.7548, -71.7548, -75.7868, -75.7868, -66.4770, -59.6898,
    ]
memb_par_max.loc[dict(param='delta_T')] = [
    45.3814, 40.4333, 40.4333, 31.3533, 31.3533, 45.3814, 47.6214,  
    43.5882, 40.4333, 40.4333, 31.3533, 31.3533, 43.5882, 47.6214,
    ]
memb_par_max.loc[dict(param='V_up')] = [
    -30.0577, -36.5929, -36.5929, -45.6445, -45.6445, -30.0577, -30.7977,  
    -25.2891, -36.5929, -36.5929, -45.6445, -45.6445, -25.2891, -30.7977,
    ]
memb_par_max.loc[dict(param='tau_w')] = [
    232.8699, 21.9964, 21.9964, 120.5043, 120.5043, 232.8699, 102.4180,  
    909.5520, 21.9964, 21.9964, 120.5043, 120.5043, 909.5520, 102.4180,
    ]
memb_par_max.loc[dict(param='b')] = [
    40.2930, 196.7634, 196.7634, 71.0958, 71.0958, 40.2930, 54.2781,  
    325.7906, 196.7634, 196.7634, 71.0958, 71.0958, 325.7906, 54.2781,
    ]
memb_par_max.loc[dict(param='V_r')] = [
    -45.0393, -56.5047, -56.5047, -56.8682, -56.8682, -45.0393, -35.7409,  
    -35.1145, -56.5047, -56.5047, -56.8682, -56.8682, -35.1145, -35.7409,
    ]
memb_par_max.loc[dict(param='V_T')] = [
    -36.8701, -39.1085, -39.1085, -49.0974, -49.0974, -36.8701, -20.6720,  
    -27.8669, -39.1085, -39.1085, -49.0974, -49.0974, -27.8669, -20.6720,
    ]

memb_tau_min = np.asarray(
    [10.3876, 7.3511, 7.3511, 9.2264, 9.2264, 10.3876, 5.8527,
     16.7015, 7.3511, 7.3511, 9.2264, 9.2264, 16.7015, 5.8527]
    )
memb_tau_min = xr.DataArray(
    data=memb_tau_min,
    coords=[group_names], 
    dims='group',
    name='Memb_param tau_m min',
    )   

memb_tau_max = np.asarray(
    [42.7304, 15.9128, 15.9128, 25.9839, 25.9839, 42.7304, 48.7992, 
     67.7062, 15.9128, 15.9128, 25.9839, 25.9839, 67.7062, 48.7992]
    )
memb_tau_max =  xr.DataArray(
    data=memb_tau_max,
    coords=[group_names], 
    dims='group',
    name='Memb_param tau_m max',
    )

interstripe_sets ={}

interstripe_sets['A']={
    'pair': ['PC_L23','PC_L23'],
    'coefficient_0': 0.909,
    'coefficient_1': 1.4587,
    'connections': [-4, -2, 2, 4],
    }

interstripe_sets['B']={
    'pair': ['PC_L5','IN_CC_L5'],
    'coefficient_0': 0.909,
    'coefficient_1': 1.0375,
    'connections':[-1, 1],
    }

interstripe_sets['C']={
    'pair':  ['PC_L5', 'IN_F_L5'],
    'coefficient_0': 0.909,
    'coefficient_1': 1.0375,
    'connections': [-1, 1],
    }

syn_pcon = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))),
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='pCon',
    )


syn_pcon.loc[dict(target='PC_L23', source='PC_L23')] = 0.1393
syn_pcon.loc[dict(target='PC_L23', source='PC_L5')] = 0.0449
syn_pcon.loc[dict(target='PC_L5', source='PC_L23')] = 0.2333
syn_pcon.loc[dict(target='PC_L5', source='PC_L5')] = 0.0806

syn_pcon.loc[dict(target=group_sets['IN_L_both_L23'], 
                  source='PC_L23')] = 0.3247
syn_pcon.loc[dict(target=group_sets['IN_L_both_L23'], 
                  source='PC_L5')] = 0.1875
syn_pcon.loc[dict(target=group_sets['IN_L_both_L5'], 
                  source='PC_L23')] = 0.0870
syn_pcon.loc[dict(target=group_sets['IN_L_both_L5'], 
                  source='PC_L5')] = 0.3331  
syn_pcon.loc[dict(target=group_sets['IN_CL_both_L23'], 
                  source='PC_L23')] = 0.1594
syn_pcon.loc[dict(target=group_sets['IN_CL_both_L23'], 
                  source='PC_L5')] = 0.0920
syn_pcon.loc[dict(target=group_sets['IN_CL_both_L5'], 
                  source='PC_L23')] = 0.0800
syn_pcon.loc[dict(target=group_sets['IN_CL_both_L5'], 
                  source='PC_L5')] = 0.0800   

syn_pcon.loc[dict(target='IN_CC_L23', source='PC_L23')] = 0.3247
syn_pcon.loc[dict(target='IN_CC_L23', source='PC_L5')] = 0.1875
syn_pcon.loc[dict(target='IN_CC_L5', source='PC_L23')] = 0.0870
syn_pcon.loc[dict(target='IN_CC_L5', source='PC_L5')] = 0.3331
syn_pcon.loc[dict(target='IN_F_L23', source='PC_L23')] = 0.2900
syn_pcon.loc[dict(target='IN_F_L23', source='PC_L5')] = 0.1674
syn_pcon.loc[dict(target='IN_F_L5', source='PC_L23')] = 0.1500
syn_pcon.loc[dict(target='IN_F_L5', source='PC_L5')] = 0.3619

syn_pcon.loc[dict(target='PC_L23', 
                  source=group_sets['IN_L_both_L23'])] = 0.4586
syn_pcon.loc[dict(target='PC_L23', 
                  source=group_sets['IN_L_both_L5'])] = 0.0991
syn_pcon.loc[dict(target='PC_L5', 
                  source=group_sets['IN_L_both_L23'])] = 0.2130
syn_pcon.loc[dict(target='PC_L5', 
                  source=group_sets['IN_L_both_L5'])] = 0.7006
syn_pcon.loc[dict(target='PC_L23', 
                  source=group_sets['IN_CL_both_L23'])] = 0.4164
syn_pcon.loc[dict(target='PC_L23', 
                  source=group_sets['IN_CL_both_L5'])] = 0.0321
syn_pcon.loc[dict(target='PC_L5', 
                  source=group_sets['IN_CL_both_L23'])] = 0.1934
syn_pcon.loc[dict(target='PC_L5', 
                  source=group_sets['IN_CL_both_L5'])] = 0.2271
syn_pcon.loc[dict(target='PC_L23', source='IN_CC_L23')] = 0.4586
syn_pcon.loc[dict(target='PC_L23', source='IN_CC_L5')] = 0.0991
syn_pcon.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.2130
syn_pcon.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.7006
syn_pcon.loc[dict(target='PC_L23', source='IN_F_L23')] = 0.6765
syn_pcon.loc[dict(target='PC_L23', source='IN_F_L5')] = 0.1287
syn_pcon.loc[dict(target='PC_L5', source='IN_F_L23')] = 0.3142
syn_pcon.loc[dict(target='PC_L5', source='IN_F_L5')] = 0.9096

syn_pcon.loc[dict(target=group_sets['IN_L23'], 
                  source=group_sets['IN_L23'])] = 0.25
syn_pcon.loc[dict(target=group_sets['IN_L5'], 
                  source=group_sets['IN_L5'])] = 0.60

syn_clusterflag = xr.DataArray(
    data=np.zeros((len(group_names),len(group_names))),
    coords=[group_names,group_names],
    dims=['target', 'source'],
    name='Clustering flag',
    )                      

syn_clusterflag.loc[dict(target='PC_L23', source='PC_L23')] = 1
syn_clusterflag.loc[dict(target='PC_L5', source='PC_L5')] = 1

synSTSP_par_dict = {
    'U': dict(unit=1, value='U'), 'tau_rec': dict(unit='ms', value='tau_rec'), 
    'tau_fac': dict(unit='ms', value='tau_rec'),
    }
synSTSP_var_dict = {
    'u_temp': dict(unit=1, value='U'), 'u': dict(unit=1, value='U'), 
    'R_temp': dict(unit=1, value=1),'R': dict(unit=1, value=1), 
    'a_syn':dict(unit=1, value='U')
    }
synSTSP_all_dict = synSTSP_par_dict | synSTSP_var_dict

synSTSP_par_names = list(synSTSP_par_dict.keys())
synSTSP_all_names = list(synSTSP_all_dict.keys())

synSTSP_par_statmeasures= [
    '{}_{}'.format(var, stat) for stat in ['mean', 'std'] 
    for var in synSTSP_par_names
    ]

synspike_par = {'gmax': dict(unit='nS', value='gmax'), 
                'pfail': dict(unit=1, value='pfail')}
synspike_par_names = list(synspike_par.keys())

synaux_par = {'failure': dict(unit=1), 'gsyn_amp': dict(unit=1)}
synaux_par_names = list(synaux_par.keys())

synSTSP_type_dict = {
    'E1': np.array([0.28, 194, 507, 0.02, 18, 37]),
    'E2': np.array([0.25, 671, 17, 0.02, 17, 5]),
    'E3': np.array([0.29, 329, 326, 0.03, 53, 66]),
    'I1': np.array([0.16, 45, 376,0.10, 21, 253]),
    'I2': np.array([0.25, 706, 21, 0.13, 405, 9]),
    'I3': np.array([0.32, 144, 62, 0.14, 80, 31]),
    }

synSTSP_types = xr.DataArray(
    data=np.array(list(synSTSP_type_dict.values())), 
    coords=[list(synSTSP_type_dict.keys()), synSTSP_par_statmeasures], 
    dims=['kind', 'param'],
    name='STSP param sets',
    )

synSTSP_set_dict = {
    'A': np.asarray([0.45, 0.38, 0.17, np.NaN, np.NaN, np.NaN]),
    'B': np.asarray([1, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
    'C': np.asarray([np.NaN, 1, np.NaN, np.NaN, np.NaN, np.NaN]),
    'D': np.asarray([np.NaN, np.NaN, np.NaN, 0.25, 0.5, 0.25]),
    'E': np.asarray([np.NaN, np.NaN, np.NaN, np.NaN, 1, np.NaN]),
    'F': np.asarray([np.NaN, np.NaN, np.NaN, 0.29, 0.58, 0.13])
    }

synSTSP_set_probs = xr.DataArray(
    data=np.array(list(synSTSP_set_dict.values())), 
    coords=[list(synSTSP_set_dict.keys()), list(synSTSP_type_dict.keys())],
    dims=['set', 'kind'],
    name='STSP set distribution',
    )

synSTSP_set_distrib = np.asarray(
    [['' for i in range(len(group_names))] for j in range(len(group_names))], 
    dtype='U16',
    )

synSTSP_set_distrib = xr.DataArray(
    data=synSTSP_set_distrib,
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='STSP distribution groups',
    )

synSTSP_set_distrib.loc[dict(target=group_sets['PC'], 
                             source=group_sets['PC'])] = 'A'
synSTSP_set_distrib.loc[dict(target=group_sets['IN_L'], 
                             source=group_sets['PC'])] = 'B'
synSTSP_set_distrib.loc[dict(target=group_sets['IN_L_d'], 
                             source=group_sets['PC'])] = 'C'
synSTSP_set_distrib.loc[dict(target=group_sets['IN_CL'], 
                             source=group_sets['PC'])] = 'C'
synSTSP_set_distrib.loc[dict(target=group_sets['IN_CL_AC'], 
                             source=group_sets['PC'])] = 'B'
synSTSP_set_distrib.loc[dict(target=group_sets['IN_CC'], 
                             source=group_sets['PC'])] = 'B'
synSTSP_set_distrib.loc[dict(target=group_sets['IN_F'], 
                             source=group_sets['PC'])] = 'C'

synSTSP_set_distrib.loc[dict(target=group_sets['PC'], 
                             source=group_sets['IN_L'])] = 'D'
synSTSP_set_distrib.loc[dict(target=group_sets['PC'], 
                             source=group_sets['IN_L_d'])] = 'E'
synSTSP_set_distrib.loc[dict(target=group_sets['PC'], 
                             source=group_sets['IN_CL'])] = 'E'
synSTSP_set_distrib.loc[dict(target=group_sets['PC'], 
                             source=group_sets['IN_CL_AC'])] = 'E'
synSTSP_set_distrib.loc[dict(target=group_sets['PC'], 
                             source=group_sets['IN_CC'])] = 'E'
synSTSP_set_distrib.loc[dict(target=group_sets['PC'], 
                             source=group_sets['IN_F'])] = 'E'

synSTSP_set_distrib.loc[dict(target=group_sets['IN_L23'], 
                             source=group_sets['IN_L23'])] = 'F'
synSTSP_set_distrib.loc[dict(target=group_sets['IN_L5'], 
                             source=group_sets['IN_L5'])] = 'F'

syn_delay = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))), 
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='Syn delay',
    )

syn_delay.loc[dict(target='PC_L23', source='PC_L23')] = 1.5465
syn_delay.loc[dict(target='PC_L23', source='PC_L5')] = 2.7533
syn_delay.loc[dict(target='PC_L5', source='PC_L23')] = 1.9085
syn_delay.loc[dict(target='PC_L5', source='PC_L5')] = 1.5667

syn_delay.loc[dict(target='PC_L23', source=group_sets['IN_L23'])] = 1.2491
syn_delay.loc[dict(target='PC_L23', source=group_sets['IN_L5'])] = 1.4411
syn_delay.loc[dict(target='PC_L5', source=group_sets['IN_L23'])] = 1.5415  
syn_delay.loc[dict(target='PC_L5', source=group_sets['IN_L5'])]  = 0.82

syn_delay.loc[dict(target=group_sets['IN_L23'], source='PC_L23')] = 0.9581
syn_delay.loc[dict(target=group_sets['IN_L23'], source='PC_L5')]  = 1.0544
syn_delay.loc[dict(target=group_sets['IN_L5'], source='PC_L23')]  = 1.1825
syn_delay.loc[dict(target=group_sets['IN_L5'], source='PC_L5')] = 0.6
syn_delay.loc[dict(target=group_sets['IN_L23'], 
                   source=group_sets['IN_L23'])]   = 1.1
syn_delay.loc[dict(target=group_sets['IN_L5'], 
                   source=group_sets['IN_L5'])]    = 1.1

syn_gmax = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))),
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='Syn gmax',
    )
syn_gmax.loc[dict(target='PC_L23', source=group_sets['ALL'])] = [
    0.8405, 2.2615, 2.2615, 0.18, 0.18, 2.2615, 1.8218,
    0.8378, 0.2497, 0.2497, 0.0556, 0.0556, 0.2497, 0.2285,
    ]
syn_gmax.loc[dict(target='PC_L5', source=group_sets['ALL'])]  = [
    0.9533, 1.0503, 1.0503, 0.0836, 0.0836, 1.0503, 0.8461, 
    0.8818, 1.7644, 1.7644, 0.3932, 0.3932, 1.7644, 1.6146,
    ]
syn_gmax.loc[dict(target=group_sets['IN_L23'], source='PC_L23')] = [
    1.3403, 1.3403, 0.4710, 0.4710, 1.3403, 0.2500,
    ]
syn_gmax.loc[dict(target=group_sets['IN_L5'], source='PC_L23')]  = [
    1.5201, 1.5201, 0.5342, 0.5342, 1.5201, 0.2835,
    ]
syn_gmax.loc[dict(target=group_sets['IN_L23'], source='PC_L5')]  = [
    0.7738, 0.7738, 0.2719, 0.2719, 0.7738, 0.1443,
    ]
syn_gmax.loc[dict(target=group_sets['IN_L5'], source='PC_L5')] = [
    1.7431, 1.7431, 0.88, 0.88, 1.7431, 0.28,
    ]
syn_gmax.loc[dict(target=group_sets['IN_L23'], 
                  source=group_sets['IN_L23'])]   = 1.35    
syn_gmax.loc[dict(target=group_sets['IN_L5'], 
                  source=group_sets['IN_L5'])]    = 1.35

syn_gmax_adjustment = {
    'PC':np.array(
        [[1.0569, 0.5875, 0.6587, 0.7567, 0.6728, 0.9899, 0.6294,
          1.6596, 0.5941, 0.6661, 0.7647, 0.6799, 1.5818, 0.6360]]
        ).transpose(),
    'IN':np.array(
        [[2.3859, 1.6277, 1.6277, 1.6671, 1.6671, 2.3142, 1.4363, 
          3.5816, 1.6277, 1.6277, 1.6671, 1.6671, 3.4016, 1.4363]]
        ).transpose()
    }

for kind in ['PC', 'IN']: 
    gmax_fac = repmat(syn_gmax_adjustment[kind], 1, len(group_sets[kind]))
    gmax_adj = syn_gmax.loc[dict(target=group_sets['ALL'], 
                                 source=group_sets[kind])] * gmax_fac
    syn_gmax.loc[dict(target=group_sets['ALL'], 
                      source=group_sets[kind])] = gmax_adj

syn_pfail = xr.DataArray(
    data= np.ones((len(group_names), len(group_names)))*0.3,
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='syn pfail',
    )
synapse_kind = xr.DataArray(
    data=np.array([[tp for tp in group_kinds.values()] 
                   for i in range(len(group_names))]),
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='synapse kinds',
    )

AMPA_par = np.array([1.4, 10, 0, 0, 0, 0], dtype='float64') # in ms and mV
GABA_par = np.asarray([3, 40, -70, 0, 0, 0], dtype='float64') # in ms and mV
NMDA_par = np.array([4.3, 75, 0, 1, 0.0625, 0], dtype='float64') # in ms and mV

channel_par = xr.DataArray(
    data=np.asarray([AMPA_par, GABA_par, NMDA_par]), 
    coords=[list(channel_names), channel_par_names], 
    dims=['channel', 'param'],
    name='channel param',
    )

channel_syn_gmax_factor = xr.DataArray(
    data=np.asarray([[1],[1],[1.09]]), 
    coords=[list(channel_names), ['factor']],
    dims=['channel', 'param'],
    name='gmax factor',
    )

syn_gmax_sigma = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))), 
    coords=[group_names, group_names],
    dims=['target', 'source'],
    name='gmax sigma',
    )      
syn_gmax_sigma.loc[dict(target='PC_L23', source='PC_L23')] = 0.4695
syn_gmax_sigma.loc[dict(target='PC_L23', source='PC_L5')] = 0.1375
syn_gmax_sigma.loc[dict(target='PC_L5', source='PC_L23')] = 0.3530
syn_gmax_sigma.loc[dict(target='PC_L5', source='PC_L5')] = 0.9653
syn_gmax_sigma.loc[dict(target=group_sets['IN_L_both_L23'], 
                        source='PC_L23')] = 1.0855
syn_gmax_sigma.loc[dict(target=group_sets['IN_L_both_L23'], 
                        source='PC_L5')] = 0.6267
syn_gmax_sigma.loc[dict(target=group_sets['IN_L_both_L5'], 
                        source='PC_L23')] = 0.8588
syn_gmax_sigma.loc[dict(target=group_sets['IN_L_both_L5'], 
                        source='PC_L5')] = 1.1194
syn_gmax_sigma.loc[dict(target=group_sets['IN_CL_both_L23'], 
                        source='PC_L23')] = 0.1999
syn_gmax_sigma.loc[dict(target=group_sets['IN_CL_both_L23'], 
                        source='PC_L5')] = 0.1154
syn_gmax_sigma.loc[dict(target=group_sets['IN_CL_both_L5'], 
                        source='PC_L23')] = 0.1581
syn_gmax_sigma.loc[dict(target=group_sets['IN_CL_both_L5'], 
                        source='PC_L5')] = 0.7033
syn_gmax_sigma.loc[dict(target='IN_CC_L23', source='PC_L23')] = 1.0855
syn_gmax_sigma.loc[dict(target='IN_CC_L23', source='PC_L5')] = 0.6267
syn_gmax_sigma.loc[dict(target='IN_CC_L5', source='PC_L23')] = 0.8588
syn_gmax_sigma.loc[dict(target='IN_CC_L5', source='PC_L5')] = 1.1194
syn_gmax_sigma.loc[dict(target='IN_F_L23', source='PC_L23')] = 0.2000
syn_gmax_sigma.loc[dict(target='IN_F_L23', source='PC_L5')] = 0.1155
syn_gmax_sigma.loc[dict(target='IN_F_L5', source='PC_L23')] = 0.1582
syn_gmax_sigma.loc[dict(target='IN_F_L5', source='PC_L5')] = 0.3000
syn_gmax_sigma.loc[dict(target='PC_L23', 
                        source=group_sets['IN_L_both_L23'])] = 1.9462
syn_gmax_sigma.loc[dict(target='PC_L23', 
                        source=group_sets['IN_L_both_L5'])] = 0.0362
syn_gmax_sigma.loc[dict(target='PC_L5', 
                        source=group_sets['IN_L_both_L23'])] = 0.9038
syn_gmax_sigma.loc[dict(target='PC_L5',
                        source=group_sets['IN_L_both_L5'])] = 0.2557
syn_gmax_sigma.loc[dict(target='PC_L23', 
                        source=group_sets['IN_CL_both_L23'])] = 0.6634
syn_gmax_sigma.loc[dict(target='PC_L23', 
                        source=group_sets['IN_CL_both_L5'])] = 0.0093
syn_gmax_sigma.loc[dict(target='PC_L5', 
                        source=group_sets['IN_CL_both_L23'])] = 0.3081
syn_gmax_sigma.loc[dict(target='PC_L5', 
                        source=group_sets['IN_CL_both_L5'])] = 0.0655
syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_CC_L23')] = 1.9462
syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_CC_L5')] = 0.0362
syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.9038
syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_CC_L23')] = 0.2557
syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_F_L23')] = 3.6531
syn_gmax_sigma.loc[dict(target='PC_L23', source='IN_F_L5')] = 0.1828
syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_F_L23')] = 1.6966
syn_gmax_sigma.loc[dict(target='PC_L5', source='IN_F_L5')] = 1.2919
syn_gmax_sigma.loc[dict(target=group_sets['IN_L23'], 
                        source=group_sets['IN_L23'])] = 0.35
syn_gmax_sigma.loc[dict(target=group_sets['IN_L5'], 
                        source=group_sets['IN_L5'])] = 0.35

syn_gmax_min = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))), 
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='gmax min',
    )

syn_gmax_max = xr.DataArray(
    np.zeros((len(group_names), len(group_names))), 
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='gmax max',
    )
syn_gmax_max[:,:] = 100    
           
syn_delay_sigma = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))),
    coords=[group_names, group_names], 
    dims=['target', 'source'],
    name='delay sigma',
    )
syn_delay_sigma.loc[dict(target='PC_L23', source='PC_L23')] = 0.3095
syn_delay_sigma.loc[dict(target='PC_L23', source='PC_L5')] = 0.1825
syn_delay_sigma.loc[dict(target='PC_L5', source='PC_L23')] = 0.1651
syn_delay_sigma.loc[dict(target='PC_L5', source='PC_L5')] = 0.4350  
syn_delay_sigma.loc[dict(target=group_sets['IN_L23'], 
                         source='PC_L23')] = 0.2489
syn_delay_sigma.loc[dict(target=group_sets['IN_L23'], 
                         source='PC_L5')]  = 0.0839
syn_delay_sigma.loc[dict(target=group_sets['IN_L5'], 
                         source='PC_L23')]  = 0.1327
syn_delay_sigma.loc[dict(target=group_sets['IN_L5'], 
                         source='PC_L5')] = 0.2000
syn_delay_sigma.loc[dict(target='PC_L23', 
                         source=group_sets['IN_L23'])] = 0.1786
syn_delay_sigma.loc[dict(target='PC_L23', source=group_sets['IN_L5'])] = 0.0394
syn_delay_sigma.loc[dict(target='PC_L5', source=group_sets['IN_L23'])] = 0.0940
syn_delay_sigma.loc[dict(target='PC_L5', source=group_sets['IN_L5'])] = 0.0940
syn_delay_sigma.loc[dict(target=group_sets['IN_L23'], 
                         source=group_sets['IN_L23'])] = 0.4
syn_delay_sigma.loc[dict(target=group_sets['IN_L5'], 
                         source=group_sets['IN_L5'])] = 0.4

syn_delay_min = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))),
    coords=[group_names, group_names],
    dims=['target', 'source'],
    name='Delay min',
    )
  
syn_delay_max = xr.DataArray(
    data=np.zeros((len(group_names), len(group_names))),
    coords=[group_names, group_names],
    dims=['target', 'source'],
    name='delay max',
    )
syn_delay_max[:,:] = 2    

spiking_params = {}
for par in synspike_par_names:
    spiking_params[par] = dict()
spiking_params['pfail'] = syn_pfail
spiking_params['gmax'] = dict(mean=syn_gmax, sigma=syn_gmax_sigma, 
                              min=syn_gmax_min, max=syn_gmax_max)
neuron_eq_memb_par = ['{}: {}'.format(
    name, baseunit_dict[memb_par_dict[name]['unit']])
    for name in memb_par_names]
neuron_eq_memb_par = '\n'.join(neuron_eq_memb_par)

neuron_eq_channel_par = [
    '{}: {}'.format(param, baseunit_dict[channelpar_units[param]['unit']]) 
    for param in channelpar_units]

neuron_eq_channel_par = '\n'.join(neuron_eq_channel_par)

neuron_eq_auxiliary_var = (
    'I_ref: amp\n'
    'last_spike: second'
    )

neuron_eq_channel_Mgfactor = (
    'Mg_{0} = (1/(1 +  Mg_fac_{0} * 0.33 * exp(Mg_slope_{0} * '
    '(Mg_half_{0}*mV - V)/mV))):1'
    )
neuron_eq_channel_Mgfactor = [neuron_eq_channel_Mgfactor.format(name) 
                              for name in channel_names]
neuron_eq_channel_Mgfactor = '\n'.join(neuron_eq_channel_Mgfactor)

neuron_eq_channel_I = (
    'I_{0} = g_{0} * (E_{0} - V) * Mg_{0}: amp'
    )
neuron_eq_channel_I = [neuron_eq_channel_I.format(name) 
                       for name in channel_names]
neuron_eq_channel_I = '\n'.join(neuron_eq_channel_I)

neuron_eq_channel_g = (
    'g_{0} = g_{0}_off - g_{0}_on: siemens'
    )
neuron_eq_channel_g = [neuron_eq_channel_g.format(name) 
                       for name in channel_names]
neuron_eq_channel_g = '\n'.join(neuron_eq_channel_g)

neuron_eq_channel_dgdt  = (
    'dg_{0}_off/dt = - (1/tau_off_{0}) * g_{0}_off: siemens\n'
    'dg_{0}_on/dt = - (1/tau_on_{0}) * g_{0}_on: siemens'
    )

neuron_eq_channel_dgdt = [neuron_eq_channel_dgdt.format(name) 
                          for name in channel_names]
neuron_eq_channel_dgdt = '\n'.join(neuron_eq_channel_dgdt)

neuron_eq_memb_I = (
    'I_DC: amp\n'
    'I_AC = {}*pA: amp\n'
    'I_syn = ' + ' + '.join(['I_{0}'.format(name) 
                             for name in channel_names]) + ': amp\n'
    'I_inj = I_DC + I_AC: amp\n'
    'I_tot =  I_syn + I_inj: amp'
    )


neuron_eq_membr_state = (
    'I_exp = g_L * delta_T * exp((V - V_T)/delta_T): amp\n'
    'w_V = I_tot + I_exp -g_L * (V - E_L): amp\n'
    
    'dV = int(I_tot >= I_ref) * int(t-last_spike < 5*ms) * (-g_L/C) * (V-V_r)'
    '+ (1 - int(I_tot >= I_ref) * int(t-last_spike < 5*ms))'
    '* (I_tot + I_exp - g_L * (V-E_L) - w)/C: volt/second\n'   
    'dV/dt = dV: volt\n'
    
    'D0 = (C/g_L) * w_V:  coulomb\n'
    'dD0 = C *(exp((V - V_T)/delta_T)-1): farad\n'
    'dw/dt = int(w > w_V-D0/tau_w) * int(w < w_V+D0/tau_w) * int(V <= V_T)'
    '* int(I_tot < I_ref) *'
    ' -(g_L * (1 - exp((V-V_T)/delta_T)) + dD0/tau_w)*dV: amp'
    )

neuron_eq_model = '\n\n'.join(
    [neuron_eq_memb_par, neuron_eq_channel_par, neuron_eq_channel_Mgfactor, 
     neuron_eq_channel_I, neuron_eq_channel_g, neuron_eq_channel_dgdt, 
     neuron_eq_auxiliary_var, neuron_eq_memb_I, neuron_eq_membr_state]
    )

neuron_eq_thres = "V > V_up"
neuron_eq_reset = "V = V_r; w += b"

neuron_eq_event = {}
neuron_eq_event['w_cross'] = {}

neuron_eq_event['w_cross']['condition'] = ('w > w_V - D0/tau_w and '
                                           'w < w_V + D0/tau_w and '
                                           'V <= V_T')

neuron_eq_event['w_cross']['vars'] = ["V", "w"]
neuron_eq_event['w_cross']['reset'] = "w=w_V - D0/tau_w"

def neuron_rheobase(g_L, V_T, E_L ,delta_T): return g_L * (V_T - E_L - delta_T)

syn_eq_channel = '\n'.join(['{0}: 1'.format(name) for name in channel_names])
syn_eq_STSP_var = '\n'.join(
    ['{}: {}'.format(var, baseunit_dict[synSTSP_all_dict[var]['unit']])
     for var in synSTSP_all_names]
    )
syn_eq_spike_par = '\n'.join(
    ['{}: {}'.format(par, baseunit_dict[synspike_par[par]['unit']]) 
    for par in synspike_par_names]
    )
syn_eq_aux_var = '\n'.join(
    ['{}: {}'.format(par, baseunit_dict[synaux_par[par]['unit']]) 
    for par in synaux_par_names]
    )

syn_eq_model = '\n\n'.join(
    [syn_eq_channel,syn_eq_STSP_var, syn_eq_spike_par, syn_eq_aux_var]
    )

extsyn_eq_model = '\n\n'.join(
    [syn_eq_channel, syn_eq_spike_par,syn_eq_aux_var]
    )

syn_eq_pathway = []
syn_eq_pathway.append(dict(
    eq='failure = int(rand()<pfail)',
    order=0, 
    delay=False,
    ))
syn_eq_pathway.append(dict(
    eq="u_temp = U + u * (1 - U) * exp(-(t - last_spike_pre)/tau_fac)",
    order=1,
    delay=False,
    ))
syn_eq_pathway.append(dict(
    eq="R_temp = 1 + (R - u * R - 1) * exp(- (t - last_spike_pre)/tau_rec)",
    order=2,
    delay=False,
    ))
syn_eq_pathway.append(dict(
    eq='u = u_temp',
    order=3,
    delay=False,
    ))
syn_eq_pathway.append(dict(
    eq='R = R_temp',
    order=4,
    delay=False,
    ))
syn_eq_pathway.append(dict(
    eq='last_spike_pre = t',
    order=5,
    delay=False,
    ))
syn_eq_pathway.append(dict(
    eq='a_syn = u * R',
    order=6,
    delay=False,
    ))

for name in channel_names:    
    syn_eq_pathway.append(dict(
        eq="g_{0}_on_post += {0} * gmax * "
        "a_syn * (1-failure) * gsyn_amp".format(name),
        order=7,
        delay=True
        ))
    syn_eq_pathway.append(dict(
        eq="g_{0}_off_post += {0} * gmax "
        "* a_syn * (1 - failure) * gsyn_amp".format(name),
        order=7,
        delay=True
        ))
    
eq_extern_syn_pathway = []
eq_extern_syn_pathway.append(dict(eq='failure = int(rand()<pfail)',
                                  order=0, 
                                  delay=False))
for name in channel_names:    
    eq_extern_syn_pathway.append(dict(
        eq="g_{0}_on_post += {0} * gmax * (1-failure) * gsyn_amp".format(name), 
        order=0, 
        delay=False
        ))
    eq_extern_syn_pathway.append(dict(
        eq="g_{0}_off_post += {0} * gmax * "
        "(1-failure) * gsyn_amp".format(name), 
        order=0,
        delay=False
        ))

eq_var_units = dict(V ='mV', w='pA', t='ms', I_tot='pA', I_syn='pA', 
                    I_inj='pA', I_DC='pA') 
for name in channel_names:
    eq_var_units['I_{0}'.format(name)]='pA'
for var_unit in [memb_par_dict, synSTSP_all_dict, synspike_par, synaux_par]:
    for var in var_unit:
        eq_var_units[var]=var_unit[var]['unit']



@time_report('Basics setup')
def basics_setup(Ncells_prompted, Nstripes, basics_scales=None, 
                 alternative_pcells=None, disp=True):
    
    # Copies of basic_scales targets are necessary when many simulations are 
    # carried out without
    # re-importing this module; otherwise, the modifications by basics_scales 
    # would remain
    # in the DataArrays between subsequent simulations
    PCON_COPY = syn_pcon.copy()
    CELL_STD_COPY = memb_par_std.copy()
    syn_gmax_COPY = syn_gmax.copy()
    spiking_params_copy = spiking_params.copy()
    spiking_params_copy['gmax'] = dict(mean=syn_gmax_COPY, 
                                       sigma=syn_gmax_sigma, 
                                       min=syn_gmax_min,
                                       max=syn_gmax_max)

    SCALABLES = {'pCon': PCON_COPY, 
                 'membr_param_std': CELL_STD_COPY, 
                 'gmax_mean': syn_gmax_COPY}
    
    
    if alternative_pcells is None:
        pcells = pcells_per_group
    else:
        pcells = alternative_pcells
    
    if basics_scales is not None:
    
        for param in basics_scales:              
            for TS_dict, scale in basics_scales[param]:
                new_param = SCALABLES[param].loc[TS_dict].values * scale
                SCALABLES[param].loc[TS_dict] = new_param
         
    memb_par_covar=xr.DataArray(
        np.zeros((len(group_names), len(memb_par_names), len(memb_par_names))),
        coords=[group_names, memb_par_names, memb_par_names], 
        dims=['group', 'memb_par0', 'memb_par1'],
        name='memb_par covariance',
        )  
    
    for group in memb_par_std.coords['group'].values:
        for param0 in memb_par_std.coords['param'].values:
            for param1 in memb_par_std.coords['param'].values:
                    cov = memb_par_covar_normalized.loc[
                        dict(group=group, memb_par0=param0, memb_par1=param1)
                        ].values
                    std0 = CELL_STD_COPY.loc[
                        dict(group=group, param=param0)
                        ].values
                    std1 = CELL_STD_COPY.loc[
                        dict(group=group, param=param1)
                        ].values
                    memb_par_covar.loc[
                        dict(group=group, memb_par0=param0, memb_par1=param1)
                        ] = cov*std0*std1
            
    group_setup = GroupSetup(group_kinds, group_sets)          
    stripes_setup = StripeSetup(Nstripes, interstripe_sets)
    connection_setup = ConnectionSetup(PCON_COPY, syn_clusterflag)          
    cellsetup = StructureSetup(Ncells_prompted, pcells, group_setup, 
                               stripes_setup, connection_setup)
    
    membrane_setup = MembraneSetup(memb_par_names, memb_par_dict, unitbr2_dict, 
                                   baseunit_dict, memb_par_mean, 
                                   memb_par_covar, memb_par_ktransf, 
                                   CELL_STD_COPY, memb_par_min, memb_par_max, 
                                   memb_tau_min, memb_tau_max) 
    
    STSP_setup=STSPSetup(decl_vars=synSTSP_all_dict,kinds=synSTSP_types, 
                         sets=synSTSP_set_probs, distr=synSTSP_set_distrib)
    channels_setup = ChannelSetup(channel_kinds,channel_par, 
                                  channel_syn_gmax_factor, channelpar_units)      
    
    spike_params_data = ParamSetup()
    for par in synspike_par_names:
        spike_params_data[par] = spiking_params_copy[par]
        
    spike_params_setup = SpikeParamSetup(synspike_par, spike_params_data)
    delay_setup = DelaySetup(syn_delay, syn_delay_sigma, syn_delay_min, 
                             syn_delay_max)
    synapses_setup = SynapseSetup(synapse_kind, spike_params_setup, 
                                  delay_setup, STSP_setup, channels_setup)
    eqs_setup = EquationsSetup(neuron_eq_model, neuron_eq_thres, 
                               neuron_eq_reset, neuron_eq_event, 
                               neuron_rheobase, syn_eq_model, syn_eq_pathway, 
                               extsyn_eq_model, 
                               eq_extern_syn_pathway, eq_var_units)
    
    if cellsetup.Ncells != Ncells_prompted and disp:
        print(('REPORT: The number of neurons was adjusted from {} to {} due '
               'to roundings.\n'.format(Ncells_prompted, cellsetup.Ncells)))
             
    return BasicsSetup(cellsetup, membrane_setup, synapses_setup, 
                       eqs_setup, SCALABLES, basics_scales)

    
@dataclass
class ChannelSetup(BaseClass):
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
class SynapseSetup(BaseClass):
    kinds: any
    spiking: any
    delay: any
    STSP: any
    channels: any


@dataclass
class SpikeParamSetup(BaseClass):
    names: any
    params: any

    
@dataclass
class DelaySetup(BaseClass):
    delay_mean: any
    delay_sigma: any
    delay_min: any
    delay_max: any
    

@dataclass
class StripeSetup(BaseClass):
    N: int
    inter: any


@dataclass
class ConnectionSetup(BaseClass):
    pCon: any
    cluster: any
    

@dataclass
class STSPSetup(BaseClass):
    decl_vars: any
    kinds: dict
    sets: any
    distr: dict


@dataclass
class MembraneSetup(BaseClass):  
    names: any
    name_units: any
    unitbr2_dict: any
    unit_main_dict: any
    mean: float
    covariance: float
    k: float
    std: float
    min: float
    max:float
    tau_m_min: any
    tau_m_max: any

 
@dataclass
class GroupSetup(BaseClass):   
    kinds: any
    sets:any
    names: list[str] = field(default_factory=list)
    N:int = 0
    
    def __post_init__(self):
        self.names = list(self.kinds.keys())
        self.N = len(self.names)
        self.idcs = {}
        for name_idc in range(len(self.names)):
            self.idcs[self.names[name_idc]] = name_idc


@dataclass
class StructureSetup(BaseClass):
         
    Ncells_prompt: int
    Pcells_per_group: list or np.array
    groups: GroupSetup
    stripes: dict
    conn: any
    Ncells: int = 0
    Ncells_total: int = 0
    Ncells_per_group: xr.DataArray = xr.DataArray([])
        
    def __post_init__(self):
        Ncells_per_group = np.ceil(
            (self.Ncells_prompt*self.Pcells_per_group)/100
            ).astype(int)
        self.Ncells_per_group = xr.DataArray(Ncells_per_group, 
                                             coords=[self.groups.names], 
                                             dims='group')
        self.Ncells = int(sum(self.Ncells_per_group))
        self.Ncells_total = self.stripes.N*self.Ncells
        
    
@dataclass
class ParamSetup(BaseClass):
    pass


@dataclass
class EquationsSetup(BaseClass):
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
class BasicsSetup(BaseClass):
    struct: any
    membr: any
    syn: any
    equations: any
    scalables: any
    scales: any

    
if __name__ == '__main__':
   
    N1=1000
    Nstripes=1

    membr_param_std = [(dict(group=group_sets['ALL'], 
                             param=memb_par_names), 1)]
    basics_scales = {'membr_param_std': membr_param_std}

    basics = basics_setup(N1, Nstripes, basics_scales=basics_scales, 
                          disp=False)