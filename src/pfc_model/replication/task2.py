import brian2 as br2
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from pfc_model.analysis import*

__all__ = ['task2']

def task2(simulation_dir):
    if not os.path.isdir(os.path.join(simulation_dir, 'Figures')):
        os.mkdir(os.path.join(simulation_dir, 'Figures'))
        
    Varr = np.arange(-100, 50)
    Sarr = [_S(V) for V in Varr]
    
    _fig02(Varr, Sarr, simulation_dir)
    
    gmax = 10
    rate=50

    a_syn= 0.26

    eq1 = """
    g_{0} = g_{0}_off - g_{0}_on: siemens
    dg_{0}_off/dt = - (1/tau_off_{0}) * g_{0}_off: siemens
    dg_{0}_on/dt = - (1/tau_on_{0}) * g_{0}_on: siemens
    tau_off_{0}: second
    tau_on_{0}: second
    """.format('AMPA')
    neuron = br2.NeuronGroup(1, model=eq1, method='rk4')
    neuron.tau_off_AMPA = 10*br2.ms
    neuron.tau_on_AMPA = 1.4*br2.ms


    state1 = br2.StateMonitor(neuron, True, True)

    spikes= np.arange(0,100, 1)*br2.ms
    sp = br2.SpikeGeneratorGroup(1,np.zeros(len(spikes)), spikes)

    eq2 = """
    dx/dt = rate*Hz: 1 
    rate: 1
    last_spike: second
    """
    n2 = br2.NeuronGroup(1, model=eq2, threshold='x>=1', reset='x=0', method='rk4')
    n2.last_spike = -10000*br2.second

    n2.rate = rate

    state2= br2.StateMonitor(n2, True, True)
    sp2 = br2.SpikeMonitor(n2)


    syn_eq = """
    a_syn: 1
    gmax: siemens
    """

    dict_pre = {}
    dict_pre['p6'] = "g_{0}_on_post +=  gmax * a_syn  ".format('AMPA')
    dict_pre['p7'] = "g_{0}_off_post += gmax * a_syn ".format('AMPA')


    syn = br2.Synapses(n2, neuron, model=syn_eq, on_pre=dict_pre, method='rk4')

    br2.defaultclock.dt = 0.05*br2.ms

    syn.p6=6
    syn.p7=7

    syn.connect()

    syn.gmax= gmax*br2.nS
    syn.a_syn = a_syn



    br2.run(200*br2.ms)

    t_0 = state1.t/br2.ms
    g_0 = state1.g_AMPA[0]/br2.nS

    label_list = ['E$_{fac}$', 'E$_{dep}$', 'E$_{comb}$']
    letter_list = ['(A)', '(B)', '(C)']
    tau_rec_list = [194, 671, 329]
    tau_fac_list = [507, 17, 326]
    U_list = [0.28, 0.25, 0.29]

    t_list = []
    g_list = []


    for i in range(len(tau_rec_list)):
        
        
        tau_rec = tau_rec_list[i]
        tau_fac = tau_fac_list[i]
        U= U_list[i]
        
        
        
        eq1 = """
        g_{0} = g_{0}_off - g_{0}_on: siemens
        dg_{0}_off/dt = - (1/tau_off_{0}) * g_{0}_off: siemens
        dg_{0}_on/dt = - (1/tau_on_{0}) * g_{0}_on: siemens
        tau_off_{0}: second
        tau_on_{0}: second
        """.format('AMPA')
        neuron = br2.NeuronGroup(1, model=eq1, method='rk4')
        neuron.tau_off_AMPA = 10*br2.ms
        neuron.tau_on_AMPA = 1.4*br2.ms
        
        
        state1 = br2.StateMonitor(neuron, True, True)
        
        spikes= np.arange(0,100, 1)*br2.ms
        sp = br2.SpikeGeneratorGroup(1,np.zeros(len(spikes)), spikes)
        
        eq2 = """
        dx/dt = rate*Hz: 1 
        rate: 1
        last_spike: second
        """
        n2 = br2.NeuronGroup(1, model=eq2, threshold='x>=1', reset='x=0', method='rk4')
        n2.last_spike = -10000*br2.second
        
        n2.rate = rate
        
        state2= br2.StateMonitor(n2, True, True)
        sp2 = br2.SpikeMonitor(n2)
        
        
        syn_eq = """
        U: 1
        u: 1
        u_temp: 1
        R: 1
        R_temp: 1
        a_syn: 1
        tau_rec: second
        tau_fac: second
        gmax: siemens
        """
        
        dict_pre = {}
        dict_pre['p0'] = 'u_temp = U + u * (1 - U) * exp(-(t - last_spike_pre)/tau_fac)'
        dict_pre['p1'] = "R_temp = 1 + (R - u * R - 1) * exp(- (t - last_spike_pre)/tau_rec)"
        dict_pre['p2'] = 'u = u_temp'
        dict_pre['p3'] = 'R = R_temp'
        dict_pre['p4'] = 'last_spike_pre = t'
        dict_pre['p5'] = 'a_syn = u * R'
        dict_pre['p6'] = "g_{0}_on_post +=  gmax * a_syn  ".format('AMPA')
        dict_pre['p7'] = "g_{0}_off_post += gmax * a_syn ".format('AMPA')
        
        
        syn = br2.Synapses(n2, neuron, model=syn_eq, on_pre=dict_pre, method='rk4')
        
        br2.defaultclock.dt = 0.05*br2.ms
        
        syn.p0=0
        syn.p1=1
        syn.p2=2
        syn.p3=3
        syn.p4=4
        syn.p5=5
        syn.p6=6
        syn.p7=7
        
        syn.connect()
        
        syn.gmax= gmax*br2.nS
        syn.tau_rec = tau_rec*br2.ms
        syn.tau_fac = tau_fac*br2.ms
        syn.u = U
        syn.U = U
        syn.R=1
        
        
        
        br2.run(200*br2.ms)
        
        t_list.append(state1.t/br2.ms)
        g_list.append(state1.g_AMPA[0]/br2.nS)
    
    _fig03(t_list, g_list, t_0, g_0, letter_list, label_list, simulation_dir)

def _fig02(Varr, Sarr, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
    plt.figure(figsize=(18,10))
    plt.tick_params(labelsize=24)
    plt.xlabel('V (mV)', fontsize=24)
    plt.ylabel('S(V)', fontsize=24)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=6)
    plt.plot(Varr, Sarr, lw=3)
    plt.savefig(os.path.join(path, 'Figures', 'Fig02.png'))

def _fig03(t_list, g_list, t_0, g_0, letter_list, label_list, path):
    if not os.path.isdir(os.path.join(path, 'Figures')):
        os.mkdir(os.path.join(path, 'Figures'))
        
    fig, axs = plt.subplots(1, 3, figsize=(20,12))
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)


    for i in range(len(g_list)):
        t = t_list[i]
        g = g_list[i]
        label = letter_list[i]

        ax = axs[i]
        plt.sca(ax)
        ax.text(0, 1, label, transform=ax.transAxes + trans,
            fontsize=24, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))
        plt.plot(t, g, lw=3, label=label_list[i])
        plt.plot(t_0, g_0, ls='--', color='black', label='without STSP')
        plt.tick_params(labelsize=26)
        plt.xticks([0, 100, 200])
        plt.ylim(-0.1, 2.7)
        plt.yticks([0, 2])
        plt.legend(loc='upper right', fontsize=22)

    plt.sca(axs[0])
    plt.ylabel('g$_{AMPA}$ (nS)', fontsize=26)

    plt.sca(axs[1])
    plt.xlabel('t (ms)', fontsize=26)
    
    plt.savefig(os.path.join(path, 'Figures', 'Fig03.png'))

def _S(V):
    return 1/(1+0.33*np.exp(-0.0625*V))

if __name__ == '__main__':
    simulation_dir = set_simulation_dir('Results_'+os.path.basename(__file__)[:-3])
    task2(simulation_dir)

