""" This script defines task3.

task3 compairs the fraction of spiking and not-spiking cells, as well as
theirs membrane and synaptic parameters and probabilities of connection.
"""

import numpy as np
import os
from pfc_model import *
from pfc_model.analysis import *

__all__ = ['task3']

@time_report()
def task3(simulation_dir, Ntrials=100, seed_list=None):
    """Compair the fraction of spiking and not-spiking cells, as well as
    theirs membrane and synaptic parameters and probabilities of connection in
    subsequent independent simulations.
    
    - The summary report of the fraction of spiking cells are saved to:
        "simulation_dir/Reports/Spikingcells/spikingcells_fraction.txt"
    
    - The summary report of comparisons of membrane parameters are saved to:
        "simulation_dir/Reports/Param_comparisons/spikingcells_membparams.txt"
      The individual reports are saved to:
        "simulation_dir/Reports/Param_comparisons/individuals/
        membr_params_comparison_seed_Y.txt" (Y must be replaced by seed number)
        
    - The summary report of comparisons of synaptic parameters are saved to:
        "simulation_dir/Reports/Param_comparisons/spikingcells_synparams.txt"
      The individual reports are saved to:
        "simulation_dir/Reports/Param_comparisons/individuals/
        syn_params_comparison_CHANNEL_seed_Y.txt"
        (Y must be replaced for seed number and
         CHANNEL for AMP, GABA or NMDA).
          
    - The summary report of comparisons of probabilities of connection:
        "simulation_dir/Reports/Param_comparisons/spikingcells_pcon.txt"
      The individual reports are saved to:
        "simulation_dir/Reports/Param_comparisons/individuals/
        pcon_contingency_AMPA_seed_Y.txt"
        (Y must be replaced for seed number and
         CHANNEL for AMP, GABA or NMDA).
        
    Here, "simulation_dir" is to be replaced by the actual argument. 
        
    Parameters
    ----------
    simulation_dir: str
        Path to directory where results are to be saved.
    Ntrials: int, optional
        Number of simulations to be performed and included in the analyses.
        If not given, it defaults to 100.
    seed_list: list[int]
        List of integers with seed numbers. Its length must be equal to 
        Ntrials. If not given, it default to the list from 0 to Ntrials-1.
    """
    if seed_list is None:
        seed_list = [i for i in range(Ntrials)]
    else:
        if len(seed_list) != Ntrials:
            raise ValueError('Length of seed_list must be equal to Ntrials.')
    
    n_cells=1000
    n_stripes=1
    constant_stimuli = [[('PC', 0), 250],
                        [('IN', 0), 200]]
    method = 'rk4'
    dt = 0.05
    transient = 1000      
    duration = 16000
    
    spiking_threshold = 0.33
    PC_result = []
    IN_result = []
    ALL_result = []
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports')):
        os.mkdir(os.path.join(simulation_dir, 'Reports'))
    
    
    if not os.path.isdir(os.path.join(simulation_dir, 
                                  'Reports', 'Param_comparisons')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'Param_comparisons'))
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports', 
                                      'Param_comparisons', 'individuals')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 
                              'Param_comparisons', 'individuals')) 
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports', 'Spikingcells')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'Spikingcells'))
    
    if not os.path.isdir(os.path.join(simulation_dir, 'Reports', 'Spikingcells', 
                                      'individuals')):
        os.mkdir(os.path.join(simulation_dir, 'Reports', 'Spikingcells', 
                              'individuals'))
    
    membpar_dict = {}
    
    for i in range(len(seed_list)):
        seed = seed_list[i]
        cortex = Cortex.setup(n_cells=n_cells, n_stripes=n_stripes, 
                              constant_stimuli=constant_stimuli, method=method,
                              dt=dt, transient=transient, seed=seed)    
        cortex.run(duration)
    

        spikingPC = cortex.spiking_idcs((np.greater_equal, spiking_threshold), 
                                        ('PC', 0))
        spikingIN = cortex.spiking_idcs((np.greater_equal, spiking_threshold), 
                                        ('IN', 0))
        spikingALL = cortex.spiking_idcs((np.greater_equal, spiking_threshold), 
                                        ('ALL', 0))
        PC_len = len(cortex.neuron_idcs(('PC', 0)))
        PC_result.append(len(spikingPC)*100/PC_len)
        
        IN_len = len(cortex.neuron_idcs(('IN', 0)))
        IN_result.append(len(spikingIN)*100/IN_len)
        
        ALL_len = len(cortex.neuron_idcs(('ALL', 0)))
        ALL_result.append(len(spikingALL)*100/ALL_len)
        
        # text output#
        
    
        
        get_spiking(cortex, 0.33, ('PC', 0), 
                    file=os.path.join(simulation_dir, 'Reports', 
                                      'Spikingcells', 'individuals',
                                      'spikingPC_seed_{}.txt'.format(seed)))
        
        
        get_spiking(cortex, 0.33, ('IN', 0), 
                    file=os.path.join(simulation_dir, 'Reports', 
                                      'Spikingcells', 'individuals',
                                      'spikingIN_seed_{}.txt'.format(seed)))
        
        
        get_spiking(cortex, 0.33, ('ALL', 0), 
                    file=os.path.join(simulation_dir, 'Reports', 
                                      'Spikingcells', 'individuals',
                                      'spikingALL_seed_{}.txt'.format(seed)))
        
        
        
        membpar_results = comp_membrparam_rategroup(
            cortex, 0.33, [('ALL',0), ('PC', 0), ('IN', 0)], 
            file=os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                              'individuals',
                              'membr_params_comparison_seed_{}.txt'.format(seed)))
        
        cont_results = {}
        synpar_results = {}
        for channel in cortex.network.basics.syn.channels.names:
            cont_results[channel] = contingency(
                cortex, 0.33, 
                [('ALL',0), ('PC', 0), ('IN', 0)], ('ALL', 0), 
                channel=channel,
                file=os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                                  'individuals',
                                  'pcon_contingency_{}_seed_{}.txt'.format(channel, seed)))
            
            synpar_results[channel] = comp_synparam_rategroup(
                cortex, 0.33, [('ALL',0), ('PC', 0), ('IN', 0)], 
                ('ALL', 0), channel=channel, 
                file=os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                                  'individuals',
                                  'syn_params_comparison_{}_seed_{}.txt'.format(channel, seed)))
            
            
        if i == 0:
            for par in membpar_results:
                membpar_dict[par] = {}
                for gs in membpar_results[par]:
                    membpar_dict[par][gs] = {}
                    membpar_dict[par][gs]['seed'] = []
                    membpar_dict[par][gs]['significant'] = []
                    membpar_dict[par][gs]['notspiking_mean'] = []
                    membpar_dict[par][gs]['spiking_mean'] = []
            
            cont_dict = {}
            synpar_dict = {}
            for channel in cortex.network.basics.syn.channels.names:
                cont_dict[channel] = {}
                for ts in cont_results[channel]:
                    cont_dict[channel][ts] = {}
                    cont_dict[channel][ts]['seed'] = []
                    cont_dict[channel][ts]['significant'] = []
                    cont_dict[channel][ts]['notspiking_pcon'] = []
                    cont_dict[channel][ts]['spiking_pcon'] = []
                
                synpar_dict[channel] = {}
                for par in synpar_results[channel]:
                    synpar_dict[channel][par] = {}
                    for ts in synpar_results[channel][par]:
                        synpar_dict[channel][par][ts] = {}
                        synpar_dict[channel][par][ts]['seed'] = []
                        synpar_dict[channel][par][ts]['significant'] = []
                        synpar_dict[channel][par][ts]['notspiking_mean'] = []
                        synpar_dict[channel][par][ts]['spiking_mean'] = []
                        
                     
        for par in membpar_dict:
            for gs in  membpar_dict[par]:
                pvalue = membpar_results[par][gs]['mwtest'].pvalue
                sign = 1 if pvalue < 0.05 else 0
                less_mean = membpar_results[par][gs]['less'].mean
                geq_mean = membpar_results[par][gs]['greater_equal'].mean
                
                membpar_dict[par][gs]['seed'].append(seed)
                membpar_dict[par][gs]['significant'].append(sign)
                membpar_dict[par][gs]['notspiking_mean'].append(less_mean)
                membpar_dict[par][gs]['spiking_mean'].append(geq_mean)
        
        for channel in cont_dict:
            for ts in cont_dict[channel]:
                pvalue = cont_results[channel][ts]['pvalue']
                signif = 1 if pvalue < 0.05 else 0
                pcon_less = cont_results[channel][ts]['pcon_less']
                pcon_geq = cont_results[channel][ts]['pcon_geq']
                            
                cont_dict[channel][ts]['seed'].append(seed)
                cont_dict[channel][ts]['significant'].append(signif)
                cont_dict[channel][ts]['notspiking_pcon'].append(pcon_less)
                cont_dict[channel][ts]['spiking_pcon'].append(pcon_geq)
            
            for par in synpar_dict[channel]:
                for ts in synpar_dict[channel][par]:
                    pvalue = synpar_results[channel][par][ts]['mwtest'].pvalue
                    sign = 1 if pvalue < 0.05 else 0
                    less_mean = synpar_results[channel][par][ts]['less'].mean
                    geq_mean = synpar_results[channel][par][ts]['greater_equal'].mean
                    
                    synpar_dict[channel][par][ts]['seed'].append(seed)
                    synpar_dict[channel][par][ts]['significant'].append(sign)
                    synpar_dict[channel][par][ts]['notspiking_mean'].append(less_mean)
                    synpar_dict[channel][par][ts]['spiking_mean'].append(geq_mean)
    
        
    results = [PC_result, IN_result, ALL_result]
    results_name = ['PC', 'IN', 'All cells']

    with open(os.path.join(simulation_dir, 'Reports', 'Spikingcells',
                           'spikingcells_fraction.txt'), 'w') as f:
         print('Spiking cell statistics', file=f)
         print('Spiking threshold: {} Hz'.format(spiking_threshold), file=f)
         print('Number of trials: {}'.format(Ntrials), file=f)
         print('List of seeds:', seed_list, file=f)
         print(file=f)
         for i in range(len(results)):
             print(results_name[i], file=f)
             result = results[i]
             print('Mean: {:.2f} %'.format(np.mean(result)), file=f)
             print('Standard deviation: {:.2f} %'.format(np.std(result)), 
                   file=f)
             print('Max: {:.2f} %'.format(np.max(result)), file=f)
             print('Min: {:.2f} %'.format(np.min(result)), file=f)
             print('-'*40, file=f)
             print(file=f)
        
         
    with open(os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                            'spikingcells_membparams.txt'), 'w') as f:
        print('Membrane parameter', file=f)
        print('Spiking threshold: {} Hz'.format(spiking_threshold), file=f)
        print('Number of trials: {}'.format(Ntrials), file=f)
        print('List of seeds:', seed_list, file=f)
        print(file=f)
        for par in membpar_dict:
            print('='*(len(par)+6), file=f)
            print('|| '+par+' ||', file=f)
            print('='*(len(par)+6), file=f)
            print(file=f)
            for gs in membpar_dict[par]:
                print('Group:', gs, file=f, end='\n\n')
                seeds = np.asarray(membpar_dict[par][gs]['seed'])
                less_means = np.asarray(membpar_dict[par][gs]['notspiking_mean'])
                geq_means = np.asarray(membpar_dict[par][gs]['spiking_mean'])
                sign = np.asarray(membpar_dict[par][gs]['significant'])
                sign_where = sign.astype(bool)
                sign_seeds = seeds[sign_where]
                sign_less = less_means[sign_where]
                sign_geq = geq_means[sign_where]                
                less_higher = np.where(sign_less > sign_geq)[0]
                geq_higher = np.where(sign_less < sign_geq)[0]
                

                print('Average across all simulations:', file=f)
                print('Spiking cells: '
                      '{:.2f}'.format(np.mean(geq_means)), file=f)
                print('Not-spiking cells: '
                      '{:.2f}'.format(np.mean(less_means)), end='\n\n', file=f)
                
                print('Ocurrences of significant difference:', np.sum(sign), 
                      file=f)
                if np.sum(sign) > 0:
                    print('-- Higher in spiking cells:', len(geq_higher), 
                    end='', file=f)
                    
                    if len(geq_higher)>0:
                        print(' (seeds: ', 
                              ', '.join(sign_seeds[geq_higher].astype(str)), ')', 
                              sep='',  file=f)
                    else:
                        print(file=f)
                    
                    print('-- Higher in not-spiking cells:', len(less_higher), 
                    end='', file=f)
                    
                    if len(less_higher)>0:
                        print(' (seeds: ', 
                              ', '.join(sign_seeds[less_higher].astype(str)), ')', 
                              sep='',  file=f, end='\n\n')
                    else:
                        print(file=f, end='\n\n')
                else:
                    print('\n', file=f)
                
                print('-'*40, file=f)
            print('='*40, file=f)
            print('='*40, file=f, end='\n\n')
                
                                  
                                
                
        
    with open(os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                            'spikingcells_synparams.txt'), 'w') as f:
        print('Synaptic parameter', file=f)
        print('Spiking threshold: {} Hz'.format(spiking_threshold), file=f)
        print('Number of trials: {}'.format(Ntrials), file=f)
        print('List of seeds:', seed_list, file=f)
        print(file=f)
        for channel in synpar_dict:
            print('='*(len(channel) + len('channel')+6), file=f)
            print('-'*(len(channel) + len('channel')+6), file=f)
            print('|| '+ channel, 'channel'+ ' ||', file=f)
            print('-'*(len(channel) + len('channel')+6), file=f)
            print('='*(len(channel) + len('channel')+6), file=f)
            print(file=f)
            
            for par in synpar_dict[channel]:
                print('='*(len(par)+len(channel)+7), file=f)
                print('|| '+ channel+ ' ' +par+' ||', file=f)
                print('='*(len(par)+len(channel)+7), file=f)
                print(file=f)
                for gs in synpar_dict[channel][par]:
                    print('Group:', gs, file=f, end='\n\n')
                    seeds = np.asarray(synpar_dict[channel][par][gs]['seed'])
                    less_means = np.asarray(synpar_dict[channel][par][gs]['notspiking_mean'])
                    geq_means = np.asarray(synpar_dict[channel][par][gs]['spiking_mean'])
                    sign = np.asarray(synpar_dict[channel][par][gs]['significant'])
                    sign_where = sign.astype(bool)
                    sign_seeds = seeds[sign_where]
                    sign_less = less_means[sign_where]
                    sign_geq = geq_means[sign_where]                
                    less_higher = np.where(sign_less > sign_geq)[0]
                    geq_higher = np.where(sign_less < sign_geq)[0]
                    
    
                    print('Average across all simulations:', file=f)
                    print('Spiking cells: '
                          '{:.2f}'.format(np.mean(geq_means)), file=f)
                    print('Not-spiking cells: '
                          '{:.2f}'.format(np.mean(less_means)), end='\n\n', file=f)
                    
                    print('Ocurrences of significant difference:', np.sum(sign), 
                          file=f)
                    if np.sum(sign) > 0:
                        print('-- Higher in spiking cells:', len(geq_higher), 
                        end='', file=f)
                        
                        if len(geq_higher)>0:
                            print(' (seeds: ', 
                                  ', '.join(sign_seeds[geq_higher].astype(str)), ')', 
                                  sep='',  file=f)
                        else:
                            print(file=f)
                        
                        print('-- Higher in not-spiking cells:', len(less_higher), 
                        end='', file=f)
                        
                        if len(less_higher)>0:
                            print(' (seeds: ', 
                                  ', '.join(sign_seeds[less_higher].astype(str)), ')', 
                                  sep='',  file=f, end='\n\n')
                        else:
                            print(file=f, end='\n\n')
                    else:
                        print('\n', file=f)
                    
                    print('-'*40, file=f)
                print('='*40, file=f)
                print('='*40, file=f, end='\n\n')
                    
                                      
                                
    
        
        with open(os.path.join(simulation_dir, 'Reports', 'Param_comparisons',
                            'spikingcells_pcon.txt'), 'w') as f:
            print('pcon contingency', file=f)
            print('Spiking threshold: {} Hz'.format(spiking_threshold), file=f)
            print('Number of trials: {}'.format(Ntrials), file=f)
            print('List of seeds:', seed_list, file=f)
            print(file=f)
            for channel in synpar_dict:
                print('='*(len(channel) + len('channel')+6), file=f)
                print('-'*(len(channel) + len('channel')+6), file=f)
                print('|| '+ channel, 'channel'+ ' ||', file=f)
                print('-'*(len(channel) + len('channel')+6), file=f)
                print('='*(len(channel) + len('channel')+6), file=f)
                print(file=f)
                
                
                for ts in cont_dict[channel]:
                    print(ts, file=f, end='\n\n')
                    seeds = np.asarray(cont_dict[channel][ts]['seed'])
                    less_pcon = np.asarray(cont_dict[channel][ts]['notspiking_pcon'])
                    geq_pcon = np.asarray(cont_dict[channel][ts]['spiking_pcon'])
                    sign = np.asarray(cont_dict[channel][ts]['significant'])
                    sign_where = sign.astype(bool)
                    sign_seeds = seeds[sign_where]
                    sign_less = less_pcon[sign_where]
                    sign_geq = geq_pcon[sign_where]                
                    less_higher = np.where(sign_less > sign_geq)[0]
                    geq_higher = np.where(sign_less < sign_geq)[0]
                    
    
                    print('Average across all simulations:', file=f)
                    print('Spiking cells: '
                          '{:.2f}%'.format(np.mean(geq_pcon)*100), file=f)
                    print('Not-spiking cells: '
                          '{:.2f}%'.format(np.mean(less_pcon)*100), end='\n\n', file=f)
                    
                    print('Ocurrences of significant difference:', np.sum(sign), 
                          file=f)
                    if np.sum(sign) > 0:
                        print('-- Higher in spiking cells:', len(geq_higher), 
                        end='', file=f)
                        
                        if len(geq_higher)>0:
                            print(' (seeds: ', 
                                  ', '.join(sign_seeds[geq_higher].astype(str)), ')', 
                                  sep='',  file=f)
                        else:
                            print(file=f)
                        
                        print('-- Higher in not-spiking cells:', len(less_higher), 
                        end='', file=f)
                        
                        if len(less_higher)>0:
                            print(' (seeds: ', 
                                  ', '.join(sign_seeds[less_higher].astype(str)), ')', 
                                  sep='',  file=f, end='\n\n')
                        else:
                            print(file=f, end='\n\n')
                    else:
                        print('\n', file=f)
                    
                    print('-'*40, file=f)
                print('='*40, file=f)
                print('='*40, file=f, end='\n\n')
                    
                                  
                            
            
    
    
         
if __name__ == '__main__':    
    simulation_dir = set_simulation_dir('Results_'+os.path.basename(__file__)[:-3])
    Ntrials=100
    seed_list = [i for i in range(Ntrials)]
    task3(simulation_dir, Ntrials=Ntrials, seed_list=seed_list)
    