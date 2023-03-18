import numpy as np
from scipy.integrate import quad
from scipy.optimize import fmin, fsolve
import xarray as xr
from ._basics_setup import membranetuple

__all__ = ['set_syn_params', 'redistribute', 'set_membr_params',
           'get_iref', 'setcon_commneigh', 'setcon_standard']
           
def set_syn_params(basics, gmax_fac, delay_fac, group_tgt, group_src, 
                   syn_idc, target_source, syn_params, syn_pairs, 
                   n_syn_current): 
    
    syntypes_current = str(basics.syn.kinds.loc[group_tgt, group_src].values)
    synchannels_current = basics.syn.channels.kinds_to_names[syntypes_current]
    
    bcs_gmax = basics.syn.spiking.params.gmax
    bcs_delay = basics.syn.delay
    gmax = float(bcs_gmax['mean'].loc[group_tgt, group_src])*gmax_fac
    gmax_sigma = float(bcs_gmax['sigma'].loc[group_tgt, group_src])*gmax_fac
    gmax_min = float(bcs_gmax['min'].loc[group_tgt, group_src])
    gmax_max = float(bcs_gmax['max'].loc[group_tgt, group_src])
    
    delay = float(bcs_delay.delay_mean.loc[group_tgt, group_src])+delay_fac
    delay_sigma = (float(bcs_delay.delay_sigma.loc[group_tgt, group_src])
                   +delay_fac)
    delay_min = float(bcs_delay.delay_min.loc[group_tgt, group_src])
    delay_max = float(bcs_delay.delay_max.loc[group_tgt, group_src])
    
    stsp_set =str(basics.syn.stsp.distr.loc[group_tgt, group_src].values)          
    
    stsp_sets = basics.syn.stsp.sets
    stsp_where = np.isfinite(np.array(stsp_sets
                                      .loc[dict(set=stsp_set)].values))
    stsp_kinds = np.asarray(stsp_sets.coords['kind'].values)[stsp_where]
    stsp_values = np.asarray(
        stsp_sets.loc[dict(set=stsp_set)].values)[stsp_where]

    new_delay = _set_syn_delay(syn_idc, delay, delay_sigma, 
                               delay_min, delay_max)

    for channel in synchannels_current:
        new_channel = xr.DataArray(
            [[basics.syn.channels.names.index(channel)]*len(syn_idc)],
            coords=[['channel'], syn_idc], 
            dims=['par', 'syn_index'])     
        
        new_stsp_params = _set_syn_stsp(syn_idc, stsp_kinds, stsp_values, 
                                       basics.syn.stsp.kinds)
        new_gmax = _set_syn_gmax(syn_idc, gmax, gmax_sigma, gmax_min, gmax_max)
        new_pfail = _set_syn_pfail(
            syn_idc, basics.syn.spiking.params.pfail.loc[group_tgt, group_src])
        
        new_spike_params = np.concatenate((new_gmax, new_pfail), axis=0)
        
        syn_params['channel'] = np.concatenate(
            (syn_params['channel'], new_channel), axis=1)       
        syn_params['spiking'] = np.concatenate(
            (syn_params['spiking'], new_spike_params), axis=1)     
        syn_params['STSP_params'] = np.concatenate(
            (syn_params['STSP_params'], new_stsp_params), axis=1)
        syn_params['delay'] = np.concatenate(
            (syn_params['delay'], new_delay), axis=1)
        
        syn_pairs = np.concatenate((syn_pairs, target_source), axis=1)
        n_syn_current += len(syn_idc)        

    return syn_params,  syn_pairs, n_syn_current 

def redistribute(basics, group_distr, membr_params, stripe):
    
    bcs_names = basics.struct.groups.names
    I0 = 500

    set_current = []
    for group in ['IN_L_L23', 'IN_L_d_L23']:         
        set_current.extend(
            group_distr[stripe][bcs_names.index(group)])

    set_L = []
    set_L_d = []
    for cell in set_current:
        memb = membranetuple(*membr_params[dict(cell_index=cell)].to_numpy())
        t_lat_adex = _latency_adex(memb, I0)
        t_lat_lif =  _latency_lif(memb, I0)
        
        if t_lat_adex-t_lat_lif > 0:
            set_L_d.append(cell)
        else:
            set_L.append(cell)
    
    
    group_distr[stripe][bcs_names.index('IN_L_L23')] = set_L
    group_distr[stripe][bcs_names.index('IN_L_d_L23')] = set_L_d
            
    set_current = []
    for group in ['IN_L_L5', 'IN_L_d_L5']:         
            set_current.extend(group_distr[stripe][bcs_names.index(group)])

    set_L = []
    set_L_d = []
    for cell in set_current:
        memb = membranetuple(
            *membr_params.loc[dict(cell_index=cell)].to_numpy())         
        t_lat_adex = _latency_adex(memb, I0)
        t_lat_lif =  _latency_lif(memb, I0)
        
        if t_lat_adex-t_lat_lif > 0:
            set_L_d.append(cell)
        else:
            set_L.append(cell)

    group_distr[stripe][bcs_names.index('IN_L_L5')] = set_L
    group_distr[stripe][bcs_names.index('IN_L_d_L5')] = set_L_d
    
    i_range = np.arange(25,301,25)
    adapt_cut = 1.5834
    
    set_current = []    
    for group in ['IN_CL_L23', 'IN_CL_AC_L23']:
        set_current.extend(group_distr[stripe][bcs_names.index(group)])
    
    set_CL = []
    set_CL_AC = []
    for cell in set_current:  
        f_trans = np.zeros(len(i_range))
        f_stat  = np.zeros(len(i_range))            
        for i_idc in range(len(i_range)):
            memb_par = membranetuple(
                *membr_params.loc[dict(cell_index=cell)].to_numpy())
            f_trans[i_idc] = _get_inst_rate(memb_par, i_range[i_idc],
                                           0, memb_par.E_L)
            f_stat[i_idc]  = _get_static_rate(memb_par,i_range[i_idc])
                    
        idc_valid = np.where((f_stat>0))[0]  
        
        if np.median(f_trans[idc_valid] / f_stat[idc_valid])>adapt_cut:
            set_CL_AC.append(cell)
        else:
            set_CL.append(cell)
      
    group_distr[stripe][bcs_names.index('IN_CL_L23')] = set_CL
    group_distr[stripe][bcs_names.index('IN_CL_AC_L23')] = set_CL_AC
    
    set_current = []    
    for group in ['IN_CL_L5', 'IN_CL_AC_L5']:
        set_current.extend(group_distr[stripe][bcs_names.index(group)])
    
    set_CL = []
    set_CL_AC = []
    for cell in set_current:  
        f_trans = np.zeros(len(i_range))
        f_stat  = np.zeros(len(i_range))            
        for i_idc in range(len(i_range)):
            memb_par = membranetuple(
                *membr_params.loc[dict(cell_index=cell)].to_numpy())
            f_trans[i_idc] = _get_inst_rate(
                memb_par,i_range[i_idc],0,memb_par.E_L)
            f_stat[i_idc]  = _get_static_rate(memb_par,i_range[i_idc])
                    
        idc_valid = np.where((f_stat>0))[0]  
        if np.median(f_trans[idc_valid]/f_stat[idc_valid])>adapt_cut:
            set_CL_AC.append(cell)
        else:
            set_CL.append(cell)
      
    group_distr[stripe][bcs_names.index('IN_CL_L5')] = set_CL
    group_distr[stripe][bcs_names.index('IN_CL_AC_L5')] = set_CL_AC
    
    return group_distr


def set_membr_params(membr_params, set_new, set_current, multi_param_transf, 
                     multi_lambda_, multi_minparam_inv, basics, group):
    
    def inv_transform(param_transf, lambda_, minparam_inv):
                
        if lambda_>0:
            if minparam_inv<0:
                param_inv = param_transf**(1/lambda_) + 1.1*minparam_inv
            else:
                param_inv = param_transf**(1/lambda_)
        
        else:
            if minparam_inv<0:
                param_inv = np.exp(param_transf) + 1.1*minparam_inv
            else:
                param_inv = np.exp(param_transf)

        return param_inv

    for par in basics.membr.names:
        param_transf = multi_param_transf.loc[dict(par=par)]
        lambda_ = multi_lambda_.loc[dict(par=par)]
        minparam_inv = multi_minparam_inv.loc[dict(par=par)]
        
        param_inv = inv_transform(param_transf, lambda_, minparam_inv)                  
        membr_params.loc[dict(par=par, cell_index=set_new)] = param_inv

    membr_params.loc['C',set_new] =  (membr_params.loc['C',set_new] 
                                      * membr_params.loc['g_L',set_new])
                                    
    set_outer = np.asarray([])
    #AQUI_2
    for par in basics.membr.names:
        out_min = (membr_params.loc[dict(par=par, cell_index=set_current)]
                    < basics.membr.min.loc[dict(par=par, group=group)])
        out_max = (membr_params.loc[dict(par=par,cell_index=set_current)] 
                    > basics.membr.max.loc[dict(par=par, group=group)])
        where_minmax = np.where(out_min | out_max)[0]
        set_outer = np.concatenate((set_outer, where_minmax))
    
    out_V = (membr_params.loc['V_r',set_current] 
             >= membr_params.loc['V_T',set_current])
    where_V = np.where(out_V)[0]   
    
    taum = (membr_params.loc['C',set_current]
            /membr_params.loc['g_L',set_current])

    out_taum_min = (taum < basics.membr.tau_m_min.loc[group])
    out_taum_max = (taum > basics.membr.tau_m_max.loc[group])    
    where_taum = np.where(out_taum_min | out_taum_max)[0]
    
    where_tauw = np.where((membr_params.loc['tau_w', set_current] <= taum))[0]
    where_nan = np.argwhere(np.sum(np.isnan(
        membr_params.loc[dict(cell_index=set_current)]), axis=0)
        .to_numpy())[:,0]
   
    set_outer = np.concatenate(
        (set_outer,where_V, where_taum, where_tauw, where_nan))
    set_new = set_current[np.unique(set_outer).astype(int)]  

    return membr_params, set_new

def _setcon(n_syn, target_arr, source_arr, pcon):
    
    n_target, n_source = len(target_arr), len(source_arr)
    conn_idc = np.arange(n_target*n_source)
    np.random.shuffle(conn_idc)
    conn_idc=conn_idc[:int(np.round(pcon*n_target*n_source, 0))]
    syn_idc = np.arange(int(np.round(pcon*n_target*n_source, 0))) + n_syn
   
    return _get_targetsource(conn_idc, n_source), syn_idc

def _get_targetsource(conn_idc, n_source):
    return np.asarray([conn_idc//n_source,conn_idc%n_source]).astype(int)


def setcon_standard(n_syn, target_arr, source_arr, pcon): 
    target_arr = np.asarray(target_arr)
    source_arr = np.asarray(source_arr)
    target_source, syn_idc = _setcon(n_syn, target_arr, source_arr, pcon)
    
    return (np.array([target_arr[target_source[0,:]], 
                      source_arr[target_source[1,:]]]), 
            syn_idc.astype(int))


def setcon_commneigh(n_syn, cells_arr, pcon, p_selfcon):
     
    cells_arr = np.array(cells_arr)
    n_cells = len(cells_arr)
    target_source, syn_idc = _setcon(n_syn, cells_arr, cells_arr, pcon)    
    neigh_mat = get_commonneigh_recur(target_source, n_cells)
    slope = 20*3.9991/n_cells
    
    (n_neighbours, n_neigh_connections,
     all_neigh_count, connected_neigh_count)  = _p_calc_recur(
         target_source, neigh_mat, pcon, p_selfcon, slope)

    connections= _get_connections(target_source, n_cells)
         
    curr_n_diag = int(round(p_selfcon*n_cells))
    
    pairs_select = np.array([[],[]])
    
    for n in [n for n in range(len(n_neighbours)) 
              if n_neigh_connections[n] > 0]: # para cada número de conexões
        
        curr_all_pairs = np.array(np.where(neigh_mat == n_neighbours[n]))
        curr_pairs_disconnected = _get_intersect(
            connections['pairs_disconnected'], curr_all_pairs, n_cells)
        
        curr_pairs_disconnected_tril = _get_tril(curr_pairs_disconnected)
        curr_pairs_connected_recur = _get_intersect(
            connections['pairs_connected_recur'], curr_all_pairs, n_cells)
        
        curr_pairs_connected_recur_tril = _get_tril(curr_pairs_connected_recur)
 
        curr_pairs_connected_uni = _get_intersect(
            connections['pairs_connected_uni'], curr_all_pairs, n_cells)
        
        curr_n_recur = np.floor(n_neigh_connections[n]*p_selfcon/2).astype(int)
        curr_n_uni = n_neigh_connections[n] - 2*curr_n_recur 
        
        if curr_pairs_connected_recur_tril.shape[1] >= curr_n_recur:
            idc_curr = np.arange(curr_pairs_connected_recur_tril.shape[1])
            np.random.shuffle(idc_curr)        
            idc_select = np.asarray(idc_curr[:curr_n_recur])
            pairs_select = np.concatenate(
                (pairs_select, 
                 _get_bidirect(
                     curr_pairs_connected_recur_tril[:, idc_select])), 
                 axis=1)           
        else:
            pairs_select = np.concatenate(
                (pairs_select, curr_pairs_connected_recur), axis=1)
                    
            n_new_idc = curr_n_recur - curr_pairs_connected_recur_tril.shape[1] 
            
            idc_new = np.arange(curr_pairs_disconnected_tril.shape[1])
            np.random.shuffle(idc_new)
            idc_select = np.asarray(idc_new[:n_new_idc])
            pairs_select = np.concatenate(
                (pairs_select, 
                 _get_bidirect(curr_pairs_disconnected_tril[:, idc_select])),
                axis=1)
            curr_pairs_disconnected = _get_diff(
                curr_pairs_disconnected, 
                _get_bidirect(curr_pairs_disconnected_tril[:, idc_select]))
         
        if curr_pairs_connected_uni.shape[1] > curr_n_uni:      
            idc_curr = np.arange(curr_pairs_connected_uni.shape[1])
            np.random.shuffle(idc_curr)
            idc_select = np.asarray(idc_curr[:curr_n_uni])
            pairs_select = np.concatenate(
                (pairs_select, curr_pairs_connected_uni[:, idc_select]),
                axis=1)
        
        else:
            pairs_select = np.concatenate(
                (pairs_select, curr_pairs_connected_uni),
                axis=1)
            
            n_new_idc = curr_n_uni - curr_pairs_connected_uni.shape[1] 

            idc_new = np.arange(curr_pairs_disconnected.shape[1])
            np.random.shuffle(idc_new)

            idc_select = np.asarray(idc_new[:n_new_idc])
            pairs_select = np.concatenate(
                (pairs_select, curr_pairs_disconnected[:, idc_select]),
                axis=1)
        
    curr_diag_connected = connections['diag_connected']
    curr_diag_disconnected = connections['diag_disconnected']
    
    if curr_diag_connected.shape[1] > curr_n_diag:
        idc_old = np.arange(curr_diag_connected.shape[1])
        np.random.shuffle(idc_old)
        idc_select = idc_old[:curr_n_diag]
        pairs_select = np.concatenate(
            (pairs_select, curr_diag_connected[:, idc_select]),
            axis=1)
    else:
        n_new_idc = curr_n_diag-curr_diag_connected.shape[1]
        
        pairs_select = np.concatenate(
            (pairs_select, curr_diag_connected),
            axis=1)
    
        idc_new = np.arange(curr_diag_disconnected.shape[1])
        np.random.shuffle(idc_new)
        idc_select = idc_new[:n_new_idc]

        pairs_select  = np.concatenate(
            (pairs_select, curr_diag_disconnected[:,idc_select]),
            axis=1)
        
    syn_idc = n_syn + np.arange(pairs_select.shape[1]) 
    
    pairs_select = pairs_select.astype(int)

    target_source = np.array([cells_arr[pairs_select[0,:]],
                              cells_arr[pairs_select[1,:]]])
    
    return target_source, syn_idc.astype(int)
                       
def _get_static_rate(memb, I0):
    
    tau_m = memb.C/memb.g_L
    tau_ratio = tau_m/memb.tau_w
    Dist_VT = (tau_ratio 
               * (I0 + memb.g_L*memb.delta_T - memb.g_L * (memb.V_T-memb.E_L)))                 
    w_end = (- memb.g_L * (memb.V_T - memb.E_L) +
             memb.g_L * memb.delta_T + I0 - Dist_VT)
  
    if memb.b != 0:
        w_r = w_end+memb.b
    else:
        w_r = 0    
        
    return _get_rate(memb, I0, w_r, memb.V_r)

def _get_inst_rate(memb, I0, w0, V0):
    return _get_rate(memb, I0, w0, V0)

def _get_rate(memb, I0, w_r, V_r):
    
    def dt_dv_regime1and3(V, I, memb, w_r):
        
        dv_dt = ((1/memb.C)
             * (I - w_r 
                + memb.g_L*memb.delta_T*np.exp((V-memb.V_T) / memb.delta_T)
                - memb.g_L * (V-memb.E_L)))
        dt_dv = 1/dv_dt
        
        return dt_dv

    def dt_dv_regime2(V, I0, memb):
        
        tau_ratio = memb.C / (memb.tau_w*memb.g_L)
        
        k0 = (tau_ratio-1)*memb.g_L
        k1 = (1-tau_ratio)*I0-k0*memb.E_L
        k2 = (1-tau_ratio)*memb.g_L*memb.delta_T
        
        dv_dt = ((1/memb.C) 
             * (I0 - (k0*V + k1 + k2*np.exp((V-memb.V_T) / memb.delta_T)) 
                + memb.g_L*memb.delta_T*np.exp((V-memb.V_T) / memb.delta_T) 
                - memb.g_L * (V-memb.E_L)))
        dt_dv = 1/dv_dt
        
        return dt_dv

    def get_w_nullbound_inters(memb, I, w0, w_ref, signal):

        
        if w0 > w_ref:
            V_nullbound_inters = [None, None]
            tau_ratio = memb.C / (memb.g_L*memb.tau_w)
         
            nullcl_bound = lambda V: ((1+signal*tau_ratio) 
                                      * (I - memb.g_L * (V-memb.E_L) 
                                         + memb.g_L*memb.delta_T
                                         *np.exp((V-memb.V_T) / memb.delta_T)) 
                                      - w0)
            lo_guess = (memb.E_L 
                        + (I - (w0 / (1 + signal*tau_ratio)))/memb.g_L 
                        - 0.1)
            hi_guess = (memb.E_L + memb.delta_T 
                        + (I - (w0 / (1 + signal*tau_ratio))) / memb.g_L)
            
            for trial in range(1000):         
                if (np.sign(nullcl_bound(hi_guess))
                    *np.sign(nullcl_bound(lo_guess))) == -1:
                    break
                lo_guess -= 1
            else:
                print('Error in numcor')
            
         
            V_nullbound_inters[0] = fsolve(
                nullcl_bound, (lo_guess+hi_guess) / 2)
           
            lo_guess = (memb.E_L + memb.delta_T + 
                        (I - (w0 / (1 + signal*tau_ratio))) / memb.g_L)
            hi_guess = memb.V_up
            
            for trial in range(1000):
                if (np.sign(nullcl_bound(hi_guess))
                        *np.sign(nullcl_bound(lo_guess))) == -1:
                    break
                lo_guess -= 1
            else:
                print('Error')
            
            V_nullbound_inters[1] = fsolve(
                nullcl_bound, (lo_guess+hi_guess) / 2)    
            
        elif w0 == w_ref:
            V_nullbound_inters = [memb.V_T,]      
        else:
            V_nullbound_inters = [] 
        
        return V_nullbound_inters

    #AQUI_5
    tau_m = memb.C/memb.g_L
    tau_ratio = tau_m/memb.tau_w
    Dist_VT = (tau_ratio 
               * (I0 + memb.g_L*memb.delta_T - memb.g_L * (memb.V_T-memb.E_L)))
    w_end = (- memb.g_L * (memb.V_T-memb.E_L) 
             + memb.g_L*memb.delta_T + I0 - Dist_VT)
    
    if (Dist_VT<=0 or tau_ratio>=1 or memb.C<=0 or memb.g_L<=0 
            or memb.tau_w<=0 or memb.delta_T<=0):
        firing_rate=0
    else:
        dist_V_r = (tau_ratio * 
                    (I0 + memb.g_L*memb.delta_T
                     *np.exp((V_r-memb.V_T) / memb.delta_T) 
                     - memb.g_L * (V_r-memb.E_L)))
        wV_V_r = (- memb.g_L * (V_r-memb.E_L) + memb.g_L*memb.delta_T
                  *np.exp((V_r-memb.V_T) / memb.delta_T) + I0)
        w1 = wV_V_r-dist_V_r
        w2 = wV_V_r+dist_V_r

        delta_t1 = 0
        delta_t2 = 0
        if V_r >= memb.V_T:
            if w_r >= wV_V_r:
                w_ref = (- memb.g_L * (memb.V_T-memb.E_L) 
                         + memb.g_L*memb.delta_T + I0 + Dist_VT)
                V_nullbound_inters = get_w_nullbound_inters(
                    memb, I0, w_r, w_ref, 1)            
                if len(V_nullbound_inters) < 2:
                    delta_t1 = quad(
                        lambda x: dt_dv_regime1and3(x, I0, memb, w_r),
                        V_r, memb.V_T)[0]
                    delta_t2 = 0
                else:
                    V_cross_nullbound = np.min(V_nullbound_inters)
                    delta_t1 = quad(
                        lambda x: dt_dv_regime1and3(x, I0, memb, w_r), 
                        V_r, V_cross_nullbound)[0]
                    delta_t2 = quad(
                        lambda x: dt_dv_regime2(x, I0, memb),
                        V_cross_nullbound, memb.V_T)[0]
                
                w_stop = w_end
                V1b = memb.V_T
                
            else:
                V1b = V_r
                w_stop = w_r
                   
        else:
            if w_r < w2 and w_r > w1:
                delta_t2 = quad(
                    lambda x: dt_dv_regime2(x, I0, memb), 
                    V_r, memb.V_T)[0]
                w_stop = w_end
            else:
                if w_r <= w1:
                    signal = -1
                    w_ref = (- memb.g_L * (memb.V_T-memb.E_L) 
                             + memb.g_L*memb.delta_T + I0 - Dist_VT)
                else:
                    signal = 1
                    w_ref = (- memb.g_L * (memb.V_T-memb.E_L) 
                             + memb.g_L*memb.delta_T + I0 + Dist_VT) # w_end
                
                V_nullbound_inters = get_w_nullbound_inters(
                    memb, I0, w_r, w_ref, signal)
                
                if not V_nullbound_inters or len(V_nullbound_inters) == 1:
                    delta_t1 = quad(
                        lambda x: dt_dv_regime1and3(x, I0, memb, w_r), 
                        V_r, memb.V_T)[0]
                    w_stop = w_r
                else:
                    V_bound = np.min(V_nullbound_inters)
                    delta_t1 = quad(
                        lambda x: dt_dv_regime1and3(x, I0, memb, w_r), 
                        V_r, V_bound)[0]
                    delta_t2 = quad(
                        lambda x: dt_dv_regime2(x, I0, memb), 
                        V_bound, memb.V_T)[0]
                    w_stop = w_end
                           
            V1b = memb.V_T

        if V1b >= memb.V_up:
            delta_t3 = 0
        else:
            
            delta_t3= quad(
                lambda x: dt_dv_regime1and3(x, I0, memb, w_stop), 
                V1b, memb.V_up)[0]

        ISI = np.asarray(delta_t1+delta_t2+delta_t3)
        firing_rate = 1000/ISI

        return firing_rate


def _cost_iref(I, memb):
    re = _get_inst_rate(memb, I, 0, memb.V_r)

    if re is None:
        re = 0
    q = (re-200)**2
    
    return q

def get_iref(memb):
    Irheo = memb.g_L * (memb.V_T-memb.E_L) - memb.g_L*memb.delta_T                  
    return fmin(_cost_iref, Irheo+100, disp=False, args=(memb,))[0]

def _latency_adex(memb, I):
    return quad(lambda V: memb.C/(I - memb.g_L * (V-memb.E_L) 
                                  + memb.g_L*memb.delta_T
                                  *np.exp((V-memb.V_T) / memb.delta_T)),
                memb.E_L, memb.V_T/2)[0]
    
def _latency_lif(memb, I):
    return memb.C*np.log(I / (I + memb.g_L * (memb.E_L-memb.V_T/2))) / memb.g_L

def get_commonneigh_recur(ts, n_cell):
    target, source = ts
    conn_mat = np.zeros((n_cell, n_cell))
    conn_mat[target, source] = 1
    conn_mat = conn_mat + conn_mat.transpose()
    conn_mat = conn_mat.astype(bool).astype(int)
    
    comneigh_mat = np.zeros((n_cell, n_cell))
    
    for tgt in range(n_cell):
        for src in range(tgt):
            comneigh_mat[tgt,src] = (sum(conn_mat[tgt]*conn_mat[src]) 
                                     - conn_mat[tgt, src]*conn_mat[tgt, tgt] 
                                     - conn_mat[src, src]*conn_mat[src, tgt])

    comneigh_mat = comneigh_mat + comneigh_mat.transpose()
    
    return comneigh_mat.astype(int)    
    

def commonneighbour_report(target_source, n_cells):
    neigh_mat = get_commonneigh_recur(target_source, n_cells)   
    n_neigh_connected = neigh_mat[target_source[0,:], target_source[1, :]]  
    
    common_neigh = {}
    common_neigh['N'] = np.unique(neigh_mat)
    common_neigh['all'] = np.zeros(len(common_neigh['N']))
    common_neigh['connected'] = np.zeros(len(common_neigh['N']))
    for i in range(len(common_neigh['N'])):
        common_neigh['all'][i] = np.sum(
            (neigh_mat == common_neigh['N'][i]).astype(int)
            )
        common_neigh['connected'][i] = np.sum(
            (n_neigh_connected==common_neigh['N'][i]).astype(int)
            )
    
    common_neigh['ratio'] = common_neigh['connected']/common_neigh['all']

    return common_neigh


def _random_param(n, par_mean, par_std, par_min, par_max, distr_flag):

    if par_std == 0:
        if par_max == 0:
            par = par_mean*np.ones(n)    
        else:
            par = par_min + (par_max-par_min) * np.random.random(size=n)   
    else:  
                         
        if distr_flag == 'normal':
            par = par_mean + par_std* np.random.normal(size=n)
            exc_ind = np.where((par<par_min) | (par>par_max))[0]

            par[exc_ind] =  (par_min + (par_max-par_min)
                             * np.random.random(size=len(exc_ind)))
        elif distr_flag == 'uniform':
            par = par_min + (par_max-par_min)* np.random.random(size=n)
        elif distr_flag == 'log_normal':
            par = np.exp(np.random.normal(size=n)*par_std + par_mean)
            exc_ind = np.where((par < par_min) | (par > par_max))[0]
            par[exc_ind] = (par_min + (par_max-par_min) 
                            * np.random.random(size=len(exc_ind)))
        
    return par


def _set_syn_gmax(new_idc, gmax, gmax_sigma, gmax_min, gmax_max):
       
    syn_params = xr.DataArray(np.zeros((1, len(new_idc))), 
                              coords=[['gmax'], new_idc], 
                              dims=['par', 'syn_index'])

    mean_gmax = np.log(gmax**2/np.sqrt(gmax_sigma**2 + gmax**2))
    std_gmax = np.sqrt(np.log(gmax_sigma**2/gmax**2 + 1))
    
    syn_params.loc['gmax', new_idc] = _random_param(
        len(new_idc), mean_gmax, std_gmax, gmax_min*gmax, gmax_max * gmax,
        'log_normal')
           
    return syn_params

def _set_syn_pfail(new_idc, pfail):
       
    syn_params = xr.DataArray(
        np.zeros((1, len(new_idc))), coords=[['pfail'], new_idc],
        dims=['par', 'syn_index'])
    syn_params.loc['pfail', new_idc]=pfail
    
    return syn_params

def _set_syn_delay(new_idc, delay, delay_sigma, delay_min, delay_max):
   
    syn_params = xr.DataArray(
        np.zeros((1, len(new_idc))),
        coords=[['delay'], new_idc],
        dims=['par', 'syn_index'])
    syn_params.loc['delay', new_idc]= _random_param(
        len(new_idc), delay, delay_sigma, delay_min*delay, delay_max*delay,
        'normal')
    
    return syn_params


def _set_syn_stsp(new_idc, stsp_kinds, stsp_values, stsp_params):
    
    syn_params = xr.DataArray(
        np.zeros((3, len(new_idc))),
        coords=[['U', 'tau_rec', 'tau_fac'], new_idc],
        dims=['par', 'syn_index'])
   
    stsp_idc = np.round(len(new_idc)*stsp_values, 0).astype(int)
    
    while sum(stsp_idc) != len(new_idc):   
        if sum(stsp_idc) > len(new_idc):
            change_idc = np.argmin(
                (len(new_idc)*stsp_values 
                 - np.floor(len(new_idc)*stsp_values)))
            stsp_idc[change_idc] -= 1
        else:
            change_idc = np.argmax(
                (len(new_idc)*stsp_values 
                 - np.floor(len(new_idc)*stsp_values)))
            stsp_idc[change_idc] += 1
    
    stsp_idc_cumm = np.cumsum(stsp_idc).astype(int)
    stsp_idc_limits = (np.concatenate([np.asarray([0]), stsp_idc_cumm]) 
                      + new_idc[0])

    
    for name_idc in range(len(stsp_kinds)):
        
        idc_curr = np.arange(stsp_idc_limits[name_idc], 
                             stsp_idc_limits[name_idc+1])

        syn_params.loc['U', idc_curr] = _random_param(
            len(idc_curr), 
            float(stsp_params['mean'].loc[
                dict(par='U', kind=stsp_kinds[name_idc])].values), 
            float(stsp_params['std'].loc['U'].values),  0,    1, 'normal')   
        
        syn_params.loc['tau_rec', idc_curr] = _random_param(
            len(idc_curr), 
            float(stsp_params['mean'].loc[
                dict(par='tau_rec', kind=stsp_kinds[name_idc])].values),
            float(stsp_params['std'].loc['tau_rec'].values), 0, 1500, 'normal')
        
        syn_params.loc['tau_fac', idc_curr] = _random_param(
            len(idc_curr), 
            float(stsp_params['mean'].loc[
                dict(par='tau_fac', kind=stsp_kinds[name_idc])].values), 
            float(stsp_params['std'].loc['tau_fac'].values), 0, 1500, 'normal')

    return syn_params


def _get_tril(a, output='pairs'):
    if output == 'pairs':
        return a[:, np.where((a[0,:] > a[1,:]))[0]]
    elif output=='indices':
        return np.where((a[0,:] > a[1,:]))[0]


def _get_bidirect(a):
    temp_a = a.copy()
    inverted = a.copy()
    inverted[0,:], inverted[1,:] = temp_a[1,:], temp_a[0,:]
    concatenated = np.concatenate((a, inverted), axis=1)
    return np.unique(concatenated, axis=1)


def _get_connections(target_source, n_cells):
    connections = {}
    
    X = np.zeros((n_cells, n_cells))
    X[target_source[0,:], target_source[1,:]] = 1
    Xnondiag0 = (np.ones((n_cells,n_cells))-np.identity(n_cells))*X
    Xnondiag1 = X + np.identity(n_cells)
    
    Xrecur0 = Xnondiag0 + Xnondiag0.transpose()
    Xrecur1 = Xnondiag1 + Xnondiag1.transpose()
    
    connections['pairs_connected_recur'] = np.array(np.where(Xrecur0 == 2))
    connections['pairs_connected_all'] = np.array(np.where(Xrecur0 >= 1))
    connections['pairs_disconnected'] = np.array(np.where(Xrecur1 == 0))
    conn0 = connections['pairs_connected_recur'][0,:]
    conn1 = connections['pairs_connected_recur'][1,:]
    Xnondiag0[conn0, conn1] = 0
    connections['pairs_connected_uni'] = np.array(np.where(Xnondiag0 == 1))
   
    Xdiag0 = X * np.identity(n_cells)
    Xdiag1 = X + (np.ones((n_cells,n_cells))-np.identity(n_cells))
    connections['diag_connected'] = np.array(np.where(Xdiag0 == 1))
    connections['diag_disconnected'] = np.array(np.where(Xdiag1 == 0))
    
    return connections 


def _get_intersect(a, b, n_cells):
    a_mat = np.zeros((n_cells, n_cells))
    b_mat = a_mat.copy()
    a_mat[a[0,:],a[1,:]] = 1
    b_mat[b[0,:],b[1,:]] = 1
    return np.array(np.where(a_mat*b_mat > 0))    

def _get_diff(a,b):
    
    if a.shape[1] == 0:
        return np.array([[],[]])
    elif b.shape[1] == 0:
        return a
    
    a = np.array(a).astype(int)
    b = np.array(b).astype(int)
    
    n_cells = max(np.max(a), np.max(b))+1
    a_mat = np.zeros((n_cells, n_cells))
    b_mat = a_mat.copy()
    a_mat[a[0,:],a[1,:]] = 1
    b_mat[b[0,:],b[1,:]] = 1
    
    return np.array(np.where(a_mat-b_mat==1))


def _p_calc_recur(target_source, neigh_mat, pcon, p_selfcon, slope):
    
    def p_min(n0, n_neighbours, all_neigh_count, pcon, slope):
        n1 = np.round(min(max(n_neighbours), n0 + 1/slope/pcon))
        
        cond1 = n_neighbours > n0
        cond2 = n_neighbours <= n1
        cond3 = n_neighbours > n1
        n_2 = n_neighbours[np.where(cond1 & cond2)]
        pn_2 = (all_neigh_count[np.where(cond1 & cond2)]
                /np.sum(all_neigh_count))
        pn_3 = all_neigh_count[np.where(cond3)]/np.sum(all_neigh_count)
        
        return abs(np.sum(pn_2 * slope * (n_2-n0)) + np.sum(pn_3) - 1)
    
    n_cell = neigh_mat.shape[0]    
    n_neigh_connected = neigh_mat[target_source[0,:], target_source[1, :]]  
    
    n_neighbours = np.unique(neigh_mat)
    all_neigh_count = np.zeros(len(n_neighbours))
    neigh_factor = np.zeros(len(n_neighbours))
    connected_neigh_count = np.zeros(len(n_neighbours))
    for i in range(len(n_neighbours)):
        all_neigh_count[i] = np.sum(
            (neigh_mat == n_neighbours[i]).astype(int))
        connected_neigh_count[i] = np.sum(
            (n_neigh_connected==n_neighbours[i]).astype(int))

    offset = fmin(lambda n0: p_min(n0, n_neighbours, all_neigh_count,
                                   pcon, slope), 
                  np.max(n_neighbours), disp=False)
     
    n1 = np.floor(min(np.max(n_neighbours), offset + 1 / (slope*pcon)))
    n0 = max(np.min(n_neighbours), np.ceil(offset))
    ind = np.isin(n_neighbours, np.arange(n0, n1+1))
    neigh_factor[ind] = pcon * slope * (n_neighbours[ind]-offset)
    
    if len(ind) > 0 and len(np.where(ind)[0]) > 0:
        neigh_factor[np.where(ind)[0][-1]+1:] = 1
    else:
        neigh_factor[:] = 1
    
    n_original = n_cell**2 * pcon - n_cell*p_selfcon

    n_act = all_neigh_count@neigh_factor
    n_neigh_connections =np.round(
        neigh_factor*n_original*all_neigh_count/n_act).astype(int)
    
    return (n_neighbours.astype(int), n_neigh_connections, all_neigh_count,
            connected_neigh_count)
  