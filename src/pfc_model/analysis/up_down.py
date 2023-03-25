from hmmlearn import hmm
import numpy as np

__all__ = ['get_hidden_UD', 'get_UD_plots', 'get_updown_intervals', 
         'set_updown_time', 'separateUD']

def get_hidden_UD(var):
    """Get UP and DOWN states evolution through a Poisson Hidden Markov
    Model.
    
    Parameters
    ----------
    var: array_like
        Variable used to define UP and DOWN states (e.g. populational
        mean rate)
    
    Returns
    -------
    out: array
        States (1 represents UP and 0 represents DOWN).
    """
    
    var = np.asarray(var)
    scores = list()
    models = list()
    n_states = 2
    for n_components in range(1,n_states+1):
        for idx in range(10):  
            model = hmm.PoissonHMM(n_components=n_components, random_state=idx,
                                   n_iter=10)
            model.fit(var[:, None])
            models.append(model)
            scores.append(model.score(var[:, None]))
           
    model = models[np.argmax(scores)]
    states = model.predict(var[:, None])
    
    if model.lambdas_[1,0]<model.lambdas_[0,0]:
        states = 1 - states
    
    return states

def get_UD_plots(states, t0, t1, dt, down_value=0, up_value=1):
    """Get UP-DOWN plots. The values corresponding to UP and DOWN states
    can be redifined. The resulting plots can be added to other figures.
    
    Parameters
    ----------
    states: array_like
        UP and DOWN states sequence.
    t0: int or float
        Start time (in ms).
    t1: int or float
        Stop time (in ms).
    dt: int or float
        Time steps (in ms).
    down_value: int or float, optional
        Value correponding to DOWN states. If not given, it defaults
        to 0.
    up_value: int or float, optional
        Value correponding to UP states. If not given, it defaults
        to 1.
    
    Returns
    -------
    This function returns a 2-tuple.
    out1: array
        Array of time instantes corresponding to UP-DOWN states.
    out2: array
        Array of UP-DOWN state sequence.
    """
    
    states = np.asarray(states)
    new_states = states.copy()
    new_states = new_states.astype(float)
    t = np.arange(t0, t1, dt)
    
    new_states[new_states==0] = down_value
    new_states[new_states==1] = up_value
    
    return t, new_states

def get_updown_intervals(tarr, states, min_interval=0, min_t=0):
    start=0
    stop=0
    up = []
    down=[]
    
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            stop = tarr[i]
            if stop-start >= min_interval and start>=min_t:
                if states[i-1]==1:
                    up.append([start, stop])
                else:
                    down.append([start, stop])
            start=stop
    stop=tarr[-1]+tarr[-1]-tarr[-2]
    if stop-start >= min_interval and start>=min_t:
        if states[i-1]==1:
            up.append([start, stop])
        else:
            down.append([start, stop])
    return up, down

def set_updown_time(tarr, intervals):
    """Set a bool array based on a time array indicating the periods
    when a requested state happens.
    
    Parameters
    ----------
    tarr: array_like
        Test time array (values in ms).
    intervals: list[tuple]
        List of 2d-tuples representing the intervals when the requested
        state happens. Each 2d-tuple contains the start and stop time
        values (in ms) for each interval.
    
    Returns
    -------
    out: array
        Bool array indicating which instants in tarr fall inside one
        of the periods defined  in intervals. Each value corresponds
        to the instant in the same position in tarr. True indicates
        that the corresponding instant is contained in one of the
        intervals.
    """
    
    tarr = np.asarray(tarr)
    updown = np.asarray([False for i in range(len(tarr))])
    for start, stop in intervals:
        updown[np.where((tarr>=start)&(tarr<stop))[0]]=True
    return updown

def separateUD(state):
    """Get indices of an UP-DOWN state array corresponding to UP and 
    DOWN periods.
    
    Parameters
    ----------
    state: array or list
        Sequence of UP-DOWN states. UP must be indicated as 1 and DOWN
        as 0.
    
    Returns
    -------
    This function returns a 2-tuple.
    out1: list[list]
        List of DOWN periods. Each period is represented by a list
        containing indices of the period in state sequence.
    out2: list[list]
        List of UP periods. Each period is represented by a list
        containing indices of the period in state sequence.
    """
    
    last = state[0]
    temp_i = 0
    state_up = []
    state_down = []
    row = 0
    for i in range(1,len(state),1):
        if state[i]==last:           
            if row==0:
                if state[i] == 0:
                    state_down.append([temp_i, i])
                else:
                    state_up.append([temp_i, i])
            else:
                if state[i] == 0:
                    state_down[-1].append(i)
                else:
                    state_up[-1].append(i)
            row+=1
        else:
            row=0
        temp_i = i
        last=state[i]
    return state_down, state_up
