from pfc_model import *

seed = 2

  
constant_stimuli = [
                    [('PC', 0), 250],
                    [('IN', 0), 200]
                    ]

cortex=Cortex.setup(n_cells=1000, n_stripes=1, 
                    constant_stimuli=constant_stimuli, method='rk4',
                    dt=0.05,seed=seed,            
                    )

cortex.run(3000)

from pfc_model.analysis import *
raster_plot(cortex)
