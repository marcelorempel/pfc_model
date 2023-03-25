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

PC = cortex.neuron_idcs([('PC', 0)])

cortex.set_poisson_stimuli('poisson', 20, ['AMPA', 'NMDA'], 
                        PC, 0.3, 2, 1000, 2000, 1, 0)

cortex.set_neuron_monitors('V', 'V', ('ALL', 0))
cortex.run(3000)

from pfc_model.analysis import *
raster_plot(cortex)
