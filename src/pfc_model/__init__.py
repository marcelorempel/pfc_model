"""
pfc_model
=========

A package that implements a detailed computational model of prefrontal 
cortex.

Subpackage
----------
replication_hass
    Set of tasks to replicate the results of Hass et al. (2016).
    
analysis
    A toolbox for network analysis.

Utilities
---------
cortex_setup
    Set cortex network inside Brian2, define monitors and stimuli,
    run simulations.
"""

from importlib.metadata import version

__version__ = version(__name__)

from .cortex_setup import *
from ._auxiliary import *
from ._basics_setup import group_sets, membranetuple
import numpy as np
from matplotlib import pyplot as plt
import brian2 as br2


