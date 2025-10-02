import sys
sys.path.append("/Users/grecco/Documents/code/jablonski/src")

import numpy as np
import pint

from jablonski import SpectroscopicSystem, transitions, util, initial
from jablonski.states import SingletState, TripletState

ureg = pint.get_application_registry()

class TwoStateSystem2(SpectroscopicSystem):

    high3: TripletState = initial(energy=3 * ureg.eV, default=1, spin_multiplicity="triplet")
    high1: SingletState = initial(energy=3 * ureg.eV, default=1, spin_multiplicity="singlet")
    low: SingletState = initial(energy=0 * ureg.eV, default=0)

    # Ok    
    pump1 = transitions.Absorption(ground=low, excited=high1, rate=1 / ureg.s)

    # Typing Error
    pump = transitions.Absorption(ground=low, excited=high3, rate=1 / ureg.s)

    # ok
    fluo = transitions.Fluorescence(ground=low, excited=high1, rate=1 / ureg.s)
    phos = transitions.Phosphorescence(ground=low, excited=high3, rate=1 / ureg.s)

    # ok
    fluo_nok = transitions.Phosphorescence(ground=low, excited=high1, rate=1 / ureg.s)
    phos_nok = transitions.Fluorescence(ground=low, excited=high3, rate=1 / ureg.s)



