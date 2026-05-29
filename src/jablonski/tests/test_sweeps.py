import numpy as np

from jablonski import (
    SingletState,
    SpectroscopicSystem,
    initial,
)

from ..sweeps import sweep_spectral_steady_state_emission
from ..transitions import Absorption, Fluorescence
from ..util import ureg


class Model(SpectroscopicSystem):
    low: SingletState = initial(4 * ureg.eV, "singlet", default=10)
    mid: SingletState = initial(5 * ureg.eV, "singlet", default=10)
    high: SingletState = initial(6 * ureg.eV, "singlet", default=1)

    absorption_1 = Absorption(ground=low, excited=high, rate=1e15 / ureg.s)
    absorption_2 = Absorption(ground=low, excited=mid, rate=2e15 / ureg.s)
    absorption_3 = Absorption(ground=mid, excited=high, rate=1.2e15 / ureg.s)
    emission_1 = Fluorescence(ground=low, excited=mid, rate=2e8 / ureg.s)
    emission_2 = Fluorescence(ground=mid, excited=high, rate=0.5e8 / ureg.s)
    emission_3 = Fluorescence(ground=low, excited=high, rate=1e8 / ureg.s)


def test_sweep_spectral_steady_state_emission():
    values = np.linspace(0, 10, 5)
    sweep = sweep_spectral_steady_state_emission(
        Model, excitation_transition=Model.absorption_1, heights=values
    )

    assert np.all(np.asarray(list(sweep.data_vars.keys())) == values)


def test_sweep_emission_spectra():
    values = np.linspace(0, 10, 5)
    sweep = sweep_spectral_steady_state_emission(
        Model, excitation_transition=Model.absorption_1, heights=values
    )

    assert np.all(np.asarray(list(sweep.data_vars.keys())) == values)
