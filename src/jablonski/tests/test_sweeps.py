import numpy as np

from jablonski import (
    SingletState,
    SpectroscopicSystem,
    initial,
)
from poincare import Simulator

from ..sweeps import sweep_spectral_steady_state_emission, sweep_emission_spectra
from ..transitions import Absorption, Fluorescence
from ..util import ureg


class Model(SpectroscopicSystem):
    low: SingletState = initial(4 * ureg.eV, "singlet", default=10)
    mid: SingletState = initial(5 * ureg.eV, "singlet", default=10)
    high: SingletState = initial(6 * ureg.eV, "singlet", default=1)

    absorption_1 = Absorption(ground=low, excited=high, rate=1e-15 * ureg.cm**2)
    absorption_2 = Absorption(ground=low, excited=mid, rate=2e15 * ureg.cm**2)
    absorption_3 = Absorption(ground=mid, excited=high, rate=1.2e15 * ureg.cm**2)
    emission_1 = Fluorescence(ground=low, excited=mid, rate=2e8 / ureg.s)
    emission_2 = Fluorescence(ground=mid, excited=high, rate=0.5e8 / ureg.s)
    emission_3 = Fluorescence(ground=low, excited=high, rate=1e8 / ureg.s)

sim = Simulator(Model)

def test_sweep_spectral_steady_state_emission():
    values = np.linspace(0, 1e20, 5) / (ureg.cm**2 * ureg.s)
    sweep = sweep_spectral_steady_state_emission(
        sim, excitations=[{Model.absorption_1: value} for value in values]
    )
    assert np.all(
        np.asarray([key.magnitude for key in sweep.data_vars.keys()])
        / (ureg.cm**2 * ureg.s)
        == values
    )


def test_sweep_emission_spectra():
    values = np.linspace(0, 10e20, 5) / (ureg.cm**2 * ureg.s)
    sweep = sweep_emission_spectra(
        sim, excitations=[{Model.absorption_1: value} for value in values]
    )

    assert np.all(
        np.asarray([key.magnitude for key in sweep.data_vars.keys()])
        / (ureg.cm**2 * ureg.s)
        == values
    )
