import numpy as np
import xarray as xr

from jablonski import (
    SingletState,
    SpectroscopicSystem,
    initial,
)
from poincare import Simulator

from ..simulation import quantum_yield
from ..helpers import pump_from_laser
from ..sweeps import sweep_quantum_yield, sweep_spectra, sweep_spectral_steady_state
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
    sweep = sweep_spectral_steady_state(
        sim, excitations=[{Model.absorption_1: value} for value in values], kind="emission"
    )
    assert np.all(
        np.asarray([key.magnitude for key in sweep.data_vars.keys()])
        / (ureg.cm**2 * ureg.s)
        == values
    )


def test_sweep_emission_spectra():
    values = np.linspace(0, 10e20, 5) / (ureg.cm**2 * ureg.s)
    sweep = sweep_spectra(
        sim, excitations=[{Model.absorption_1: value} for value in values], kind="emission"
    )

    assert np.all(
        np.asarray([key.magnitude for key in sweep.data_vars.keys()])
        / (ureg.cm**2 * ureg.s)
        == values
    )


def test_sweep_quantum_yield():
    values = np.linspace(1e10, 1e20, 5) / (ureg.cm**2 * ureg.s)
    excitations = [{Model.absorption_1: value} for value in values]
    sweep = sweep_quantum_yield(sim, excitations=excitations)

    assert sweep.pint.units == ureg.dimensionless
    assert sweep.coords["excitation"].pint.units == ureg.Unit("1 / (cm**2 * s)")
    np.testing.assert_allclose(
        sweep.coords["excitation"].pint.dequantify().values,
        values.magnitude,
    )

    expected_values = [
        quantum_yield(sim, excitation=exc).pint.dequantify().values.item()
        for exc in excitations
    ]
    np.testing.assert_allclose(
        sweep.pint.dequantify().values,
        expected_values,
    )
    custom_keys = ["a", "b", "c", "d", "e"]
    sweep_custom = sweep_quantum_yield(sim, excitations=excitations, keys=custom_keys)
    assert isinstance(sweep_custom, xr.DataArray)
    assert sweep_custom.dims == ("excitation",)
    assert list(sweep_custom.coords["excitation"].values) == custom_keys
    np.testing.assert_allclose(
        sweep_custom.pint.dequantify().values,
        expected_values,
    )

    powers = np.logspace(-3, 1, 50) * ureg.W
    excitations = [
        pump_from_laser(
            system=Model,
            power=power,
            wavelength=500 * ureg.nm,
            width=1 * ureg.um,
            linewidth=500 * ureg.nm,
        )
        for power in powers
    ]
    sweep_powers = sweep_quantum_yield(sim, excitations=excitations, keys=powers)
    assert sweep_powers.coords["excitation"].pint.units == powers.units
    np.testing.assert_allclose(
        sweep_powers.coords["excitation"].pint.dequantify().values,
        powers.magnitude,
    )

