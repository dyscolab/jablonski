import numpy as np
import scipy.constants as constants
import xarray as xr
from poincare import Simulator

from jablonski import (
    SingletState,
    SpectroscopicSystem,
    initial,
)
from jablonski._typing import Pumper, RadiativeDecay

from .._units import ureg
from ..simulation import (
    delta_excitation,
    emission_spectra,
    piecewise,
    pulse_excitation,
    spectral_steady_state_emission,
    spectral_time_resolved_emission,
    steady_state_emission,
    time_resolved_emission,
)
from ..transitions import Absorption, Fluorescence


class Model(SpectroscopicSystem):
    low: SingletState = initial(4 * ureg.eV, "singlet", default=10)
    mid: SingletState = initial(5 * ureg.eV, "singlet", default=10)
    high: SingletState = initial(6 * ureg.eV, "singlet", default=1)

    absorption_1 = Absorption(ground=low, excited=high, rate=1e-15 * ureg.cm**2)
    absorption_2 = Absorption(ground=low, excited=mid, rate=2e-15 * ureg.cm**2)
    absorption_3 = Absorption(ground=mid, excited=high, rate=1.2e-15 * ureg.cm**2)
    emission_1 = Fluorescence(ground=low, excited=mid, rate=2e8 / ureg.s)
    emission_2 = Fluorescence(ground=mid, excited=high, rate=0.5e8 / ureg.s)
    emission_3 = Fluorescence(ground=low, excited=high, rate=1e8 / ureg.s)


def test_piecewise():
    sim = Simulator(Model)
    pulse = pulse_excitation(
        excitation={Model.absorption_1: 1e10 / (ureg.cm**2 * ureg.s)},
        start=1e-8 * ureg.s,
        width=2e-8 * ureg.s,
    )
    result_1 = piecewise(sim, events=pulse, save_at=np.linspace(0, 4e-8, 100) * ureg.s)
    times = result_1.pint.dequantify().indexes["time"].values
    sim_1_times = times[times < 1e-8] * ureg.s
    result_2_1 = sim.solve(save_at=sim_1_times)
    sim_2_times = times[(times >= 1e-8) & (times < 3e-8)] * ureg.s
    result_2_2 = sim.solve(
        save_at=sim_2_times,
        values={
            Model.absorption_1.pump: 1e10 / (ureg.cm**2 * ureg.s),
            Model.high: result_2_1["high"].values[-1],
            Model.mid: result_2_1["mid"].values[-1],
            Model.low: result_2_1["low"].values[-1],
        },
    )
    sim_3_times = times[times >= 3e-8] * ureg.s
    result_2_3 = sim.solve(
        save_at=sim_3_times,
        values={
            Model.high: result_2_2["high"].values[-1],
            Model.mid: result_2_2["mid"].values[-1],
            Model.low: result_2_2["low"].values[-1],
        },
    )
    result_2 = xr.concat(
        [result.pint.dequantify() for result in (result_2_1, result_2_2, result_2_3)],
        dim="time",
    )
    array_1 = np.asarray(result_1.pint.dequantify().to_dataarray())
    array_2 = np.asarray(result_2.to_dataarray())  # result_2 is already dequantified
    assert np.all(array_1 - array_2 <= (array_1 + array_2) / 2 * 0.01)


def test_piecewise_with_units():
    class UnitsModel(SpectroscopicSystem):
        low: SingletState = initial(
            4 * ureg.eV, "singlet", default=10 * ureg.mol / ureg.L
        )
        mid: SingletState = initial(
            5 * ureg.eV, "singlet", default=10 * ureg.mol / ureg.L
        )
        high: SingletState = initial(
            6 * ureg.eV, "singlet", default=1 * ureg.mol / ureg.L
        )

        absorption_1 = Absorption(ground=low, excited=high, rate=1e-15 * ureg.cm**2)
        absorption_2 = Absorption(ground=low, excited=mid, rate=2e-15 * ureg.cm**2)
        absorption_3 = Absorption(ground=mid, excited=high, rate=1.2e-15 * ureg.cm**2)
        emission_1 = Fluorescence(ground=low, excited=mid, rate=2e8 / ureg.s)
        emission_2 = Fluorescence(ground=mid, excited=high, rate=0.5e8 / ureg.s)
        emission_3 = Fluorescence(ground=low, excited=high, rate=1e8 / ureg.s)

    pulse = pulse_excitation(
        excitation={Model.absorption_1: 1e10 / (ureg.cm**2 * ureg.s)},
        start=1e-8 * ureg.s,
        width=2e-8 * ureg.s,
    )

    sim = Simulator(Model)
    units_sim = Simulator(UnitsModel)

    result_1 = piecewise(sim, events=pulse, save_at=np.linspace(0, 4e-8, 100) * ureg.s)
    result_2 = piecewise(
        units_sim, events=pulse, save_at=np.linspace(0, 4e-8, 100) * ureg.s
    )

    array_1 = np.asarray(result_1.to_dataarray())
    array_2 = np.asarray(result_2.pint.dequantify().to_dataarray())
    assert np.all(array_1 - array_2 <= (array_1 + array_2) / 2 * 0.01)


def test_time_resolved_emission():
    delta = delta_excitation(Model.absorption_3, start=0 * ureg.s, area=1 / ureg.cm**2)

    result = spectral_time_resolved_emission(
        system=Model,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
    )
    assert set([str(emission) for emission in result.data_vars.keys()]) == set(
        [
            "line_" + str(emission)
            for emission in [Model.emission_1, Model.emission_2, Model.emission_3]
        ]
    )
    joined_result = spectral_time_resolved_emission(
        system=Model,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
        join_by_energy=True,
    )
    assert set(joined_result.data_vars.keys()) == set(
        ["1 electron_volt", "2 electron_volt"]
    )

    non_spectral = time_resolved_emission(
        system=Model,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
    )
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )


def test_steady_state_emission():
    result = spectral_steady_state_emission(
        system=Model,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
    )

    assert set([str(emission) for emission in result.data_vars.keys()]) == set(
        [
            "line_" + str(emission)
            for emission in [Model.emission_1, Model.emission_2, Model.emission_3]
        ]
        + ["event"]
    )
    joined_result = spectral_steady_state_emission(
        system=Model,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        join_by_energy=True,
    )
    assert set(joined_result.data_vars.keys()) == set(
        ["1 electron_volt", "2 electron_volt"]
    )
    non_spectral = steady_state_emission(
        system=Model,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
    )
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )


def test_emission_spectra():
    result = emission_spectra(
        system=Model,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
    )
    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = np.array(
        [
            (c * h / radiative.energy_difference).to(ureg.nm).magnitude
            for radiative in Model._yield(RadiativeDecay)
        ]
    )
    print(result.indexes["wavelenght"])
    print(wavelenghts)
    assert set(result.indexes["wavelenght"].values) == set(wavelenghts)


def test_absorption_spectra():
    result = emission_spectra(
        system=Model,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
    )
    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = np.array(
        [
            (c * h / radiative.energy_difference).to(ureg.nm).magnitude
            for radiative in Model._yield(Pumper)
        ]
    )
    assert set(result.indexes["wavelenght"].values) == set(wavelenghts)
