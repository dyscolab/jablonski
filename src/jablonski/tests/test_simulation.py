import numpy as np
import scipy.constants as constants
import xarray as xr
import pytest
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
    excitation_emission_matrix,
    excitation_spectra,
    piecewise,
    pulse_excitation,
    quantum_yield,
    spectra,
    spectral_quantum_yield,
    spectral_steady_state,
    spectral_steady_state_emission,
    spectral_time_resolved,
    spectral_time_resolved_emission,
    steady_state,
    steady_state_emission,
    step_excitation,
    time_resolved,
    time_resolved_emission,
    widened_emission_spectra,
    widened_spectra,
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

sim = Simulator(Model)


def test_piecewise():
    pulse = pulse_excitation(
        excitation={Model.absorption_1: 1e10 / (ureg.cm**2 * ureg.s)},
        start=1e-8 * ureg.s,
        width=2e-8 * ureg.s,
    )
    result_1 = piecewise(sim, events=pulse, save_at=np.linspace(0, 4e-8, 100) * ureg.s)
    times = result_1.pint.dequantify().indexes["time"].values
    sim_1_times = times[times <= 1e-8] * ureg.s
    result_2_1 = sim.solve(save_at=sim_1_times)
    sim_2_times = times[(times > 1e-8) & (times <= 3e-8)] * ureg.s
    result_2_2 = sim.with_values(
        {
            Model.absorption_1.pump: 1e10 / (ureg.cm**2 * ureg.s),
            Model.high: result_2_1["high"].values[-1],
            Model.mid: result_2_1["mid"].values[-1],
            Model.low: result_2_1["low"].values[-1],
        }
    ).solve(t_span=(1e-8, 3e-8) * ureg.s, save_at=sim_2_times)
    sim_3_times = times[times > 3e-8] * ureg.s
    result_2_3 = sim.with_values(
        {
            Model.high: result_2_2["high"].values[-1],
            Model.mid: result_2_2["mid"].values[-1],
            Model.low: result_2_2["low"].values[-1],
        }
    ).solve(t_span=(3e-8, times[-1]) * ureg.s, save_at=sim_3_times)
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

    result = spectral_time_resolved(
        sim,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
        kind="emission",
    )
    assert set([str(emission) for emission in result.data_vars.keys()]) == set(
        [
            "emission_" + str(emission)
            for emission in [Model.emission_1, Model.emission_2, Model.emission_3]
        ]
    )
    joined_result = spectral_time_resolved(
        sim,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
        kind="emission",
        join_by_energy=True,
    )
    assert set(joined_result.data_vars.keys()) == set(
        ["1 electron_volt", "2 electron_volt"]
    )

    non_spectral = time_resolved(
        sim,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
        kind="emission",
    )
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )


def test_steady_state_emission():
    result = spectral_steady_state(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="emission",
    )

    assert set([str(emission) for emission in result.data_vars.keys()]) == set(
        [
            "emission_" + str(emission)
            for emission in [Model.emission_1, Model.emission_2, Model.emission_3]
        ]
        + ["event"]
    )
    joined_result = spectral_steady_state(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="emission",
        join_by_energy=True,
    )
    assert set(joined_result.data_vars.keys()) == set(
        ["1 electron_volt", "2 electron_volt"]
    )
    non_spectral = steady_state(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="emission",
    )
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )


def test_emission_spectra():
    result = spectra(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="emission",
    )
    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = np.array(
        [
            (c * h / radiative.energy_difference).to(ureg.nm).magnitude
            for radiative in Model._yield(RadiativeDecay)
        ]
    )
    assert set(result.pint.dequantify().indexes["wavelenght"].values) == set(wavelenghts)
    assert result.pint.units == ureg.Unit("1/s")
    assert result.coords["wavelenght"].pint.units == ureg.nm


def test_time_resolved_absorption():
    delta = delta_excitation(Model.absorption_3, start=0 * ureg.s, area=1 / ureg.cm**2)

    result = spectral_time_resolved(
        sim,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
        kind="absorption",
    )
    assert set([str(absorption) for absorption in result.data_vars.keys()]) == set(
        [
            "absorption_" + str(absorption)
            for absorption in [Model.absorption_1, Model.absorption_2, Model.absorption_3]
        ]
    )
    joined_result = spectral_time_resolved(
        sim,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
        kind="absorption",
        join_by_energy=True,
    )
    assert set(joined_result.data_vars.keys()) == set(
        ["1 electron_volt", "2 electron_volt"]
    )

    non_spectral = time_resolved(
        sim,
        excitation=delta,
        save_at=np.linspace(0, 5, 20) * ureg.s,
        kind="absorption",
    )
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )


def test_steady_state_absorption():
    result = spectral_steady_state(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="absorption",
    )

    assert set([str(absorption) for absorption in result.data_vars.keys()]) == set(
        [
            "absorption_" + str(absorption)
            for absorption in [Model.absorption_1, Model.absorption_2, Model.absorption_3]
        ]
        + ["event"]
    )
    joined_result = spectral_steady_state(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="absorption",
        join_by_energy=True,
    )
    assert set(joined_result.data_vars.keys()) == set(
        ["1 electron_volt", "2 electron_volt"]
    )
    non_spectral = steady_state(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="absorption",
    )
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )


def test_absorption_spectra():
    result = spectra(
        sim,
        excitation={Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)},
        kind="absorption",
    )
    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = np.array(
        [
            (c * h / radiative.energy_difference).to(ureg.nm).magnitude
            for radiative in Model._yield(Pumper)
        ]
    )
    assert set(result.pint.dequantify().indexes["wavelenght"].values) == set(wavelenghts)
    assert result.pint.units == ureg.Unit("1/s")
    assert result.coords["wavelenght"].pint.units == ureg.nm


def test_spectral_quantum_yield():
    exc = {Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)}
    result = spectral_quantum_yield(sim, excitation=exc)
    qy = quantum_yield(sim, excitation=exc)

    assert result.pint.units == ureg.dimensionless
    assert result.coords["wavelenght"].pint.units == ureg.nm
    assert result.dims == ("wavelenght",)

    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = np.array(
        [
            (c * h / radiative.energy_difference).to(ureg.nm).magnitude
            for radiative in Model._yield(Fluorescence)
        ]
    )
    assert set(result.pint.dequantify().indexes["wavelenght"].values) == set(wavelenghts)
    np.testing.assert_allclose(result.sum().values.item(), qy.values.item())

    # Test kind as list
    result_fl = spectral_quantum_yield(sim, excitation=exc, kind=["fluorescence"])
    np.testing.assert_allclose(result_fl.values, result.values)

    with pytest.raises(ValueError):
        spectral_quantum_yield(sim, excitation=exc, kind="absorption")


def test_excitations():
    height = 1e10 / (ureg.cm**2 * ureg.s)
    step = step_excitation({Model.absorption_1: height}, start=1e-8 * ureg.s)
    assert step == {1e-8 * ureg.s: {Model.absorption_1.pump: height}}

    pulse = pulse_excitation(
        {Model.absorption_1: height}, start=1e-8 * ureg.s, width=2e-8 * ureg.s
    )
    (start, on), (end, off) = pulse.items()
    assert start == 1e-8 * ureg.s
    np.testing.assert_allclose(end.to(ureg.s).magnitude, 3e-8)
    assert on == {Model.absorption_1.pump: height}
    assert off == {Model.absorption_1.pump: None}

    delta = delta_excitation(Model.absorption_1, area=1 / ureg.cm**2)
    (start, on), (end, off) = delta.items()
    assert start == 0 * ureg.s
    assert off == {Model.absorption_1.pump: None}
    np.testing.assert_allclose(
        (on[Model.absorption_1.pump] * (end - start)).to(1 / ureg.cm**2).magnitude, 1
    )


def test_step_excitation_piecewise():
    height = 1e10 / (ureg.cm**2 * ureg.s)
    step = step_excitation({Model.absorption_1: height}, start=1e-8 * ureg.s)
    # A pulse that ends at the last saved time is equivalent to a step.
    pulse = pulse_excitation(
        {Model.absorption_1: height}, start=1e-8 * ureg.s, width=3e-8 * ureg.s
    )
    save_at = np.linspace(0, 4e-8, 100) * ureg.s
    result_1 = piecewise(sim, events=step, save_at=save_at)
    result_2 = piecewise(sim, events=pulse, save_at=save_at)
    np.testing.assert_allclose(
        np.asarray(result_1.pint.dequantify().to_dataarray()),
        np.asarray(result_2.pint.dequantify().to_dataarray()),
    )


def test_quantum_yield():
    exc = {Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)}
    qy = quantum_yield(sim, excitation=exc)
    assert qy.name == "quantum_yield"
    assert qy.pint.units == ureg.dimensionless

    emission = steady_state(sim, excitation=exc, kind="emission")["emission"]
    absorption = steady_state(sim, excitation=exc, kind="absorption")["absorption"]
    np.testing.assert_allclose(
        qy.values, (emission / absorption).pint.to(ureg.dimensionless).values
    )

    # Test kind as list
    qy_fl = quantum_yield(sim, excitation=exc, kind=["fluorescence"])
    np.testing.assert_allclose(qy_fl.values, qy.values)

    with pytest.raises(ValueError):
        quantum_yield(sim, excitation=exc, kind="absorption")


def test_excitation_emission_matrix():
    height = 5e10 / (ureg.cm**2 * ureg.s)
    result = excitation_emission_matrix(sim, height=height)

    assert set(result.data_vars.keys()) == set(
        str(pumper) for pumper in Model._yield(Pumper)
    )
    assert result.coords["wavelenght"].pint.units == ureg.nm
    for pumper in Model._yield(Pumper):
        expected = spectra(sim, excitation={pumper: height}, kind="emission")
        np.testing.assert_allclose(
            result[str(pumper)].pint.dequantify().values,
            expected.pint.dequantify().values,
        )
        assert result[str(pumper)].pint.units == expected.pint.units


def test_excitation_spectra():
    height = 5e10 / (ureg.cm**2 * ureg.s)
    matrix = excitation_emission_matrix(sim, height=height)

    for emission in matrix.pint.dequantify().indexes["wavelenght"].values:
        result = excitation_spectra(sim, emission=emission, height=height)
        assert result.dims == ("pumper",)
        assert set(result.coords["pumper"].values) == set(matrix.data_vars.keys())
        assert result.pint.units == ureg.Unit("1/s")
        for pumper in result.coords["pumper"].values:
            np.testing.assert_allclose(
                result.sel(pumper=pumper).pint.dequantify().values,
                matrix[pumper].sel(wavelenght=emission * ureg.nm).pint.dequantify().values,
            )

        result_quantity = excitation_spectra(
            sim, emission=emission * ureg.nm, height=height
        )
        np.testing.assert_allclose(
            result_quantity.pint.dequantify().values,
            result.pint.dequantify().values,
        )

    # emission is matched to the nearest line, e.g. after unit round-off
    for emission in matrix.pint.dequantify().indexes["wavelenght"].values:
        result = excitation_spectra(sim, emission=emission, height=height)
        for shifted in (
            (emission * ureg.nm).to(ureg.um),
            emission + 1,
            emission - 1,
        ):
            result_shifted = excitation_spectra(sim, emission=shifted, height=height)
            np.testing.assert_allclose(
                result_shifted.pint.dequantify().values,
                result.pint.dequantify().values,
            )

    # unit given as a string
    matrix_um = excitation_emission_matrix(sim, height=height, unit="um")
    for emission in matrix_um.pint.dequantify().indexes["wavelenght"].values:
        result = excitation_spectra(sim, emission=emission, height=height, unit="um")
        for pumper in result.coords["pumper"].values:
            np.testing.assert_allclose(
                result.sel(pumper=pumper).pint.dequantify().values,
                matrix_um[pumper].sel(wavelenght=emission * ureg.um).pint.dequantify().values,
            )


def test_widened_spectra():
    exc = {Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)}
    samples = np.linspace(500, 1400, 9001) * ureg.nm
    result = widened_spectra(sim, excitation=exc, samples=samples, width=5)
    lines = spectra(sim, excitation=exc)

    assert result.name == "spectrum"
    assert result.dims == ("wavelenght",)
    assert result.shape == (9001,)
    assert result.pint.units == lines.pint.units
    assert result.coords["wavelenght"].pint.units == ureg.nm

    # Lines are far apart compared to the width, so each peak has the line height.
    widened = result.pint.dequantify()
    for wavelenght, height in zip(
        lines.pint.dequantify().indexes["wavelenght"].values,
        lines.pint.dequantify().values,
    ):
        np.testing.assert_allclose(
            widened.sel(wavelenght=wavelenght, method="nearest").values,
            height,
            rtol=1e-3,
        )
    np.testing.assert_allclose(
        widened.sel(wavelenght=800, method="nearest").values, 0, atol=1e-12
    )


def test_backward_compatibility_wrappers():
    exc = {Model.absorption_1: 5e10 / (ureg.cm**2 * ureg.s)}
    delta = delta_excitation(Model.absorption_3, start=0 * ureg.s, area=1 / ureg.cm**2)
    save_at = np.linspace(0, 5, 20) * ureg.s

    xr.testing.assert_identical(
        spectral_time_resolved_emission(sim, delta, save_at).pint.dequantify(),
        spectral_time_resolved(sim, delta, save_at).pint.dequantify(),
    )
    xr.testing.assert_identical(
        time_resolved_emission(sim, delta, save_at).pint.dequantify(),
        time_resolved(sim, delta, save_at).pint.dequantify(),
    )
    xr.testing.assert_identical(
        spectral_steady_state_emission(sim, exc, join_by_energy=True).pint.dequantify(),
        spectral_steady_state(sim, exc, join_by_energy=True).pint.dequantify(),
    )
    xr.testing.assert_identical(
        steady_state_emission(sim, exc).pint.dequantify(),
        steady_state(sim, exc).pint.dequantify(),
    )
    xr.testing.assert_identical(
        emission_spectra(sim, exc).pint.dequantify(),
        spectra(sim, exc).pint.dequantify(),
    )
    xr.testing.assert_identical(
        widened_emission_spectra(sim, exc).pint.dequantify(),
        widened_spectra(sim, exc).pint.dequantify(),
    )
