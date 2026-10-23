import numpy as np
from poincare import Simulator, solvers

from jablonski import SingletState, SpectroscopicSystem, initial

from .._units import ureg
from ..fits import (
    fit_emission_spectra,
    fit_spectra,
    fit_spectral_time_resolved,
    fit_spectral_time_resolved_emission,
    make_spectra_target,
    make_time_resolved_target,
)
from ..simulation import pulse_excitation, spectra, spectral_time_resolved
from ..transitions import Absorption, Fluorescence


class Model(SpectroscopicSystem):
    low: SingletState = initial(4 * ureg.eV, "singlet", default=10)
    mid: SingletState = initial(5 * ureg.eV, "singlet", default=0)
    high: SingletState = initial(6 * ureg.eV, "singlet", default=0)

    absorption = Absorption(ground=low, excited=high, rate=1e-15 * ureg.cm**2)
    fluo_1 = Fluorescence(ground=low, excited=mid, rate=2e8 / ureg.s)
    fluo_2 = Fluorescence(ground=mid, excited=high, rate=0.5e8 / ureg.s)
    fluo_3 = Fluorescence(ground=low, excited=high, rate=1e8 / ureg.s)


sim = Simulator(Model).with_solver(solvers.LSODA(rtol=1e-10, atol=1e-12))
rtol = 1e-3

# The pulse starts and ends at times not in save_at
excitation = pulse_excitation(
    {Model.absorption: 1e22 / (ureg.cm**2 * ureg.s)},
    width=1e-9 * ureg.s,
    start=1e-10 * ureg.s,
)
save_at = np.linspace(0, 5e-8, 60) * ureg.s
cw_excitation = {Model.absorption: 1e20 / (ureg.cm**2 * ureg.s)}


def time_resolved_target():
    result = spectral_time_resolved(sim, excitation=excitation, save_at=save_at)
    result = result.isel(time=np.isin(result["time"].values, save_at.magnitude))
    return make_time_resolved_target(
        {
            transition: result["emission_" + str(transition)].data
            for transition in [Model.fluo_1, Model.fluo_2, Model.fluo_3]
        },
        save_at=save_at,
    )


def spectra_target():
    result = spectra(sim, excitation=cw_excitation).pint.dequantify()
    (index,) = result.dims
    return make_spectra_target(
        result[index].values * ureg.nm, result.values / ureg.s
    )


def test_fit_spectral_time_resolved():
    target = time_resolved_target()
    p0 = {Model.fluo_1.rate: 3e8 / ureg.s}

    fit = fit_spectral_time_resolved(sim, target, excitation=excitation, p0=p0)
    assert np.isclose(fit[Model.fluo_1.rate].to(1 / ureg.s).magnitude, 2e8, rtol=rtol)

    sigma = 0.01 * target + 1e3 / ureg.s
    fit = fit_spectral_time_resolved_emission(
        sim, target, excitation=excitation, p0=p0, sigma=sigma
    )
    assert np.isclose(fit[Model.fluo_1.rate].to(1 / ureg.s).magnitude, 2e8, rtol=rtol)
    assert np.isfinite(fit.pcov["fluo_1.rate", "fluo_1.rate"].magnitude)


def test_fit_spectra():
    # The steady-state spectra depend on the branching from high (fluo_2 vs fluo_3)
    target = spectra_target()
    p0 = {Model.fluo_2.rate: 0.8e8 / ureg.s}

    fit = fit_spectra(sim, target, excitation=cw_excitation, p0=p0)
    assert np.isclose(fit[Model.fluo_2.rate].to(1 / ureg.s).magnitude, 0.5e8, rtol=rtol)

    # The unnamed target is called "spectra", so scale can refer to it
    fit = fit_emission_spectra(
        sim,
        target,
        excitation=cw_excitation,
        p0=p0,
        scale={"spectra": 2},
        sigma=0.01 * target,
    )
    assert np.isclose(fit[Model.fluo_2.rate].to(1 / ureg.s).magnitude, 0.5e8, rtol=rtol)
    assert np.isfinite(fit.pcov["fluo_2.rate", "fluo_2.rate"].magnitude)


def test_time_resolved_target_names():
    """Transitions named with the emission_/absorption_ prefix get it again,
    like in the simulation output."""

    class Prefixed(SpectroscopicSystem):
        low: SingletState = initial(4 * ureg.eV, "singlet", default=10)
        high: SingletState = initial(6 * ureg.eV, "singlet", default=0)

        absorption_1 = Absorption(ground=low, excited=high, rate=1e-15 * ureg.cm**2)
        emission_1 = Fluorescence(ground=low, excited=high, rate=1e8 / ureg.s)

    psim = Simulator(Prefixed).with_solver(solvers.LSODA(rtol=1e-10, atol=1e-12))
    pexcitation = pulse_excitation(
        {Prefixed.absorption_1: 1e22 / (ureg.cm**2 * ureg.s)},
        width=1e-9 * ureg.s,
        start=1e-10 * ureg.s,
    )

    targets = {}
    for kind, transition in [
        ("emission", Prefixed.emission_1),
        ("absorption", Prefixed.absorption_1),
    ]:
        result = spectral_time_resolved(
            psim, excitation=pexcitation, save_at=save_at, kind=kind
        )
        result = result.isel(time=np.isin(result["time"].values, save_at.magnitude))
        (name,) = result.data_vars
        assert name == f"{kind}_{transition}"  # e.g. emission_emission_1

        targets[kind] = make_time_resolved_target(
            {transition: result[name].data}, save_at=save_at
        )
        assert list(targets[kind].data_vars) == [name]

    # Previously failed with a KeyError: the target was named emission_1
    fit = fit_spectral_time_resolved(
        psim,
        targets["emission"],
        excitation=pexcitation,
        p0={Prefixed.emission_1.rate: 2e8 / ureg.s},
    )
    assert np.isclose(
        fit[Prefixed.emission_1.rate].to(1 / ureg.s).magnitude, 1e8, rtol=rtol
    )
