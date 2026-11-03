import numpy as np
import pytest
import pint
import xarray as xr
from poincare.reactions.rebop import RebopSimulator


from jablonski import (
    ConstantGroundState,
    Simulator,
    SingletState,
    SpectroscopicSystem,
    TripletState,
    initial,
)

from ..simulation import (
    delta_excitation,
    piecewise,
    pulse_excitation,
    spectral_time_resolved_emission,
    time_resolved_emission,
)
from ..transitions import (
    Absorption,
    EnergyTransferUpconversion,
    Fluorescence,
    IntersystemCrossing,
    Phosphorescence,
    TripletTripletAbsorption,
)
from .rebop import (
    rebop_piecewise,
    rebop_spectral_time_resolved,
    rebop_time_resolved,
)

ureg = pint.get_application_registry()


class Model(SpectroscopicSystem):
    # Define states with energies, mid is a triplet state
    low: SingletState = initial(0 * ureg.eV, "singlet", default=100)
    mid: TripletState = initial(2 * ureg.eV, "triplet", default=0)
    high: SingletState = initial(3 * ureg.eV, "singlet", default=0)

    # Define absorptions and emissions
    absorption_1 = Absorption(ground=low, excited=high, rate=1e-15 * ureg.cm**2)
    emission_1 = Phosphorescence(ground=low, excited=mid, rate=2e8 / ureg.s)
    emission_2 = IntersystemCrossing(source=high, target=mid, rate=0.5e8 / ureg.s)
    emission_3 = Fluorescence(ground=low, excited=high, rate=1e8 / ureg.s)
    etu = EnergyTransferUpconversion(
        sensitizer_high=mid,
        sensitizer_low=low,
        activator_low=mid,
        activator_high=high,
        rate=1e7 / ureg.s,
    )

sim = Simulator(Model)
rsim = RebopSimulator(Model)

delta = delta_excitation(
    Model.absorption_1, start=0.1e-8 * ureg.s, area=1e14 / ureg.cm**2
)
pulse = pulse_excitation(
    excitation={Model.absorption_1: 1e23 / (ureg.cm**2 * ureg.s)},
    start=5e-8 * ureg.s,
    width=2e-8 * ureg.s,
)
seeds = [1, 43, 56, 67, 78, 45, 90, 35, 45, 81]

def test_rebop_simulation():
    r_sols = []
    sols = []
    for seed in seeds:
        r_sol_raw = rebop_piecewise(
            rsim, events=delta | pulse, upto_t=100 * ureg.ns, n_points=1000, rng=seed
        )
        assert r_sol_raw.time.pint.units == ureg.ns
        r_sol = r_sol_raw.pint.dequantify()

        sol = piecewise(
            sim, events=delta | pulse, save_at=r_sol.time.values * ureg.ns
        ).pint.dequantify()
        r_sols.append(r_sol)
        sols.append(sol)

    mean_r_sol = xr.concat(r_sols, dim="sample").mean(dim="sample")
    mean_sol = xr.concat(sols, dim="sample").mean(dim="sample")
    rtol = 0.1
    atol = 2
    assert np.all(
        (
            np.abs((mean_r_sol - mean_r_sol) / (mean_r_sol + mean_r_sol) * 2).to_array()
            <= rtol
        )
        | (np.abs(mean_r_sol - mean_sol).to_array() <= atol)
    )


def test_rebop_time_resolved_emission():
    result = rebop_spectral_time_resolved(
        rsim,
        excitation=delta|pulse,
        upto_t= 100* ureg.ns,
        n_points=100,
        rng = seeds[0]
    )
    assert result.time.pint.units == ureg.ns
    assert set([str(emission) for emission in result.data_vars.keys()]) == set(
        [
            "emission_" + str(emission)
            for emission in [Model.emission_1, Model.emission_3]
        ]
    )
    joined_result = rebop_spectral_time_resolved(
        rsim,
        excitation=delta|pulse,
        upto_t= 100* ureg.ns,
        n_points=100,
        rng = seeds[0],
        join_by_energy=True,
    )
    assert joined_result.time.pint.units == ureg.ns
    assert set(joined_result.data_vars.keys()) == set(
        ["3 electron_volt", "2 electron_volt"]
    )

    non_spectral = rebop_time_resolved(
        rsim,
        excitation=delta|pulse,
        upto_t= 100* ureg.ns,
        n_points=100,
        rng = seeds[0]
    )
    assert non_spectral.time.pint.units == ureg.ns
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )

def test_rebop_n_runs():
    kwargs = dict(events=delta | pulse, upto_t=100 * ureg.ns, n_points=200)
    averaged = rebop_piecewise(rsim, **kwargs, rng=seeds[0], n_runs=20)
    assert averaged.time.pint.units == ureg.ns
    assert "run" not in averaged.dims

    # Same seed gives the same average, runs are independent of each other.
    repeated = rebop_piecewise(rsim, **kwargs, rng=seeds[0], n_runs=20)
    xr.testing.assert_identical(averaged, repeated)
    single = rebop_piecewise(rsim, **kwargs, rng=seeds[0], n_runs=1)
    assert not averaged.pint.dequantify().identical(single.pint.dequantify())

    # The average of many runs approaches the deterministic solution.
    r_sol = averaged.pint.dequantify()
    sol = (
        piecewise(sim, events=delta | pulse, save_at=r_sol.time.values * ureg.ns)
        .pint.to({"time": ureg.ns})
        .pint.dequantify()
    )
    # Skip event instants, where the ODE output holds values before and after.
    event_times = [t.m_as(ureg.ns) for t in delta | pulse]
    t = r_sol.time.values
    t = t[~np.isclose(t[:, None], event_times).any(axis=1)]
    np.testing.assert_allclose(
        r_sol.sel(time=t).to_array().values,
        sol.sel(time=t, method="nearest").to_array().values,
        rtol=0.1,
        atol=5,
    )

    emission = rebop_time_resolved(
        rsim,
        excitation=delta | pulse,
        upto_t=100 * ureg.ns,
        n_points=100,
        rng=seeds[0],
        n_runs=5,
    )
    assert emission.time.pint.units == ureg.ns

    with pytest.raises(ValueError):
        rebop_piecewise(rsim, **kwargs, n_runs=0)


def test_rebop_constant_ground_state():
    class ConstantModel(SpectroscopicSystem):
        ground: ConstantGroundState = ConstantGroundState(default=100)
        excited: TripletState = initial(2 * ureg.eV, "triplet", default=0)

        absorption = TripletTripletAbsorption(
            ground=ground,
            excited=excited,
            rate=1 * ureg.cm**2,
            pump=50 / (ureg.cm**2 * ureg.s),
        )
        emission = Phosphorescence(excited=excited, ground=ground, rate=1 / ureg.s)

    t = np.linspace(0, 20, 21) * ureg.s
    ode = Simulator(ConstantModel).solve(save_at=t)
    stochastic = RebopSimulator(ConstantModel).solve(
        upto_t=20 * ureg.s, n_points=20, rng=1
    )
    # Steady state: pump * rate * ground / k = 5000
    np.testing.assert_allclose(ode["excited"].values[-1], 5000, rtol=1e-3)
    np.testing.assert_allclose(
        stochastic["excited"].values[5:], ode["excited"].values[5:], rtol=0.05
    )
