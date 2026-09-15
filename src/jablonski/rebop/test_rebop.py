import numpy as np
import pint
import xarray as xr
from poincare.reactions.rebop import RebopSimulator


from jablonski import (
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
)
from .rebop import rebop_piecewise, rebop_spectral_time_resolved_emission, rebop_time_resolved_emission

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
        sensitizer=mid, activator=low, relaxator=high, rate=1e7 / ureg.s
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
        r_sol = rebop_piecewise(
            rsim, events=delta | pulse, upto_t=100 * ureg.ns, n_points=1000, rng=seed
        ).pint.dequantify()

        sol = piecewise(
            sim, events=delta | pulse, save_at=r_sol.time.values * ureg.s
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
    result = rebop_spectral_time_resolved_emission(
        rsim,
        excitation=delta|pulse,
        upto_t= 100* ureg.ns,
        n_points=100,
        rng = seeds[0]
    )
    assert set([str(emission) for emission in result.data_vars.keys()]) == set(
        [
            "line_" + str(emission)
            for emission in [Model.emission_1, Model.emission_3]
        ]
    )
    joined_result = rebop_spectral_time_resolved_emission(
        rsim,
        excitation=delta|pulse,
        upto_t= 100* ureg.ns,
        n_points=100,
        rng = seeds[0],
        join_by_energy=True,
    )
    assert set(joined_result.data_vars.keys()) == set(
        ["3 electron_volt", "2 electron_volt"]
    )

    non_spectral = rebop_time_resolved_emission(
        rsim,
        excitation=delta|pulse,
        upto_t= 100* ureg.ns,
        n_points=100,
        rng = seeds[0]
    )
    assert np.all(
        np.asarray(non_spectral.to_array())
        == np.asarray(result.to_array().sum(dim="variable"))
    )