from itertools import chain, pairwise
from typing import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pint
import pint_xarray
import scipy.constants as constants
import xarray as xr
from poincare import Simulator, SteadyState
from poincare.simulator import Components, Initial
from poincare.solvers import Solver, LSODA
from poincare.reactions.rebop import RebopSimulator
from poincare.reactions import Reactant
from rebop.gillespie import RNGLike, SeedLike
from symbolite import Real

from .. import util
from .._typing import Pumper, Time
from .._units import DEFAULT_DELTA, ureg
from ..states import SpectroscopicSystem
from ..util import SpectraKind

pint.get_application_registry().force_ndarray_like = False
# When pint-xarray is imported it sets it to true, it can break poincare compilation


def rebop_piecewise(
    rsim: RebopSimulator,
    *,
    events: dict[Time, Mapping[Components, Initial | Real | None]],
    upto_t: pint.Quantity,
    n_points: int | None = None,
    rng: RNGLike | SeedLike | None = None,
    sparse: bool = True,
    var_names: Iterable[Reactant] | None = None,
) -> xr.Dataset:
    try:
        event_keys = np.array([key.to(ureg.s).magnitude for key in events.keys()])
    except (AttributeError, pint.DimensionalityError):
        raise pint.PintError(
            "events keys must be pint Quantities and have time dimensionality."
        )

    try:
        adimensional_upto_t = upto_t.to(ureg.s).magnitude
    except (AttributeError, pint.DimensionalityError):
        raise pint.PintError(
            "upto_t must be pint Quantity and have time dimensionality."
        )
    upto_ts = np.concatenate([event_keys, [adimensional_upto_t]])
    upto_ts = np.sort(upto_ts)
    point_distribution = distribute_points(upto_ts, n_points)
    upto_ts = [upto_t * ureg.s for upto_t in upto_ts]
    dss = []
    state = {}

    previous = 0 * ureg.s
    for upto_t, n_points in zip(upto_ts, point_distribution):
        ds = rsim.with_values(state).solve(
            upto_t=upto_t - previous,
            n_points=n_points,
            rng=rng,
            sparse=sparse,
            var_names=var_names,
        )
        for k, v in events.get(upto_t, {}).items():
            if v is None and k in state:
                del state[k]
            else:
                state[k] = v
            # str(k) porque en el output no usamos el objeto Variable aun
            as_str = str(k)
            if as_str in ds:
                ds[as_str][-1] = v

        state.update({k: ds[str(k)][-1].item() for k in rsim._sim.compiled.variables})
        if previous >= 0 * ureg.s:
            ds = ds.isel(time=slice(1, None))
            ds = ds.assign_coords(time=ds.time + previous)
        previous = upto_t
        dss.append(ds.pint.dequantify())
    ds = xr.concat(dss, dim="time")

    pint_xarray.setup_registry(ureg)
    ds = ds.pint.quantify()
    ureg.force_ndarray_like = False
    return ds


def distribute_points(upto_ts: Sequence[np.float64], n_points: int) -> Sequence[int]:
    effective_points = n_points - len(upto_ts)
    time_proportions = np.diff(np.concat([[0], upto_ts])) / upto_ts[-1]
    remainders, points = np.modf(time_proportions * effective_points)
    points += 1  # At least one point per simulation period
    leftover = int(n_points - np.sum(points))
    order = np.argsort(remainders)
    # TODO: siwtch to descending = True and normal args in argsort and order[leftover:]
    # in next line once numpy 2.5.0 is not so new
    points[order[-leftover:]] += 1
    return points.astype(int)
