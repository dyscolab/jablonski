"""
    jablonski.simulation
    ~~~~~~~~~~~~~~~~~~~~

    Simulation functions.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from itertools import chain, pairwise
from typing import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from poincare import Simulator
from poincare.simulator import Components, Initial, Symbol

from ._typing import Pumper, Time
from ._units import DEFAULT_DELTA, ureg


def piecewise(
    sim: Simulator,
    *,
    events: dict[Time, Mapping[Components, Initial | Symbol | None]],
    save_at: npt.NDArray[np.float64],
) -> pd.DataFrame:
    assert sim.transform is None

    t_events = np.sort(list(events.keys()))
    save_at = np.union1d(save_at, t_events)
    pos = np.searchsorted(save_at, t_events)
    save_ats = np.split(save_at, pos + 1)
    t_spans = pairwise(chain((0,), t_events, (save_at[-1],)))

    dfs = []
    state = {}
    for t_span, save_at in zip(t_spans, save_ats):
        df = sim.solve(t_span=t_span, save_at=save_at, values=state)
        for k, v in events.get(save_at[-1], {}).items():
            if v is None and k in state:
                del state[k]
            else:
                state[k] = v
            # str(k) porque en el output no usamos el objeto Variable aun
            k = str(k)
            if k in df:
                df[k].values[-1] = v

        for k, v in df.iloc[-1].items():
            try:
                k = getattr(sim.model, k)
            except AttributeError:
                raise AttributeError(f"Could not find attribute {k} in model")
                # some transformed variables might not be part of the state
                continue
            state[k] = v
        dfs.append(df)

    return pd.concat(dfs)


def step_excitation(
    excitation_transition: Pumper, height: float, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Symbol | None]]:
    return {
        start: {excitation_transition.pump: height},
    }


def pulse_excitation(
    excitation_transition: Pumper, height: float, width: Time, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Symbol | None]]:
    return {
        start: {excitation_transition.pump: height},
        (start + width): {excitation_transition.pump: None},
    }


def delta_excitation(
    excitation_transition: Pumper, area: Time, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Symbol | None]]:
    width = DEFAULT_DELTA
    height = area / width
    return pulse_excitation(excitation_transition, height, width, start)
