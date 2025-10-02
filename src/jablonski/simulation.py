"""
    jablonski.simulation
    ~~~~~~~~~~~~~~~~~~~~

    Simulation functions.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from itertools import chain, pairwise
from typing import Mapping

import pint
import numpy as np
import numpy.typing as npt
import pandas as pd
from poincare import Simulator
from poincare.simulator import Components, Initial, Symbol

from . import util
from .util import SpectraKind
from ._typing import Pumper, Time
from ._units import DEFAULT_DELTA, ureg
from .states import SpectroscopicSystem


def piecewise(
    sim: Simulator,
    *,
    events: dict[Time, Mapping[Components, Initial | Symbol | None]],
    save_at: npt.NDArray[np.float64],
) -> pd.DataFrame:
    event_keys = [k.m_as("s") for k in events.keys()]
    t_events = np.sort(event_keys)
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

        state.update(df.attrs["last_state"])
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    # df = df.d
    return df


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


def spectral_time_resolved_emission(
    system: SpectroscopicSystem,
    excitation: dict[Time, Mapping[Components, Initial | Symbol | None]],
    save_at: npt.NDArray[np.float64],
    kind: util.SpectraKind = "emission",
) -> pd.DataFrame:
    """Single transition square excitation."""

    
    lines = {
        f"line{ndx}": transition
        for ndx, transition in enumerate(util.emission_transitions(system, kind=kind))
    }

    transform = {k: v.radiative_decay for k, v in lines.items()}

    sim = Simulator(system, transform=transform)
    df = piecewise(sim, events=excitation, save_at=save_at)
    df.attrs["lines"] = lines
    return df


def spectral_steady_state_emission(
    system: SpectroscopicSystem,
    excitation_transition: Pumper,
    height: float,
    kind: util.SpectraKind = "emission",
) -> pd.DataFrame:
    
    lines = {
        f"line{ndx}": transition
        for ndx, transition in enumerate(util.emission_transitions(system, kind=kind))
    }

    transform = {k: v.radiative_decay for k, v in lines.items()}

    sim = Simulator(system, transform=transform)
    df = piecewise(
        sim, 
        events=step_excitation(excitation_transition, height), 
        save_at=np.asarray([0, 100]),
    ).iloc[[-1]]
    df.attrs["lines"] = lines
    return df
    

def time_resolved_emission()


def excitation_spectra(
        excitation: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
        emission: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
    ): 
    """CW excitation spectra.
    """


def emission_spectra(
        system: SpectroscopicSystem,
        excitation_transition: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
        height: float, 
        unit: str | pint.Unit = ureg.nm,
        kind: SpectraKind = "emission",
    ): 
    """CW emission spectra.
    """
    # steady_state_emission(system, excitation_transition, height, kind)
    # for k, v in util.emission_transitions(system, unit, kind).items()
