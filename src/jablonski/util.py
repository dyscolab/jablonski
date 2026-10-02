"""
    jablonski.util
    ~~~~~~~~~~~~~~

    Molecular states.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from collections.abc import Generator, Iterable
from types import UnionType
from typing import Any, Literal, TypeAlias

import pint
from pint.facets.plain import PlainQuantity

from ._typing import RadiativeDecay, Pumper
from .states import (
    DIM_ENERGY,
    DIM_FREQUENCY,
    DIM_WAVELENGTH,
    DIM_WAVENUMBER,
    SpectroscopicSystem,
)
from .transitions import Fluorescence, Phosphorescence

_ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]

ureg = pint.get_application_registry()

SpectraKind = Literal["emission", "fluorescence", "phosphorescence", "absorption"]

TYPE_MAP = {
    "emission": RadiativeDecay,
    "fluorescence": Fluorescence,
    "phosphorescence": Phosphorescence,
    "absorption": Pumper,
}
REVERSE_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}

def yield_transitions(
    system: SpectroscopicSystem,
    kind: SpectraKind | Iterable[SpectraKind] = "emission",
) -> Generator[RadiativeDecay | Pumper, None, None]:
    if isinstance(kind, str):
        kind = [kind]
    try:
        include = tuple(TYPE_MAP[k] for k in kind)
    except KeyError:
        raise ValueError(f"kind must be {SpectraKind}")

    for transition in system._yield(include):
        yield transition

def lines_and_transform(
    system: SpectroscopicSystem,
    kind: SpectraKind | Iterable[SpectraKind] = "emission",
) -> tuple[dict[str, Any], dict[str, Any]]:
    lines = {
        f"{"emission_" if isinstance(transition, RadiativeDecay) else "absorption_"}{transition}": transition
        for transition in yield_transitions(system=system, kind=kind)
    }
    transform = {
        k: v.radiative_decay.rate_law if isinstance(v, RadiativeDecay) else v.absorption.rate_law
        for k, v in lines.items()
    }
    return lines, transform


def convert():
    dim = ureg.get_dimensionality(unit)

    if dim == DIM_ENERGY:
        # energy
        return {
            transition.energy_difference.to(unit): transition
            for transition in system._yield(include)
        }

    elif dim == DIM_WAVENUMBER:
        # wavenumber
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition
                for transition in system._yield(include)
            }

    elif dim == DIM_WAVELENGTH:
        # wavelength
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition
                for transition in system._yield(include)
            }

    elif dim == DIM_FREQUENCY:
        # frequency
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition
                for transition in system._yield(include)
            }

    else:
        raise ValueError(f"Cannot provide the spectra in {unit} ({dim})")
