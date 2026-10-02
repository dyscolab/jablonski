"""
    jablonski.util
    ~~~~~~~~~~~~~~

    Molecular states.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from types import UnionType
from typing import Any, Literal, TypeAlias, Generator

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


def excitation_transitions(
    system: SpectroscopicSystem,
) -> Generator[Pumper, None, None]:
    for transition in system._yield(Pumper):
        yield transition


def emission_transitions(
    system: SpectroscopicSystem,
    kind: SpectraKind = "emission",
) -> Generator[RadiativeDecay, None, None]:
    if kind == "emission":
        include = (Fluorescence, Phosphorescence)
    elif kind == "fluorescence":
        include = Fluorescence
    elif kind == "phosphorescence":
        include = Phosphorescence
    else:
        raise ValueError(f"kind must be {SpectraKind}")

    for transition in system._yield(include):
        if isinstance(transition, RadiativeDecay):
            yield transition

def lines_and_transform(
    system: SpectroscopicSystem,
    kind: SpectraKind = "emission",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if kind == "absorption":
        lines = {
            f"line_{transition}": transition
            for transition in excitation_transitions(system)
        }
        transform = {k: v.absorption.rate_law for k, v in lines.items()}
    elif kind in ("emission", "fluorescence", "phosphorescence"):
        lines = {
            f"line_{transition}": transition
            for transition in emission_transitions(system, kind=kind)
        }
        transform = {k: v.radiative_decay.rate_law for k, v in lines.items()}
    else:
        raise ValueError(f"kind must be {SpectraKind}")

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
