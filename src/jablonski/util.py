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

SpectraKind = Literal["emission", "fluorescence", "phosphorescence"]


def excitation_transitions(
    system: SpectroscopicSystem,
) -> Generator[Pumper, None, None]:

    for transition in system._yield(SpectroscopicSystem):
        if isinstance(transition, Pumper):
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
