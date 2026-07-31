"""
jablonski._typing
~~~~~~~~~~~~~~~~~

Types and type alias.

:copyright: 2024 by jablonski Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from typing import Literal, Protocol, TypeAlias, runtime_checkable
from collections.abc import Mapping

import pint
from poincare import Parameter, Variable

Power: TypeAlias = float | int
Time: TypeAlias = float | int | pint.Quantity

SpinMultiplicity = Literal["singlet", "triplet"] | None


@runtime_checkable
class Pumper(Protocol):
    pump: Parameter

    @property
    def energy_difference(self) -> pint.Quantity: ...


Excitation: TypeAlias = Mapping[Pumper, pint.Quantity]


@runtime_checkable
class RadiativeDecay(Protocol):
    radiative_decay: Parameter


@runtime_checkable
class Drawable(Protocol):
    _source: Variable
    _target: Variable
