"""
    jablonski._typing
    ~~~~~~~~~~~~~~~~~

    Types and type alias.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from typing import Literal, Protocol, TypeAlias, runtime_checkable

import pint
from poincare import Parameter

Power: TypeAlias = float | int
Time: TypeAlias = float | int | pint.Quantity

SpinMultiplicity = Literal["singlet", "triplet"]


@runtime_checkable
class Pumper(Protocol):
    pump: Parameter


@runtime_checkable
class RadiativeDecay(Protocol):
    radiative_decay: Parameter
