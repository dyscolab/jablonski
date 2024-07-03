"""
    jablonski._typing
    ~~~~~~~~~~~~~~~~~

    Types and type alias.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from typing import Literal, Protocol, TypeAlias

import pint
from poincare import Parameter

Power: TypeAlias = float | int
Time: TypeAlias = float | int | pint.Quantity

SpinMultiplicity = Literal["singlet", "triplet"]


class Pumper(Protocol):
    pump: Parameter


class RadiativeDecay(Protocol):
    radiative_decay: Parameter
