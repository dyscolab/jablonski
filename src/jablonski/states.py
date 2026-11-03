"""
jablonski.states
~~~~~~~~~~~~~~~~

Molecular states.

:copyright: 2024 by jablonski Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from typing import Literal, TypeAlias, overload

import pint
from poincare import Independent, Parameter, System, Variable
from poincare._node import Node
from poincare.types import Initial
from typing_extensions import dataclass_transform

from ._typing import SpinMultiplicity
from ._units import DIM_ENERGY, DIM_FREQUENCY, DIM_WAVELENGTH, DIM_WAVENUMBER, ureg


class SpinState(Variable):
    """Base class for singlet and triplet states."""

    energy: pint.Quantity

    multiplicity = None

    def _copy_from(self, parent: Node):
        copy = super()._copy_from(parent=parent)
        copy.energy = self.energy
        copy.multiplicity = self.multiplicity
        return copy

    def __set__(self, obj, value):
        if isinstance(value, ConstantGroundState):
            # Replace the state by a constant species in this instance
            if value.parent is None and value.name == "":
                value.__set_name__(obj, self.name)
            obj.__dict__[self.name] = value
        else:
            super().__set__(obj, value)


class SingletState(SpinState):
    """A molecular electronic state such that all electron spins are paired;
    that is, they are antiparallel (oposite spin).
    """

    multiplicity = "singlet"


class TripletState(SpinState):
    """A molecular electronic state such that an excited electron is not
    paired with the ground state electron; that is, they are parallel (same spin).
    """

    multiplicity = "triplet"


class ConstantGroundState(Parameter):
    """A ground state whose population is held constant.

    Its population (``default``) enters the rate of every transition it
    takes part in, but it has no equation of its own. It can replace a
    singlet or a triplet state in any transition, and passes the spin
    multiplicity checks of both.

    ``energy`` is interpreted as in `initial`: a float is in eV.
    """

    energy: pint.Quantity

    multiplicity = None

    def __init__(
        self,
        *,
        default: Initial | None = None,
        energy: float | pint.Quantity = 0.0,
    ):
        super().__init__(default=default)
        self.energy = _to_energy(energy)

    def _copy_from(self, parent: Node):
        copy = super()._copy_from(parent=parent)
        copy.energy = self.energy
        return copy


SingletStateLike: TypeAlias = SingletState | ConstantGroundState
TripletStateLike: TypeAlias = TripletState | ConstantGroundState
SpinStateLike: TypeAlias = SpinState | ConstantGroundState


def _to_energy(energy: float | pint.Quantity) -> pint.Quantity:
    dim = ureg.get_dimensionality(energy)

    if dim == {}:
        energy = energy * ureg.electron_volt

    elif dim in (DIM_WAVELENGTH, DIM_WAVENUMBER, DIM_FREQUENCY):
        with ureg.context("spectroscopy"):
            energy = energy.to(ureg.eV)

    elif dim != DIM_ENERGY:
        raise ValueError(
            "energy must be a float (which is interpreted as eV) or have "
            "dimensionality of wavelength, wavenumber, frequency and energy, not {dim}"
        )

    assert isinstance(energy, pint.Quantity)
    return energy


@overload
def initial(
    energy: float | pint.Quantity,
    spin_multiplicity: Literal["singlet"] = "singlet",
    *,
    default: Initial | None = None,
) -> SingletState: ...


@overload
def initial(
    energy: float | pint.Quantity,
    spin_multiplicity: Literal["triplet"],
    *,
    default: Initial | None = None,
) -> TripletState: ...


def initial(
    energy: float | pint.Quantity,
    spin_multiplicity: SpinMultiplicity = "singlet",
    *,
    default: Initial | None = None,
) -> SingletState | TripletState:
    if spin_multiplicity == "singlet":
        state = SingletState(initial=default)
    elif spin_multiplicity == "triplet":
        state = TripletState(initial=default)
    elif spin_multiplicity is None:
        state = SpinState(initial=default)
    else:
        raise ValueError("spin_multiplicity must be 'singlet', 'triplet' or None")
    state.energy = _to_energy(energy)
    return state


@dataclass_transform(field_specifiers=(initial, ConstantGroundState))
class SpectroscopicSystem(System, abstract=True):
    time = Independent(default=0 * ureg.s)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check()

    def _check(self, check_range=False):
        pass
