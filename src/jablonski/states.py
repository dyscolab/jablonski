"""
    jablonski.states
    ~~~~~~~~~~~~~~~~

    Molecular states.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from typing import Literal, overload

import pint
from poincare import Independent, System, Variable
from poincare.types import Initial
from typing_extensions import dataclass_transform

from ._typing import SpinMultiplicity
from ._units import DIM_ENERGY, DIM_FREQUENCY, DIM_WAVELENGTH, DIM_WAVENUMBER, ureg


class SingletState(Variable):
    """A molecular electronic state such that all electron spins are paired;
    that is, they are antiparallel (oposite spin).
    """

    energy: pint.Quantity


class TripletState(Variable):
    """A molecular electronic state such that an excited electron is not
    paired with the ground state electron; that is, they are parallel (same spin).
    """

    energy: pint.Quantity


@overload
def initial(
    energy: float | pint.Quantity,
    spin_multiplicity: Literal["singlet"] = "singlet",
    *,
    default: Initial | None = None,
) -> SingletState:
    ...


@overload
def initial(
    energy: float | pint.Quantity,
    spin_multiplicity: Literal["triplet"],
    *,
    default: Initial | None = None,
) -> TripletState:
    ...


def initial(
    energy: float | pint.Quantity,
    spin_multiplicity: SpinMultiplicity = "singlet",
    *,
    default: Initial | None = None,
) -> SingletState | TripletState:
    if spin_multiplicity == "singlet":
        state = SingletState(initial=default)
    else:
        state = TripletState(initial=default)

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
    state.energy = energy
    return state


@dataclass_transform(field_specifiers=(initial,))
class SpectroscopicSystem(System, abstract=True):
    time = Independent(default=0 * ureg.s)
