"""
    jablonski.states
    ~~~~~~~~~~~~~~~~

    Molecular states.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


import pint
from poincare import Independent, System, Variable
from poincare.types import Initial
from typing_extensions import dataclass_transform

ureg = pint.get_application_registry()

DIM_WAVELENGTH = {"[length]": 1}
DIM_WAVENUMBER = {"[length]": -1}
DIM_FREQUENCY = {"[time]": -1}
DIM_ENERGY = ureg.get_dimensionality("eV")


class State(Variable):
    energy: pint.Quantity


def initial(
    energy: float | pint.Quantity, *, default: Initial | None = None, init: bool = True
) -> State:
    state = State(initial=default)

    dim = ureg.get_dimensionality(energy)

    if dim == {}:
        energy = energy * ureg.electron_volt

    elif dim in (DIM_WAVELENGTH, DIM_WAVENUMBER, DIM_FREQUENCY):
        with ureg.context("spectroscopy"):
            energy = energy.to("eV")

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
