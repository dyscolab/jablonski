"""
    jablonski.transitions
    ~~~~~~~~~~~~~~~~~~~~~

    Transitions between molecular states.

    A few rules to make the classes:
    - the states appear in the class (and in the init) from source to target.


    # Ideas taken from https://www.edinst.com/de/blog/jablonski-diagram/

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


import pint
from poincare import Parameter, System, assign
from typing_extensions import dataclass_transform

from jablonski.states import SingletState, TripletState, initial

ureg = pint.get_application_registry()


@dataclass_transform(field_specifiers=(initial,))
class Absorption(System):
    """A radiative transition from a lower to a higher electronic state of a molecule.

    The energy of the photon is converted to the internal energy of the molecule.
    """

    ground: SingletState = initial(0.0, default=0)
    excited: SingletState = initial(0.0, default=0)

    # timescale 10^-15 s
    rate: Parameter = assign(default=1e15 / ureg.s)

    pump = rate * ground

    down = ground.derive() << -pump
    up = excited.derive() << pump

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        assert self.energy_difference >= 0


@dataclass_transform(field_specifiers=(initial,))
class VibrationalRelaxation(System):
    """A non-radiative transition to a lower vibrational level
    within the same electronic state.
    """

    high: SingletState = initial(0.0, default=0)
    low: SingletState = initial(0.0, default=0)

    # timescale 10^-12 s and 10^-10 s
    rate: Parameter = assign(default=1e12 / ureg.s)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self):
        assert self.energy_difference >= 0


@dataclass_transform(field_specifiers=(initial,))
class InternalConversion(System):
    """A non-radiative transition between two electronic states
    of the same spin multiplicity.
    """

    high: SingletState = initial(0.0, default=0)
    low: SingletState = initial(0.0, default=0)

    # timescale 10^-11 s and 10^-9 s, sometimes slower.
    rate: Parameter = assign(default=1e12 / ureg.s)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self):
        assert self.energy_difference >= 0


@dataclass_transform(field_specifiers=(initial,))
class Fluorescence(System):
    """A radiative transition between two electronic states
    of the same spin multiplicity.
    """

    excited: SingletState = initial(0.0, default=0)
    ground: SingletState = initial(0.0, default=0)

    # timescale 10^-10 s and 10^-7 s.
    rate: Parameter = assign(default=1e10 / ureg.s)

    radiative_decay = rate * excited

    down = ground.derive() << radiative_decay
    up = excited.derive() << -radiative_decay

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        assert self.excited.energy > 0


@dataclass_transform(field_specifiers=(initial,))
class IntersystemCrossing(System):
    """A non-radiative transition between two isoenergetic vibrational levels belonging
    to electronic states of different spin multiplicity, from singlet to triplet.
    """

    source: SingletState = initial(0.0, default=0)
    target: TripletState = initial(0.0, "triplet", default=0)

    # timescale 10^−8 s to 10^−3 s
    rate: Parameter = assign(default=1e8 / ureg.s)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.source.energy - self.target.energy

    def _check(self):
        assert self.excited.energy == 0


@dataclass_transform(field_specifiers=(initial,))
class ReverseIntersystemCrossing(System):
    """A non-radiative transition between two isoenergetic vibrational levels belonging
    to electronic states of different spin multiplicity, from triple to singlet.
    """

    source: TripletState = initial(0.0, "triplet", default=0)
    target: SingletState = initial(0.0, default=0)

    # timescale 10^−8 s to 10^−3 s
    rate: Parameter = assign(default=1e8 / ureg.s)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.source.energy - self.target.energy

    def _check(self):
        assert self.excited.energy == 0


@dataclass_transform(field_specifiers=(initial,))
class Phosphorescence(System):
    """A radiative transition between two electronic
    states of different spin multiplicity.
    """

    excited: TripletState = initial(0.0, "triplet", default=0)
    ground: SingletState = initial(0.0, default=0)

    # timescale 10^-6 s to 10 s range.
    rate: Parameter = assign(default=1e6 / ureg.s)

    radiative_decay = rate * excited

    down = ground.derive() << radiative_decay
    up = excited.derive() << -radiative_decay

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        assert self.excited.energy > 0


@dataclass_transform(field_specifiers=(initial,))
class EnergyTransferUpconversion(System):
    sensitizer = initial(0.0, default=0)
    activator = initial(0.0, default=0)
    relaxator = initial(0.0, default=0)

    rate: Parameter = assign(default=0 / ureg.s)

    value = rate * sensitizer**2

    sensitization = sensitizer.derive() << -2 * value
    activation = activator.derive() << value
    relaxation = relaxator.derive() << value
