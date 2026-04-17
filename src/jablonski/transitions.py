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

import warnings

import pint
from poincare import Parameter, assign
from poincare.reactions import MassAction

from jablonski.states import (
    SingletState,
    SpectroscopicSystem,
    SpinState,
    TripletState,
    initial,
)

ureg = pint.get_application_registry()


def _check_range(obj, attr, mn, mx):
    rate = getattr(obj, attr).m_as("Hz")
    if not (mx >= rate >= mn):
        warnings.warn(
            "{obj!r} rate ({rate} Hz) is not within "
            "expected range high ({mn}-{mx} Hz).",
            stacklevel=2,
        )


class Absorption(SpectroscopicSystem):
    """A radiative transition from a lower to a higher electronic state
    of a molecule (both singlets).

    The energy of the photon is converted to the internal energy of the molecule.
    """

    ground: SingletState = initial(0.0, default=0)
    excited: SingletState = initial(0.0, default=0)

    # timescale 10^-15 s
    rate: Parameter = assign(default=1e15 / ureg.s)

    pump: Parameter = assign(default=0)

    absorption = MassAction(reactants=[ground], products=[excited], rate=rate * pump)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        assert self.energy_difference >= 0
        _check_range(self, "rate", 1e14, 1e16)


class TripletTripletAbsorption(SpectroscopicSystem):
    """A radiative transition from a lower to a higher electronic state
    of a molecule (both triplets).

    The energy of the photon is converted to the internal energy of the molecule.
    """

    ground: TripletState = initial(0.0, default=0, spin_multiplicity="triplet")
    excited: TripletState = initial(0.0, default=0, spin_multiplicity="triplet")

    # TODO: what is the correct timescale
    rate: Parameter = assign(default=1e15 / ureg.s)

    pump: Parameter = assign(default=0)

    absorption = MassAction(reactants=[ground], products=[excited], rate=pump * rate)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        assert self.energy_difference >= 0
        # TODO: what is the correct timescale
        _check_range(self, "rate", 1e14, 1e16)


class VibrationalRelaxation(SpectroscopicSystem):
    """A non-radiative transition to a lower vibrational level
    within the same electronic state.
    """

    high: SpinState = initial(0.0, default=0)
    low: SpinState = initial(0.0, default=0)

    # timescale 10^-12 s and 10^-10 s
    rate: Parameter = assign(default=1e12 / ureg.s)

    non_radiative_decay = MassAction(reactants=[high], products=[low], rate=rate)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self):
        assert self.energy_difference >= 0
        _check_range(self, "rate", 1e10, 1e12)


class InternalConversion(SpectroscopicSystem):
    """A non-radiative transition between two electronic states
    of the same spin multiplicity.
    """

    # TODO: what does this do if it doesn't have equations?
    high: SingletState = initial(0.0, default=0)
    low: SingletState = initial(0.0, default=0)

    # timescale 10^-11 s and 10^-9 s, sometimes slower.
    rate: Parameter = assign(default=1e12 / ureg.s)

    non_radiative_decay = MassAction(reactants=[high], products=[low], rate=rate)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self):
        assert self.energy_difference >= 0
        _check_range(self, "rate", 1e11, 1e9)


class Fluorescence(SpectroscopicSystem):
    """A radiative transition between two electronic states
    of the same spin multiplicity.
    """

    excited: SingletState = initial(0.0, default=0)
    ground: SingletState = initial(0.0, default=0)

    # timescale 10^-10 s and 10^-7 s.
    rate: Parameter = assign(default=1e10 / ureg.s)

    radiative_decay = MassAction(reactants=[excited], products=[ground], rate=rate)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        assert self.excited.energy > 0
        _check_range(self, "rate", 1e10, 1e7)


class IntersystemCrossing(SpectroscopicSystem):
    """A non-radiative transition between two isoenergetic vibrational levels belonging
    to electronic states of different spin multiplicity, from singlet to triplet.
    """

    source: SingletState = initial(0.0, default=0)
    target: TripletState = initial(0.0, "triplet", default=0)

    # timescale 10^−8 s to 10^−3 s
    rate: Parameter = assign(default=1e8 / ureg.s)

    non_radiative_transition = MassAction(
        reactants=[source], products=[target], rate=rate
    )

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.source.energy - self.target.energy

    def _check(self):
        assert self.energy_difference == 0
        _check_range(self, "rate", 1e10, 1e8)


class ReverseIntersystemCrossing(SpectroscopicSystem):
    """A non-radiative transition between two isoenergetic vibrational levels belonging
    to electronic states of different spin multiplicity, from triple to singlet.
    """

    source: TripletState = initial(0.0, "triplet", default=0)
    target: SingletState = initial(0.0, default=0)

    # timescale 10^−8 s to 10^−3 s
    rate: Parameter = assign(default=1e8 / ureg.s)

    non_radiative_transition = MassAction(
        reactants=[source], products=[target], rate=rate
    )

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.source.energy - self.target.energy

    def _check(self):
        assert self.energy_difference == 0
        _check_range(self, "rate", 1e10, 1e8)


class Phosphorescence(SpectroscopicSystem):
    """A radiative transition between two electronic
    states of different spin multiplicity.
    """

    excited: TripletState = initial(0.0, "triplet", default=0)
    ground: SingletState = initial(0.0, default=0)

    # timescale 10^-6 s to 10 s range.
    rate: Parameter = assign(default=1e6 / ureg.s)
    radiative_decay = MassAction(reactants=[excited], products=[ground], rate=rate)

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        assert self.excited.energy > 0
        _check_range(self, "rate", 1e6, 1)


class EnergyTransferUpconversion(SpectroscopicSystem):
    sensitizer: SingletState = initial(0.0, default=0)
    activator: SingletState = initial(0.0, default=0)
    relaxator: SingletState = initial(0.0, default=0)

    rate: Parameter = assign(default=0 / ureg.s)

    upconversion = MassAction(
        reactants=[2 * sensitizer], products=[activator, relaxator], rate=rate
    )
