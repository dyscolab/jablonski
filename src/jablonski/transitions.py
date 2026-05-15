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
    rate = getattr(obj, attr).default.m_as("Hz")
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.ground
        self._target = self.excited

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        if self.energy_difference < 0:
            raise ValueError(
                "Excited state energy must be higher than ground state energy"
            )
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.ground
        self._target = self.excited

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        if self.energy_difference < 0:
            raise ValueError(
                "excited state energy must be higher than ground state energy."
            )
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.high
        self._target = self.low

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self):
        if self.energy_difference < 0:
            raise ValueError("High state energy must be higher than low state energy")
        if self.high.multiplicity != self.low.multiplicity:
            raise TypeError(
                "Both states must have the same multiplicity in a vibrational relaxation"
            )
        _check_range(self, "rate", 1e10, 1e12)


class InternalConversion(SpectroscopicSystem):
    """A non-radiative transition between two electronic states
    of the same spin multiplicity.
    """

    high: SpinState = initial(0.0, default=0)
    low: SpinState = initial(0.0, default=0)

    # timescale 10^-11 s and 10^-9 s, sometimes slower.
    rate: Parameter = assign(default=1e11 / ureg.s)

    non_radiative_decay = MassAction(reactants=[high], products=[low], rate=rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.high
        self._target = self.low

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self):
        if self.energy_difference < 0:
            raise ValueError("High state energy must be higher than low state energy")
        if self.high.multiplicity != self.low.multiplicity:
            raise TypeError(
                "Both states must have the same multiplicity in an internal conversion"
            )
        _check_range(self, "rate", 1e9, 1e11)


class Fluorescence(SpectroscopicSystem):
    """A radiative transition between two electronic states
    of the same spin multiplicity.
    """

    excited: SpinState = initial(0.0, default=0)
    ground: SpinState = initial(0.0, default=0)

    # timescale 10^-10 s and 10^-7 s.
    rate: Parameter = assign(default=1e10 / ureg.s)

    radiative_decay = MassAction(reactants=[excited], products=[ground], rate=rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.excited
        self._target = self.ground

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        if self.energy_difference <= 0:
            raise ValueError(
                "Excited state energy must be higher than ground state energy"
            )
        if self.excited.multiplicity != self.ground.multiplicity:
            raise TypeError(
                "Both states must have the same multiplicity in fluorescence, use Phosphorescence for radiative transitions with different spin multiplicity"
            )
        _check_range(self, "rate", 1e7, 1e10)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.source
        self._target = self.target

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.source.energy - self.target.energy

    def _check(self):
        if self.energy_difference != 0:
            raise ValueError(
                "Source and target states energy must be equal in an intersystem crossing."
            )
        _check_range(self, "rate", 1e8, 1e10)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.source
        self._target = self.target

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.source.energy - self.target.energy

    def _check(self):
        if self.energy_difference != 0:
            raise ValueError(
                "Source and target states energy must be equal in an intersystem crossing."
            )
        _check_range(self, "rate", 1e8, 1e10)


class Phosphorescence(SpectroscopicSystem):
    """A radiative transition between two electronic
    states of different spin multiplicity.
    """

    excited: SpinState = initial(0.0, "triplet", default=0)
    ground: SpinState = initial(0.0, default=0)

    # timescale 10^-6 s to 10 s range.
    rate: Parameter = assign(default=1e6 / ureg.s)
    radiative_decay = MassAction(reactants=[excited], products=[ground], rate=rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.excited
        self._target = self.ground

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self):
        if self.energy_difference <= 0:
            raise ValueError(
                "Excited state energy must be higher than ground state energy"
            )
        if self.excited.multiplicity == self.ground.multiplicity:
            raise TypeError(
                "Both states must have different multiplicity in phosphorescence, use Fluorescence for radiative transitions with the same spin multiplicity"
            )
        _check_range(self, "rate", 1, 1e6)


class EnergyTransferUpconversion(SpectroscopicSystem):
    sensitizer: SingletState = initial(0.0, default=0)
    activator: SingletState = initial(0.0, default=0)
    relaxator: SingletState = initial(0.0, default=0)

    rate: Parameter = assign(default=0 / ureg.s)

    upconversion = MassAction(
        reactants=[2 * sensitizer], products=[activator, relaxator], rate=rate
    )
