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
from poincare import Constant, Parameter, assign
from poincare.reactions import MassAction

from jablonski.states import (
    ConstantGroundState,
    SingletStateLike,
    SpectroscopicSystem,
    SpinStateLike,
    TripletStateLike,
    initial,
)

ureg = pint.get_application_registry()


def check_all(system: SpectroscopicSystem | type[SpectroscopicSystem]) -> None:
    """Warn about transition rates outside their typical range.

    Every transition in system (including those of subsystems) is checked.
    Rates given as an expression of other components are not checked.
    """
    for spectroscopic in system._yield(SpectroscopicSystem):
        spectroscopic._check(check_range=True)


def _resolve_value(value):
    """Follow linked parameters to the value they hold."""
    while isinstance(value, Parameter | Constant):
        value = value.default
    return value


def _check_range(obj, attr: str, mn: float, mx: float, unit: pint.Unit = ureg.s**-1):
    """Warn if the value of obj.<attr> is not between mn and mx (in unit).

    A plain number is taken to be in unit. Expressions are not checked.
    """
    value = _resolve_value(getattr(obj, attr))
    if isinstance(value, pint.Quantity):
        try:
            magnitude = value.to(unit).magnitude
        except pint.DimensionalityError:
            warnings.warn(
                f"{obj} {attr} ({value}) does not have the expected units ({unit}).",
                stacklevel=3,
            )
            return
    elif isinstance(value, int | float):
        magnitude = value
    else:
        return

    low, high = sorted((mn, mx))
    if not low <= magnitude <= high:
        warnings.warn(
            f"{obj} {attr} ({value}) is not within the expected range "
            f"({low:g} to {high:g} {unit:~}).",
            stacklevel=3,
        )


def _same_multiplicity(a: SpinStateLike, b: SpinStateLike) -> bool:
    """A ConstantGroundState matches any multiplicity."""
    if isinstance(a, ConstantGroundState) or isinstance(b, ConstantGroundState):
        return True
    return a.multiplicity == b.multiplicity


def _different_multiplicity(a: SpinStateLike, b: SpinStateLike) -> bool:
    """A ConstantGroundState matches any multiplicity."""
    if isinstance(a, ConstantGroundState) or isinstance(b, ConstantGroundState):
        return True
    return a.multiplicity != b.multiplicity


class Absorption(SpectroscopicSystem):
    """A radiative transition from a lower to a higher electronic state
    of a molecule (both singlets).

    The energy of the photon is converted to the internal energy of the molecule.
    """

    ground: SingletStateLike = initial(0.0, spin_multiplicity="singlet")
    excited: SingletStateLike = initial(0.0, spin_multiplicity="singlet")

    # timescale 10^-15 s
    rate: Parameter = assign()

    pump: Parameter = assign(default=0 / (ureg.cm**2 * ureg.s))

    absorption = MassAction(reactants=[ground], products=[excited], rate=rate * pump)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.ground
        self._target = self.excited

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self, check_range=False):
        if self.energy_difference < 0:
            raise ValueError(
                "Excited state energy must be higher than ground state energy"
            )
        if check_range:
            _check_range(
                self, "rate", 1e-14, 1e-16, unit=ureg.cm**2
            )  # TODO: what are the correct limits?


class TripletTripletAbsorption(SpectroscopicSystem):
    """A radiative transition from a lower to a higher electronic state
    of a molecule (both triplets).

    The energy of the photon is converted to the internal energy of the molecule.
    """

    ground: TripletStateLike = initial(0.0, spin_multiplicity="triplet")
    excited: TripletStateLike = initial(0.0, spin_multiplicity="triplet")

    # TODO: what is the correct timescale
    rate: Parameter = assign()

    pump: Parameter = assign(default=0 / (ureg.cm**2 * ureg.s))

    absorption = MassAction(reactants=[ground], products=[excited], rate=pump * rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.ground
        self._target = self.excited

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self, check_range=False):
        if self.energy_difference < 0:
            raise ValueError(
                "excited state energy must be higher than ground state energy."
            )
        # TODO: what is the correct timescale
        if check_range:
            _check_range(self, "rate", 1e-14, 1e-16, unit=ureg.cm**2)


class VibrationalRelaxation(SpectroscopicSystem):
    """A non-radiative transition to a lower vibrational level
    within the same electronic state.
    """

    high: SpinStateLike = initial(0.0, spin_multiplicity=None)
    low: SpinStateLike = initial(0.0, spin_multiplicity=None)

    # timescale 10^-12 s and 10^-10 s
    rate: Parameter = assign()

    non_radiative_decay = MassAction(reactants=[high], products=[low], rate=rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.high
        self._target = self.low

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self, check_range=False):
        if self.energy_difference < 0:
            raise ValueError("High state energy must be higher than low state energy")
        if not _same_multiplicity(self.high, self.low):
            raise TypeError(
                "Both states must have the same multiplicity in a vibrational relaxation"
            )
        if check_range:
            _check_range(self, "rate", 1e10, 1e12)


class InternalConversion(SpectroscopicSystem):
    """A non-radiative transition between two electronic states
    of the same spin multiplicity.
    """

    high: SpinStateLike = initial(0.0, spin_multiplicity=None)
    low: SpinStateLike = initial(0.0, spin_multiplicity=None)

    # timescale 10^-11 s and 10^-9 s, sometimes slower.
    rate: Parameter = assign()

    non_radiative_decay = MassAction(reactants=[high], products=[low], rate=rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.high
        self._target = self.low

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.high.energy - self.low.energy

    def _check(self, check_range=False):
        if self.energy_difference < 0:
            raise ValueError("High state energy must be higher than low state energy")
        if not _same_multiplicity(self.high, self.low):
            raise TypeError(
                "Both states must have the same multiplicity in an internal conversion"
            )
        if check_range:
            _check_range(self, "rate", 1e9, 1e11)


class Fluorescence(SpectroscopicSystem):
    """A radiative transition between two electronic states
    of the same spin multiplicity.
    """

    excited: SpinStateLike = initial(0.0, spin_multiplicity=None)
    ground: SpinStateLike = initial(0.0, spin_multiplicity=None)

    # timescale 10^-10 s and 10^-7 s.
    rate: Parameter = assign()

    radiative_decay = MassAction(reactants=[excited], products=[ground], rate=rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.excited
        self._target = self.ground

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self, check_range=False):
        if self.energy_difference <= 0:
            raise ValueError(
                "Excited state energy must be higher than ground state energy"
            )
        if not _same_multiplicity(self.excited, self.ground):
            raise TypeError(
                "Both states must have the same multiplicity in fluorescence, use Phosphorescence for radiative transitions with different spin multiplicity"
            )
        if check_range:
            _check_range(self, "rate", 1e7, 1e10)


class IntersystemCrossing(SpectroscopicSystem):
    """A non-radiative transition between two isoenergetic vibrational levels belonging
    to electronic states of different spin multiplicity, from singlet to triplet.
    """

    source: SingletStateLike = initial(0.0)
    target: TripletStateLike = initial(0.0, "triplet")

    # timescale 10^−8 s to 10^−3 s
    rate: Parameter = assign()

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

    def _check(self, check_range=False):
        if self.energy_difference < 0:
            raise ValueError(
                "Source energy must be higher than target's in Intersystem Crossing"
            )
        if check_range:
            _check_range(self, "rate", 1e8, 1e10)


class ReverseIntersystemCrossing(SpectroscopicSystem):
    """A non-radiative transition between two isoenergetic vibrational levels belonging
    to electronic states of different spin multiplicity, from triple to singlet.
    """

    source: TripletStateLike = initial(0.0, "triplet")
    target: SingletStateLike = initial(0.0)

    # timescale 10^−8 s to 10^−3 s
    rate: Parameter = assign()

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

    def _check(self, check_range=False):
        if self.energy_difference < 0:
            raise ValueError(
                "Source energy must be higher than target's in Intersystem Crossing"
            )
        if check_range:
            _check_range(
                self, "rate", 1e8, 1e10
            )  # TODO: above it says timescale to 10**-3 s, should it be S


class Phosphorescence(SpectroscopicSystem):
    """A radiative transition between two electronic
    states of different spin multiplicity.
    """

    excited: SpinStateLike = initial(0.0, spin_multiplicity=None)
    ground: SpinStateLike = initial(0.0, spin_multiplicity=None)

    # timescale 10^-6 s to 10 s range.
    rate: Parameter = assign()
    radiative_decay = MassAction(reactants=[excited], products=[ground], rate=rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = self.excited
        self._target = self.ground

    @property
    def energy_difference(self) -> pint.Quantity:
        return self.excited.energy - self.ground.energy

    def _check(self, check_range=False):
        if self.energy_difference <= 0:
            raise ValueError(
                "Excited state energy must be higher than ground state energy"
            )
        if not _different_multiplicity(self.excited, self.ground):
            raise TypeError(
                "Both states must have different multiplicity in phosphorescence, use Fluorescence for radiative transitions with the same spin multiplicity"
            )
        if check_range:
            _check_range(self, "rate", 1, 1e6)


class EnergyTransferUpconversion(SpectroscopicSystem):
    sensitizer_high: SpinStateLike = initial(
        0.0, spin_multiplicity=None
    )  # TODO: should this be a generic SpinState?
    sensitizer_low: SpinStateLike = initial(0.0, spin_multiplicity=None)

    activator_high: SpinStateLike = initial(0.0, spin_multiplicity=None)
    activator_low: SpinStateLike = initial(0.0, spin_multiplicity=None)

    rate: Parameter = assign()

    upconversion = MassAction(
        reactants=[sensitizer_high, activator_low],
        products=[sensitizer_low, activator_high],
        rate=rate,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = [self.sensitizer_high, self.activator_low]
        self._target = [self.sensitizer_low, self.activator_high]

    @property
    def energy_difference(self) -> pint.Quantity:
        return (
            self.sensitizer_high.energy
            + self.activator_low.energy
            - self.sensitizer_low.energy
            - self.activator_high.energy
        )

    def _check(self, check_range=False):
        pass
