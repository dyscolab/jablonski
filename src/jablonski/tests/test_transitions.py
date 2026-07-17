from poincare import Simulator

from jablonski.states import SpectroscopicSystem, initial

from .._units import ureg
from ..transitions import (
    Absorption,
    EnergyTransferUpconversion,
    Fluorescence,
    InternalConversion,
    IntersystemCrossing,
    Phosphorescence,
    ReverseIntersystemCrossing,
    TripletTripletAbsorption,
    VibrationalRelaxation,
)


def test_absorption():

    class Correct(SpectroscopicSystem):
        ground = initial(1 * ureg.eV, "singlet", default=10)
        excited = initial(2 * ureg.eV, "singlet", default=10)

        transition = Absorption(ground=ground, excited=excited, rate=1e-15 * ureg.cm**2)

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            ground = initial(1 * ureg.eV, "triplet", default=10)
            excited = initial(2 * ureg.eV, "singlet", default=10)

            transition = Absorption(
                ground=ground, excited=excited, rate=1e-15 * ureg.cm**2
            )
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            ground = initial(2 * ureg.eV, "singlet", default=10)
            excited = initial(1 * ureg.eV, "singlet", default=10)

            transition = Absorption(
                ground=ground, excited=excited, rate=1e-15 * ureg.cm**2
            )
    except ValueError:
        pass
    else:
        assert False


def test_triplet_triplet_absorption():
    class Correct(SpectroscopicSystem):
        ground = initial(1 * ureg.eV, "triplet", default=10)
        excited = initial(2 * ureg.eV, "triplet", default=10)

        transition = TripletTripletAbsorption(
            ground=ground, excited=excited, rate=1e15 * ureg.cm**2
        )

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            ground = initial(1 * ureg.eV, "singlet", default=10)
            excited = initial(2 * ureg.eV, "triplet", default=10)

            transition = TripletTripletAbsorption(
                ground=ground, excited=excited, rate=1e-15 * ureg.cm**2
            )
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            ground = initial(2 * ureg.eV, "triplet", default=10)
            excited = initial(1 * ureg.eV, "triplet", default=10)

            transition = TripletTripletAbsorption(
                ground=ground, excited=excited, rate=1e-15 * ureg.cm**2
            )
    except ValueError:
        pass
    else:
        assert False


def test_vibrational_relaxation():
    class Correct(SpectroscopicSystem):
        high = initial(2 * ureg.eV, "singlet", default=10)
        low = initial(1 * ureg.eV, "singlet", default=10)

        transition = VibrationalRelaxation(high=high, low=low, rate=1e12 / ureg.s)

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            high = initial(2 * ureg.eV, "singlet", default=10)
            low = initial(1 * ureg.eV, "triplet", default=10)

            transition = VibrationalRelaxation(high=high, low=low, rate=1e12 / ureg.s)
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            high = initial(1 * ureg.eV, "singlet", default=10)
            low = initial(2 * ureg.eV, "singlet", default=10)

            transition = VibrationalRelaxation(high=high, low=low, rate=1e12 / ureg.s)
    except ValueError:
        pass
    else:
        assert False


def test_internal_conversion():
    class Correct(SpectroscopicSystem):
        high = initial(2 * ureg.eV, "singlet", default=10)
        low = initial(1 * ureg.eV, "singlet", default=10)

        transition = InternalConversion(high=high, low=low, rate=1e11 / ureg.s)

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            high = initial(2 * ureg.eV, "singlet", default=10)
            low = initial(1 * ureg.eV, "triplet", default=10)

            transition = InternalConversion(high=high, low=low, rate=1e11 / ureg.s)
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            high = initial(1 * ureg.eV, "singlet", default=10)
            low = initial(2 * ureg.eV, "singlet", default=10)

            transition = InternalConversion(high=high, low=low, rate=1e11 / ureg.s)
    except ValueError:
        pass
    else:
        assert False


def test_fluorescence():
    class Correct(SpectroscopicSystem):
        excited = initial(2 * ureg.eV, "singlet", default=10)
        ground = initial(1 * ureg.eV, "singlet", default=10)

        transition = Fluorescence(excited=excited, ground=ground, rate=1e10 / ureg.s)

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            excited = initial(2 * ureg.eV, "singlet", default=10)
            ground = initial(1 * ureg.eV, "triplet", default=10)

            transition = Fluorescence(
                excited=excited, ground=ground, rate=1e10 / ureg.s
            )
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            excited = initial(1 * ureg.eV, "singlet", default=10)
            ground = initial(2 * ureg.eV, "singlet", default=10)

            transition = Fluorescence(
                excited=excited, ground=ground, rate=1e10 / ureg.s
            )
    except ValueError:
        pass
    else:
        assert False


def test_intersystem_crossing():
    class Correct(SpectroscopicSystem):
        source = initial(2 * ureg.eV, "singlet", default=10)
        target = initial(2 * ureg.eV, "triplet", default=10)

        transition = IntersystemCrossing(
            source=source, target=target, rate=1e8 / ureg.s
        )

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            source = initial(2 * ureg.eV, "triplet", default=10)
            target = initial(2 * ureg.eV, "triplet", default=10)

            transition = IntersystemCrossing(
                source=source, target=target, rate=1e8 / ureg.s
            )
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            source = initial(1 * ureg.eV, "singlet", default=10)
            target = initial(2 * ureg.eV, "triplet", default=10)

            transition = IntersystemCrossing(
                source=source, target=target, rate=1e8 / ureg.s
            )
    except ValueError:
        pass
    else:
        assert False


def test_reverse_intersystem_crossing():
    class Correct(SpectroscopicSystem):
        source = initial(2 * ureg.eV, "triplet", default=10)
        target = initial(2 * ureg.eV, "singlet", default=10)

        transition = ReverseIntersystemCrossing(
            source=source, target=target, rate=1e8 / ureg.s
        )

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            source = initial(2 * ureg.eV, "singlet", default=10)
            target = initial(2 * ureg.eV, "singlet", default=10)

            transition = ReverseIntersystemCrossing(
                source=source, target=target, rate=1e8 / ureg.s
            )
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            source = initial(1 * ureg.eV, "triplet", default=10)
            target = initial(2 * ureg.eV, "singlet", default=10)

            transition = ReverseIntersystemCrossing(
                source=source, target=target, rate=1e8 / ureg.s
            )
    except ValueError:
        pass
    else:
        assert False


def test_phosphorescence():
    class Correct(SpectroscopicSystem):
        excited = initial(2 * ureg.eV, "triplet", default=10)
        ground = initial(1 * ureg.eV, "singlet", default=10)

        transition = Phosphorescence(excited=excited, ground=ground, rate=1e6 / ureg.s)

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
    try:

        class Incorrect1(SpectroscopicSystem):
            excited = initial(2 * ureg.eV, "singlet", default=10)
            ground = initial(1 * ureg.eV, "singlet", default=10)

            transition = Phosphorescence(
                excited=excited, ground=ground, rate=1e6 / ureg.s
            )
    except TypeError:
        pass
    else:
        assert False

    try:

        class Incorrect2(SpectroscopicSystem):
            excited = initial(1 * ureg.eV, "triplet", default=10)
            ground = initial(2 * ureg.eV, "singlet", default=10)

            transition = Phosphorescence(
                excited=excited, ground=ground, rate=1e6 / ureg.s
            )
    except ValueError:
        pass
    else:
        assert False


def test_energy_transfer_upconversion():
    class Correct(SpectroscopicSystem):
        sensitizer = initial(2 * ureg.eV, "singlet", default=10)
        activator = initial(3 * ureg.eV, "singlet", default=10)
        relaxator = initial(1 * ureg.eV, "singlet", default=10)

        transition = EnergyTransferUpconversion(
            sensitizer=sensitizer,
            activator=activator,
            relaxator=relaxator,
            rate=1e10 / ureg.s,
        )

    sim = Simulator(Correct)
    sim.solve(save_at=[1, 2, 3])
