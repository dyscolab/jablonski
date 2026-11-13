import numpy as np
import pytest
from poincare import Parameter, Simulator, assign
from poincare.solvers import LSODA

from jablonski.states import ConstantGroundState, SpectroscopicSystem, initial

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
    TripletTripletAnnihilation,
    VibrationalRelaxation,
)


def test_absorption():

    class Correct(SpectroscopicSystem):
        ground = initial(1 * ureg.eV, "singlet", default=10)
        excited = initial(2 * ureg.eV, "singlet", default=10)

        transition = Absorption(ground=ground, excited=excited, rate=1e-15 * ureg.cm**2)

    sim = Simulator(Correct)
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)
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
        sensitizer_high = initial(2 * ureg.eV, "singlet", default=10)
        sensitizer_low = initial(1 * ureg.eV, "singlet", default=10)
        activator_low = initial(2 * ureg.eV, "singlet", default=10)
        activator_high = initial(3 * ureg.eV, "singlet", default=10)

        transition = EnergyTransferUpconversion(
            sensitizer_high=sensitizer_high,
            sensitizer_low=sensitizer_low,
            activator_low=activator_low,
            activator_high=activator_high,
            rate=1e10 / ureg.s,
        )

    sim = Simulator(Correct)
    sim.solve(save_at=np.array([1, 2, 3]) * ureg.us)


def test_triplet_triplet_annihilation():
    # Same state twice: 2 T -> 2 S
    class Same(SpectroscopicSystem):
        t1 = initial(2 * ureg.eV, "triplet", default=10)
        s1 = initial(1 * ureg.eV, "singlet", default=0)

        transition = TripletTripletAnnihilation(
            triplet_1=t1, triplet_2=t1, singlet_1=s1, singlet_2=s1, rate=0.1 / ureg.s
        )

    assert Same.transition.energy_difference == 2 * ureg.eV

    # dT/dt = -2 k T**2, so T = T0 / (1 + 2 k T0 t)
    t = np.linspace(0, 2, 9) * ureg.s
    solver = LSODA(rtol=1e-8, atol=1e-10)
    result = Simulator(Same).with_solver(solver).solve(save_at=t)
    triplet = 10 / (1 + 2 * 0.1 * 10 * t.m)
    np.testing.assert_allclose(result["t1"].values, triplet, rtol=1e-5)
    np.testing.assert_allclose(result["s1"].values, 10 - triplet, rtol=1e-5, atol=1e-8)

    # Different levels: T1 + T1 -> S1 + S0, with S1 above T1 (upconversion)
    class Upconversion(SpectroscopicSystem):
        s0 = initial(0 * ureg.eV, "singlet", default=0)
        s1 = initial(3 * ureg.eV, "singlet", default=0)
        t1 = initial(2 * ureg.eV, "triplet", default=10)

        transition = TripletTripletAnnihilation(
            triplet_1=t1, triplet_2=t1, singlet_1=s1, singlet_2=s0, rate=0.1 / ureg.s
        )

    assert Upconversion.transition.energy_difference == 1 * ureg.eV
    result = Simulator(Upconversion).with_solver(solver).solve(save_at=t)
    np.testing.assert_allclose(result["t1"].values, triplet, rtol=1e-5)
    # Each annihilation consumes two triplets and makes one S1 and one S0
    for name in ["s0", "s1"]:
        np.testing.assert_allclose(
            result[name].values, (10 - triplet) / 2, rtol=1e-5, atol=1e-8
        )

    # Two different triplets: dTa/dt = dTb/dt = -k Ta Tb
    class TwoTriplets(SpectroscopicSystem):
        ta = initial(2 * ureg.eV, "triplet", default=10)
        tb = initial(2.5 * ureg.eV, "triplet", default=10)
        s0 = initial(0 * ureg.eV, "singlet", default=0)
        s1 = initial(3 * ureg.eV, "singlet", default=0)

        transition = TripletTripletAnnihilation(
            triplet_1=ta, triplet_2=tb, singlet_1=s1, singlet_2=s0, rate=0.1 / ureg.s
        )

    result = Simulator(TwoTriplets).with_solver(solver).solve(save_at=t)
    expected = 10 / (1 + 0.1 * 10 * t.m)
    for name in ["ta", "tb"]:
        np.testing.assert_allclose(result[name].values, expected, rtol=1e-5)

    with pytest.raises(ValueError):

        class EndothermicAnnihilation(SpectroscopicSystem):
            t1 = initial(1 * ureg.eV, "triplet", default=10)
            s0 = initial(0 * ureg.eV, "singlet", default=0)
            s1 = initial(3 * ureg.eV, "singlet", default=0)

            transition = TripletTripletAnnihilation(
                triplet_1=t1,
                triplet_2=t1,
                singlet_1=s1,
                singlet_2=s0,
                rate=0.1 / ureg.s,
            )

    with pytest.raises(TypeError):

        class WrongMultiplicity(SpectroscopicSystem):
            t1 = initial(2 * ureg.eV, "singlet", default=10)
            s1 = initial(1 * ureg.eV, "singlet", default=0)

            transition = TripletTripletAnnihilation(
                triplet_1=t1,
                triplet_2=t1,
                singlet_1=s1,
                singlet_2=s1,
                rate=0.1 / ureg.s,
            )


def test_constant_ground_state():
    class Model(SpectroscopicSystem):
        ground = ConstantGroundState(default=10)
        s1 = initial(3 * ureg.eV, "singlet", default=0)
        t1 = initial(2 * ureg.eV, "triplet", default=0)

        absorption = Absorption(
            ground=ground,
            excited=s1,
            rate=1 * ureg.cm**2,
            pump=2 / (ureg.cm**2 * ureg.s),
        )
        # Same and different multiplicity checks both accept the constant state
        fluorescence = Fluorescence(excited=s1, ground=ground, rate=5 / ureg.s)
        isc = IntersystemCrossing(source=s1, target=t1, rate=1 / ureg.s)
        phosphorescence = Phosphorescence(excited=t1, ground=ground, rate=1 / ureg.s)
        relaxation = InternalConversion(high=t1, low=ground, rate=1 / ureg.s)
        tta = TripletTripletAbsorption(ground=ground, excited=t1, rate=0 * ureg.cm**2)

    assert Model.ground.energy == 0 * ureg.eV
    assert Model.absorption.ground is Model.ground
    assert Model.phosphorescence.energy_difference == 2 * ureg.eV

    sim = Simulator(Model)
    assert sim.compiled.variables == [Model.s1, Model.t1]

    # ds1/dt = pump * rate * ground - (k_f + k_isc) * s1
    t = np.linspace(0, 2, 9) * ureg.s
    result = sim.with_solver(LSODA(rtol=1e-8, atol=1e-10)).solve(save_at=t)
    s1 = 20 / 6 * (1 - np.exp(-6 * t.m))
    np.testing.assert_allclose(result["s1"].values, s1, rtol=1e-5, atol=1e-8)

    # The population of the constant state can be changed as a parameter
    result = sim.with_values({Model.ground: 20}).solve(save_at=t)
    np.testing.assert_allclose(result["s1"].values, 2 * s1, rtol=1e-3)


def test_constant_ground_state_energy():
    state = ConstantGroundState(default=1, energy=500 * ureg.nm)
    np.testing.assert_allclose(state.energy.to("eV").m, 2.4797, rtol=1e-4)

    with pytest.raises(ValueError):
        ConstantGroundState(default=1, energy=1 * ureg.s)

    with pytest.raises(ValueError):

        class Incorrect(SpectroscopicSystem):
            ground = ConstantGroundState(default=1, energy=3)
            excited = initial(2 * ureg.eV, "singlet", default=0)

            transition = Absorption(ground=ground, excited=excited, rate=1 * ureg.cm**2)


def test_parameter_is_not_a_state():
    with pytest.raises(TypeError):

        class Incorrect(SpectroscopicSystem):
            ground: Parameter = assign(default=1)
            excited = initial(2 * ureg.eV, "singlet", default=0)

            transition = Absorption(ground=ground, excited=excited, rate=1 * ureg.cm**2)


def _range_model(fluorescence_rate, absorption_rate=1e-15 * ureg.cm**2):
    class Model(SpectroscopicSystem):
        k: Parameter = assign(default=fluorescence_rate)
        S0 = initial(0 * ureg.eV, "singlet", default=1)
        S1 = initial(2 * ureg.eV, "singlet", default=0)
        absorption = Absorption(ground=S0, excited=S1, rate=absorption_rate)
        fluorescence = Fluorescence(excited=S1, ground=S0, rate=k)

    return Model


def test_check_all_in_range():
    import warnings

    from ..transitions import check_all

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_all(_range_model(1e8 / ureg.s))
        # Converted to 1/s before comparing: 1e5 / ms = 1e8 / s
        check_all(_range_model(1e5 / ureg.ms))
        # Absorption limits are cross-sections in cm²: 1e-19 m² = 1e-15 cm²
        check_all(_range_model(1e8 / ureg.s, absorption_rate=1e-19 * ureg.m**2))
        # Rates that are expressions of other components are not checked
        check_all(_range_model(2 * Parameter(default=1e3 / ureg.s)))


def test_check_all_out_of_range():
    from ..transitions import check_all

    with pytest.warns(UserWarning, match="fluorescence rate"):
        check_all(_range_model(1e3 / ureg.s))
    with pytest.warns(UserWarning, match="absorption rate"):
        check_all(_range_model(1e8 / ureg.s, absorption_rate=1e-10 * ureg.cm**2))


def test_check_all_nested():
    from ..transitions import check_all

    Inner = _range_model(1e3 / ureg.s)

    class Outer(SpectroscopicSystem):
        inner = Inner()

    with pytest.warns(UserWarning, match="fluorescence rate"):
        check_all(Outer)
