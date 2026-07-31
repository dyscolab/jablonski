import pint
import numpy as np

from typing import Literal

from ._typing import Pumper, Excitation
from ._units import ureg
from .states import SpectroscopicSystem

c = 1 * ureg.speed_of_light
h = 1 * ureg.planck_constant  # Not reduced, h not h bar


def pump_from_laser(
    system: SpectroscopicSystem,
    wavelength: pint.Quantity,
    linewidth: pint.Quantity,
    power: pint.Quantity,
    width: pint.Quantity,
) -> Excitation:
    """Creates excitation affecting all pumpers whose corresponding wavelenght is within linewidth of laser wavelenght.
    Photon flux is calculated assuming it is at the peak of a gaussian beam of the corresponding power."""
    affected_pumpers = []
    for pumper in system._yield(Pumper):
        pumper_wavelenght = c * h / (pumper.energy_difference)
        if (
            np.abs((wavelength - pumper_wavelenght).to(ureg.nm).magnitude)
            <= linewidth.to(ureg.nm).magnitude
        ):
            affected_pumpers.append(pumper)

    photon_energy = c / wavelength * h
    total_photons_per_second = power / photon_energy
    photon_flux_at_peak = total_photons_per_second / (2 * np.pi * width**2)

    return {pumper: photon_flux_at_peak for pumper in affected_pumpers}
