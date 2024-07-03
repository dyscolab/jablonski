"""
    jablonski._units
    ~~~~~~~~~~~~~~~~

    Units and dimensions.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import pint

ureg = pint.get_application_registry()

DEFAULT_DELTA = 50 * ureg.picosecond

DIM_WAVELENGTH = {"[length]": 1}
DIM_WAVENUMBER = {"[length]": -1}
DIM_FREQUENCY = {"[time]": -1}
DIM_ENERGY = ureg.get_dimensionality("eV")
