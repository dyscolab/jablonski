"""
    jablonski
    ~~~~~~~~~

    Write and simulate ODE systems to describe the transitions in spectroscopic
    molecular systems.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from poincare import Parameter, assign

from . import transitions, util
from .states import SingletState, SpectroscopicSystem, TripletState, initial

__all__ = [
    "transitions",
    "util",
    "SpectroscopicSystem",
    "initial",
    "TripletState",
    "SingletState",
    "assign",
    "Parameter",
]
