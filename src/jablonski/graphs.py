from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pint
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ._typing import Pumper
from ._units import ureg
from .simulation import widened_emission_spectra
from .states import SpectroscopicSystem
from .util import SpectraKind


def graph_spectra(
    system: SpectroscopicSystem,
    excitation_transition: Pumper | Iterable[Pumper],
    height: float,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
    samples: Iterable[float] = np.linspace(380, 700, 1000),
    width: float = 5,  # TODO: what is the right width?
):
    spectra = widened_emission_spectra(
        system, excitation_transition, height, unit, kind, samples, width=width
    )

    points = spectra["wavelenght"].values
    spectrum = spectra.values
    plot_points = np.array([points, spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([plot_points[:-1], plot_points[1:]], axis=1)
    lc = LineCollection(segments, cmap="nipy_spectral")
    lc.set_array(points)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_xlim(points.min(), points.max())
    ax.set_ylim(spectrum.min(), spectrum.max())
    ax.set_xlabel(f"Wavelenght [ {unit} ]")
    ax.set_ylabel("Emission [ photons/s ]")
    return fig, ax


class JablonskiDiagram:
    def __init__(
        self, singlets, triplets, radiative=None, non_radiative=None, aspect_ratio=2
    ):
        self.singlets = singlets
        self.triplets = triplets
        self.radiative = radiative or []
        self.non_radiative = non_radiative or []

        self.positions = {}

        self.energies = list(self.singlets.values()) + list(self.triplets.values())
        self.max_energy = np.max(self.energies)
        self.min_energy = np.min(self.energies)
        self.yscale = self.max_energy - self.min_energy
        self.xscale = self.yscale / aspect_ratio

    def _build_positions(self):
        x_singlet = self.xscale * 0.4
        x_triplet = self.xscale * 1.2

        for label, energy in self.singlets.items():
            self.positions[label] = (x_singlet, energy)

        for label, energy in self.triplets.items():
            self.positions[label] = (x_triplet, energy)

    def plot(
        self,
        figsize=(6, 8),  # TODO: how to set with aspect ratio?
        level_width=0.4,  # TODO: major and minor levels with dirfferent linewidths?
        fontsize=12,
        show_energy_axis=True,
    ):
        level_width = level_width * self.xscale
        self._build_positions()

        fig, ax = plt.subplots(figsize=figsize)

        # Draw energy levels
        for label, (x, y) in self.positions.items():
            ax.hlines(
                y,
                x - level_width / 2,
                x + level_width / 2,
                linewidth=3,
                color="black",
            )

            ax.text(
                x + level_width / 2 + 0.05,
                y,
                label,
                va="center",
                fontsize=fontsize,
            )

        # Draw radiative transitions
        for start, end, label in self.radiative:
            self._draw_transition(
                ax,
                start,
                end,
                label=label,
                linestyle="-",
                color="blue",
            )

        # Draw non-radiative transitions
        for start, end, label in self.non_radiative:
            self._draw_transition(
                ax,
                start,
                end,
                label=label,
                linestyle="--",
                color="green",
            )

        # Formatting
        ax.set_xlim(0, self.yscale)
        ax.set_xticks([])

        ax.set_ylim(
            self.min_energy - self.yscale * 0.05,
            self.max_energy + self.yscale * 0.05,
        )

        if show_energy_axis:
            ax.set_ylabel("Energy")
            ax.set_yticks(self.energies)
        else:
            ax.set_yticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        plt.tight_layout()
        return fig, ax

    def _draw_transition(
        self,
        ax: Axes,
        start: float,
        end: float,
        label: str | None,
        linestyle: str = "-",
        color="blue",
    ):
        x1, y1 = self.positions[start]
        x2, y2 = self.positions[end]

        if linestyle == "-":
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    linestyle="-",
                    color=color,
                    lw=2,
                    shrinkA=10,
                    shrinkB=10,
                ),
            )

        else:
            t = np.linspace(0, 1, 300)

            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t

            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)

            if length == 0:
                return

            px = -dy / length
            py = dx / length

            amplitude = 0.015 * self.xscale
            frequency = 12

            wiggle = amplitude * np.sin(2 * np.pi * frequency * t)

            x_wiggle = x + px * wiggle
            y_wiggle = y + py * wiggle

            ax.plot(
                x_wiggle,
                y_wiggle,
                color=color,
                lw=2,
            )

            # Arrow head
            ax.annotate(
                "",
                xy=(x_wiggle[-1], y_wiggle[-1]),
                xytext=(x_wiggle[-5], y_wiggle[-5]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=2,
                ),
            )
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2

        if label:
            ax.text(
                xm,
                ym,
                label,
                fontsize=10,
                color=color,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )


# @dataclass(frozen=True)
# class Level:
#     label: str
#     energy: Number
#     column: str

# Number = float | int


# @dataclass(frozen=True)
# class Transition:
#     source: Level
#     target: Level
#     radiative: bool
#     label: str | None = None


# class JablonskiDiagram:
#     def __init__(
#         self,
#         levels: Iterable[Level],
#         transitions: Iterable[Transition],
#         aspect_ratio: Number = 2,
#         columns: Iterable[str] = ["singlet", "triplet"],
#     ):
#         self.levels = levels
#         self.transitions = transitions
#         self.aspect_ratio = aspect_ratio

#         self.positions = self._build_positions()
#         self.intracolumn, self.multicolumn = self._sort_transitions()

#         self.energies = [level.energy for level in self.levels]
#         self.max_energy = np.max(self.energies)
#         self.min_energy = np.min(self.energies)
#         self.yscale = self.max_energy - self.min_energy
#         self.xscale = self.yscale / aspect_ratio

#     def _build_positions(self):
#         # TODO: adapt for more columns than triplet and singlet
#         x_singlet = self.xscale * 0.4
#         x_triplet = self.xscale * 0.8
#         positions = {}
#         for level in self.levels:
#             positions[level.label] = (
#                 x_singlet if level.column == "singlet" else x_triplet,
#                 level.energy,
#             )

#     def _sort_transitions(self, transitions):
#         intracolumn_transitions = {
#             column: [
#                 t
#                 for t in transitions
#                 if t.source.column == column and t.source.column == column
#             ]
#             for column in self.columns
#         }
#         multicolumn_transitions = {
#             column: [
#                 t
#                 for t in transitions
#                 if t.source.column == column and t.source.column != t.source.column
#             ]
#             for column in self.columns
#         }
#         sorted_intracolumn_transitions = {
#             column: {
#                 "up": sorted(
#                     [
#                         t
#                         for t in intracolumn_transitions[column]
#                         if t.target.energy >= t.source.energy
#                     ],
#                     key=lambda t: (-t.source.energy, t.target.energy),
#                 ),
#                 "down": sorted(
#                     [
#                         t
#                         for t in intracolumn_transitions[column]
#                         if t.target.energy < t.source.energy
#                     ],
#                     key=lambda t: (-t.source.energy, t.target.energy),
#                 ),
#             }
#             for column in self.columns
#         }
#         return sorted_intracolumn_transitions, multicolumn_transitions

#     def plot(
#         self,
#         figsize=(5, 8),  # TODO: how to set with aspect ratio?
#         level_width=0.4,  # TODO: major and minor levels with dirfferent linewidths?
#         fontsize=12,
#         show_energy_axis=True,
#     ):
#         level_width = level_width * self.xscale
#         self._build_positions()

#         fig, ax = plt.subplots(figsize=figsize)

#         # Draw energy levels
#         for label, (x, y) in self.positions.items():
#             ax.hlines(
#                 y,
#                 x - level_width / 2,
#                 x + level_width / 2,
#                 linewidth=3,
#                 color="black",
#             )

#             ax.text(
#                 x + level_width / 2 + 0.05,
#                 y,
#                 label,
#                 va="center",
#                 fontsize=fontsize,
#             )

#         # Draw intracolumn transitions
#         for column in self.columns:


#         # Draw radiative transitions
#         # for start, end, label in self.radiative:
#         #     self._draw_transition(
#         #         ax,
#         #         start,
#         #         end,
#         #         label=label,
#         #         linestyle="-",
#         #         color="blue",
#         #     )

#         # # Draw non-radiative transitions
#         # for start, end, label in self.non_radiative:
#         #     self._draw_transition(
#         #         ax,
#         #         start,
#         #         end,
#         #         label=label,
#         #         linestyle="--",
#         #         color="green",
#         #     )

#         # Formatting
#         ax.set_xlim(0, self.yscale)
#         ax.set_xticks([])

#         ax.set_ylim(
#             self.min_energy - self.yscale * 0.05,
#             self.max_energy + self.yscale * 0.05,
#         )

#         # ax.xaxis.set_visible(False)

#         if show_energy_axis:
#             ax.set_ylabel("Energy")
#             ax.set_yticks(self.energies)
#         else:
#             ax.set_yticks([])

#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)

#         plt.tight_layout()
#         return fig, ax

#     def _draw_transition(
#         self,
#         ax: Axes,
#         start: float,
#         end: float,
#         label: str | None,
#         radiative: bool,
#         color="blue",
#     ):
#         x1, y1 = self.positions[start]
#         x2, y2 = self.positions[end]

#         if linestyle == "-":
#             ax.annotate(
#                 "",
#                 xy=(x2, y2),
#                 xytext=(x1, y1),
#                 arrowprops=dict(
#                     arrowstyle="->",
#                     linestyle="-",
#                     color=color,
#                     lw=2,
#                     shrinkA=10,
#                     shrinkB=10,
#                 ),
#             )

#         else:
#             t = np.linspace(0, 1, 300)

#             x = x1 + (x2 - x1) * t
#             y = y1 + (y2 - y1) * t

#             dx = x2 - x1
#             dy = y2 - y1
#             length = np.hypot(dx, dy)

#             if length == 0:
#                 return

#             px = -dy / length
#             py = dx / length

#             amplitude = 0.015 * self.xscale
#             frequency = 12

#             wiggle = amplitude * np.sin(2 * np.pi * frequency * t)

#             x_wiggle = x + px * wiggle
#             y_wiggle = y + py * wiggle

#             ax.plot(
#                 x_wiggle,
#                 y_wiggle,
#                 color=color,
#                 lw=2,
#             )

#             # Arrow head
#             ax.annotate(
#                 "",
#                 xy=(x_wiggle[-1], y_wiggle[-1]),
#                 xytext=(x_wiggle[-5], y_wiggle[-5]),
#                 arrowprops=dict(
#                     arrowstyle="->",
#                     color=color,
#                     lw=2,
#                 ),
#             )
#         xm = (x1 + x2) / 2
#         ym = (y1 + y2) / 2

#         if label:
#             ax.text(
#                 xm,
#                 ym,
#                 label,
#                 fontsize=10,
#                 color=color,
#                 bbox=dict(
#                     facecolor="white",
#                     edgecolor="none",
#                     alpha=0.7,
#                 ),
#             )
