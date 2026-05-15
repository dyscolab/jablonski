from itertools import chain
from typing import Iterable, Literal, Mapping, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Number = float | int
column = str


class Level(NamedTuple):
    label: str
    energy: Number
    column: str
    color: str = "black"
    linewidth: Number = 2.5


class Transition(NamedTuple):
    source: Level
    target: Level
    radiative: bool
    label: str | None = None
    color: str = "royalblue"
    linewidth: Number = 1


class JablonskiDiagram:
    def __init__(
        self,
        levels: Iterable[Level],
        transitions: Iterable[Transition],
        columns: Iterable[column] = ["singlet", "triplet"],
    ):
        self.levels = levels
        self.transitions = transitions
        self.columns = columns

    def _place_columns(
        self, columns: Iterable[column]
    ) -> Mapping[column, tuple[Number, Number]]:
        num_columns = len(columns)
        # Spaces between levels are 1/4 level width, calculated automatically to use all of xscale
        # level_lenght l is the solution to  n*l + (n-1)*l/4 = xscale, n = num_column
        level_length = self._xscale / (5 / 4 * num_columns - 1 / 4)
        interval_length = level_length / 4
        column_positions = {
            col: (
                i * interval_length + i * level_length,
                i * interval_length + (i + 1) * level_length,
            )
            for i, col in enumerate(columns)
        }
        return column_positions

    def _build_positions(
        self, column_positions: Mapping[column, tuple[Number, Number]]
    ) -> Mapping[column, tuple[tuple[Number, Number], Number]]:

        positions = {}
        for level in self.levels:
            positions[level] = (
                column_positions[level.column],
                level.energy,
            )
        return positions

    def _sort_transitions(
        self, column_positions: Mapping[column, tuple[Number, Number]]
    ) -> tuple[
        Mapping[column, Mapping[Literal["up", "down"], Iterable[Transition]]],
        Mapping[column, Mapping[Literal["forward", "backward"], Iterable[Transition]]],
    ]:
        intracolumn_transitions = {
            column: [
                t
                for t in self.transitions
                if t.source.column == column and t.target.column == column
            ]
            for column in self.columns
        }
        multicolumn_transitions = {
            column: [
                t
                for t in self.transitions
                if t.source.column == column and t.source.column != t.target.column
            ]
            for column in self.columns
        }
        sorted_intracolumn_transitions = {
            column: {
                "up": sorted(
                    [
                        t
                        for t in intracolumn_transitions[column]
                        if t.target.energy >= t.source.energy
                    ],
                    key=lambda t: (t.source.energy, -t.target.energy),
                ),
                "down": sorted(
                    [
                        t
                        for t in intracolumn_transitions[column]
                        if t.target.energy < t.source.energy
                    ],
                    key=lambda t: (t.target.energy, -t.source.energy),
                ),
            }
            for column in self.columns
        }
        sorted_multicolumn_transitions = {
            column: {
                "backward": [
                    t
                    for t in multicolumn_transitions[column]
                    if column_positions[t.source.column][0]
                    >= column_positions[t.target.column][0]
                ],
                "forward": [
                    t
                    for t in multicolumn_transitions[column]
                    if column_positions[t.source.column][0]
                    < column_positions[t.target.column][0]
                ],
            }
            for column in self.columns
        }
        return sorted_intracolumn_transitions, sorted_multicolumn_transitions

    def plot(
        self,
        figsize: tuple[Number, Number] = (
            6.4,
            4.8,
        ),
        fontsize: Number = 10,
        show_energy_axis: bool = True,
    ) -> tuple[Axes, Figure]:

        # Use energy ranges and figure size
        energies = [level.energy for level in self.levels]
        max_energy = np.max(energies)
        min_energy = np.min(energies)
        self._yscale = max_energy - min_energy
        self._xscale = self._yscale * figsize[0] / figsize[1]

        column_positions = self._place_columns(self.columns)
        positions = self._build_positions(column_positions)
        intracolumn, multicolumn = self._sort_transitions(column_positions)

        fig, ax = plt.subplots(figsize=figsize)

        # Draw energy levels
        for level, ((x_start, x_end), y) in positions.items():
            ax.hlines(
                y,
                x_start,
                x_end,
                linewidth=level.linewidth,
                color=level.color,
            )

            ax.text(
                x_end + 0.01,
                y,
                level.label,
                va="center",
                fontsize=fontsize,
            )

        # Draw  transitions
        for column in self.columns:
            x_start, x_end = column_positions[column]
            n_intra = len(intracolumn[column]["up"]) + len(intracolumn[column]["down"])
            spacing = (x_end - x_start) / (n_intra + 1)

            # draw intracolumn transitions
            for i, transition in enumerate(
                chain(intracolumn[column]["up"], intracolumn[column]["down"])
            ):
                self._draw_transition(
                    ax,
                    (
                        x_start + (i + 1) * spacing,
                        transition.source.energy,
                    ),
                    (
                        x_start + (i + 1) * spacing,
                        transition.target.energy,
                    ),
                    label=transition.label,
                    radiative=transition.radiative,
                    color=transition.color,
                    linewidth=transition.linewidth,
                )
            for transition in multicolumn[column]["backward"]:
                self._draw_transition(
                    ax,
                    (x_start, transition.source.energy),
                    (
                        column_positions[transition.target.column][1],
                        transition.target.energy,
                    ),
                    label=transition.label,
                    radiative=transition.radiative,
                    color=transition.color,
                    linewidth=transition.linewidth,
                )
            for transition in multicolumn[column]["forward"]:
                self._draw_transition(
                    ax,
                    (x_end, transition.source.energy),
                    (
                        column_positions[transition.target.column][0],
                        transition.target.energy,
                    ),
                    label=transition.label,
                    radiative=transition.radiative,
                    color=transition.color,
                    linewidth=transition.linewidth,
                )

        # Formatting
        ax.set_xlim(-0.05 * self._xscale, self._xscale * 1.1)
        ax.set_xticks([])

        ax.set_ylim(
            min_energy - self._yscale * 0.05,
            max_energy + self._yscale * 0.05,
        )

        ax.xaxis.set_visible(False)

        if show_energy_axis:
            ax.set_ylabel("Energy")
            ax.set_yticks(energies)
        else:
            ax.set_yticks([])

        fig.tight_layout()
        return fig, ax

    def _draw_transition(
        self,
        ax: Axes,
        start: tuple[Number, Number],
        end: tuple[Number, Number],
        label: str | None,
        radiative: bool,
        color="royalblue",
        linewidth: Number = 1,
    ):
        x1, y1 = start
        x2, y2 = end

        if radiative:
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    linestyle="-",
                    color=color,
                    lw=linewidth,
                ),
            )

        else:
            # "Squiggly" lines are a straight line with a pieceswise modulation: 1. sine wave, 2: interpolating spline, 3: straight line.
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)

            if length == 0:
                return

            px = -dy / length
            py = dx / length
            amplitude = 0.01 * (self._xscale * np.abs(px) + self._yscale * np.abs(py))
            frequency = 12

            # Define the time splits (t goes from 0 to 1)
            t1 = 0.85  # Spline start
            t2 = 0.90  # Straight line start

            t_sine = np.linspace(0, t1, 200)
            wiggle_sine = amplitude * np.sin(2 * np.pi * frequency * t_sine)

            # Boundary conditions for spline
            w1 = amplitude * np.sin(2 * np.pi * frequency * t1)
            dw1_dt = (
                amplitude * (2 * np.pi * frequency) * np.cos(2 * np.pi * frequency * t1)
            )

            # Calculate spline
            t_blend = np.linspace(t1, t2, 50)
            u = (t_blend - t1) / (t2 - t1)

            c0 = w1
            c1 = dw1_dt * (t2 - t1)
            c2 = -3 * c0 - 2 * c1
            c3 = 2 * c0 + c1
            wiggle_blend = c0 + c1 * u + c2 * u**2 + c3 * u**3

            # Straight line
            t_straight = np.linspace(t2, 1.0, 50)
            wiggle_straight = np.zeros_like(t_straight)

            t_all = np.concatenate([t_sine, t_blend[1:], t_straight[1:]])
            wiggle_all = np.concatenate(
                [wiggle_sine, wiggle_blend[1:], wiggle_straight[1:]]
            )

            x_wiggle = x1 + dx * t_all + px * wiggle_all
            y_wiggle = y1 + dy * t_all + py * wiggle_all

            ax.plot(
                x_wiggle,
                y_wiggle,
                color=color,
                lw=linewidth,
            )

            # Arrow head
            ax.annotate(
                "",
                xy=(x_wiggle[-1], y_wiggle[-1]),
                xytext=(x_wiggle[-5], y_wiggle[-5]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=linewidth,
                ),
            )

            ax.plot(
                x_wiggle,
                y_wiggle,
                color=color,
                lw=linewidth,
            )

            # Arrow head (now perfectly tracking the final straight segment)
            ax.annotate(
                "",
                xy=(x_wiggle[-1], y_wiggle[-1]),
                xytext=(x_wiggle[-5], y_wiggle[-5]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=linewidth,
                ),
            )
        if label:
            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2
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
