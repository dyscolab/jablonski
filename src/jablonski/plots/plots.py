from io import StringIO
from itertools import cycle
from collections.abc import Sequence, Callable, Iterable, Mapping

from cycler import Cycler
import matplotlib.pyplot as plt
import numpy as np
import pint
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from poincare.printing.latex import Latex, ToLatex, default_packages, default_sections
from poincare.printing.latex import model_report as _model_report
from poincare.simulator import Simulator

from .._typing import Drawable, Pumper, RadiativeDecay, Excitation
from .._units import ureg
from ..simulation import widened_emission_spectra
from ..states import SpectroscopicSystem, SpinState
from ..transitions import EnergyTransferUpconversion, EnergyTransferUpconversion4
from ..util import SpectraKind
from .jablonski_diagrams import (
    ETU_COLORS,
    JablonskiDiagram,
    Level,
    Number,
    Transition,
)


def graph_spectra(
    sim: Simulator,
    excitation: Excitation,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
    samples: Iterable[float] = np.linspace(380, 700, 1000),
    width: float = 5,  # TODO: what is the right width?
    figsize: tuple[Number, Number] = (6.4, 4.8),
):
    spectra = widened_emission_spectra(
        sim, excitation, unit, kind, samples, width=width
    )

    points = spectra["wavelenght"].values
    spectrum = spectra.values
    plot_points = np.array([points, spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([plot_points[:-1], plot_points[1:]], axis=1)
    lc = LineCollection(segments, cmap="nipy_spectral")
    lc.set_array(points)
    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.set_xlim(points.min(), points.max())
    ax.set_ylim(spectrum.min(), spectrum.max())
    ax.set_xlabel(f"Wavelenght [ {unit} ]")
    ax.set_ylabel("Emission [ photons/s ]")
    return fig, ax


def _is_conflict_free(reactions_subset: Sequence[dict]) -> bool:
    downs = set()
    ups = set()
    for r in reactions_subset:
        downs.update(r["downs"])
        ups.update(r["ups"])
    return len(downs) <= 1 or len(ups) <= 1


def _solve_min_colors(reactions_data: Sequence[dict]) -> list[list[int]]:
    n = len(reactions_data)
    if n <= 1:
        return [list(range(n))]
    if _is_conflict_free(reactions_data):
        return [list(range(n))]

    for k in range(2, n + 1):
        groups: list[list[dict]] = [[] for _ in range(k)]
        group_indices: list[list[int]] = [[] for _ in range(k)]
        best_partition = None

        def backtrack(idx: int, num_used_groups: int):
            nonlocal best_partition
            if idx == n:
                if num_used_groups == k:
                    score = sum(len(g) ** 2 for g in groups)
                    if best_partition is None or score > best_partition[0]:
                        best_partition = (score, [list(g) for g in group_indices if g])
                return

            r = reactions_data[idx]
            for g_i in range(min(num_used_groups + 1, k)):
                new_used = max(num_used_groups, g_i + 1)
                if k - new_used > n - (idx + 1):
                    continue
                groups[g_i].append(r)
                group_indices[g_i].append(idx)
                if _is_conflict_free(groups[g_i]):
                    backtrack(idx + 1, new_used)
                group_indices[g_i].pop()
                groups[g_i].pop()

        backtrack(0, 0)
        if best_partition is not None:
            return best_partition[1]

    return [[i] for i in range(n)]


def color_code_etu(
    etu_reactions: Iterable[Drawable],
    levels: Mapping[SpinState, Level],
    etu_colors: Iterable[str] | Cycler = ETU_COLORS,
) -> set[Transition]:
    reactions = list(etu_reactions)

    if not reactions:
        return set()

    if isinstance(etu_colors, Cycler):
        if "color" in etu_colors.keys:
            color_list = list(etu_colors.by_key()["color"])
        else:
            color_list = [list(d.values())[0] for d in etu_colors]
    else:
        color_list = list(etu_colors)
    color_cycle = cycle(color_list)

    reactions_data = []
    for r in reactions:
        sources = (
            r._source
            if isinstance(r._source, Sequence)
            else [r._source]
        )
        targets = (
            r._target
            if isinstance(r._target, Sequence)
            else [r._target]
        )
        pairs = list(zip(sources, targets))
        downs = frozenset(
            (s, t) for s, t in pairs if levels[t].energy < levels[s].energy
        )
        ups = frozenset(
            (s, t) for s, t in pairs if levels[t].energy > levels[s].energy
        )
        reactions_data.append({
            "reaction": r,
            "pairs": pairs,
            "downs": downs,
            "ups": ups,
        })

    visited = set()
    components = []
    for i in range(len(reactions_data)):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in range(len(reactions_data)):
                    if neighbor not in visited:
                        share_down = bool(
                            reactions_data[curr]["downs"]
                            & reactions_data[neighbor]["downs"]
                        )
                        share_up = bool(
                            reactions_data[curr]["ups"]
                            & reactions_data[neighbor]["ups"]
                        )
                        if share_down or share_up:
                            visited.add(neighbor)
                            queue.append(neighbor)
            components.append(comp)

    transitions: set[Transition] = set()
    for comp in components:
        comp_reactions = [reactions_data[idx] for idx in comp]
        comp_groups = _solve_min_colors(comp_reactions)

        for group in comp_groups:
            color = next(color_cycle)
            for rel_idx in group:
                orig_idx = comp[rel_idx]
                for s, t in reactions_data[orig_idx]["pairs"]:
                    transitions.add(
                        Transition(
                            source=levels[s],
                            target=levels[t],
                            kind="ETU",
                            color=color,
                        )
                    )

    return transitions



def jablonski_diagram(
    system: SpectroscopicSystem,
    figsize: tuple[Number, Number] = (6.4, 4.8),
    fontsize: Number = 10,
    show_energy_axis: bool = True,
    unit: str | pint.Unit = ureg.eV,
    etu_colors: Iterable[str] | Cycler = ETU_COLORS,
) -> tuple[Axes, Figure]:
    if isinstance(unit, str):
        unit = ureg[unit]
    columns = set()
    levels = {
        level: Level(
            label=level.name,
            energy=level.energy.to(unit).magnitude,
            column=get_column(level=level, system=system, columns=columns),
        )
        for level in system._yield(SpinState)
    }
    transitions: set[Transition] = set()
    etu_reactions = []

    for transition in system._yield(Drawable):
        if isinstance(
            transition, EnergyTransferUpconversion | EnergyTransferUpconversion4
        ):
            etu_reactions.append(transition)
        elif isinstance(transition, Pumper | RadiativeDecay):
            sources = (
                transition._source
                if isinstance(transition._source, Sequence)
                else [transition._source]
            )
            targets = (
                transition._target
                if isinstance(transition._target, Sequence)
                else [transition._target]
            )
            for s, t in zip(sources, targets):
                transitions.add(
                    Transition(
                        source=levels[s],
                        target=levels[t],
                        kind="radiative",
                    )
                )
        else:
            sources = (
                transition._source
                if isinstance(transition._source, Sequence)
                else [transition._source]
            )
            targets = (
                transition._target
                if isinstance(transition._target, Sequence)
                else [transition._target]
            )
            for s, t in zip(sources, targets):
                transitions.add(
                    Transition(
                        source=levels[s],
                        target=levels[t],
                        kind="non radiative",
                    )
                )

    if etu_reactions:
        transitions.update(color_code_etu(etu_reactions, levels, etu_colors=etu_colors))


    jd = JablonskiDiagram(
        levels=list(levels.values()),
        transitions=transitions,
        columns=sorted(list(columns), key=sort_by_multiplicity),
    )
    fig, ax = jd.plot(
        figsize=figsize,
        fontsize=fontsize,
        show_energy_axis=show_energy_axis,
    )
    ax.set_ylabel(f"Energy [{str(unit)} ]", fontsize=fontsize)
    return fig, ax



def jablonski_diagram_section(model: SpectroscopicSystem, latex: ToLatex):
    backend = plt.get_backend()
    plt.switch_backend("pgf")
    fig, ax = jablonski_diagram(model, figsize=(6, 4))

    with StringIO() as plot_buffer:
        fig.savefig(plot_buffer, format="pgf")
        plt.switch_backend(backend)
        return (
            "\\begin{figure}[H]\n\\centering\n"
            + plot_buffer.getvalue()
            + "\n\\end{figure}"
        )


def model_report(
    model: type[SpectroscopicSystem],
    path: str | None = None,
    transform: dict | None = None,
    descriptions: dict | None = None,
    standalone: bool = True,
    replace_algebraics: bool = False,
    sections: Mapping[
        str, Callable[[SpectroscopicSystem, ToLatex], str]
    ] = default_sections | {"Jablonski diagram": jablonski_diagram_section},
    packages: Iterable[str] = default_packages + ["pgf"],
) -> Latex | None:
    return _model_report(
        model=model,
        path=path,
        transform=transform,
        descriptions=descriptions,
        standalone=standalone,
        replace_algebraics=replace_algebraics,
        sections=sections,
        packages=packages,
    )


def get_column(level: SpinState, system: SpectroscopicSystem, columns: set) -> str:
    if level.parent == system:
        col = level.multiplicity
    else: 
        col = f"{level.parent}-{level.multiplicity}"
    columns.add(col)
    return col

def sort_by_multiplicity(s: str) -> tuple[str,int]:
    base, _, suffix = s.rpartition('-')
    order = 0 if suffix == 'singlet' else 1
    return (base, order)