from io import StringIO
from typing import Callable, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pint
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from poincare.printing.latex import Latex, ToLatex, default_packages, default_sections
from poincare.printing.latex import model_report as _model_report

from .._typing import Drawable, Pumper, RadiativeDecay
from .._units import ureg
from ..simulation import widened_emission_spectra
from ..states import SpectroscopicSystem, SpinState
from ..util import SpectraKind
from .jablonski_diagrams import JablonskiDiagram, Level, Number, Transition


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


def jablonski_diagram(
    system: SpectroscopicSystem,
    figsize: tuple[Number, Number] = (6.4, 4.8),
    fontsize: Number = 10,
    show_energy_axis: bool = True,
    unit: str | pint.Unit = ureg.eV,
) -> tuple[Axes, Figure]:
    if isinstance(unit, str):
        unit = ureg[unit]
    levels = {
        level: Level(
            label=level.name,
            energy=level.energy.to(unit).magnitude,
            column=level.multiplicity,
        )
        for level in system._yield(SpinState)
    }
    transitions = [
        Transition(
            source=levels[transition._source],
            target=levels[transition._target],
            radiative=isinstance(transition, Pumper | RadiativeDecay),
        )
        for transition in system._yield(Drawable)
    ]
    columns = []
    has_singlet = any(level.column == "singlet" for level in levels.values())
    has_triplet = any(level.column == "triplet" for level in levels.values())
    if has_singlet:
        columns.append("singlet")
    if has_triplet:
        columns.append("triplet")
    jd = JablonskiDiagram(
        levels=list(levels.values()),
        transitions=transitions,
        columns=columns,
    )
    fig, ax = jd.plot(
        figsize=figsize,
        fontsize=fontsize,
        show_energy_axis=show_energy_axis,
    )
    ax.set_ylabel(f"Energy [{str(unit)} ]")
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
    # packages: Iterable[str] = default_packages + ["graphicx", "inline-images"],
) -> Latex | None:
    return _model_report(
        model=model,
        path=path,
        descriptions=descriptions,
        standalone=standalone,
        replace_algebraics=replace_algebraics,
        sections=sections,
        packages=packages,
    )
