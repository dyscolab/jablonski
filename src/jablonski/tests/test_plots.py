import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from jablonski import SingletState, SpectroscopicSystem, initial
from jablonski.plots import model_report, notebook_report
from jablonski.transitions import Absorption, Fluorescence
from jablonski.util import ureg


class Model(SpectroscopicSystem):
    ground: SingletState = initial(0 * ureg.eV, "singlet", default=1)
    excited: SingletState = initial(2 * ureg.eV, "singlet", default=0)

    absorption = Absorption(ground=ground, excited=excited, rate=1e-15 * ureg.cm**2)
    emission = Fluorescence(ground=ground, excited=excited, rate=1e8 / ureg.s)


def test_notebook_report():
    output = notebook_report(Model, show=False)
    assert output.startswith(
        "## Jablonski diagram\n\n![Jablonski diagram](data:image/png;base64,"
    )
    assert "## Equations" in output
    assert not plt.get_fignums()


@pytest.mark.skipif(
    shutil.which(mpl.rcParams["pgf.texsystem"]) is None,
    reason="the PGF backend needs a TeX installation",
)
def test_model_report_closes_its_figure():
    user_fig = plt.figure()
    output = model_report(Model)
    assert "\\begin{pgfpicture}" in output
    assert output.index("Jablonski diagram") < output.index("{Equations}")
    assert plt.get_fignums() == [user_fig.number]
    plt.close(user_fig)
