"""jablonski's part of the dyscolab-modeling agent skill (see poincare.skill)."""

from pathlib import Path
from runpy import run_path

import pytest
from poincare import skill

SOURCE = Path(__file__).parents[3] / skill.SOURCE_PATH
EXAMPLES = sorted((SOURCE / "examples").glob("*.py"))


def test_part_is_found():
    part = skill.find_part("jablonski")
    assert {"references/jablonski.md", "references/jablonski-api.md"} <= set(part.files)
    assert all(Path(f).name.startswith("jablonski") for f in part.files)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example(example: Path):
    run_path(str(example), run_name="test")
