"""Install and check the ``dyscolab-modeling`` agent skill.

An alias of :mod:`poincare.skill`, which merges the skill files of poincare, simbio and
jablonski::

    python -m jablonski.skill install --dest .claude/skills
    python -m jablonski.skill check .claude/skills/dyscolab-modeling
"""

from poincare.skill import assemble, check, install, main

__all__ = ["assemble", "check", "install", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
