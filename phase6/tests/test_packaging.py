"""Proves `phase6` resolves as a real, declared, installed Python
package -- the production import/launch path this checkpoint's
Dockerfile actually uses (`pip install --no-deps .`, then
`streamlit run phase6/app.py`) -- rather than through any
PYTHONPATH/sys.path/cwd-based trick.

A prior draft briefly added a `sys.path.insert(...)` bootstrap at the
top of `phase6/app.py` to paper over `streamlit run phase6/app.py` only
adding its own script directory to `sys.path`, not `/app` (the real
`ModuleNotFoundError: No module named 'phase6'` this project's
"no production sys.path/PYTHONPATH hacks" rule exists specifically to
prevent). That workaround was removed; `services/frontend/pyproject.toml`
+ a real `pip install` is the fix. This test is the regression guard for
that fix: it runs a fresh Python subprocess from a directory that has no
special relationship to this repository, with `PYTHONPATH` explicitly
absent, and proves `import phase6` still works purely because the
package was actually installed (exactly what `pip show voyager-frontend`
confirms is the case in this environment, since the checkpoint's own
`pip install -e .` step already ran).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def test_phase6_importable_from_an_unrelated_directory_with_no_pythonpath():
    """The core proof: a subprocess started in a directory that is not
    this repository, with PYTHONPATH removed entirely, can still
    `import phase6` -- possible only because `phase6` is a real
    installed package (site-packages / an editable-install .pth/finder),
    never because of the caller's own working directory or an
    environment-variable path injection."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    with tempfile.TemporaryDirectory() as unrelated_dir:
        result = subprocess.run(
            [sys.executable, "-c", "import phase6, phase6.results, phase6.api_client; print(phase6.__name__)"],
            cwd=unrelated_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "phase6"


def test_installed_distribution_metadata_confirms_a_real_package_install():
    """A second, independent signal: `importlib.metadata` must be able
    to find a real installed distribution backing `phase6` -- proving
    this is a genuine package install, not an accidental sys.path hit
    (e.g. a stray copy on disk some other mechanism happened to find)."""
    import importlib.metadata as metadata

    dist = metadata.distribution("voyager-frontend")
    assert dist is not None
    assert dist.metadata["Name"] == "voyager-frontend"


def test_app_py_source_contains_no_sys_path_or_pythonpath_mutation():
    """Static regression guard, independent of the two runtime proofs
    above: the production script itself must never re-introduce a
    sys.path/PYTHONPATH workaround -- this is a source-text check, not a
    behavioral one, so it is a direct guard against exactly the pattern
    this checkpoint's correction removed."""
    import pathlib

    app_source = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "sys.path" not in app_source
    assert "PYTHONPATH" not in app_source
    assert "import sys" not in app_source
