"""Test runner usable with or without pytest.

With pytest installed this simply runs ``pytest`` on this folder (so
parametrized tests and fixtures work).  Without pytest, plain test functions
that take no arguments are run directly; tests that need pytest
(parametrize, fixtures such as ``tmp_path``) are reported as SKIP instead of
failing, and modules that import pytest are skipped as a whole.
"""
import importlib
import inspect
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).parent

try:
    import pytest
except ImportError:  # pragma: no cover - exercised only without pytest
    pytest = None

if pytest is not None:
    sys.exit(pytest.main(["-q", str(HERE)]))

sys.path.insert(0, str(HERE))
failed = skipped = 0
for path in sorted(HERE.glob("test_*.py")):
    try:
        mod = importlib.import_module(path.stem)
    except ImportError as exc:
        skipped += 1
        print(f"SKIP {path.stem} (needs {exc.name})")
        continue
    for name in dir(mod):
        if not name.startswith("test_"):
            continue
        func = getattr(mod, name)
        if inspect.signature(func).parameters:
            skipped += 1
            print(f"SKIP {path.stem}::{name} (needs pytest)")
            continue
        t = time.time()
        try:
            func()
            print(f"PASS {path.stem}::{name} ({time.time() - t:.1f}s)")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {path.stem}::{name}")
            traceback.print_exc()
print("failed:", failed, " skipped:", skipped)
sys.exit(1 if failed else 0)
