"""Dependency-free test runner (so the suite can run without pytest)."""
import importlib
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent))
failed = 0
for path in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
    mod = importlib.import_module(path.stem)
    for name in dir(mod):
        if name.startswith("test_"):
            t = time.time()
            try:
                getattr(mod, name)()
                print(f"PASS {path.stem}::{name} ({time.time() - t:.1f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {path.stem}::{name}")
                traceback.print_exc()
print("failed:", failed)
sys.exit(1 if failed else 0)
