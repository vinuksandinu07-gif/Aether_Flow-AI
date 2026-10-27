"""Minimal runner so the suite works without pytest:  python tests/run_tests.py [-k substring]

Discovers ``tests/test_*.py`` and runs every top-level ``test_*`` function (no fixtures used).
With pytest installed, simply run ``pytest`` instead - the tests are plain asserts.
"""
import importlib
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    kw = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    passed = failed = 0
    failures = []
    t0 = time.time()
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        mod = importlib.import_module(f.stem)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            if kw and kw not in f"{f.stem}::{name}":
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except Exception:                                  # noqa: BLE001
                failed += 1
                failures.append((f"{f.stem}::{name}", traceback.format_exc()))
    for nm, tb in failures:
        print(f"\nFAILED {nm}\n{tb}")
    print(f"\n{passed} passed, {failed} failed in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
