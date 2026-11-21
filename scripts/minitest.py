#!/usr/bin/env python3
"""Minimal pytest stand-in for machines where pytest cannot be installed (no package index access).

Supports what tests/ uses: fixtures (conftest + module, with yield), tmp_path, monkeypatch, capsys,
pytest.raises, pytest.fixture, pytest.mark.parametrize, pytest.approx.

    PYTHONPATH=src python3 scripts/minitest.py [tests/unit/test_x.py ...] [-k substring]
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import math
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---- fake pytest module -------------------------------------------------------------------

class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match, self.value = exc, match, None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match:
            import re
            if not re.search(self.match, str(ev)):
                raise AssertionError(f"pattern {self.match!r} not found in {ev!r}")
        self.value = ev
        return True


def _fixture(fn=None, **_kw):
    def mark(f):
        f._is_fixture = True
        return f
    return mark(fn) if callable(fn) else mark


class _Mark:
    @staticmethod
    def parametrize(names, values, **_kw):
        def deco(f):
            f._params = ([n.strip() for n in names.split(",")] if isinstance(names, str) else list(names), list(values))
            return f
        return deco

    def __getattr__(self, _name):
        return lambda *a, **k: (lambda f: f)


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        return math.isclose(other, self.expected, rel_tol=self.rel, abs_tol=self.abs)


pytest = types.ModuleType("pytest")
pytest.raises = lambda exc, match=None: _Raises(exc, match)
pytest.fixture = _fixture
pytest.mark = _Mark()
pytest.approx = _Approx
pytest.skip = lambda msg="": (_ for _ in ()).throw(_Skip(msg))
sys.modules["pytest"] = pytest


class _Skip(Exception):
    pass


# ---- builtin fixtures -----------------------------------------------------------------------

class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None, raising=True):
        if isinstance(target, str) and value is None:
            module, attr = target.rsplit(".", 1)
            target, name, value = importlib.import_module(module), attr, name
        old = getattr(target, name, None)
        self._undo.append(lambda: setattr(target, name, old))
        setattr(target, name, value)

    def setenv(self, key, value):
        import os
        old = os.environ.get(key)
        self._undo.append(lambda: os.environ.__setitem__(key, old) if old is not None else os.environ.pop(key, None))
        os.environ[key] = str(value)

    def delenv(self, key, raising=True):
        import os
        old = os.environ.pop(key, None)
        if old is not None:
            self._undo.append(lambda: os.environ.__setitem__(key, old))

    def chdir(self, path):
        import os
        old = os.getcwd()
        self._undo.append(lambda: os.chdir(old))
        os.chdir(path)

    def setitem(self, mapping, key, value):
        old = mapping.get(key, _MISSING)
        self._undo.append(lambda: mapping.pop(key, None) if old is _MISSING else mapping.__setitem__(key, old))
        mapping[key] = value

    def undo(self):
        for fn in reversed(self._undo):
            fn()


_MISSING = object()


class CapSys:
    def __init__(self):
        self._out, self._err = io.StringIO(), io.StringIO()
        self._old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._out, self._err

    def readouterr(self):
        out, err = self._out.getvalue(), self._err.getvalue()
        self._out.seek(0), self._out.truncate(), self._err.seek(0), self._err.truncate()
        return types.SimpleNamespace(out=out, err=err)

    def close(self):
        sys.stdout, sys.stderr = self._old


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(paths: list[Path], keyword: str | None) -> int:
    sys.path.insert(0, str(ROOT))
    conftest = _load(ROOT / "tests" / "conftest.py", "tests.conftest") if (ROOT / "tests" / "conftest.py").exists() else None
    passed = failed = skipped = 0
    for path in paths:
        module = _load(path, f"tests.unit.{path.stem}")
        fixtures = {}
        for src in (conftest, module):
            if src:
                fixtures.update({n: f for n, f in vars(src).items() if getattr(f, "_is_fixture", False)})
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != module.__name__ or (keyword and keyword not in name):
                continue
            cases = [{}]
            if hasattr(fn, "_params"):
                names, values = fn._params
                cases = [dict(zip(names, v if isinstance(v, (tuple, list)) and len(names) > 1 else (v,))) for v in values]
            for case in cases:
                stack = contextlib.ExitStack()
                cache: dict = {}
                tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                mp = MonkeyPatch()
                stack.callback(mp.undo)

                def resolve(arg):
                    if arg in case:
                        return case[arg]
                    if arg in cache:
                        return cache[arg]
                    if arg == "tmp_path":
                        value = tmp
                    elif arg == "monkeypatch":
                        value = mp
                    elif arg == "capsys":
                        cap = CapSys()
                        stack.callback(cap.close)
                        value = cap
                    elif arg in fixtures:
                        f = fixtures[arg]
                        result = f(**{a: resolve(a) for a in inspect.signature(f).parameters})
                        if inspect.isgenerator(result):
                            value = next(result)
                            stack.callback(lambda g=result: next(g, None))
                        else:
                            value = result
                    else:
                        raise LookupError(f"unknown fixture '{arg}'")
                    cache[arg] = value
                    return value

                label = f"{path.name}::{name}" + (f"[{case}]" if case else "")
                try:
                    with stack:
                        fn(**{a: resolve(a) for a in inspect.signature(fn).parameters})
                    passed += 1
                except _Skip:
                    skipped += 1
                except Exception:  # noqa: BLE001
                    failed += 1
                    print(f"FAIL {label}")
                    traceback.print_exc(limit=6)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    kw = None
    if "-k" in argv:
        i = argv.index("-k")
        kw = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    files = [Path(a) for a in argv] or sorted((ROOT / "tests" / "unit").glob("test_*.py"))
    sys.exit(run([f.resolve() for f in files], kw))
