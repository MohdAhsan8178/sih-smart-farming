"""
tests/test_python36_ast_guard.py — Standing guard preventing Python 3.7+ regressions.

Audits every module under edge/, gateway/, core/, and configs/ using AST and lexical analysis.
Fails loudly if any Python 3.7+ construct is reintroduced:
  - Walrus operator (:=) (Python 3.8+)
  - Self-documenting f-strings f"{var=}" (Python 3.8+)
  - Positional-only parameters (/) (Python 3.8+)
  - Pattern matching (match / case) (Python 3.10+)
  - PEP 585 generic builtins (list[str], dict[str, Any]) (Python 3.9+)
  - PEP 604 union syntax (int | None, str | int) (Python 3.10+)
  - Dataclasses (@dataclass, import dataclasses) (Python 3.7+)
  - Subprocess keywords (text=True, capture_output=True) (Python 3.7+)
  - datetime.fromisoformat (Python 3.7+)
  - ThreadingHTTPServer (Python 3.7+)
  - from __future__ import annotations (Python 3.7+)
"""

import ast
from pathlib import Path
import re
from typing import List, Tuple
import pytest

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIRS = ["edge", "gateway", "core", "configs"]


class Python36Auditor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = str(filepath)
        self.issues: List[Tuple[int, str]] = []

    def visit_NamedExpr(self, node: ast.AST) -> None:
        self.issues.append((getattr(node, "lineno", 0), "Walrus operator (:=) used (Python 3.8+)"))
        self.generic_visit(node)

    def visit_Match(self, node: ast.AST) -> None:
        self.issues.append((getattr(node, "lineno", 0), "Pattern matching (match/case) used (Python 3.10+)"))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if hasattr(node.args, "posonlyargs") and node.args.posonlyargs:
            self.issues.append((node.lineno, "Positional-only parameters (/) used (Python 3.8+)"))
        if node.returns:
            self._check_annotation(node.returns)
        for arg in node.args.args + getattr(node.args, "kwonlyargs", []):
            if arg.annotation:
                self._check_annotation(arg.annotation)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if hasattr(node.args, "posonlyargs") and node.args.posonlyargs:
            self.issues.append((node.lineno, "Positional-only parameters (/) used (Python 3.8+)"))
        if node.returns:
            self._check_annotation(node.returns)
        for arg in node.args.args + getattr(node.args, "kwonlyargs", []):
            if arg.annotation:
                self._check_annotation(arg.annotation)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.annotation:
            self._check_annotation(node.annotation)
        self.generic_visit(node)

    def _check_annotation(self, node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                self.issues.append((getattr(sub, "lineno", 0), "PEP 604 union syntax (|) used in type annotation (Python 3.10+)"))
            elif isinstance(sub, ast.Subscript):
                if isinstance(sub.value, ast.Name) and sub.value.id in ("list", "dict", "tuple", "set", "frozenset", "type"):
                    self.issues.append((getattr(sub, "lineno", 0), "PEP 585 generic builtin %s[...] used (Python 3.9+)" % sub.value.id))

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Check PEP 585 generics used outside annotations (e.g. type aliases)
        if isinstance(node.value, ast.Name) and node.value.id in ("list", "dict", "tuple", "set", "frozenset", "type"):
            self.issues.append((node.lineno, "PEP 585 generic builtin %s[...] used (Python 3.9+)" % node.value.id))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "dataclasses":
                self.issues.append((node.lineno, "import dataclasses used (Python 3.7+)"))
            elif alias.name.endswith("ThreadingHTTPServer"):
                self.issues.append((node.lineno, "ThreadingHTTPServer imported (Python 3.7+)"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "dataclasses":
            self.issues.append((node.lineno, "from dataclasses import ... used (Python 3.7+)"))
        elif node.module == "__future__" and any(a.name == "annotations" for a in node.names):
            self.issues.append((node.lineno, "from __future__ import annotations used (Python 3.7+)"))
        elif any(a.name == "ThreadingHTTPServer" for a in node.names):
            self.issues.append((node.lineno, "ThreadingHTTPServer imported (Python 3.7+)"))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "fromisoformat":
            self.issues.append((node.lineno, "datetime.fromisoformat attribute used (Python 3.7+)"))
        elif node.attr == "ThreadingHTTPServer":
            self.issues.append((node.lineno, "ThreadingHTTPServer attribute used (Python 3.7+)"))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "ThreadingHTTPServer":
            self.issues.append((node.lineno, "ThreadingHTTPServer name used (Python 3.7+)"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name in ("run", "Popen", "check_output", "check_call"):
            for kw in node.keywords:
                if kw.arg in ("text", "capture_output"):
                    self.issues.append((node.lineno, "subprocess keyword argument %s= used (Python 3.7+)" % kw.arg))
        self.generic_visit(node)


def audit_code_string(code: str, filename: str = "<string>") -> List[Tuple[int, str]]:
    """Runs AST and regex checks on a Python code snippet."""
    tree = ast.parse(code, filename=filename)
    auditor = Python36Auditor(filename)
    auditor.visit(tree)
    for idx, line in enumerate(code.splitlines(), 1):
        if re.search(r'f[\'\"][^\"\'\n]*\{[^}\n]+=\s*\}', line):
            auditor.issues.append((idx, "f-string with = specifier used (Python 3.8+): %s" % line.strip()))
    return auditor.issues


def test_no_python37_plus_constructs_in_production_code():
    """Exhaustively asserts that NO Python 3.7+ syntax or constructs exist across edge, gateway, core, configs."""
    files_to_check: List[Path] = []
    for d in AUDIT_DIRS:
        target_dir = ROOT / d
        if target_dir.exists():
            files_to_check.extend(sorted(target_dir.glob("**/*.py")))

    assert len(files_to_check) >= 20, "Expected at least 20 production files to audit, found %d" % len(files_to_check)

    all_violations: List[str] = []
    for fpath in files_to_check:
        content = fpath.read_text(encoding="utf-8")
        issues = audit_code_string(content, filename=str(fpath.relative_to(ROOT)))
        for lineno, msg in issues:
            all_violations.append("%s:%d: %s" % (str(fpath.relative_to(ROOT)), lineno, msg))

    if all_violations:
        pytest.fail(
            "Found %d Python 3.7+ compatibility violation(s) that will break Python 3.6 on the Jetson Nano:\n%s"
            % (len(all_violations), "\n".join(all_violations))
        )


# --- Unit Tests verifying that every guard in Python36Auditor catches violations ---

def test_guard_catches_walrus():
    code = "if (x := 10) > 5:\n    pass\n"
    issues = audit_code_string(code)
    assert any("Walrus operator" in msg for _, msg in issues)


def test_guard_catches_debug_fstring():
    code = 'x = 42\nmsg = f"{x=}"\n'
    issues = audit_code_string(code)
    assert any("f-string with =" in msg for _, msg in issues)


def test_guard_catches_positional_only_params():
    code = "def foo(a, b, /, c=0):\n    return a + b + c\n"
    issues = audit_code_string(code)
    assert any("Positional-only" in msg for _, msg in issues)


def test_guard_catches_match_case():
    code = "match val:\n    case 1:\n        pass\n"
    issues = audit_code_string(code)
    assert any("Pattern matching" in msg for _, msg in issues)


def test_guard_catches_pep585_generics():
    code1 = "def bar(items: list[str]) -> None:\n    pass\n"
    issues1 = audit_code_string(code1)
    assert any("PEP 585" in msg for _, msg in issues1)

    code2 = "MyType = dict[str, int]\n"
    issues2 = audit_code_string(code2)
    assert any("PEP 585" in msg for _, msg in issues2)


def test_guard_catches_pep604_unions():
    code1 = "def baz(val: int | None) -> str | int:\n    return val\n"
    issues1 = audit_code_string(code1)
    assert any("PEP 604" in msg for _, msg in issues1)

    code2 = "count: int | float = 0\n"
    issues2 = audit_code_string(code2)
    assert any("PEP 604" in msg for _, msg in issues2)


def test_guard_catches_dataclasses():
    code1 = "import dataclasses\n"
    issues1 = audit_code_string(code1)
    assert any("dataclasses" in msg for _, msg in issues1)

    code2 = "from dataclasses import dataclass\n"
    issues2 = audit_code_string(code2)
    assert any("dataclasses" in msg for _, msg in issues2)


def test_guard_catches_subprocess_kwargs():
    code = "import subprocess\nsubprocess.run(['ls'], text=True, capture_output=True)\n"
    issues = audit_code_string(code)
    assert any("text=" in msg for _, msg in issues)
    assert any("capture_output=" in msg for _, msg in issues)


def test_guard_catches_datetime_fromisoformat():
    code = "from datetime import datetime\ndt = datetime.fromisoformat('2026-09-15T18:00:00')\n"
    issues = audit_code_string(code)
    assert any("fromisoformat" in msg for _, msg in issues)


def test_guard_catches_threading_http_server():
    code1 = "from http.server import ThreadingHTTPServer\n"
    issues1 = audit_code_string(code1)
    assert any("ThreadingHTTPServer" in msg for _, msg in issues1)

    code2 = "import http.server\nsrv = http.server.ThreadingHTTPServer(('127.0.0.1', 80), None)\n"
    issues2 = audit_code_string(code2)
    assert any("ThreadingHTTPServer" in msg for _, msg in issues2)


def test_guard_catches_future_annotations():
    code = "from __future__ import annotations\n"
    issues = audit_code_string(code)
    assert any("__future__ import annotations" in msg for _, msg in issues)
