"""Tiny dependency-free lint: unused imports and names used but never defined."""
import ast
import builtins
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems = []

    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported[a.asname or a.name] = node.lineno

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    src_lines = path.read_text(encoding="utf-8").splitlines()

    for name, line in imported.items():
        if name in used or name == "annotations":
            continue
        if "noqa" in src_lines[line - 1]:
            continue
        # allow names listed in __all__ or used in string annotations
        if f'"{name}' in "\n".join(src_lines) or f"'{name}" in "\n".join(src_lines):
            continue
        problems.append(f"{path}:{line}: unused import '{name}'")

    defined = set(dir(builtins)) | set(imported)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if not isinstance(node, ast.ClassDef):
                a = node.args
                for arg in a.args + a.kwonlyargs + a.posonlyargs:
                    defined.add(arg.arg)
                if a.vararg:
                    defined.add(a.vararg.arg)
                if a.kwarg:
                    defined.add(a.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for arg in a.args + a.kwonlyargs + a.posonlyargs:
                defined.add(arg.arg)
            if a.vararg:
                defined.add(a.vararg.arg)
            if a.kwarg:
                defined.add(a.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.alias):
            pass

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined and node.id not in {"__file__", "__name__"}:
                problems.append(f"{path}:{node.lineno}: undefined name '{node.id}'")

    return problems


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]] or sorted(Path("src").glob("*.py"))
    issues = [i for f in files for i in check(f)]
    print("\n".join(issues) or "lint_check: no problems found")
    sys.exit(1 if issues else 0)