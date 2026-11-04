"""Shared pieces of the notebook builders (build_notebook.py, build_faster_rcnn_notebook.py).

Each package module marks its sections with "# %% <name>" comments; a builder places those sections in reading
order, drops the package-relative imports (every name lives in the notebook's one namespace) and gathers the
third-party imports into the setup cell.
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "workbench_detection"
RELATIVE_IMPORT = re.compile(r"^\s*from \.+[\w.]* import [^(]*$")


def read_module(relative):
    """Return (import lines, {section name: code}) for one module."""
    text = (PACKAGE / relative).read_text(encoding="utf-8")
    parts = re.split(r"^# %% (\S+)\n", text, flags=re.M)
    header, sections = parts[0], dict(zip(parts[1::2], parts[2::2]))
    tree = ast.parse(header)
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                continue
            imports.append(ast.get_source_segment(header, node))
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            raise ValueError(f"{relative}: move code above the first '# %%' section into a section")
    cleaned = {}
    for name, code in sections.items():
        lines = [line for line in code.splitlines() if not RELATIVE_IMPORT.match(line)]
        cleaned[name] = "\n".join(lines).strip("\n")
    return imports, cleaned


def read_modules(modules):
    """({module: {section: code}}, the modules' third-party imports without repeats)."""
    sources, imports = {}, []
    for module in modules:
        module_imports, sections = read_module(module)
        imports += [line for line in module_imports if line not in imports]
        sources[module] = sections
    return sources, imports


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def check_and_write(output, cells, sources, used, required=None):
    """Check the notebook (every required section placed, no name defined twice, valid Python) and write it.

    required: the modules whose every section must appear (default: all of them).
    Keeps the saved outputs of an already-run notebook when the cell structure is unchanged.
    """
    required = sources if required is None else required
    unused = {(module, name) for module in required for name in sources[module]} - used
    if unused:
        raise SystemExit(f"Sections not placed in the notebook: {sorted(unused)}")
    defined = {}
    for module, name in used:
        for node in ast.parse(sources[module][name]).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in defined:
                    raise SystemExit(f"{node.name} defined in both {defined[node.name]} and {module}")
                defined[node.name] = module
    for number, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook cell {number}", "exec")
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))["cells"]
        kinds = [cell["cell_type"] for cell in previous]
        if kinds == [cell["cell_type"] for cell in cells] and any(cell.get("outputs") for cell in previous):
            for old, new in zip(previous, cells):
                if new["cell_type"] == "code":
                    new["outputs"], new["execution_count"] = old.get("outputs", []), old.get("execution_count")
            print("Kept the saved outputs of the existing notebook.")
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
                "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}}}
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output.name}: {len(cells)} cells, {len(used)} source sections from {len(sources)} modules")
