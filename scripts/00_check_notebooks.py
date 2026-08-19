#!/usr/bin/env python3
"""Execute the non-interactive notebooks headlessly against the smoke dataset.

Catches the failure mode that a notebook cannot report on itself: a cell that
raises three days after it was written, because a helper was renamed.

``03_labeling`` is skipped - it waits for human clicks by design. ``00`` and
``01`` need real sources, so they are compile-checked only.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADLESS = {
    "02_back_geometry_inspector.ipynb": {
        "MANIFEST_CSV": 'PROJECT_ROOT / "data" / "smoke" / "manifest_smoke.csv"',
    },
    "04_training_evaluation.ipynb": {
        "MANIFEST_CSV": 'PROJECT_ROOT / "data" / "smoke" / "manifest_smoke.csv"',
        "RUN_DIR": 'PROJECT_ROOT / "outputs" / "smoke_nb"',
        "PREVIEW_ONLY": "False",
        "SAVE_OUTPUTS": "True",
        "MODELS": '["geometry"]',
    },
}
COMPILE_ONLY = ("00_dataset_browser.ipynb", "01_detection_segmentation_inspector.ipynb",
                "03_labeling.ipynb")


def strip_magics(source: str) -> str:
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()
    )


def compile_check(path: Path) -> list[str]:
    problems = []
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(strip_magics("".join(cell["source"])))
        except SyntaxError as error:
            problems.append(f"{path.name} cell {index}: {error.msg} (line {error.lineno})")
    return problems


def execute(path: Path, overrides: dict[str, str]) -> None:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = strip_magics(cell.source)
        for key, value in overrides.items():
            pattern = rf"^{re.escape(key)}\s*=.*$"
            if re.search(pattern, source, flags=re.M):
                source = re.sub(pattern, f"{key} = {value}", source, count=1, flags=re.M)
        cell.source = source
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    notebooks = ROOT / "notebooks"
    problems: list[str] = []
    for path in sorted(notebooks.glob("*.ipynb")):
        problems.extend(compile_check(path))
    if problems:
        for problem in problems:
            print("SYNTAX", problem)
        raise SystemExit(1)
    print(f"compile check passed for {len(list(notebooks.glob('*.ipynb')))} notebooks")
    for name in COMPILE_ONLY:
        print(f"  compile-only: {name}")

    if args.compile_only:
        return

    manifest = ROOT / "data" / "smoke" / "manifest_smoke.csv"
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} is missing. Run scripts/00_smoke_test.py first so the "
            "headless notebooks have a dataset to execute against."
        )

    for name, overrides in HEADLESS.items():
        print(f"\nexecuting {name} ...", flush=True)
        execute(notebooks / name, overrides)
        print(f"OK {name}")

    print("\nNOTEBOOK CHECK PASSED - plumbing only, on synthetic data.")


if __name__ == "__main__":
    main()
