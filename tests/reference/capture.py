"""Run the reference scenarios and save their published parquets under tests/reference/.

    uv run python tests/reference/capture.py [scenario ...]

With no arguments every scenario in ``tests/_fixtures.SCENARIOS`` is captured. Each scenario's
directory is replaced wholesale, and ``MANIFEST.json`` records the commit the outputs came from
(``+dirty``: that commit's working tree, i.e. the commit that adds these outputs).
Regenerate a scenario only in a commit that deliberately changes its outputs (and explains why).

A scenario's ``global/`` subdirectory (the former embeddings global stages' outputs, the
baseline of ``tests/test_global_parity.py``) is kept as it is: runs no longer write it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from _compare import load  # noqa: E402
from _fixtures import (  # noqa: E402
    GLOBAL_REFERENCE,
    REFERENCE_DIR,
    ROOT,
    SCENARIOS,
    published_parquets,
    run_scenario,
)


def _commit() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "packages", "nextflow"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return sha + ("+dirty" if dirty else "")


def capture(scenario: str) -> Path:
    dest = REFERENCE_DIR / scenario
    with tempfile.TemporaryDirectory(prefix=f"ref_{scenario}_") as tmp:
        run_root = Path(tmp)
        pipeline_dir = run_scenario(scenario, run_root)
        files = published_parquets(pipeline_dir, scenario)
        kept = dest / GLOBAL_REFERENCE
        saved = Path(tmp) / "kept_global"
        if kept.exists():
            shutil.move(kept, saved)
        if dest.exists():
            shutil.rmtree(dest)
        if saved.exists():
            dest.mkdir(parents=True)
            shutil.move(saved, dest / GLOBAL_REFERENCE)
        for rel, src in files.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            # The run's directory, which path columns contain, is saved as <RUN>.
            load(src, run_root).write_parquet(target)
    kept_files = [
        p.relative_to(dest).as_posix()
        for p in (dest / GLOBAL_REFERENCE).rglob("*.parquet")
    ]
    manifest = {
        "scenario": scenario,
        "commit": _commit(),
        "files": sorted([*files, *kept_files]),
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "scenarios", nargs="*", metavar="scenario", help=f"any of {SCENARIOS}"
    )
    args = parser.parse_args()
    unknown = set(args.scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f"unknown scenarios: {sorted(unknown)}")
    for scenario in args.scenarios or SCENARIOS:
        dest = capture(scenario)
        print(f"{scenario}: {len(list(dest.rglob('*.parquet')))} files -> {dest}")


if __name__ == "__main__":
    main()
