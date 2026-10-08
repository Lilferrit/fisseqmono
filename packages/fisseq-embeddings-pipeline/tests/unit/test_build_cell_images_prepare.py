"""Tests for BUILD_CELL_IMAGES' prepare phase
(fisseq_embeddings_pipeline.build_cell_images_prepare).

Covers `resolve_data_dir` (phenotyping_dir/segmentation_dir/sequencing_dir
resolution against a real or absent starcall-workflow project config),
`snakemake_config` (the nested snakemake's --configfile) and the cluster
jobscript. The tile list itself is snakemake/fisseq_targets.py's --
tests/unit/test_snakemake_targets.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fisseq_embeddings_pipeline import build_cell_images_prepare as mod

# ---------------------------------------------------------------------------
# resolve_data_dir -- phenotyping_dir/segmentation_dir/sequencing_dir
# resolution against starcall-workflow's own project config
# ---------------------------------------------------------------------------


def test_resolve_data_dir_returns_explicit_value_without_reading_any_config(
    tmp_path: Path,
):
    # No config.yaml/default-config.yaml written at all -- an explicit
    # value must never require one to exist.
    assert (
        mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", "/elsewhere")
        == "/elsewhere"
    )


def test_resolve_data_dir_falls_back_to_bare_subdir_when_no_config_file_exists(
    tmp_path: Path,
):
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "phenotyping"
    )
    assert mod.resolve_data_dir(str(tmp_path), "segmentation_dir", None) == str(
        tmp_path / "segmentation"
    )
    assert mod.resolve_data_dir(str(tmp_path), "sequencing_dir", None) == str(
        tmp_path / "sequencing"
    )


def test_resolve_data_dir_reads_absolute_value_from_config_yaml(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "/mnt/other-storage/phenotyping"})
    )
    assert (
        mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None)
        == "/mnt/other-storage/phenotyping"
    )


def test_resolve_data_dir_resolves_relative_value_from_config_yaml_against_starcall_workflow_dir(
    tmp_path: Path,
):
    # starcall-workflow's own default-config.yaml convention: a relative,
    # trailing-slash value.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "custom_pheno/"})
    )
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "custom_pheno"
    )


def test_resolve_data_dir_falls_back_to_default_config_yaml_when_config_yaml_absent(
    tmp_path: Path,
):
    (tmp_path / "default-config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "phenotyping/"})
    )
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "phenotyping"
    )


def test_resolve_data_dir_does_not_fall_through_to_default_config_yaml_when_config_yaml_exists(
    tmp_path: Path,
):
    """config.yaml existing at all stops default-config.yaml from being
    consulted, even for a key config.yaml doesn't itself set -- matching
    workflow/Snakefile's own either/or (never both) precedence exactly."""
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"some_other_key": 1}))
    (tmp_path / "default-config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "/should-not-be-used"})
    )
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "phenotyping"
    )


def test_resolve_data_dir_explicit_value_wins_over_config_yaml(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "/from-config-yaml"})
    )
    assert (
        mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", "/explicit-override")
        == "/explicit-override"
    )


# ---------------------------------------------------------------------------
# snakemake_config -- the nested snakemake's --configfile
# ---------------------------------------------------------------------------


_DIRS = {
    "phenotyping_dir": "/data/e1/phenotyping",
    "segmentation_dir": "/data/e1/segmentation",
    "sequencing_dir": "/data/e1/sequencing/",
}


def _config(**overrides):
    fields = dict(starcall_workflow_dir="/data/e1", window=224)
    fields.update(overrides)
    return mod.snakemake_config(
        mod.BuildCellImagesPrepareConfig(**fields),
        _DIRS,
        "/work/tiles_manifest.csv",
        "/venv/bin/python",
    )


def test_snakemake_config_gives_every_dir_one_trailing_slash():
    """starcall concatenates paths onto these, so the '/' is load-bearing."""
    config = _config()
    assert config["phenotyping_dir"] == "/data/e1/phenotyping/"
    assert config["segmentation_dir"] == "/data/e1/segmentation/"
    assert config["sequencing_dir"] == "/data/e1/sequencing/"


def test_snakemake_config_carries_the_manifest_rule_settings():
    config = _config(use_corrected=True, window=180, cp_features=True)
    assert config["fisseq_python"] == "/venv/bin/python"
    assert config["fisseq_manifest"] == "/work/tiles_manifest.csv"
    assert config["fisseq_image"] == "corrected"
    assert config["fisseq_window"] == 180
    assert config["fisseq_cp_features"] is True
    assert config["fisseq_segmentation_type"] == "cells"


def test_snakemake_config_leaves_wells_and_grid_size_to_starcall_when_unset():
    """Unset, the Snakefile falls back to starcall's own `wells` and
    `phenotyping_grid_size`."""
    config = _config()
    assert "fisseq_wells" not in config
    assert "fisseq_grid_size" not in config


def test_snakemake_config_passes_explicit_wells_and_grid_size():
    config = _config(wells=["well1", "well2"], grid_size=4)
    assert config["fisseq_wells"] == ["well1", "well2"]
    assert config["fisseq_grid_size"] == 4


def test_snakemake_config_is_plain_yaml():
    """Snakemake loads it with its own YAML reader: no Python-specific tags."""
    text = yaml.safe_dump(_config(wells=["well1"]))
    assert yaml.safe_load(text)["fisseq_wells"] == ["well1"]


# ---------------------------------------------------------------------------
# render_starcall_jobscript -- child-job image re-entry
# ---------------------------------------------------------------------------


def _fill(template: str, exec_job: str = "cd /work && python -m snakemake x") -> str:
    """What snakemake 7's ClusterExecutor.write_jobscript does."""
    return template.format(properties='{"rule": "r"}', exec_job=exec_job)


def test_render_starcall_jobscript_formats_like_snakemake():
    script = mod.render_starcall_jobscript(
        "apptainer", "/imgs/pipe.sif", ["/data", "/work"], gpu=False
    )
    filled = _fill(script)
    assert filled.startswith("#!/bin/sh\n# properties = {")
    assert (
        "exec apptainer exec --bind /data:/data,/work:/work /imgs/pipe.sif /bin/sh"
        in filled
    )
    assert filled.rstrip().endswith("cd /work && python -m snakemake x")
    assert "--nv" not in filled


def test_render_starcall_jobscript_passes_nv_for_gpu():
    script = mod.render_starcall_jobscript("singularity", "/i.sif", ["/d"], gpu=True)
    assert "exec singularity exec --nv --bind /d:/d /i.sif" in _fill(script)


def test_render_starcall_jobscript_quotes_paths_with_spaces():
    script = mod.render_starcall_jobscript(
        "apptainer", "/my imgs/p.sif", ["/a b"], False
    )
    assert "'/a b:/a b' '/my imgs/p.sif'" in _fill(script)


def test_render_starcall_jobscript_rejects_braces():
    with pytest.raises(ValueError, match="brace"):
        mod.render_starcall_jobscript("apptainer", "/i.sif", ["/data/{x}"], False)


def test_render_starcall_jobscript_reenters_image_once(tmp_path: Path):
    """Run the rendered script for real, with a fake container runtime
    that just runs its trailing command: the job body runs exactly once,
    inside the 'image'."""
    fake_bin = tmp_path / "fakeapptainer"
    # Drop "exec [--nv] --bind X IMAGE" and run the rest.
    fake_bin.write_text(
        '#!/bin/sh\nshift\n[ "$1" = --nv ] && shift\nshift 3\n'
        'echo entered >> "$LOG"\nexec "$@"\n'
    )
    fake_bin.chmod(0o755)
    script = mod.render_starcall_jobscript(str(fake_bin), "/i.sif", ["/d"], False)
    jobscript = tmp_path / "job.sh"
    jobscript.write_text(_fill(script, exec_job='echo ran >> "$LOG"'))

    import os
    import subprocess

    log = tmp_path / "log"
    env = {**os.environ, "LOG": str(log)}
    env.pop("FISSEQ_STARCALL_IN_IMAGE", None)
    subprocess.run(["/bin/sh", str(jobscript)], check=True, env=env)
    assert log.read_text().split() == ["entered", "ran"]


def test_render_starcall_jobscript_reenters_without_seeing_its_own_path(
    tmp_path: Path,
):
    """A scheduler may keep the jobscript in a node-local spool (SGE:
    /var/spool/uge/<host>/job_scripts/<id>) that isn't bound into the
    image, so the re-entered shell must not need to open "$0". The fake
    runtime deletes the script before running the inner command."""
    fake_bin = tmp_path / "fakeapptainer"
    fake_bin.write_text(
        '#!/bin/sh\nshift\n[ "$1" = --nv ] && shift\nshift 3\n'
        'rm -f "$JOBSCRIPT"\necho entered >> "$LOG"\nexec "$@"\n'
    )
    fake_bin.chmod(0o755)
    script = mod.render_starcall_jobscript(str(fake_bin), "/i.sif", ["/d"], False)
    jobscript = tmp_path / "job.sh"
    jobscript.write_text(_fill(script, exec_job='echo ran "$0" >> "$LOG"'))

    import os
    import subprocess

    log = tmp_path / "log"
    env = {**os.environ, "LOG": str(log), "JOBSCRIPT": str(jobscript)}
    env.pop("FISSEQ_STARCALL_IN_IMAGE", None)
    subprocess.run(["/bin/sh", str(jobscript)], check=True, env=env)
    assert log.read_text().split() == ["entered", "ran", str(jobscript)]


def test_jobscript_bind_paths_includes_cwd_and_dedupes():
    binds = mod.jobscript_bind_paths(
        {"phenotyping_dir": "/s/p", "sequencing_dir": "/s/q"},
        ["/s/p", "/cache"],
        "/work/ab/123",
    )
    assert binds == ["/cache", "/s/p", "/s/q", "/work/ab/123"]


def test_jobscript_bind_paths_adds_resolved_symlinks(tmp_path):
    # snakemake cd's each child job into os.getcwd() after entering
    # --directory, i.e. starcall_workflow_dir with symlinks resolved.
    real = tmp_path / "nobackup" / "exp"
    real.mkdir(parents=True)
    link = tmp_path / "exp"
    link.symlink_to(real)

    binds = mod.jobscript_bind_paths({}, [str(link)], "/work/ab/123")

    assert str(link) in binds
    assert os.path.realpath(link) in binds
    assert os.path.realpath(link).endswith("/nobackup/exp")
