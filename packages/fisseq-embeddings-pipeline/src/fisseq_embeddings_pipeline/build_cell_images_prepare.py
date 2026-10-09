"""BUILD_CELL_IMAGES, phase 1: prepare the nested snakemake run.

Hydra entry point (`python -m
fisseq_embeddings_pipeline.build_cell_images_prepare`), backing the first
of BUILD_CELL_IMAGES' three phases. Writes what has to exist before the
nested snakemake starts:

- `resolved_dirs_out`: the fully resolved `phenotyping_dir`/
  `segmentation_dir`/`sequencing_dir` (see :func:`resolve_data_dir`), a
  shell-sourceable `key='value'` file. The process sources it for its
  `env("phenotyping_dir")` output, EMBED_CELLS' bind path.
- `snakemake_config_out`: the `--configfile` of both snakemake calls --
  the same three directories (overriding the project's own config.yaml),
  the interpreter `make_cell_shard` runs this package with, and the
  `fisseq_*` settings of `snakemake/Snakefile`'s `fisseq_tiles_manifest`
  rule. That rule, not this phase, lists every tile and every well's shards
  and writes the two manifests phase 3 (`build_cell_images_table.py`)
  reads.
- `jobscript_out`, only when `starcall_job_image` is set (i.e. the run
  passes a `starcall_profile` for per-rule cluster submission): the
  `--jobscript` template every starcall child job runs through. See
  :func:`render_starcall_jobscript`.

`resolve_data_dir` resolves each of the three directories independently:
an explicit value always wins; otherwise starcall-workflow's own project
config (`{starcall_workflow_dir}/config.yaml`, or `default-config.yaml` if
that's absent -- the same files, in the same precedence, `workflow/
Snakefile` itself consults) is read for that key; only once neither file
sets it does this fall back to starcall-workflow's documented default
subdirectory (`phenotyping/`, `segmentation/`, `sequencing/`).
"""

import dataclasses
import logging
import os
import pathlib
import shlex
import sys
from typing import Any, Dict, List, Optional

import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.utils.log import setup_logging

from .config import AppConfig

# starcall-workflow's own default-config.yaml default for each directory
# key, relative to its own working directory -- see resolve_data_dir.
_DEFAULT_SUBDIR = {
    "phenotyping_dir": "phenotyping",
    "segmentation_dir": "segmentation",
    "sequencing_dir": "sequencing",
}

# The two project-config filenames workflow/Snakefile itself consults, in
# the same either/or precedence order (never both -- confirmed by reading
# that file directly: `if os.path.exists('config.yaml'): configfile:
# 'config.yaml' elif os.path.exists('default-config.yaml'): configfile:
# 'default-config.yaml'`).
_PROJECT_CONFIG_FILENAMES = ("config.yaml", "default-config.yaml")

_RESOLVED_DIR_KEYS = ("phenotyping_dir", "segmentation_dir", "sequencing_dir")


@dataclasses.dataclass
class BuildCellImagesPrepareConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CELL_IMAGES' prepare phase.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    starcall_workflow_dir : str
        The Snakemake checkout/working directory this experiment's
        `phenotyping_dir`/`segmentation_dir`/`sequencing_dir` default
        under, and whose own project config (`config.yaml`/
        `default-config.yaml`) is consulted first -- see
        :func:`resolve_data_dir`.
    phenotyping_dir, segmentation_dir, sequencing_dir : str or None
        starcall-workflow's own per-experiment output trees (real,
        unredirected paths -- see the `build_cell_images` rule's module docstring
        on why). All optional: ``None`` (the default) resolves via
        :func:`resolve_data_dir` against `starcall_workflow_dir`; set one
        explicitly only when that tree isn't colocated under
        `starcall_workflow_dir`.
    wells : list[str] or None
        Wells to build cell images for. ``None`` (the default) takes
        starcall-workflow's own (`wells`, from the project config or
        detected from its input tree).
    grid_size : int or None
        The tile grid each well is cut into. ``None`` (the default) takes
        the project config's `phenotyping_grid_size`.
    segmentation_type : str
        Defaults to ``"cells"``.
    use_corrected : bool
        Cut each tile's shard from the background-corrected whole-tile
        phenotype image (`corrected_pt.tif`) instead of the raw one
        (`raw_pt.tif`) -- mirrors starcall-workflow's own
        `get_phenotyping_pt`. Defaults to ``False``.
    window : int
        Crop size each cell is cut at, in the shards' directory name -- see
        ``tile_shard.TileShardConfig``. Must match the Cell-DINO
        checkpoint's expected input (`cell_dino_crop_size`).
    shard_size : int or None
        Cells per WebDataset shard, each well's shards counted separately
        (see ``well_shards.WellShardsConfig``); also in the shards'
        directory name. ``None`` (the default) gives each well one shard.
    barcode_col_name, aa_changes_col_name, edit_distance_col_name : str
        Which columns of each tile's reads table are the barcode, amino-acid
        changes and edit distance (starcall's aux tables name them per
        experiment). Both the shards' ``meta.json`` and
        ``cell_table.parquet`` rename them to ``meta_barcode``/
        ``meta_aa_changes``/``meta_edit_distance``. Default to
        ``"upBarcode"``/``"aaChanges"``/``"editDistance"``.
    sequencing_reads_params : str
        Suffix threaded into the reads CSV filename
        (`{segmentation_type}_reads{sequencing_reads_params}.csv`).
        Defaults to ``""``.
    cp_features : bool
        Whether to also build this experiment's CellProfiler CSV.
        Defaults to ``False``.
    cellprofiler_cycle, cellprofiler_pipeline : str
        Threaded into the CellProfiler CSV filename when `cp_features` is
        set. Default to ``""``.
    manifest_out : str
        The tile manifest the nested snakemake's `fisseq_tiles_manifest`
        rule writes (under `output_dir`), passed to it as an absolute
        path. Defaults to ``"tiles_manifest.csv"``.
    shards_manifest_out : str
        The shard manifest the same rule writes beside it, one row per
        shard. Defaults to ``"shards_manifest.csv"``.
    snakemake_config_out : str
        Output filename (under `output_dir`) for the nested snakemake's
        `--configfile` -- see :func:`snakemake_config`.
    starcall_job_image : str or None
        The image file (a ``.sif``) each starcall child job re-enters. Set
        only in cluster mode; when set, `jobscript_out` is written.
    starcall_container_bin : str
        The container runtime a child job re-enters the image with.
        Defaults to ``"apptainer"``; some nodes only ship ``singularity``.
    starcall_job_gpu : bool
        Pass ``--nv`` to that runtime. Apptainer only warns on a node with
        no GPU, so this is safe to leave on. Defaults to ``False``.
    jobscript_binds : list[str]
        Extra host paths to bind into each child job, on top of the ones
        this stage derives itself (see :func:`jobscript_bind_paths`). The
        process passes ``starcall_workflow_dir``, the snakemake cache dir and
        ``params.starcall_job_binds``.
    jobscript_out : str
        Output filename for the jobscript, written under `output_dir`.
    resolved_dirs_out : str
        Output filename (under `output_dir`) for the fully-resolved
        `phenotyping_dir`/`segmentation_dir`/`sequencing_dir` -- a
        shell-sourceable `key='value'` file, one line per key, so the
        process's `env("phenotyping_dir")` output is the exact path this
        phase resolved, without duplicating the resolution in Groovy.
    """

    starcall_workflow_dir: str = MISSING
    phenotyping_dir: Optional[str] = None
    segmentation_dir: Optional[str] = None
    sequencing_dir: Optional[str] = None
    wells: Optional[List[str]] = None
    grid_size: Optional[int] = None
    segmentation_type: str = "cells"
    use_corrected: bool = False
    window: int = MISSING
    shard_size: Optional[int] = None
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"
    sequencing_reads_params: str = ""
    cp_features: bool = False
    cellprofiler_cycle: str = ""
    cellprofiler_pipeline: str = ""
    manifest_out: str = "tiles_manifest.csv"
    shards_manifest_out: str = "shards_manifest.csv"
    snakemake_config_out: str = "snakemake_config.yaml"
    starcall_job_image: Optional[str] = None
    starcall_container_bin: str = "apptainer"
    starcall_job_gpu: bool = False
    jobscript_binds: List[str] = dataclasses.field(default_factory=list)
    jobscript_out: str = "starcall_jobscript.sh"
    resolved_dirs_out: str = "resolved_dirs.env"


def resolve_data_dir(
    starcall_workflow_dir: str, dir_key: str, explicit: Optional[str]
) -> str:
    """Resolve one of phenotyping_dir/segmentation_dir/sequencing_dir.

    Precedence, matching `workflow/Snakefile`'s own resolution exactly (so
    this stays correct even when a project's own config.yaml remaps these
    paths -- "handles cases where the output looks different for whatever
    reason", not just the documented-default case):

    1. ``explicit``, if given.
    2. ``dir_key`` read from ``starcall_workflow_dir``'s own project
       config -- ``config.yaml`` if present, else ``default-config.yaml``
       (never both; a project's ``config.yaml`` existing at all stops
       ``default-config.yaml`` from being consulted, exactly like
       `workflow/Snakefile`'s own `if os.path.exists('config.yaml'): ...
       elif os.path.exists('default-config.yaml'): ...`). A relative value
       there (starcall-workflow's own convention, e.g. ``'phenotyping/'``)
       is resolved against ``starcall_workflow_dir`` -- the same directory
       Snakemake itself runs relative to (`--directory`); an absolute
       value is used as-is.
    3. ``{starcall_workflow_dir}/{bare_name}`` (e.g. ``.../phenotyping``),
       matching starcall-workflow's own ``default-config.yaml`` default
       for the common case where a project doesn't override this key at
       all (confirmed by reading `rules/config.smk`'s own
       ``config.get(dir_key, '<bare_name>/')`` -- this fallback is exactly
       that default, resolved against the same working directory).

    Parameters
    ----------
    dir_key : str
        One of ``"phenotyping_dir"``, ``"segmentation_dir"``,
        ``"sequencing_dir"``.
    """
    if explicit:
        return explicit

    bare_name = _DEFAULT_SUBDIR[dir_key]
    for config_filename in _PROJECT_CONFIG_FILENAMES:
        config_path = os.path.join(starcall_workflow_dir, config_filename)
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as f:
            project_config = yaml.safe_load(f) or {}
        value = project_config.get(dir_key)
        if value:
            value = str(value).rstrip("/")
            return (
                value
                if os.path.isabs(value)
                else os.path.join(starcall_workflow_dir, value)
            )
        break  # config.yaml exists (even without dir_key set) -- don't
        # also fall through to default-config.yaml, matching the
        # Snakefile's own either/or (never both) precedence.

    return os.path.join(starcall_workflow_dir, bare_name)


def snakemake_config(
    cfg: BuildCellImagesPrepareConfig,
    resolved_dirs: Dict[str, str],
    output_dir: str,
    fisseq_python: str,
) -> Dict[str, Any]:
    """The nested snakemake's ``--configfile`` contents.

    The three resolved directories, each with a trailing ``/``: starcall's
    rules build every path by plain string concatenation onto them,
    matching its own ``'phenotyping/'``-style defaults, so without it a
    path wildcard silently comes out malformed. Command-line config wins
    over the project's own config.yaml, so these are the directories
    starcall uses.

    ``fisseq_python`` is the interpreter ``make_cell_shard`` and
    ``make_well_shards`` run this package with; the other ``fisseq_*`` keys
    are the settings of the ``fisseq_shards`` and ``fisseq_tiles_manifest``
    rules (``snakemake/Snakefile``). The two manifests go under
    ``output_dir``, which must be absolute: the nested snakemake runs in
    ``starcall_workflow_dir``. ``wells``/``grid_size`` are left out when
    unset, so the Snakefile falls back to starcall's own. Phase 3
    (``build_cell_images_table``) reads the ``fisseq_*_col`` keys back
    from the same file, so the table and the shards' ``meta.json`` rename
    the same columns.
    """
    config: Dict[str, Any] = {k: v.rstrip("/") + "/" for k, v in resolved_dirs.items()}
    config["fisseq_python"] = fisseq_python
    config["fisseq_manifest"] = os.path.join(output_dir, cfg.manifest_out)
    config["fisseq_shards_manifest"] = os.path.join(output_dir, cfg.shards_manifest_out)
    if cfg.wells:
        config["fisseq_wells"] = list(cfg.wells)
    if cfg.grid_size is not None:
        config["fisseq_grid_size"] = cfg.grid_size
    config.update(
        {
            "fisseq_segmentation_type": cfg.segmentation_type,
            "fisseq_image": "corrected" if cfg.use_corrected else "raw",
            "fisseq_window": cfg.window,
            "fisseq_shard_size": cfg.shard_size,
            "fisseq_barcode_col": cfg.barcode_col_name,
            "fisseq_aa_changes_col": cfg.aa_changes_col_name,
            "fisseq_edit_distance_col": cfg.edit_distance_col_name,
            "fisseq_sequencing_reads_params": cfg.sequencing_reads_params,
            "fisseq_cp_features": cfg.cp_features,
            "fisseq_cellprofiler_cycle": cfg.cellprofiler_cycle,
            "fisseq_cellprofiler_pipeline": cfg.cellprofiler_pipeline,
        }
    )
    return config


def jobscript_bind_paths(
    resolved_dirs: Dict[str, str], extra: List[str], cwd: str
) -> List[str]:
    """Every host path a starcall child job can touch, deduplicated.

    Each path is bound both as given and at its ``os.path.realpath``:
    starcall's rules use the paths as given, but snakemake prefixes every
    cluster job with ``cd <workdir>`` (``ClusterExecutor.
    get_job_exec_prefix``), and it records that workdir with ``os.getcwd()``
    after changing into ``--directory`` (``starcall_workflow_dir``), so
    symlinks in it come out resolved. A ``starcall_workflow_dir`` reached
    through a symlink is otherwise missing from the image at the path the
    job ``cd``s into.

    ``cwd`` (this task's work directory, where the jobscript, snakemake
    config and tile manifest live) is bound as well.
    """
    paths = {*resolved_dirs.values(), *extra, cwd}
    return sorted({q for p in paths for q in (p, os.path.realpath(p))})


def render_starcall_jobscript(
    container_bin: str, image: str, binds: List[str], gpu: bool
) -> str:
    """The ``--jobscript`` template each starcall child job runs through.

    A child job lands on a bare node, but its snakemake command
    (``{exec_job}``) names the image's own ops-env interpreter, and
    starcall's ``run:`` rule bodies execute inside that interpreter -- so
    the job has to run inside the pipeline image. The script re-executes
    itself there once (guarded by an environment variable, which Apptainer
    passes through by default), then runs the job as usual.

    It re-enters with its contents (``sh -c "$(cat "$0")"``), not its
    path: a scheduler may keep the script in a node-local spool (SGE's
    ``/var/spool/<cell>/<host>/job_scripts/``) that isn't bound into the
    image.

    Every path is bound at its own unchanged location because starcall's
    rules build output paths by concatenating strings onto phenotyping_dir
    and friends: a path that merely reaches the data isn't enough.

    Snakemake fills in ``{properties}``/``{exec_job}`` with ``str.format``,
    so the script must contain no other braces.
    """
    bind_arg = ",".join(f"{p}:{p}" for p in binds)
    runtime_args = ["exec"]
    if gpu:
        runtime_args.append("--nv")
    runtime_args += ["--bind", bind_arg, image]
    command = " ".join(shlex.quote(a) for a in [container_bin, *runtime_args])
    script = f"""#!/bin/sh
# properties = {{properties}}
# Written by fisseq_embeddings_pipeline.build_cell_images_prepare.
if [ -z "$FISSEQ_STARCALL_IN_IMAGE" ]; then
    FISSEQ_STARCALL_IN_IMAGE=1
    export FISSEQ_STARCALL_IN_IMAGE
    exec {command} /bin/sh -c "$(cat "$0")" "$0" "$@"
fi
{{exec_job}}
"""
    if "{" in script.replace("{properties}", "").replace("{exec_job}", ""):
        raise ValueError(
            "starcall jobscript would contain a literal brace (from a bind path "
            f"or image name), which snakemake's str.format would reject: {script!r}"
        )
    return script


_cs = ConfigStore.instance()
_cs.store(name="build_cell_images_prepare_main", node=BuildCellImagesPrepareConfig)


@hydra.main(
    version_base=None,
    config_path=None,
    config_name="build_cell_images_prepare_main",
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: resolve the data directories, write the nested
    snakemake's config (and, in cluster mode, its jobscript).

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.build_cell_images_prepare \\
            output_dir=./out \\
            starcall_workflow_dir=/data/experiment1 \\
            segmentation_type=cells \\
            window=224
    """
    prep_cfg: BuildCellImagesPrepareConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(prep_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_cfg.output_dir = str(output_dir)
    setup_logging(prep_cfg, "build_cell_images_prepare")

    resolved_dirs = {
        dir_key: resolve_data_dir(
            prep_cfg.starcall_workflow_dir, dir_key, getattr(prep_cfg, dir_key)
        )
        for dir_key in _RESOLVED_DIR_KEYS
    }
    resolved_dirs_path = output_dir / prep_cfg.resolved_dirs_out
    with open(resolved_dirs_path, "w") as f:
        for dir_key in _RESOLVED_DIR_KEYS:
            f.write(f"{dir_key}='{resolved_dirs[dir_key]}'\n")
    logging.info("Resolved starcall-workflow directories: %s", resolved_dirs)

    config_path = output_dir / prep_cfg.snakemake_config_out
    with open(config_path, "w") as f:
        yaml.safe_dump(
            snakemake_config(
                prep_cfg, resolved_dirs, os.path.abspath(output_dir), sys.executable
            ),
            f,
            sort_keys=False,
        )
    logging.info("Wrote snakemake config %s", config_path)

    if prep_cfg.starcall_job_image:
        binds = jobscript_bind_paths(
            resolved_dirs,
            list(prep_cfg.jobscript_binds),
            os.path.abspath(os.getcwd()),
        )
        jobscript_path = output_dir / prep_cfg.jobscript_out
        jobscript_path.write_text(
            render_starcall_jobscript(
                prep_cfg.starcall_container_bin,
                prep_cfg.starcall_job_image,
                binds,
                prep_cfg.starcall_job_gpu,
            )
        )
        jobscript_path.chmod(0o755)
        logging.info("Wrote starcall jobscript %s (binds: %s)", jobscript_path, binds)


if __name__ == "__main__":
    main()
