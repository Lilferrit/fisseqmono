"""Tests for fisseq_common.stages.config.

Covers a ConfigStore-registered stage config's default random_seed (0) and
that it's overridable via Hydra CLI override syntax -- the one shared seed
field, reused by every stage -- that every shared stage is an entry point
without a seed of its own, and the cell-identity helpers.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import fisseq_common.stages
from fisseq_common.schema import EMBEDDING_SELECTOR
from fisseq_common.stages.config import (
    DATA_JOIN_KEYS,
    EMBEDDINGS_JOIN_KEYS,
    AppConfig,
    CellsInput,
    feature_selector,
    row_keys,
)


@dataclasses.dataclass
class _DummyStageConfig(AppConfig):
    """A minimal stage config, standing in for a real pipeline stage config,
    to confirm AppConfig's random_seed default/override behavior end to end
    through Hydra's own config-composition machinery rather than just
    dataclass defaults."""

    pass


def test_appconfig_default_random_seed_is_zero():
    assert AppConfig().random_seed == 0


def test_dummy_stage_config_inherits_random_seed_default():
    assert _DummyStageConfig(output_dir="/tmp/out").random_seed == 0


def test_random_seed_overridable_via_omegaconf_merge():
    """Stands in for a Hydra CLI override (`random_seed=42`) -- OmegaConf.merge
    is what Hydra's CLI-override parsing ultimately calls into."""
    base = OmegaConf.structured(_DummyStageConfig(output_dir="/tmp/out"))
    overridden = OmegaConf.merge(base, {"random_seed": 42})
    assert overridden.random_seed == 42
    assert base.random_seed == 0  # base config untouched


def test_random_seed_overridable_via_hydra_compose():
    """The real end-to-end path: a ConfigStore-registered node, composed via
    Hydra with a CLI-override-style string, the way every stage's `main()`
    actually runs."""
    cs = ConfigStore.instance()
    cs.store(name="dummy_stage_test", node=_DummyStageConfig)

    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="dummy_stage_test",
            overrides=["random_seed=42", "output_dir=/tmp/out"],
        )
    assert cfg.random_seed == 42


# ---------------------------------------------------------------------------
# Every shared stage: a Hydra entry point, and no stage-local seed
# ---------------------------------------------------------------------------

#: The modules ``python -m fisseq_common.stages.<stage>`` runs.
STAGES = [
    "qcfilter",
    "filter",
    "ovwt",
    "aggregate",
    "generatesplit",
    "correlatefeatures",
    "blocklist",
    "combineblocklists",
    "finalize",
]

# Field names that would reintroduce a second, independent seed.
_BANNED_SEED_FIELDS = {"random_state", "seed", "downsample_seed", "umap_random_state"}


def _stage_config_classes() -> list[type]:
    """Every AppConfig subclass defined in fisseq_common.stages."""
    found: dict[str, type] = {}
    for info in pkgutil.iter_modules(fisseq_common.stages.__path__):
        module = importlib.import_module(f"fisseq_common.stages.{info.name}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, AppConfig)
                and dataclasses.is_dataclass(obj)
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return sorted(found.values(), key=lambda c: (c.__module__, c.__qualname__))


def test_every_stage_is_an_entry_point():
    for stage in STAGES:
        module = importlib.import_module(f"fisseq_common.stages.{stage}")
        assert callable(getattr(module.main, "__wrapped__", None)), stage


def test_discovery_finds_the_stage_configs():
    """Guards the guard: an import failure must not silently empty the sweep."""
    names = {c.__name__ for c in _stage_config_classes()}
    assert {"QcFilterParams", "FilterParams", "OvwtConfig", "AggregateConfig"} <= names
    assert {"GenerateSplitParams", "FinalizeConfig"} <= names


@pytest.mark.parametrize(
    "cfg_cls", _stage_config_classes(), ids=lambda c: f"{c.__module__}.{c.__name__}"
)
def test_no_stage_local_seed_field(cfg_cls):
    """``AppConfig.random_seed`` is the one seed; a stage that must differ from its siblings
    derives a fixed offset from it (e.g. GENERATE_SPLIT uses ``random_seed + bootstrap_idx``)."""
    fields = {f.name for f in dataclasses.fields(cfg_cls)}
    offenders = fields & _BANNED_SEED_FIELDS
    assert not offenders, (
        f"{cfg_cls.__module__}.{cfg_cls.__name__} declares stage-local seed "
        f"field(s) {sorted(offenders)}. Use AppConfig.random_seed instead -- "
        f"derive a fixed offset from it if this stage must differ from its "
        f"siblings."
    )


def test_row_keys_add_the_variant_tag_once():
    assert row_keys(list(EMBEDDINGS_JOIN_KEYS)) == list(EMBEDDINGS_JOIN_KEYS) + [
        "meta_variant_tag"
    ]
    assert row_keys(DATA_JOIN_KEYS) == list(DATA_JOIN_KEYS)


def test_cells_input_defaults_to_the_data_join_keys():
    assert CellsInput().join_keys == list(DATA_JOIN_KEYS)
    assert CellsInput().feature_selector == "features"


def test_unknown_feature_selector_raises():
    assert feature_selector("embeddings") is EMBEDDING_SELECTOR
    with pytest.raises(ValueError, match="Unknown feature_selector"):
        feature_selector("nope")
