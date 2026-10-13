"""EMBED_CELLS.

Hydra entry point (`python -m fisseq_embeddings_pipeline.embed`), backing
the pipeline's only GPU-bound stage. Streams every cell in an experiment's
per-well WebDataset shards (written by the nested snakemake's
``make_well_shards`` rule -- see ``well_shards.py`` -- and listed in
BUILD_CELL_IMAGES' ``shards.parquet``) through a pretrained Cell-DINO
checkpoint (Meta's dinov2) and writes one row per cell to
embeddings.parquet. Not gated by QC_FILTER: this GPU pass runs once per
experiment regardless of how many times QC thresholds get retuned
afterward.

Each sample's ``meta.json`` carries the cell's key, QC fields and variant
class (``tile_shard.py``); ``embeddings.parquet`` gets the same seven
``meta_*`` columns QC_FILTER saw (``utils.cell_table.cell_metadata_exprs``),
with ``meta_batch`` from ``batch_stem`` -- a run-level name the cached
shards don't carry.

`load_cell_dino()` builds the backbone via dinov2's architecture factory
functions (dict-dispatched by `cfg.arch`) and ports
`dinov2.utils.utils.load_pretrained_weights()`'s loading logic. The vendored
model (`.vendor.dinov2`) is not installed as the `dinov2` package -- see
`vendor/dinov2/VENDORED_FROM.md`.

Real Cell-DINO checkpoints are not all shaped the same way, so
`load_cell_dino()` recovers `in_chans`, `img_size`, `block_chunks` and
LayerScale presence from the checkpoint's own state dict instead of
hardcoding one architecture. `embed_batch()` branches on the loaded model's
`patch_embed.in_chans`: `1` (bag-of-channels) gets the per-channel
split-and-pool treatment (`cfg.channel_pool`); anything else is fed to the
model jointly, requiring the crop's channel count to match exactly. See
docs/architecture.md for the two real checkpoints this was verified against.

`load_embedding_dataloader()` expands a non-brace `shard_pattern` via
`glob.glob()` itself: the installed `webdataset` only expands brace
patterns, so a bare `*` glob would pass through as a literal filename.
"""

import dataclasses
import glob
import logging
import pathlib
import re
from typing import Callable, Dict, List, Optional

import hydra
import polars as pl
import torch
import webdataset as wds
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from polars import selectors as cs

from fisseq_common.utils.log import setup_logging

from .config import AppConfig
from .utils.cell_table import (
    CELL_METADATA_SCHEMA,
    cell_metadata_exprs,
)
from .vendor.dinov2.models.vision_transformer import (
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)

# Named architectures this pipeline supports, mirroring dinov2's own
# vits.__dict__[arch] dispatch (build_model is not vendored).
_ARCHS: Dict[str, Callable[..., torch.nn.Module]] = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_giant2": vit_giant2,
}

# Key real Cell-DINO/dinov2 teacher checkpoints are stored under.
_CHECKPOINT_KEY = "teacher"


@dataclasses.dataclass
class EmbedCellsConfig(AppConfig):
    """
    Hydra structured configuration for EMBED_CELLS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    EMBED_CELLS's own logic doesn't consume random_seed itself (inference
    is deterministic given a fixed checkpoint), but every stage config
    inherits it uniformly.

    Attributes
    ----------
    shards_path : str or None
        BUILD_CELL_IMAGES' ``shards.parquet``: every row's ``shard_tar`` is
        one of this experiment's shards. Exactly one of ``shards_path`` and
        ``shard_pattern`` must be set; the pipeline uses this one.
    shard_pattern : str or None
        Path/brace pattern for a set of shards instead, e.g.
        ``"dataset-{000000..000042}.tar"``. A bare glob (``"dataset-*.tar"``)
        also works; ``load_embedding_dataloader`` expands it itself.
    batch_stem : str
        This experiment's identifier, written into every row as
        ``meta_batch`` (the shards' ``meta.json`` doesn't carry it).
    checkpoint_path : str
        Path to the Cell-DINO checkpoint (``.pth``), e.g.
        ``weights/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth``,
        a bag-of-channels ``vit_large``/patch-16 checkpoint matching this
        config's defaults (see docs/architecture.md's "Real checkpoint"
        section). Deliberately not defaulted here or in ``params.yaml``: the
        path is deployment-specific and ``weights/`` is gitignored.
    arch : str
        Backbone architecture, one of ``vit_small``/``vit_base``/
        ``vit_large``/``vit_giant2``. Defaults to ``"vit_large"``
        (embed dim 1024).
    patch_size : int
        ViT patch size. Defaults to ``16``.
    crop_size : int
        Expected crop size -- must match the shards' ``window``.
        Defaults to ``224``. Only a fallback for model construction when
        the checkpoint's ``pos_embed`` can't be inspected; it doesn't
        constrain the crop size that can be embedded.
    channels : list[int]
        Which of the crop's channel indices to feed into the model, in
        this order -- lets a crop carry more channels (e.g. multiple
        imaging cycles, flattened cycle-major per tile_shard.py) than the
        model should see. Defaults to ``[0, 1, 2, 3]`` (the first imaging
        cycle's four phenotyping channels, starcall-workflow's own
        default -- see docs/configuration.md).
    apply_mask : bool
        Whether ``mask.npy``-based background zeroing is applied before
        embedding -- to every selected channel, or to none. A single flag
        because each cell has exactly one shared ``mask.npy``, cut from
        starcall's single-plane segmentation mask (see
        ``tile_shard.crop_cell``). Defaults to ``True``.
    channel_pool : str
        How per-channel CLS embeddings are pooled into one per-cell
        embedding: ``"mean"`` or ``"max"``. Only consulted when the loaded
        model is bag-of-channels (``patch_embed.in_chans == 1`` -- see
        :func:`embed_batch`); ignored for a fixed-channel-count backbone.
        Defaults to ``"mean"``.
    device : str
        torch device string. Defaults to ``"cuda"``.
    batch_size : int
        Cells per dataloader batch. Defaults to ``256``.
    num_workers : int
        ``webdataset``/``DataLoader`` worker processes. Defaults to ``4``.
    """

    shards_path: Optional[str] = None
    shard_pattern: Optional[str] = None
    batch_stem: str = MISSING
    checkpoint_path: str = MISSING
    arch: str = "vit_large"
    patch_size: int = 16
    crop_size: int = 224
    channels: List[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3])
    apply_mask: bool = True
    channel_pool: str = "mean"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 4


def shard_urls(cfg: EmbedCellsConfig) -> "str | List[str]":
    """This run's shards: ``shards_path``'s ``shard_tar`` column, or
    ``shard_pattern`` (a brace pattern passed through, a glob expanded).

    Raises
    ------
    ValueError
        Unless exactly one of ``shards_path``/``shard_pattern`` is set; if a
        non-brace ``shard_pattern`` matches no files; or if ``shards_path``
        lists a shard that doesn't exist (its cells would otherwise just
        silently go missing).
    """
    if (cfg.shards_path is None) == (cfg.shard_pattern is None):
        raise ValueError("Set exactly one of shards_path and shard_pattern.")
    if cfg.shards_path is not None:
        urls = pl.read_parquet(cfg.shards_path)["shard_tar"].to_list()
        missing = [u for u in urls if not pathlib.Path(u).is_file()]
        if missing:
            raise ValueError(
                f"{len(missing)} shard(s) listed in {cfg.shards_path!r} don't "
                f"exist, e.g. {missing[0]!r}"
            )
        return urls
    if "{" in cfg.shard_pattern:
        return cfg.shard_pattern
    urls = sorted(glob.glob(cfg.shard_pattern))
    if not urls:
        raise ValueError(
            f"shard_pattern={cfg.shard_pattern!r} matched no files "
            "(not a brace pattern, and glob.glob found nothing)"
        )
    return urls


def load_embedding_dataloader(cfg: EmbedCellsConfig) -> "torch.utils.data.DataLoader":
    """Stream ``(key, crop, mask, meta)`` batches from this run's shards.

    Reads the shards ``write_tile_shard()`` (tile_shard.py) writes and
    ``write_well_shards()`` (well_shards.py) packs. Batches come in shard
    order via ``.batched()``/``DataLoader(batch_size=None)``, since tar
    shards don't support random access efficiently. ``mask.npy`` is always
    fetched; :func:`embed_batch` decides via ``cfg.apply_mask`` whether to
    apply it.

    Which shards are read is :func:`shard_urls`'s call.

    Parameters
    ----------
    cfg : EmbedCellsConfig
        Supplies ``shards_path``/``shard_pattern``, ``batch_size``,
        ``num_workers``.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(keys, crops, masks, metas)`` per batch: ``keys`` is a
        ``list[str]``, ``crops``/``masks`` are stacked ``torch.Tensor``
        (``(B, C, H, W)`` ``uint16`` / ``(B, H, W)`` ``uint8``), ``metas``
        is a ``list[dict]``.

    """
    urls = shard_urls(cfg)
    dataset = (
        wds.WebDataset(urls, shardshuffle=False)
        .decode()
        .to_tuple("__key__", "crop.npy", "mask.npy", "meta.json")
        .batched(cfg.batch_size)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=cfg.num_workers
    )


# Chunked-block state-dict key, e.g. "blocks.1.3.norm1.weight" (dinov2's
# block_chunks>0 layout: chunk index, then position in the unchunked sequence).
_CHUNKED_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.(\d+)\.")
# Matches a flat (block_chunks=0) state-dict key, e.g. "blocks.3.norm1.weight".
_FLAT_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.")
# LayerScale only registers these params when a block was built with a truthy
# init_values; their presence tells us to pass a placeholder nonzero
# init_values, rather than silently dropping every learned residual scale to
# nn.Identity().
_LAYERSCALE_KEY_SUFFIXES = (".ls1.gamma", ".ls2.gamma")


def _infer_patch_embed_shape(
    state: dict, fallback_patch_size: int
) -> "tuple[int, Optional[int]]":
    """Recover ``(in_chans, patch_size)`` from ``patch_embed.proj.weight``'s
    shape (``(embed_dim, in_chans, patch_size, patch_size)``), the most
    direct source of what the checkpoint expects. Returns
    ``patch_size=None`` (skip the cross-check) if the key is missing.
    """
    weight = state.get("patch_embed.proj.weight")
    if weight is None:
        logging.warning(
            "checkpoint has no patch_embed.proj.weight key -- assuming "
            "bag-of-channels (in_chans=1); this may be wrong"
        )
        return 1, None
    return weight.shape[1], weight.shape[-1]


def _infer_img_size(state: dict, patch_size: int, fallback_crop_size: int) -> int:
    """Recover the square ``img_size`` the checkpoint's ``pos_embed`` was
    trained at from its patch-token count. Not ``cfg.crop_size``: it only
    reproduces ``pos_embed``'s shape so ``load_state_dict`` doesn't raise;
    ``interpolate_pos_encoding`` reconciles any difference with the real
    input size at inference time.
    """
    pos_embed = state.get("pos_embed")
    if pos_embed is None:
        logging.warning(
            "checkpoint has no pos_embed key -- falling back to "
            "cfg.crop_size=%d for model construction",
            fallback_crop_size,
        )
        return fallback_crop_size
    num_patches = pos_embed.shape[1] - 1
    grid = round(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(
            f"checkpoint's pos_embed has {num_patches} patch positions, not "
            "a perfect square -- can't recover a square img_size from it"
        )
    return grid * patch_size


def _infer_block_chunks(state: dict) -> int:
    """Recover ``block_chunks`` from whether block keys are chunked
    (``blocks.<chunk>.<pos>....``) or flat (``blocks.<pos>....``); the
    chunk-index count is ``block_chunks``. Defaults to ``1`` (the class's
    own default) if no block keys are found.
    """
    chunk_ids = {int(m.group(1)) for k in state if (m := _CHUNKED_BLOCK_KEY.match(k))}
    if chunk_ids:
        return max(chunk_ids) + 1
    flat_ids = {int(m.group(1)) for k in state if (m := _FLAT_BLOCK_KEY.match(k))}
    return 0 if flat_ids else 1


def load_cell_dino(cfg: EmbedCellsConfig) -> torch.nn.Module:
    """Build a DINOv2 ViT matching the checkpoint's own architecture and load it.

    Built through the named architecture factory (see ``_ARCHS``), not
    ``dinov2.eval.setup``/``build_model_from_cfg``, which need a
    training-style cfg this pipeline doesn't have.

    ``in_chans``, ``img_size``, ``block_chunks`` and LayerScale presence are
    inferred from the checkpoint's state dict rather than assumed. See
    docs/architecture.md for the two real checkpoints this was verified
    against, one of which is not bag-of-channels.

    Loading ports ``dinov2.utils.utils.load_pretrained_weights``: index by
    ``"teacher"`` if present, strip the ``module.``/``backbone.`` prefixes
    the training-time multicrop/DDP wrapper adds, then a **non-strict**
    ``load_state_dict``. ``unexpected_keys`` (head/EMA-only keys) are only
    logged. ``missing_keys`` raise: this pipeline's ``head`` is always
    ``nn.Identity()``, so any missing key means the architecture didn't
    match and backbone weights would silently stay randomly initialized.

    Parameters
    ----------
    cfg : EmbedCellsConfig
        Supplies ``arch``, ``patch_size``, ``crop_size``, ``checkpoint_path``,
        ``device``.

    Returns
    -------
    torch.nn.Module
        The loaded backbone, on ``cfg.device`` and in ``eval()`` mode.

    Raises
    ------
    ValueError
        If ``cfg.arch`` isn't one of ``_ARCHS``, if ``cfg.patch_size``
        doesn't match the checkpoint's own patch_embed kernel size, or if
        the checkpoint's ``pos_embed`` can't be reduced to a square grid.
    RuntimeError
        If any parameter the constructed model needs is missing from the
        (non-strict-loaded) checkpoint.
    """
    if cfg.arch not in _ARCHS:
        raise ValueError(f"Unknown arch {cfg.arch!r}, expected one of {sorted(_ARCHS)}")

    state = torch.load(cfg.checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and _CHECKPOINT_KEY in state:
        logging.info("Taking key %r from checkpoint dict", _CHECKPOINT_KEY)
        state = state[_CHECKPOINT_KEY]
    state = {
        k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()
    }

    in_chans, checkpoint_patch_size = _infer_patch_embed_shape(state, cfg.patch_size)
    if checkpoint_patch_size is not None and checkpoint_patch_size != cfg.patch_size:
        raise ValueError(
            f"cfg.patch_size={cfg.patch_size} but checkpoint_path="
            f"{cfg.checkpoint_path!r}'s patch_embed kernel is "
            f"{checkpoint_patch_size}x{checkpoint_patch_size}"
        )
    img_size = _infer_img_size(state, cfg.patch_size, cfg.crop_size)
    block_chunks = _infer_block_chunks(state)
    has_layerscale = any(k.endswith(_LAYERSCALE_KEY_SUFFIXES) for k in state)

    model = _ARCHS[cfg.arch](
        patch_size=cfg.patch_size,
        in_chans=in_chans,
        channel_adaptive=(in_chans == 1),
        img_size=img_size,
        block_chunks=block_chunks,
        init_values=(1.0 if has_layerscale else None),
    )

    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        raise RuntimeError(
            f"checkpoint_path={cfg.checkpoint_path!r} left "
            f"{len(msg.missing_keys)} backbone parameter(s) unloaded -- "
            f"arch/patch_size/in_chans/block_chunks mismatch? first few: "
            f"{msg.missing_keys[:10]}"
        )
    logging.info(
        "Loaded checkpoint %s (arch=%s, in_chans=%d, img_size=%d, "
        "block_chunks=%d, layerscale=%s), %d unexpected (ignored) key(s): %s",
        cfg.checkpoint_path,
        cfg.arch,
        in_chans,
        img_size,
        block_chunks,
        has_layerscale,
        len(msg.unexpected_keys),
        msg.unexpected_keys,
    )
    return model.to(cfg.device).eval()


@torch.no_grad()
def embed_batch(
    model: torch.nn.Module,
    crops: torch.Tensor,
    masks: torch.Tensor,
    cfg: EmbedCellsConfig,
) -> torch.Tensor:
    """
    Embed a batch of multi-channel cell crops.

    Selects/reorders ``cfg.channels`` out of the crop's channel axis (it
    may carry more than the model should see, e.g. several imaging cycles,
    cycle-major per tile_shard.py), optionally applies the shared cell mask
    (``cfg.apply_mask``), then branches on the model's ``patch_embed.in_chans``
    (set by ``load_cell_dino()`` from the checkpoint, not a config flag):

    - ``in_chans == 1`` (bag-of-channels, as
      ``weights/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth``
      is): the crop is split into ``C`` single-channel images run through the
      same backbone, and the per-channel CLS tokens are pooled
      (``cfg.channel_pool``) into one embedding per cell.
    - any other ``in_chans``: the channels are stacked jointly in one
      forward pass; ``len(cfg.channels)`` must match exactly.

    Parameters
    ----------
    model : torch.nn.Module
        As built by :func:`load_cell_dino` (or an equivalent, e.g. a
        shrunk test model); its ``patch_embed.in_chans`` selects the mode.
    crops : torch.Tensor
        Shape ``(B, C, crop_size, crop_size)`` -- ``C`` may exceed
        ``len(cfg.channels)``, since channel selection happens here, not
        upstream. Any real dtype (typically ``uint16``); cast to
        ``float32`` before the forward pass.
    masks : torch.Tensor
        Shape ``(B, crop_size, crop_size)``, ``uint8`` label mask --
        nonzero where a pixel belongs to the target cell. One shared mask
        per cell, applied to every selected channel. Only read when
        ``cfg.apply_mask`` is ``True``.
    cfg : EmbedCellsConfig
        Supplies ``channels``, ``apply_mask``, and (bag-of-channels mode
        only) ``channel_pool``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, D)`` -- one embedding per cell, on the model's device.
        ``D`` = backbone width.

    Raises
    ------
    ValueError
        If ``cfg.channels`` has an index out of range for the crop, if
        ``cfg.channel_pool`` isn't recognized, or if the model expects a
        fixed channel count ``cfg.channels`` doesn't provide.
    """
    if any(idx < 0 or idx >= crops.shape[1] for idx in cfg.channels):
        raise ValueError(
            f"cfg.channels={cfg.channels} out of range for a crop with "
            f"{crops.shape[1]} channel(s)"
        )

    device = next(model.parameters()).device
    crops = crops.to(device=device, dtype=torch.float32)
    crops = crops[:, cfg.channels, :, :]

    if cfg.apply_mask:
        masks = masks.to(device=device)
        # one shared mask per cell, broadcast over the channel axis
        crops = crops * (masks > 0).unsqueeze(1)  # (B, 1, H, W)

    b, c, h, w = crops.shape
    in_chans = model.patch_embed.in_chans

    if in_chans != 1:
        if c != in_chans:
            raise ValueError(
                f"model expects in_chans={in_chans} (not bag-of-channels) "
                f"but cfg.channels selects {c} channel(s)"
            )
        return model(crops)  # (B, D) CLS embeddings, one joint forward pass

    per_channel = crops.reshape(b * c, 1, h, w)  # bag: each channel is its own "image"
    tokens = model(per_channel)  # (B*C, D) CLS embeddings
    tokens = tokens.reshape(b, c, -1)
    if cfg.channel_pool == "mean":
        return tokens.mean(dim=1)
    elif cfg.channel_pool == "max":
        return tokens.max(dim=1).values
    raise ValueError(f"Unknown channel_pool {cfg.channel_pool!r}")


_cs = ConfigStore.instance()
_cs.store(name="embed_main", node=EmbedCellsConfig)


@hydra.main(version_base=None, config_path=None, config_name="embed_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: embed every cell in a Cell Dataset via Cell-DINO.

    Writes one row per cell to ``embeddings.parquet``: the ``meta_*``
    columns (``cell_metadata_exprs``) plus zero-padded
    ``emb_0000``..``emb_{D-1}`` columns, which ``EMBEDDING_SELECTOR``
    (``fisseq_common.schema``) matches.

    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.embed \\
            output_dir=./out \\
            shards_path=./shards.parquet \\
            batch_stem=experiment1 \\
            checkpoint_path=/data/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth \\
            device=cpu \\
            'channels=[0,1,2,3]' \\
            apply_mask=true \\
            random_seed=0
    """
    embed_cfg: EmbedCellsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(embed_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embed_cfg.output_dir = str(output_dir)
    setup_logging(embed_cfg, "embed")

    logging.info(
        "Embedding cells from %s (arch=%s, checkpoint=%s, device=%s)",
        embed_cfg.shards_path or embed_cfg.shard_pattern,
        embed_cfg.arch,
        embed_cfg.checkpoint_path,
        embed_cfg.device,
    )

    dataloader = load_embedding_dataloader(embed_cfg)
    model = load_cell_dino(embed_cfg)

    rows = []
    n_cells = 0
    for keys, crops, masks, metas in dataloader:
        embeddings = embed_batch(model, crops, masks, embed_cfg).cpu().numpy()
        n_dims = embeddings.shape[1]
        emb_cols = [f"emb_{i:04d}" for i in range(n_dims)]
        for meta, emb_row in zip(metas, embeddings):
            rows.append({**meta, **dict(zip(emb_cols, emb_row.tolist()))})
        n_cells += len(keys)

    if rows:
        frame = pl.DataFrame(rows)
        out = frame.select(
            *cell_metadata_exprs(embed_cfg.batch_stem), cs.starts_with("emb_")
        )
    else:
        out = pl.DataFrame(schema=CELL_METADATA_SCHEMA)
    logging.info("Writing embeddings.parquet (%d cells)", n_cells)
    out.write_parquet(output_dir / "embeddings.parquet")
    logging.info("Done")


if __name__ == "__main__":
    main()
