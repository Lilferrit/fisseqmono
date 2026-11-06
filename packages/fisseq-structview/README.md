# fisseq-structview

A web app that paints a pipeline run's per-position distinguishability (OvWT AUROC) scores, or
its feature profiles, onto a protein structure, linked to an interactive variant effect map of
the same run.

```bash
uv run fisseq-structview parameters.yaml                   # everything from the file
uv run fisseq-structview --params parameters.yaml --port 8051 --structure other.pdb
uv run fisseq-structview <pipeline_dir> --structure a.cif b.pdb   # no file needed
# then open http://127.0.0.1:8050
```

`<pipeline_dir>` is either pipeline's output directory, or `user@host:/path` for a remote one.
Structures are PDB (`.pdb`/`.ent`) or mmCIF (`.cif`/`.mmcif`) files, optionally gzipped; with
more than one, a dropdown in the nav bar switches between them. At launch, the app reads every
batch's `ovwt_batchwise*/<batch>/results.parquet` through `fisseqborn.OvwtScores`. For every
score column it pools each variant across batches, both corrected (`OvwtScores.correct`, which
keeps the AUROC scale) and raw, taking the median, and keeps the result in memory.

If the run also has feature-selection outputs (`feature_select_batchwise*/<batch>/aggregates/`),
the app reads every aggregate type there (median, KS, AUROC, ...) through
`fisseqborn.Profiles.from_pipeline`. It prepares them the way the PCA-embedding notebook does:

1. Z-score each batch against its own synonymous variants (`normalize(by="meta_experiment")`).
2. Take each feature's median across batches (`median_across_batches`).

Features with no value left are dropped. Every feature of every type stays in memory: for a
1,700-feature CellProfiler run with three aggregate types and 12,000 variants, that's about
230 MB, and a launch takes about 15 s.

Everything else is chosen in the browser.

## Parameters file

Every launch option can live in a YAML file, given as the argument (a `.yaml`/`.yml` path) or
with `--params` (`-p`). Any flag given on
the command line replaces the file's value for that key: `--structure` replaces the whole
structure list, and `--clinvar` replaces only the ClinVar table. Relative paths in the file
(the pipeline directory, structures, ClinVar table and cache) are relative to the file's
directory, so a folder with the file and its data can be copied or shared whole. Relative paths
on the command line are relative to the working directory, and `~` is expanded in both.

```yaml
pipeline_dir: data/2026-11-04          # or user@host:/path
structures:                            # a path, or a mapping
  - structures/AF-P02545-F1-model_v6.pdb
  - path: structures/AF-P02545-2-F1-model_v6.pdb
    label: Lamin C                     # the dropdown's name; default: the file name
    chain: A                           # default: chain, else the first protein chain
    offset: 0                          # default: offset
gene: LMNA                             # optional: list its landmark features first
track: embeddings                      # layout, batches, exclude, variant_col, cache and
offset: 0                              # refresh too: the flags below, by the same names
check_wt: true
plddt: auto
clinvar:                               # optional
  path: cache/LMNA_annotated.tsv       # a bbi-lab/variant-annotation table
  min_stars: 1                         # also: gene, release, conditions (default false)
regions:                               # optional bars over the effect map, top to bottom
  - label: Tile
    regions: {T1: [1, 96], T2: [90, 185]}
  - label: Domain
    regions: {Head: [1, 28], Coil 1A: [29, 67]}
    palette: {Head: "#4c72b0"}         # optional: a mapping or a seaborn palette name
host: 127.0.0.1
port: 8050
debug: false
```

An unknown key, a wrong type, a region whose start is after its end, or a missing
`pipeline_dir` or structure list stops the launch with an error naming the file and the key.

## The page

- **Nav bar:** controls that apply to both panels:
  - the structure, when there's more than one;
  - **Color by** distinguishability or feature, when the run has both (otherwise the one it
    has);
  - for distinguishability, the score (default `auroc_median_fold`) and corrected or raw;
  - for features, the feature. Type to search its name or column; the list shows the first
    100 matches. With `gene`, that gene's landmark features (`fisseqborn.genes`) come first
    under their readable names (e.g. "Lamin A intensity in the nucleus (median)"), and the
    first one is shown at launch;
  - the colormap: Blues, Reds or RdBu_r. Switching Color by resets it to that source's default
    (Blues for distinguishability, RdBu_r for features);
  - the numbering offset (structure residue = score position + offset).

  Its ⓘ button opens a summary of the data:
  - the parameters file;
  - the run's source and how it was read;
  - the feature profiles: how many features of each aggregate type, the landmarks and how they
    were pooled;
  - every batch used, with its number of OvWT rows and feature-profile variants;
  - variant counts by class;
  - each structure's file, chain, residues, pLDDT detection and offset;
  - the ClinVar source, release and counts, and the region tracks;
  - the current warnings.

  ☾/☀ switches between light and dark (remembered by the browser; the first visit follows the
  system setting).
- **Structure** (left), with its controls below it:
  - the aggregation per position: median, mean, min, max, or the fraction of variants above a
    deleterious threshold. For features this is the fraction with |z| above it, since a
    feature can deviate either way (default 2; 0.6 for an AUROC);
  - the variant classes aggregated (single missense by default; nonsense and synonymous
    optional);
  - the structure's color range;
  - fading of residues with pLDDT below a cutoff, when the B-factors hold pLDDT (AlphaFold
    models);
  - with ClinVar, ball-and-stick marks on the residues with ClinVar variants, by class
    (pathogenic, uncertain, benign; off by default), in the color of the most severe one.
- **Variant effect map** (right), in the layout of `fb.VariantEffectMap`:
  - one row per substituted amino acid, in fisseqborn's order, and one column per structure
    residue;
  - a black dot on each wild-type cell;
  - on top, the parameters file's region bars. Overlapping regions, such as neighbouring
    tiles, go on separate lanes of their bar;
  - with ClinVar, a triangle on each annotated variant's cell: red for
    pathogenic/likely pathogenic, gold for uncertain significance, blue for benign/likely
    benign, as in the variant-effect notebooks. A checklist under the map picks the classes,
    and a cell's hover gives its call and review stars;
  - below it, the per-position value that colors the structure.

  The map is drawn at full length and scrolls sideways under a fixed set of row labels and
  color bars: use the scrollbar, shift + mouse wheel, or a trackpad. The zoom slider below it
  sets the cell width, keeping the residue in the middle of the view in place. Next to the zoom
  slider is the map's own color range.

The default colors are those of the variant-effect notebooks: for distinguishability
`fb.VariantEffectMap(cmap="Blues", vmin=0.5, vmax=1.0)`, and for features the landmark plots'
`cmap="RdBu_r", center=0, clip=3`. Gray (`"0.6"`) means no value. The structure and the map
each have a color range, defaulting to 0.5–1.0 for scores and −3 to 3 for feature z-scores.
The fraction-deleterious aggregation defaults to 0–1.

Linking:
- clicking a map cell selects its residue in the structure and zooms to it;
- clicking a residue in the structure outlines its column and scrolls the map to it;
- hovering over a cell highlights its residue in the structure.

Warnings appear in a dialog when the page loads (and on every refresh): what was missing from
the run, scored positions outside the structure, and wild-type mismatches. A control change
that can't be applied (e.g. an offset that leaves no scored position on the structure) reopens
it with the reason.

## Options

| Flag | Default | |
|---|---|---|
| `--params`, `-p` | | parameters YAML (see above) |
| `--structure` | required (here or in the file) | one or more PDB or mmCIF files |
| `--clinvar` | | variant-annotation table of ClinVar calls to mark |
| `--layout`, `--track` | detected, `embeddings` | which pipeline wrote the run, and which embeddings-pipeline track to read |
| `--batches`, `--exclude` | all | which batches to read |
| `--variant-col` | `meta_aa_changes` | variant label column |
| `--gene` | | a `fisseqborn.genes` gene (e.g. `LMNA`) whose landmark features are listed first |
| `--cache`, `--refresh` | | for remote runs, as in fisseqborn |
| `--chain` | first protein chain | chain to color |
| `--offset` | 0 | initial numbering offset (also adjustable in the app) |
| `--no-check-wt` | | skip the wild-type residue check |
| `--plddt` | `auto` | whether the B-factors are pLDDT (`yes`/`no` overrides detection) |
| `--host`, `--port`, `--debug` | `127.0.0.1`, 8050 | server |

## Checks and missing outputs

- A structure that is missing, unreadable, unparseable or protein-less exits with an error. So
  does an unknown `--chain`.
- With several structures, one whose numbering doesn't match the run stays in the dropdown,
  marked, with its reason in the warnings; the app exits only if none match.
- A ClinVar table that can't be read logs a warning and leaves the ClinVar markers out. Its
  calls are matched with `fisseqborn.Dataset.clinvar`, offline unless `conditions: true` (NCBI)
  or no path is given (`gene`'s calls from gnomAD).
- A run without OvWT outputs logs a warning and says so in the warnings dialog. With feature
  profiles, the app colors by feature only. Without them either, it still serves the
  structure, in gray, without the score controls or the map.
- A run without feature-selection aggregates colors by distinguishability only. An unknown
  `gene` stops the launch.
- If no scored position falls on the structure's residues, the app exits with both residue
  ranges and points at `--offset`. The same applies when over 10% of the wild-type residues in
  the variant labels (`A12V` → `A`) disagree with the structure. In the app, the same problems
  open the warnings dialog instead of a recolor.
- Scored positions outside the structure (e.g. a domain-only model) are logged and listed in the
  warnings dialog.

## Limitations

- dash-molstar (the 3D viewer) reports clicks, but not hover, back to Dash. So hovering over the
  structure can't highlight the map; click instead.
- The map shows only the structure chain's residues, like the structure. Positions outside it
  are listed in the warnings.
- dash-molstar sets the viewer's background only when the viewer is created. So switching
  between light and dark rebuilds the viewer, and the camera returns to the whole chain. Mol*'s
  own panels (sequence, controls) keep their light style.
- dash-molstar can't color by value. Each color is its own component, so colors are quantized
  into 33 bins.
- A recolor re-sends the whole structure to the viewer (the camera is kept).
- Insertion-code residues are skipped (logged). NMR files use their first model.
