# FISSEQ Element Data Pipeline

[fisseq-data-pipeline](../fisseq-data-pipeline/README.md)'s workflow for Element Biosciences
CellProfiler cell tables. Only INPUT differs: the barcode is the whole `MostAbundantTarget`
target name, the variant label is `variant`, and every cell gets an edit distance of 0 (Element
reports none). QC, normalization, OvWT and feature selection are the shared fisseq-common
stages, with the data pipeline's parameters and publish layout.

## Quick start

```bash
git clone https://github.com/Lilferrit/fisseqmono.git
cd fisseqmono
uv sync
cd packages/fisseq-element-data-pipeline
nextflow run . -params-file params.yaml --pipeline_dir /path/to/experiment
```

Every process runs in `ghcr.io/lilferrit/fisseqmono/fisseq-element-data-pipeline`; add
`-profile local` to run against a local uv environment instead.

## Documentation

**[lilferrit.github.io/fisseqmono/element-data-pipeline](https://lilferrit.github.io/fisseqmono/element-data-pipeline/)**,
and the data pipeline's pages for everything after INPUT.

## License

[MIT](LICENSE.txt)
