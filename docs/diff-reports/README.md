# Output diff reports

Each refactor commit that changes a pipeline's reference outputs (`tests/reference/`) adds a
report here: which files and columns changed, by how much, and which difference in the code
causes it. The embeddings pipeline's per-experiment outputs never change in a refactor; the data
pipeline's change only where one of its stages switched to the shared (embeddings pipeline)
implementation.
