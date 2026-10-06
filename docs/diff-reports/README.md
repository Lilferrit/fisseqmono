# Output diff reports

Each commit that changes a pipeline's reference outputs (`tests/reference/`) adds a report
here: which files and columns changed, by how much, and which difference in the code causes it.
A pipeline's outputs change only in a commit that changes them deliberately, with a report
here; a refactor leaves them as they are.
