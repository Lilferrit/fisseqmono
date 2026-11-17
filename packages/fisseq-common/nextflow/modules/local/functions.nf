// Small helpers shared by every module's script block.

// Pin every threaded library to the task's own cpus. Each otherwise
// defaults to "all cores on the machine" and oversubscribes badly when
// several tasks share a node. Emitted inside the script itself (not as a
// beforeScript) so it reaches the container: beforeScript runs on the host.
def threadEnv(cpus) {
    def n = cpus ?: 1
    return [
        'POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    ].collect { name -> "export ${name}=${n}" }.join('\n')
}

// Render an ordered list as one shell-safe Hydra override token. A path
// input holding a single file arrives as one Path rather than a list.
def hydraList(key, values) {
    def items = values instanceof Collection ? values : [values]
    return "'${key}=[${items.join(',')}]'"
}

// A channel of (batch_stem, file) -- one per experiment, or several (one per aggregation
// method) -- gathered into a single (batch_stems, files) tuple, sorted by batch_stem then file
// name, for a stage that runs across experiments (COMBAT_FEATURES, COMBAT_OVWT). Emits
// nothing, with a warning, when fewer than two experiments arrive (one experiment has no batch
// effect to remove; an experiment whose task failed and was ignored just isn't in the list).
def gatherBatches(ch, what) {
    return ch
        .toSortedList { a, b -> a[0] <=> b[0] ?: a[1].name <=> b[1].name }
        .filter { items ->
            def n = items.collect { item -> item[0] }.unique().size()
            if (n < 2) {
                log.warn "${what}: skipped, it needs at least 2 experiments and got ${n}"
            }
            n >= 2
        }
        .map { items -> tuple(items.collect { item -> item[0] }, items.collect { item -> item[1] }) }
}
