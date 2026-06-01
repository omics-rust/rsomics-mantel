# rsomics-mantel

Mantel test — permutation correlation between two distance matrices.

Computes the Pearson (default) or Spearman correlation between the upper
triangles of two symmetric distance matrices and assesses significance with a
permutation test. Drop-in compatible with `skbio.stats.distance.mantel`.

```
rsomics-mantel dm1.tsv dm2.tsv [--method pearson|spearman] \
    [--permutations 999] [--alternative two-sided|greater|less] \
    [--seed S] [-o result.tsv]
```

Both inputs are lsmat-format distance matrices (a blank top-left corner, a
tab-separated id header, then one `id<TAB>values…` row per sample). The second
matrix is reordered onto the first's id order, so the ids must match but need
not be in the same order. Matrices must be at least 3×3.

Output is a two-line TSV: `method`, the correlation `statistic`, the permutation
`p_value`, sample count `n`, the number of `permutations`, and the
`alternative`.

## Statistic vs p-value

The **correlation statistic** is deterministic and reproduces scikit-bio's
`mantel()` to floating-point tolerance (tested value-exact to 1e-9 against a
committed skbio-captured golden and, where the venv is present, a live skbio
oracle). Spearman is Pearson on the average-ranked distances.

The **p-value** is a permutation Monte-Carlo estimate: the rows and columns of
the first matrix are permuted `--permutations` times and the proportion of
permuted statistics at least as extreme as the observed one (with the +1
correction, `(count+1)/(perms+1)`) is reported. The permutations come from this
crate's own seeded RNG (SplitMix64 + Lemire-bounded Fisher-Yates), reproducible
across runs and thread counts for a given `--seed`, but **not** a bit-for-bit
reproduction of numpy's PCG64 stream — so the p-value is an estimate that
converges to skbio's as permutations grow, not an identical draw.

## Origin

This crate is an independent Rust reimplementation of `skbio.stats.distance.mantel`,
informed by its BSD-3-licensed source (the inline pearsonr standardize-and-dot,
the upper-triangle condensed form, full-matrix permutation, and the
`(count+1)/(perms+1)` p-value) and by the method's primary reference:

- Mantel, N. (1967). "The detection of disease clustering and a generalized
  regression approach." *Cancer Research* 27(2): 209–220. PMID: 6018555.

License: MIT OR Apache-2.0.
Upstream credit: scikit-bio <https://scikit-bio.org> (BSD-3-Clause).
