use std::io::{BufRead, Write};

use rayon::prelude::*;
use rsomics_common::{Result, RsomicsError};
use serde::Serialize;

pub mod dm;
mod rng;

pub use dm::DistanceMatrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Method {
    Pearson,
    Spearman,
}

impl Method {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "pearson" => Ok(Method::Pearson),
            "spearman" => Ok(Method::Spearman),
            other => Err(RsomicsError::InvalidInput(format!(
                "invalid method '{other}' (pearson|spearman)"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Method::Pearson => "pearson",
            Method::Spearman => "spearman",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Alternative {
    TwoSided,
    Greater,
    Less,
}

impl Alternative {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "two-sided" => Ok(Alternative::TwoSided),
            "greater" => Ok(Alternative::Greater),
            "less" => Ok(Alternative::Less),
            other => Err(RsomicsError::InvalidInput(format!(
                "invalid alternative '{other}' (two-sided|greater|less)"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Alternative::TwoSided => "two-sided",
            Alternative::Greater => "greater",
            Alternative::Less => "less",
        }
    }
}

#[derive(Serialize)]
pub struct MantelResult {
    pub r: f64,
    pub p_value: f64,
    pub n: usize,
    pub method: Method,
    pub permutations: usize,
    pub alternative: Alternative,
}

/// Run the Mantel test. `y_data` is already reordered onto `x`'s id order.
///
/// The correlation coefficient is deterministic and matches scikit-bio's
/// `mantel()` to floating-point tolerance. The p-value is a permutation
/// estimate computed with a seeded RNG; it is Monte-Carlo and does not
/// reproduce numpy's PCG64 permutation stream bit-for-bit.
pub fn mantel(
    x_data: &[f64],
    y_data: &[f64],
    n: usize,
    method: Method,
    permutations: usize,
    alternative: Alternative,
    seed: u64,
) -> MantelResult {
    let (x_flat, y_flat) = match method {
        Method::Pearson => (
            dm::DistanceMatrix::condensed(x_data, n),
            dm::DistanceMatrix::condensed(y_data, n),
        ),
        Method::Spearman => (
            rankdata(&dm::DistanceMatrix::condensed(x_data, n)),
            rankdata(&dm::DistanceMatrix::condensed(y_data, n)),
        ),
    };

    // The permutation acts on the full matrix; for Spearman that is the
    // rank-transformed matrix. Rebuild a full square from the ranked condensed
    // form so permutation semantics stay identical to skbio.
    let x_full = match method {
        Method::Pearson => x_data.to_vec(),
        Method::Spearman => square_from_condensed(&x_flat, n),
    };

    // x's condensed mean and norm are permutation-invariant — the permutation
    // only reorders the same upper-triangle entries — so each permuted statistic
    // is a single allocation-free pass: dot(x_perm - xmean, ym_normalized)/normx.
    let xmean = mean(&x_flat);
    let normx = norm_centered(&x_flat, xmean);
    let ym = normalize(&y_flat);
    let r = match (&ym, normx) {
        (Some(ymn), Some(nx)) => dot_centered(&x_flat, xmean, nx, ymn).clamp(-1.0, 1.0),
        _ => f64::NAN,
    };

    let p_value = if permutations == 0 || r.is_nan() {
        f64::NAN
    } else {
        let ymn = ym.unwrap();
        let nx = normx.unwrap();
        let count_extreme: usize = (0..permutations)
            .into_par_iter()
            .map(|k| {
                let perm = rng::permutation(n, seed, k as u64);
                let stat = permuted_stat(&x_full, n, &perm, xmean, nx, &ymn).clamp(-1.0, 1.0);
                match alternative {
                    Alternative::TwoSided => usize::from(stat.abs() >= r.abs()),
                    Alternative::Greater => usize::from(stat >= r),
                    Alternative::Less => usize::from(stat <= r),
                }
            })
            .sum();
        (count_extreme + 1) as f64 / (permutations + 1) as f64
    };

    MantelResult {
        r,
        p_value,
        n,
        method,
        permutations,
        alternative,
    }
}

/// Center then scale to unit norm; `None` if the input has no variation.
fn normalize(v: &[f64]) -> Option<Vec<f64>> {
    let m = mean(v);
    let mut out: Vec<f64> = v.iter().map(|&x| x - m).collect();
    let norm = out.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return None;
    }
    for x in &mut out {
        *x /= norm;
    }
    Some(out)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn norm_centered(v: &[f64], m: f64) -> Option<f64> {
    let s = v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>().sqrt();
    (s != 0.0).then_some(s)
}

fn dot_centered(x: &[f64], xmean: f64, normx: f64, ym_norm: &[f64]) -> f64 {
    x.iter()
        .zip(ym_norm)
        .map(|(&xv, &yv)| (xv - xmean) * yv)
        .sum::<f64>()
        / normx
}

/// Pearson statistic of `x_full` permuted by `perm`, against the already-
/// normalized `ym_norm`, in one allocation-free pass over the upper triangle.
fn permuted_stat(
    x_full: &[f64],
    n: usize,
    perm: &[usize],
    xmean: f64,
    normx: f64,
    ym_norm: &[f64],
) -> f64 {
    let mut acc = 0.0;
    let mut k = 0;
    for i in 0..n {
        let base = perm[i] * n;
        for j in (i + 1)..n {
            acc += (x_full[base + perm[j]] - xmean) * ym_norm[k];
            k += 1;
        }
    }
    acc / normx
}

/// Average-rank of each element, scipy `rankdata` default (ties averaged).
fn rankdata(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut ranks = vec![0.0f64; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && v[order[j]] == v[order[i]] {
            j += 1;
        }
        // ranks are 1-based; the average of the tied positions
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &idx in &order[i..j] {
            ranks[idx] = avg;
        }
        i = j;
    }
    ranks
}

fn square_from_condensed(cond: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            out[i * n + j] = cond[k];
            out[j * n + i] = cond[k];
            k += 1;
        }
    }
    out
}

pub fn write_result<W: Write>(out: &mut W, res: &MantelResult) -> Result<()> {
    writeln!(
        out,
        "method\tstatistic\tp_value\tn\tpermutations\talternative"
    )
    .map_err(RsomicsError::Io)?;
    writeln!(
        out,
        "{}\t{:.12}\t{}\t{}\t{}\t{}",
        res.method.name(),
        res.r,
        fmt_p(res.p_value),
        res.n,
        res.permutations,
        res.alternative.name(),
    )
    .map_err(RsomicsError::Io)?;
    Ok(())
}

fn fmt_p(p: f64) -> String {
    if p.is_nan() {
        "nan".to_string()
    } else {
        format!("{p:.12}")
    }
}

pub fn read_matrix<R: BufRead>(reader: R, source: &str) -> Result<DistanceMatrix> {
    DistanceMatrix::read(reader, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(rows: &[&[f64]]) -> (Vec<f64>, usize) {
        let n = rows.len();
        let mut d = vec![0.0; n * n];
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                d[i * n + j] = v;
            }
        }
        (d, n)
    }

    #[test]
    fn skbio_doc_example_pearson() {
        let (x, n) = square(&[&[0.0, 1.0, 2.0], &[1.0, 0.0, 3.0], &[2.0, 3.0, 0.0]]);
        let (y, _) = square(&[&[0.0, 2.0, 7.0], &[2.0, 0.0, 6.0], &[7.0, 6.0, 0.0]]);
        let res = mantel(&x, &y, n, Method::Pearson, 0, Alternative::TwoSided, 1);
        assert!((res.r - 0.7559289460184544).abs() < 1e-12, "r={}", res.r);
        assert!(res.p_value.is_nan());
    }

    #[test]
    fn rankdata_ties_averaged() {
        assert_eq!(rankdata(&[1.0, 2.0, 2.0, 3.0]), vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn condensed_upper_triangle() {
        let (x, n) = square(&[&[0.0, 1.0, 2.0], &[1.0, 0.0, 3.0], &[2.0, 3.0, 0.0]]);
        assert_eq!(DistanceMatrix::condensed(&x, n), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_permutation_reproduces_observed_stat() {
        let (x, n) = square(&[&[0.0, 1.0, 2.0], &[1.0, 0.0, 3.0], &[2.0, 3.0, 0.0]]);
        let (y, _) = square(&[&[0.0, 2.0, 7.0], &[2.0, 0.0, 6.0], &[7.0, 6.0, 0.0]]);
        let xf = DistanceMatrix::condensed(&x, n);
        let yf = DistanceMatrix::condensed(&y, n);
        let xmean = mean(&xf);
        let normx = norm_centered(&xf, xmean).unwrap();
        let ymn = normalize(&yf).unwrap();
        let id: Vec<usize> = (0..n).collect();
        let permuted = permuted_stat(&x, n, &id, xmean, normx, &ymn);
        let observed = dot_centered(&xf, xmean, normx, &ymn);
        assert!((permuted - observed).abs() < 1e-12);
    }
}
