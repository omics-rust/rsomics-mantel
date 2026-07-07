use std::io::BufRead;

use rsomics_common::{Result, RsomicsError};

/// A square, symmetric distance matrix with sample ids, in row-major order.
pub struct DistanceMatrix {
    pub ids: Vec<String>,
    pub data: Vec<f64>,
    pub n: usize,
}

impl DistanceMatrix {
    pub fn read<R: BufRead>(reader: R, source: &str) -> Result<Self> {
        let mut lines = reader.lines();

        let header = lines
            .next()
            .ok_or_else(|| RsomicsError::InvalidInput(format!("{source}: empty matrix")))?
            .map_err(RsomicsError::Io)?;

        // lsmat header: a leading empty cell, then the column ids.
        let ids: Vec<String> = header
            .split('\t')
            .skip(1)
            .map(str::trim)
            .map(str::to_owned)
            .collect();
        let n = ids.len();
        if n < 3 {
            return Err(RsomicsError::InvalidInput(format!(
                "{source}: need at least 3 ids, found {n}"
            )));
        }

        let mut data = vec![0.0f64; n * n];
        let mut row = 0;
        for line in lines {
            let line = line.map_err(RsomicsError::Io)?;
            if line.is_empty() {
                continue;
            }
            if row >= n {
                return Err(RsomicsError::InvalidInput(format!(
                    "{source}: more rows than the {n} ids in the header"
                )));
            }
            let mut fields = line.split('\t');
            let rid = fields.next().unwrap_or("").trim();
            if rid != ids[row] {
                return Err(RsomicsError::InvalidInput(format!(
                    "{source}: row {row} id '{rid}' != header id '{}'",
                    ids[row]
                )));
            }
            let mut col = 0;
            for f in fields {
                if col >= n {
                    return Err(RsomicsError::InvalidInput(format!(
                        "{source}: row {row} has more than {n} values"
                    )));
                }
                data[row * n + col] = f.trim().parse::<f64>().map_err(|_| {
                    RsomicsError::InvalidInput(format!(
                        "{source}: row {row} col {col}: not a number: '{f}'"
                    ))
                })?;
                col += 1;
            }
            if col != n {
                return Err(RsomicsError::InvalidInput(format!(
                    "{source}: row {row} has {col} values, expected {n}"
                )));
            }
            row += 1;
        }
        if row != n {
            return Err(RsomicsError::InvalidInput(format!(
                "{source}: {row} data rows, expected {n}"
            )));
        }

        let dm = DistanceMatrix { ids, data, n };
        dm.validate(source)?;
        Ok(dm)
    }

    /// Enforce scikit-bio's `DistanceMatrix` invariants: unique ids, a hollow
    /// diagonal, and exact symmetry with no NaNs. Negative distances are
    /// permitted, matching skbio. Checked in skbio's order so the first failing
    /// invariant produces the same class of error a skbio caller would see.
    fn validate(&self, source: &str) -> Result<()> {
        let n = self.n;

        let mut seen = std::collections::HashSet::with_capacity(n);
        for id in &self.ids {
            if !seen.insert(id.as_str()) {
                return Err(RsomicsError::InvalidInput(format!(
                    "{source}: IDs must be unique. Found the following duplicate IDs: '{id}'"
                )));
            }
        }

        for i in 0..n {
            if self.data[i * n + i] != 0.0 {
                return Err(RsomicsError::InvalidInput(format!(
                    "{source}: Data must be hollow (i.e., the diagonal can only contain zeros)."
                )));
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if self.data[i * n + j] != self.data[j * n + i] {
                    return Err(RsomicsError::InvalidInput(format!(
                        "{source}: Data must be symmetric and cannot contain NaNs."
                    )));
                }
            }
        }

        Ok(())
    }

    /// Reorder rows and columns so the ids match `target`. skbio reorders the
    /// second matrix onto the first's id order before correlating.
    pub fn reorder_like(&self, target: &[String], source: &str) -> Result<Vec<f64>> {
        let n = self.n;
        let pos: std::collections::HashMap<&str, usize> = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let perm: Vec<usize> = target
            .iter()
            .map(|id| {
                pos.get(id.as_str()).copied().ok_or_else(|| {
                    RsomicsError::InvalidInput(format!(
                        "{source}: id '{id}' missing; the two matrices must share ids"
                    ))
                })
            })
            .collect::<Result<_>>()?;
        let mut out = vec![0.0f64; n * n];
        for (i, &pi) in perm.iter().enumerate() {
            for (j, &pj) in perm.iter().enumerate() {
                out[i * n + j] = self.data[pi * n + pj];
            }
        }
        Ok(out)
    }

    /// Upper-triangle (i<j) entries in row-major order — skbio's condensed form.
    pub fn condensed(data: &[f64], n: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                v.push(data[i * n + j]);
            }
        }
        v
    }
}
