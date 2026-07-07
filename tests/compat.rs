use std::io::BufReader;
use std::process::Command;

use rsomics_mantel::{Alternative, Method, mantel, read_matrix};

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");

fn load(name: &str) -> (Vec<f64>, Vec<String>, usize) {
    let path = format!("{GOLDEN}/{name}");
    let f = std::fs::File::open(&path).unwrap();
    let dm = read_matrix(BufReader::new(f), &path).unwrap();
    (dm.data, dm.ids, dm.n)
}

fn stat(method: Method) -> f64 {
    let (x, xids, n) = load("dm1.tsv");
    let dm2 = {
        let path = format!("{GOLDEN}/dm2.tsv");
        let f = std::fs::File::open(&path).unwrap();
        read_matrix(BufReader::new(f), &path).unwrap()
    };
    let y = dm2.reorder_like(&xids, "dm2.tsv").unwrap();
    mantel(&x, &y, n, method, 0, Alternative::TwoSided, 1).r
}

/// Always runs: the committed skbio-captured statistic must match to ~1e-9.
#[test]
fn statistic_matches_skbio_golden() {
    let golden = std::fs::read_to_string(format!("{GOLDEN}/golden.tsv")).unwrap();
    for line in golden.lines().skip(1) {
        let mut f = line.split('\t');
        let method = Method::parse(f.next().unwrap()).unwrap();
        let want: f64 = f.next().unwrap().parse().unwrap();
        let got = stat(method);
        assert!(
            (got - want).abs() < 1e-9,
            "{}: got {got}, skbio golden {want}",
            method.name()
        );
    }
}

/// Reordering dm2 onto dm1's ids reproduces the same statistic.
#[test]
fn reorder_is_invariant() {
    let (x, xids, n) = load("dm1.tsv");
    let path = format!("{GOLDEN}/dm2_reordered.tsv");
    let f = std::fs::File::open(&path).unwrap();
    let dm2r = read_matrix(BufReader::new(f), &path).unwrap();
    let y = dm2r.reorder_like(&xids, &path).unwrap();
    let got = mantel(&x, &y, n, Method::Pearson, 0, Alternative::TwoSided, 1).r;
    assert!((got - 0.917_039_295_978_798).abs() < 1e-9, "got {got}");
}

fn skbio_python() -> Option<String> {
    let candidates = [
        std::env::var("SKBIO_PYTHON").ok(),
        Some(format!(
            "{}/oracle-venvs/skbio/bin/python",
            std::env::var("HOME").unwrap_or_default()
        )),
    ];
    candidates.into_iter().flatten().find(|p| {
        Command::new(p)
            .args(["-c", "import skbio.stats.distance"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Live differential against scikit-bio. Loud-skips when the venv is absent
/// (so CI, which does not install skbio, stays green via the golden test).
#[test]
fn live_skbio_statistic() {
    let Some(py) = skbio_python() else {
        eprintln!("SKIP live_skbio_statistic: scikit-bio venv not found");
        return;
    };

    let dir = std::env::temp_dir().join("rsomics-mantel-compat");
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("oracle.py");
    std::fs::write(
        &script,
        format!(
            r#"
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
import sys
x = DistanceMatrix.read("{GOLDEN}/dm1.tsv")
y = DistanceMatrix.read("{GOLDEN}/dm2.tsv")
for m in ("pearson","spearman"):
    r,_,n = mantel(x,y,method=m,permutations=0)
    print(f"{{m}}\t{{float(r)!r}}\t{{n}}")
"#
        ),
    )
    .unwrap();

    let out = Command::new(&py).arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "skbio oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    for line in text.lines() {
        let mut f = line.split('\t');
        let method = Method::parse(f.next().unwrap()).unwrap();
        let want: f64 = f.next().unwrap().trim().parse().unwrap();
        let got = stat(method);
        assert!(
            (got - want).abs() < 1e-9,
            "{}: ours {got} vs live skbio {want}",
            method.name()
        );
    }
}

/// A malformed distance matrix must be rejected with a non-zero exit, matching
/// scikit-bio's `DistanceMatrixError`/`PairwiseMatrixError` rather than emitting
/// a confident wrong statistic. Covers the two structural invariants a parser is
/// most likely to silently accept: asymmetry and a non-hollow diagonal.
#[test]
fn rejects_malformed_matrix() {
    let bin = env!("CARGO_BIN_EXE_rsomics-mantel");
    let dir = std::env::temp_dir().join("rsomics-mantel-validate");
    std::fs::create_dir_all(&dir).unwrap();

    let valid = "\ta\tb\tc\na\t0\t1\t2\nb\t1\t0\t3\nc\t2\t3\t0\n";
    let non_symmetric = "\ta\tb\tc\na\t0\t1\t2\nb\t9\t0\t3\nc\t2\t3\t0\n";
    let non_hollow = "\ta\tb\tc\na\t5\t1\t2\nb\t1\t0\t3\nc\t2\t3\t0\n";

    let good = dir.join("good.tsv");
    let asym = dir.join("asym.tsv");
    let nonhollow = dir.join("nonhollow.tsv");
    std::fs::write(&good, valid).unwrap();
    std::fs::write(&asym, non_symmetric).unwrap();
    std::fs::write(&nonhollow, non_hollow).unwrap();

    for bad in [&asym, &nonhollow] {
        let out = Command::new(bin)
            .arg(bad)
            .arg(&good)
            .args(["-p", "0"])
            .output()
            .unwrap();
        assert!(
            !out.status.success(),
            "expected non-zero exit for malformed matrix {}",
            bad.display()
        );
    }
}

/// The permutation p-value is a Monte-Carlo estimate: with a strong true
/// correlation it must land near skbio's small p-value (both use 999 perms).
#[test]
fn p_value_in_expected_range() {
    let (x, xids, n) = load("dm1.tsv");
    let dm2 = {
        let path = format!("{GOLDEN}/dm2.tsv");
        let f = std::fs::File::open(&path).unwrap();
        read_matrix(BufReader::new(f), &path).unwrap()
    };
    let y = dm2.reorder_like(&xids, "dm2.tsv").unwrap();
    let res = mantel(&x, &y, n, Method::Pearson, 999, Alternative::Greater, 42);
    assert!(
        res.p_value <= 0.05,
        "strong correlation should be significant, got p={}",
        res.p_value
    );
}
