use std::io::BufReader;

use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_mantel::{Alternative, Method, mantel, read_matrix};

fn fixture() -> (Vec<f64>, Vec<f64>, usize) {
    let dir = std::env::var("MANTEL_BENCH_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden").to_string());
    let xp = format!("{dir}/dm1.tsv");
    let yp = format!("{dir}/dm2.tsv");
    let x = read_matrix(BufReader::new(std::fs::File::open(&xp).unwrap()), &xp).unwrap();
    let y = read_matrix(BufReader::new(std::fs::File::open(&yp).unwrap()), &yp).unwrap();
    let yd = y.reorder_like(&x.ids, &yp).unwrap();
    let n = x.n();
    (x.data, yd, n)
}

fn bench(c: &mut Criterion) {
    let (x, y, n) = fixture();
    c.bench_function("mantel_pearson_999", |b| {
        b.iter(|| mantel(&x, &y, n, Method::Pearson, 999, Alternative::TwoSided, 42))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
