//! Deterministic per-permutation shuffles. Each permutation index `k` seeds an
//! independent SplitMix64 stream from `(seed, k)`, so results are stable
//! regardless of how rayon schedules the work. This is the crate's own
//! permutation source — it does not reproduce numpy's PCG64 stream, so the
//! p-value is a seeded Monte-Carlo estimate rather than a numpy-bit-exact one.

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, bound)` via Lemire's multiply-shift rejection.
    fn bounded(&mut self, bound: u64) -> u64 {
        let mut x = self.next();
        let mut m = (x as u128) * (bound as u128);
        let mut lo = m as u64;
        if lo < bound {
            let threshold = bound.wrapping_neg() % bound;
            while lo < threshold {
                x = self.next();
                m = (x as u128) * (bound as u128);
                lo = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

pub fn permutation(n: usize, seed: u64, k: u64) -> Vec<usize> {
    let mut rng = SplitMix64(seed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut a: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.bounded((i + 1) as u64) as usize;
        a.swap(i, j);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_a_permutation() {
        let p = permutation(50, 7, 3);
        let mut s = p.clone();
        s.sort_unstable();
        assert_eq!(s, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn deterministic_for_same_seed_and_index() {
        assert_eq!(permutation(40, 99, 5), permutation(40, 99, 5));
    }

    #[test]
    fn different_index_gives_different_permutation() {
        assert_ne!(permutation(40, 99, 5), permutation(40, 99, 6));
    }
}
