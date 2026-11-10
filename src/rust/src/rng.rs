//! Small deterministic RNG (xoshiro256** seeded by SplitMix64).
//!
//! Streams are reproducible across platforms. They do not reproduce NumPy's
//! Mersenne Twister, so runs are not bit-identical to faiss.

pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, bound), unbiased (Lemire's method).
    pub fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let mut m = (self.next_u64() as u128) * (bound as u128);
        let mut low = m as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                m = (self.next_u64() as u128) * (bound as u128);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// First `take` entries of a uniform random permutation of 0..n.
    pub fn partial_permutation(&mut self, n: usize, take: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..n as u32).collect();
        let take = take.min(n);
        for i in 0..take {
            let j = i + self.below((n - i) as u64) as usize;
            p.swap(i, j);
        }
        p.truncate(take);
        p
    }
}

/// Per-epoch permutation seed, as in faiss `_epoch_seed`.
pub fn epoch_seed(seed: i64, epoch: usize) -> u64 {
    (seed as i128 * 1_000_003 + epoch as i128).rem_euclid(1i128 << 32) as u64
}
