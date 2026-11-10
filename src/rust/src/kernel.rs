//! Nearest-centroid assignment: blocked GEMM with a fused running argmin.
//!
//! The n x k distance matrix is never materialized. Rows are processed in
//! chunks of `ROW_CHUNK`, centroids in blocks of `CENTROID_CHUNK`.

use rayon::prelude::*;
use rayon::ThreadPool;

pub const ROW_CHUNK: usize = 256;
pub const CENTROID_CHUNK: usize = 1024;
/// Below this many multiply-adds per call, run on the calling thread.
const PARALLEL_MIN_WORK: usize = 1 << 16;

pub trait Float: Copy + Send + Sync + Default + 'static {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    /// C (m x n) = A (m x k) * B (k x n), beta = 0, arbitrary strides.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        c: *mut Self, rsc: isize, csc: isize,
    );
}

impl Float for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self { v as f32 }
    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        a: *const f32, rsa: isize, csa: isize,
        b: *const f32, rsb: isize, csb: isize,
        c: *mut f32, rsc: isize, csc: isize,
    ) {
        matrixmultiply::sgemm(m, k, n, 1.0, a, rsa, csa, b, rsb, csb, 0.0, c, rsc, csc);
    }
}

impl Float for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self { v }
    #[inline]
    fn to_f64(self) -> f64 { self }
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        a: *const f64, rsa: isize, csa: isize,
        b: *const f64, rsb: isize, csb: isize,
        c: *mut f64, rsc: isize, csc: isize,
    ) {
        matrixmultiply::dgemm(m, k, n, 1.0, a, rsa, csa, b, rsb, csb, 0.0, c, rsc, csc);
    }
}

/// Centroids prepared for assignment: row-major k x d in compute precision,
/// plus squared norms (Euclidean mode only).
pub struct Snapshot<T: Float> {
    pub k: usize,
    pub d: usize,
    pub spherical: bool,
    pub c: Vec<T>,
    pub norms: Vec<T>,
}

impl<T: Float> Snapshot<T> {
    /// Build from f64 row-major centroids. Spherical mode uses unit
    /// directions (zero stays zero), as faiss `normalize_into`.
    pub fn from_f64(centroids: &[f64], k: usize, d: usize, spherical: bool) -> Self {
        let mut c = Vec::with_capacity(k * d);
        let mut norms = Vec::with_capacity(if spherical { 0 } else { k });
        for row in centroids.chunks_exact(d) {
            if spherical {
                let scale = unit_scale(row);
                c.extend(row.iter().map(|&v| T::from_f64(v * scale)));
            } else {
                let start = c.len();
                c.extend(row.iter().map(|&v| T::from_f64(v)));
                let n2: f64 = c[start..].iter().map(|v| { let v = v.to_f64(); v * v }).sum();
                norms.push(T::from_f64(n2));
            }
        }
        Snapshot { k, d, spherical, c, norms }
    }
}

/// Factor that maps `row` to unit norm, prescaled by max |v| to avoid
/// overflow; 0 for a zero row.
pub fn unit_scale(row: &[f64]) -> f64 {
    let maximum = row.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if maximum == 0.0 {
        return 0.0;
    }
    let n2: f64 = row.iter().map(|&v| { let w = v / maximum; w * w }).sum();
    let norm = n2.sqrt();
    if norm > 0.0 { 1.0 / (maximum * norm) } else { 0.0 }
}

/// Source of rows: a contiguous row-major block, optionally indexed.
#[derive(Clone, Copy)]
pub struct Rows<'a, T> {
    pub x: &'a [T],
    pub d: usize,
    pub index: Option<&'a [u32]>,
}

impl<'a, T: Float> Rows<'a, T> {
    pub fn len(&self) -> usize {
        match self.index {
            Some(ix) => ix.len(),
            None => self.x.len() / self.d,
        }
    }
    #[inline]
    pub fn row(&self, i: usize) -> &'a [T] {
        let r = match self.index {
            Some(ix) => ix[i] as usize,
            None => i,
        };
        &self.x[r * self.d..(r + 1) * self.d]
    }
}

/// Per-thread scratch buffers.
#[derive(Default)]
struct Scratch<T> {
    gather: Vec<T>,
    gram: Vec<T>,
    best: Vec<f64>,
}

/// Assign one chunk of rows starting at `start`.
/// Writes 0-based labels and distances (Euclidean: squared L2;
/// spherical: -2 * inner product, as faiss).
fn assign_chunk<T: Float>(
    rows: &Rows<'_, T>,
    start: usize,
    snap: &Snapshot<T>,
    labels: &mut [i32],
    dist: &mut [f64],
    s: &mut Scratch<T>,
) {
    let m = labels.len();
    let d = snap.d;
    // Contiguous m x d block.
    let block: &[T] = if rows.index.is_none() {
        &rows.x[start * d..(start + m) * d]
    } else {
        s.gather.clear();
        for i in 0..m {
            s.gather.extend_from_slice(rows.row(start + i));
        }
        &s.gather
    };
    let kb_max = CENTROID_CHUNK.min(snap.k);
    s.gram.resize(m * kb_max, T::default());
    s.best.clear();
    s.best.resize(m, f64::INFINITY);
    labels.iter_mut().for_each(|l| *l = 0);

    let mut cb = 0;
    while cb < snap.k {
        let kb = kb_max.min(snap.k - cb);
        unsafe {
            T::gemm(
                m, d, kb,
                block.as_ptr(), d as isize, 1,
                snap.c.as_ptr().add(cb * d), 1, d as isize,
                s.gram.as_mut_ptr(), kb as isize, 1,
            );
        }
        for i in 0..m {
            let g = &s.gram[i * kb..(i + 1) * kb];
            let mut b = s.best[i];
            let mut l = labels[i];
            if snap.spherical {
                for (j, &v) in g.iter().enumerate() {
                    let score = -v.to_f64();
                    if score < b { b = score; l = (cb + j) as i32; }
                }
            } else {
                let norms = &snap.norms[cb..cb + kb];
                for (j, (&v, &cn)) in g.iter().zip(norms).enumerate() {
                    let score = cn.to_f64() - 2.0 * v.to_f64();
                    if score < b { b = score; l = (cb + j) as i32; }
                }
            }
            s.best[i] = b;
            labels[i] = l;
        }
        cb += kb;
    }
    for i in 0..m {
        dist[i] = if snap.spherical {
            2.0 * s.best[i]
        } else {
            // Direct distance, no cancellation
            let xr = &block[i * d..(i + 1) * d];
            let l = labels[i] as usize;
            sq_dist(xr, &snap.c[l * d..(l + 1) * d])
        };
    }
}

/// Squared Euclidean distance, f64 sum.
#[inline]
fn sq_dist<T: Float>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&u, &v)| {
            let t = u.to_f64() - v.to_f64();
            t * t
        })
        .sum()
}

/// Assign all rows. `labels` and `dist` must have `rows.len()` entries.
pub fn assign<T: Float>(
    rows: &Rows<'_, T>,
    snap: &Snapshot<T>,
    labels: &mut [i32],
    dist: &mut [f64],
    pool: Option<&ThreadPool>,
) {
    let n = rows.len();
    debug_assert_eq!(labels.len(), n);
    let work = n.saturating_mul(snap.k).saturating_mul(snap.d);
    match pool {
        Some(pool) if work >= PARALLEL_MIN_WORK && n > ROW_CHUNK => {
            pool.install(|| {
                labels
                    .par_chunks_mut(ROW_CHUNK)
                    .zip(dist.par_chunks_mut(ROW_CHUNK))
                    .enumerate()
                    .for_each_init(Scratch::<T>::default, |s, (ci, (l, dd))| {
                        assign_chunk(rows, ci * ROW_CHUNK, snap, l, dd, s);
                    });
            });
        }
        _ => {
            let mut s = Scratch::<T>::default();
            for (ci, (l, dd)) in labels
                .chunks_mut(ROW_CHUNK)
                .zip(dist.chunks_mut(ROW_CHUNK))
                .enumerate()
            {
                assign_chunk(rows, ci * ROW_CHUNK, snap, l, dd, &mut s);
            }
        }
    }
}
