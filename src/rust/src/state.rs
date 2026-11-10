//! e-means training state: Algorithm 1 of Bessa et al. (arXiv:2611.09501).
//!
//! Reference implementation: faiss/contrib/e_means.

use crate::kernel::{assign, unit_scale, Float, Rows, Snapshot};
use crate::rng::{epoch_seed, Rng};
use rayon::ThreadPool;

/// Cosine schedule floor: minimum step / initial step.
pub const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Batch updates between revival checks.
pub const CHECK_UNUSED_EVERY: usize = 5;
/// Initial EMA mass N_j.
pub const INIT_USAGE: f64 = 1.0 / 65536.0;
/// Relative coordinate perturbation for donor splits.
pub const SPLIT_EPSILON: f64 = 1e-4;

pub struct Recipe {
    pub batch_size: usize,
    pub alpha0: f64,
    pub epoch_rows: Vec<usize>,
    pub total_steps: usize,
    pub unused_threshold: f64,
    pub seed: i64,
}

pub struct EpochStat {
    pub obj: f64,
    pub imbalance: f64,
    pub nsplit: usize,
}

/// Adjacent representable value of `x` towards `dir` (C `nextafter`;
/// `f64::next_up` needs Rust 1.86).
fn next_toward(x: f64, dir: f64) -> f64 {
    if x.is_nan() || x == dir {
        return x;
    }
    if x == 0.0 {
        let tiny = f64::from_bits(1);
        return if dir > 0.0 { tiny } else { -tiny };
    }
    let bits = x.to_bits();
    let away_from_zero = (dir > x) == (x > 0.0);
    f64::from_bits(if away_from_zero { bits + 1 } else { bits - 1 })
}

pub fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

pub struct State<T: Float> {
    pub k: usize,
    pub d: usize,
    pub n: usize,
    pub spherical: bool,
    pub recipe: Recipe,
    // EMA positions c_j (row-major k x d) and masses N_j.
    centroids: Vec<f64>,
    usage: Vec<f64>,
    snapshot: Option<Snapshot<T>>,
    // Current batch statistics.
    batch_sums: Vec<f64>,
    batch_counts: Vec<f64>,
    batch_obj: f64,
    pub batch_n: usize,
    // Current epoch totals.
    pass_counts: Vec<f64>,
    pass_sums: Option<Vec<f64>>,
    pass_obj: f64,
    // Progress.
    pub step: usize,
    pub epoch: usize,
    pub epoch_done_rows: usize,
    pub n_splits: usize,
    splits_before: usize,
    split_rng: Rng,
    pub stats: Vec<EpochStat>,
    pub final_centroids: Option<Vec<f64>>,
    // In-memory training: permutation of the current epoch and labels.
    order: Vec<u32>,
    order_epoch: Option<usize>,
    pub labels: Option<Vec<i32>>,
    label_buf: Vec<i32>,
    dist_buf: Vec<f64>,
    pool: Option<ThreadPool>,
}

impl<T: Float> State<T> {
    pub fn new(
        init: Vec<f64>,
        k: usize,
        d: usize,
        n: usize,
        spherical: bool,
        recipe: Recipe,
        n_threads: usize,
    ) -> Result<Self, String> {
        if init.len() != k * d {
            return Err("initial centroids must be k x d".into());
        }
        let mut centroids = init;
        if spherical {
            for row in centroids.chunks_exact_mut(d) {
                let s = unit_scale(row);
                row.iter_mut().for_each(|v| *v *= s);
            }
        }
        let pool = if n_threads > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let split_rng = Rng::new(recipe.seed as u64);
        let mut st = State {
            k, d, n, spherical, recipe,
            centroids,
            usage: vec![INIT_USAGE; k],
            snapshot: None,
            batch_sums: vec![0.0; k * d],
            batch_counts: vec![0.0; k],
            batch_obj: 0.0,
            batch_n: 0,
            pass_counts: vec![0.0; k],
            pass_sums: None,
            pass_obj: 0.0,
            step: 0,
            epoch: 0,
            epoch_done_rows: 0,
            n_splits: 0,
            splits_before: 0,
            split_rng,
            stats: Vec::new(),
            final_centroids: None,
            order: Vec::new(),
            order_epoch: None,
            labels: None,
            label_buf: Vec::new(),
            dist_buf: Vec::new(),
            pool,
        };
        st.start_epoch();
        Ok(st)
    }

    pub fn n_epochs(&self) -> usize {
        self.recipe.epoch_rows.len()
    }

    fn is_last_epoch(&self) -> bool {
        self.epoch + 1 == self.n_epochs()
    }

    fn start_epoch(&mut self) {
        self.epoch_done_rows = 0;
        self.splits_before = self.n_splits;
        self.pass_counts.iter_mut().for_each(|v| *v = 0.0);
        self.pass_obj = 0.0;
        self.pass_sums = if self.is_last_epoch() { Some(vec![0.0; self.k * self.d]) } else { None };
    }

    fn check_running(&self) -> Result<(), String> {
        if self.final_centroids.is_some() {
            return Err("model already finalized".into());
        }
        if self.epoch >= self.n_epochs() {
            return Err("all epochs are complete; call finalize".into());
        }
        Ok(())
    }

    /// Rows the next update expects.
    pub fn expected_batch(&self) -> usize {
        self.recipe
            .batch_size
            .min(self.recipe.epoch_rows[self.epoch] - self.epoch_done_rows)
    }

    /// Current centroid readout (unit directions when spherical).
    pub fn current_centroids(&self) -> Vec<f64> {
        if let Some(f) = &self.final_centroids {
            return f.clone();
        }
        let mut c = self.centroids.clone();
        if self.spherical {
            for row in c.chunks_exact_mut(self.d) {
                let s = unit_scale(row);
                row.iter_mut().for_each(|v| *v *= s);
            }
        }
        c
    }

    /// Assign rows against the fixed centroids of this batch and add them to
    /// the batch statistics. Labels (0-based) are left in `label_buf`.
    pub fn accumulate(&mut self, rows: &Rows<'_, T>) -> Result<(), String> {
        self.check_running()?;
        let m = rows.len();
        if self.batch_n + m > self.expected_batch() {
            return Err(format!(
                "batch overflow: update expects {} rows, got {}",
                self.expected_batch(),
                self.batch_n + m
            ));
        }
        if self.snapshot.is_none() {
            self.snapshot = Some(Snapshot::from_f64(&self.centroids, self.k, self.d, self.spherical));
        }
        let snap = self.snapshot.as_ref().unwrap();
        self.label_buf.resize(m, 0);
        self.dist_buf.resize(m, 0.0);
        assign(rows, snap, &mut self.label_buf, &mut self.dist_buf, self.pool.as_ref());
        let d = self.d;
        for i in 0..m {
            let j = self.label_buf[i] as usize;
            self.batch_counts[j] += 1.0;
            let acc = &mut self.batch_sums[j * d..(j + 1) * d];
            for (a, &v) in acc.iter_mut().zip(rows.row(i)) {
                *a += v.to_f64();
            }
            self.batch_obj += self.dist_buf[i];
        }
        if self.spherical {
            self.batch_obj += 2.0 * m as f64;
        }
        self.batch_n += m;
        Ok(())
    }

    pub fn last_labels(&self) -> &[i32] {
        &self.label_buf
    }

    /// Apply the EMA update for the completed batch (Algorithm 1, line 8),
    /// then revive unused clusters every `CHECK_UNUSED_EVERY` steps.
    pub fn update(&mut self) -> Result<(), String> {
        self.check_running()?;
        let expected = self.expected_batch();
        if self.batch_n != expected {
            return Err(format!(
                "update expects a batch of {} rows, accumulated {}",
                expected, self.batch_n
            ));
        }
        let alpha = scheduled_alpha(self.recipe.alpha0, self.step, self.recipe.total_steps);
        let decay = 1.0 - alpha;
        let replace = alpha == 1.0;
        let (k, d) = (self.k, self.d);

        self.pass_obj += self.batch_obj;
        for j in 0..k {
            self.pass_counts[j] += self.batch_counts[j];
        }
        if let Some(ps) = self.pass_sums.as_mut() {
            for (p, &s) in ps.iter_mut().zip(&self.batch_sums) {
                *p += s;
            }
        }
        if !replace {
            self.usage.iter_mut().for_each(|u| *u *= decay);
        }
        for j in 0..k {
            let nj = self.batch_counts[j];
            if nj <= 0.0 {
                continue;
            }
            let retained = if replace { 0.0 } else { self.usage[j] };
            let mass = retained + alpha * nj;
            let weight = alpha * nj / mass;
            let c = &mut self.centroids[j * d..(j + 1) * d];
            let s = &self.batch_sums[j * d..(j + 1) * d];
            if weight == 1.0 {
                for (cv, &sv) in c.iter_mut().zip(s) {
                    *cv = sv / nj;
                }
            } else {
                for (cv, &sv) in c.iter_mut().zip(s) {
                    *cv += weight * (sv / nj - *cv);
                }
            }
            self.usage[j] = mass;
        }

        if (self.step + 1) % CHECK_UNUSED_EVERY == 0 {
            self.revive_unused();
        }
        self.snapshot = None;
        self.step += 1;
        self.epoch_done_rows += expected;
        self.batch_sums.iter_mut().for_each(|v| *v = 0.0);
        self.batch_counts.iter_mut().for_each(|v| *v = 0.0);
        self.batch_obj = 0.0;
        self.batch_n = 0;
        Ok(())
    }

    /// Split donors sampled in proportion to mass (Gumbel top-k, without
    /// replacement) into clusters with mass below tau * mean(N).
    fn revive_unused(&mut self) -> usize {
        let k = self.k;
        let total: f64 = self.usage.iter().sum();
        let threshold = total * self.recipe.unused_threshold / k as f64;
        let unused: Vec<bool> = self.usage.iter().map(|&u| u < threshold).collect();
        let n_unused = unused.iter().filter(|&&u| u).count();
        let n_donors = (0..k).filter(|&j| !unused[j] && self.usage[j] > 0.0).count();
        let nsplit = n_unused.min(n_donors);
        if nsplit == 0 {
            return 0;
        }
        let mut keys: Vec<(f64, usize)> = (0..k)
            .map(|j| {
                let u = self.split_rng.uniform().max(f64::MIN_POSITIVE);
                let key = if !unused[j] && self.usage[j] > 0.0 {
                    self.usage[j].ln() - (-u.ln()).ln()
                } else {
                    f64::NEG_INFINITY
                };
                (key, j)
            })
            .collect();
        keys.sort_by(|a, b| b.0.total_cmp(&a.0));
        let targets: Vec<usize> = (0..k).filter(|&j| unused[j]).take(nsplit).collect();
        let d = self.d;
        for (&(_, src), &dst) in keys.iter().take(nsplit).zip(&targets) {
            let base: Vec<f64> = self.centroids[src * d..(src + 1) * d].to_vec();
            let mut donor = vec![0.0; d];
            let mut child = vec![0.0; d];
            for p in 0..d {
                let sign = if p % 2 == 1 { -1.0 } else { 1.0 };
                let off = SPLIT_EPSILON * sign;
                donor[p] = base[p] * (1.0 - off);
                child[p] = base[p] * (1.0 + off);
            }
            if donor == child {
                let up = next_toward(child[0], f64::INFINITY);
                child[0] = if up.is_finite() { up } else { next_toward(child[0], f64::NEG_INFINITY) };
            }
            let half = self.usage[src] / 2.0;
            self.usage[src] = half;
            self.usage[dst] = half;
            self.centroids[src * d..(src + 1) * d].copy_from_slice(&donor);
            self.centroids[dst * d..(dst + 1) * d].copy_from_slice(&child);
        }
        self.n_splits += nsplit;
        nsplit
    }

    /// Finish one complete pass and record its moving-pass objective.
    pub fn end_epoch(&mut self) -> Result<(), String> {
        self.check_running()?;
        let rows = self.recipe.epoch_rows[self.epoch];
        if self.epoch_done_rows != rows || self.batch_n != 0 {
            return Err(format!(
                "epoch {} expects {} rows, updated {}",
                self.epoch + 1, rows, self.epoch_done_rows
            ));
        }
        let sq: f64 = self.pass_counts.iter().map(|c| c * c).sum();
        let rows_f = rows.max(1) as f64;
        self.stats.push(EpochStat {
            obj: self.pass_obj,
            imbalance: self.k as f64 * sq / (rows_f * rows_f),
            nsplit: self.n_splits - self.splits_before,
        });
        self.epoch += 1;
        if self.epoch < self.n_epochs() {
            self.start_epoch();
        }
        Ok(())
    }

    /// Closing centroids: exact means of last-epoch assignments; clusters
    /// never assigned in that pass keep their EMA position.
    pub fn finalize(&mut self) -> Result<Vec<f64>, String> {
        if let Some(f) = &self.final_centroids {
            return Ok(f.clone());
        }
        if self.epoch < self.n_epochs() {
            return Err("finalize called before all epochs completed".into());
        }
        let d = self.d;
        let mut out = self.centroids.clone();
        let sums = self.pass_sums.take().unwrap_or_else(|| vec![0.0; self.k * d]);
        for j in 0..self.k {
            let c = self.pass_counts[j];
            let row = &mut out[j * d..(j + 1) * d];
            if c > 0.0 {
                for (o, &s) in row.iter_mut().zip(&sums[j * d..(j + 1) * d]) {
                    *o = s / c;
                }
            }
            if self.spherical {
                let s = unit_scale(row);
                row.iter_mut().for_each(|v| *v *= s);
            }
        }
        self.final_centroids = Some(out.clone());
        self.snapshot = None;
        self.order = Vec::new();
        Ok(out)
    }

    /// In-memory training over `data` (row-major n x d) for at most
    /// `max_steps` batch updates. Returns true when all epochs are done.
    pub fn train_steps(&mut self, data: &[T], max_steps: usize, capture: bool) -> Result<bool, String> {
        if data.len() != self.n * self.d {
            return Err("data does not match the model's n and d".into());
        }
        if capture && self.labels.is_none() {
            self.labels = Some(vec![-1; self.n]);
        }
        let mut done_steps = 0;
        while done_steps < max_steps {
            if self.epoch >= self.n_epochs() {
                return Ok(true);
            }
            if self.order_epoch != Some(self.epoch) {
                let mut rng = Rng::new(epoch_seed(self.recipe.seed, self.epoch));
                self.order = rng.partial_permutation(self.n, self.n);
                self.order_epoch = Some(self.epoch);
            }
            let rows = self.recipe.epoch_rows[self.epoch];
            let pos = self.epoch_done_rows;
            if pos >= rows {
                self.end_epoch()?;
                continue;
            }
            let end = (pos + self.recipe.batch_size).min(rows);
            let order = std::mem::take(&mut self.order);
            let r = Rows { x: data, d: self.d, index: Some(&order[pos..end]) };
            let res = self.accumulate(&r);
            if res.is_ok() && capture && self.is_last_epoch() {
                let labels = self.labels.as_mut().unwrap();
                for (i, &row) in order[pos..end].iter().enumerate() {
                    labels[row as usize] = self.label_buf[i];
                }
            }
            self.order = order;
            res?;
            self.update()?;
            done_steps += 1;
        }
        Ok(self.epoch >= self.n_epochs())
    }
}
