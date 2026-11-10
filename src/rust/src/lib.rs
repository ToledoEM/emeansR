//! R bindings for the e-means core.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};

mod kernel;
mod rng;
mod state;

use kernel::{Float, Rows, Snapshot};
use state::{Recipe, State};

/// Row-major copy of an R matrix in compute precision.
pub struct Data<T: Float> {
    x: Vec<T>,
    n: usize,
    d: usize,
}

pub enum AnyData {
    Single(Data<f32>),
    Double(Data<f64>),
}

pub enum AnyState {
    Single(State<f32>),
    Double(State<f64>),
}

macro_rules! with_state {
    ($st:expr, $s:ident => $body:expr) => {
        match &mut *$st {
            AnyState::Single($s) => $body,
            AnyState::Double($s) => $body,
        }
    };
}

fn err<T>(e: String) -> Result<T> {
    Err(Error::Other(e))
}

/// Column-major R matrix -> row-major vector.
fn to_rows<T: Float>(m: &RMatrix<f64>) -> Vec<T> {
    let (n, d) = (m.nrows(), m.ncols());
    let src = m.data();
    let mut out = vec![T::default(); n * d];
    for j in 0..d {
        let col = &src[j * n..(j + 1) * n];
        for (i, &v) in col.iter().enumerate() {
            out[i * d + j] = T::from_f64(v);
        }
    }
    out
}

fn to_rmatrix(v: &[f64], nrow: usize, ncol: usize) -> RMatrix<f64> {
    RMatrix::new_matrix(nrow, ncol, |r, c| v[r * ncol + c])
}

fn build_pool(n_threads: usize) -> Result<Option<rayon::ThreadPool>> {
    if n_threads <= 1 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map(Some)
        .map_err(|e| Error::Other(e.to_string()))
}

/// Copy a numeric matrix into a row-major training buffer.
/// @noRd
#[extendr]
fn em_data(x: RMatrix<f64>, single: bool) -> ExternalPtr<AnyData> {
    let (n, d) = (x.nrows(), x.ncols());
    let data = if single {
        AnyData::Single(Data { x: to_rows::<f32>(&x), n, d })
    } else {
        AnyData::Double(Data { x: to_rows::<f64>(&x), n, d })
    };
    ExternalPtr::new(data)
}

/// Create a training state.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn em_new(
    init: RMatrix<f64>,
    n: f64,
    spherical: bool,
    single: bool,
    batch_size: f64,
    alpha0: f64,
    epoch_rows: Vec<f64>,
    unused_threshold: f64,
    seed: f64,
    n_threads: i32,
) -> Result<ExternalPtr<AnyState>> {
    let (k, d) = (init.nrows(), init.ncols());
    let epoch_rows: Vec<usize> = epoch_rows.iter().map(|&r| r as usize).collect();
    let batch_size = batch_size as usize;
    if batch_size == 0 {
        return err("batch_size must be >= 1".into());
    }
    let total_steps = epoch_rows.iter().map(|r| (r + batch_size - 1) / batch_size).sum();
    let recipe = Recipe {
        batch_size,
        alpha0,
        epoch_rows,
        total_steps,
        unused_threshold,
        seed: seed as i64,
    };
    let c: Vec<f64> = to_rows::<f64>(&init);
    let threads = n_threads.max(1) as usize;
    let st = if single {
        AnyState::Single(State::new(c, k, d, n as usize, spherical, recipe, threads).map_err(Error::Other)?)
    } else {
        AnyState::Double(State::new(c, k, d, n as usize, spherical, recipe, threads).map_err(Error::Other)?)
    };
    Ok(ExternalPtr::new(st))
}

/// Run up to `max_steps` updates on in-memory data. TRUE when done.
/// @noRd
#[extendr]
fn em_train_steps(mut st: ExternalPtr<AnyState>, data: ExternalPtr<AnyData>, max_steps: f64, capture: bool) -> Result<bool> {
    let max_steps = max_steps as usize;
    fn shape_ok<T: Float, U: Float>(s: &State<T>, x: &Data<U>) -> std::result::Result<(), String> {
        if s.n == x.n && s.d == x.d { Ok(()) } else { Err("data does not match the model's n and d".into()) }
    }
    let res = match (&mut *st, &*data) {
        (AnyState::Single(s), AnyData::Single(x)) => shape_ok(s, x).and_then(|_| s.train_steps(&x.x, max_steps, capture)),
        (AnyState::Double(s), AnyData::Double(x)) => shape_ok(s, x).and_then(|_| s.train_steps(&x.x, max_steps, capture)),
        _ => Err("model and data precision differ".into()),
    };
    res.map_err(Error::Other)
}

/// Accumulate one chunk of rows into the pending batch; returns 1-based labels.
/// @noRd
#[extendr]
fn em_accumulate(mut st: ExternalPtr<AnyState>, x: RMatrix<f64>) -> Result<Vec<i32>> {
    let d = x.ncols();
    let res: std::result::Result<Vec<i32>, String> = with_state!(st, s => {
        if d != s.d {
            Err(format!("chunk has {} columns, model has {}", d, s.d))
        } else {
            let rows = to_rows(&x);
            let r = Rows { x: &rows, d, index: None };
            s.accumulate(&r).map(|_| s.last_labels().iter().map(|&l| l + 1).collect())
        }
    });
    res.map_err(Error::Other)
}

/// Apply the EMA update for the pending batch.
/// @noRd
#[extendr]
fn em_update(mut st: ExternalPtr<AnyState>) -> Result<()> {
    with_state!(st, s => s.update()).map_err(Error::Other)
}

/// Close the current epoch.
/// @noRd
#[extendr]
fn em_end_epoch(mut st: ExternalPtr<AnyState>) -> Result<()> {
    with_state!(st, s => s.end_epoch()).map_err(Error::Other)
}

/// Closing centroids (k x d).
/// @noRd
#[extendr]
fn em_finalize(mut st: ExternalPtr<AnyState>) -> Result<RMatrix<f64>> {
    with_state!(st, s => {
        let c = s.finalize().map_err(Error::Other)?;
        Ok(to_rmatrix(&c, s.k, s.d))
    })
}

/// Current centroid readout (k x d).
/// @noRd
#[extendr]
fn em_centroids(mut st: ExternalPtr<AnyState>) -> RMatrix<f64> {
    with_state!(st, s => to_rmatrix(&s.current_centroids(), s.k, s.d))
}

/// Last-epoch labels captured by em_train_steps (1-based, NA if unseen).
/// @noRd
#[extendr]
fn em_labels(mut st: ExternalPtr<AnyState>) -> Robj {
    with_state!(st, s => match &s.labels {
        Some(l) => l.iter().map(|&v| if v < 0 { i32::MIN } else { v + 1 }).collect::<Vec<i32>>().into(),
        None => ().into(),
    })
}

/// Progress and per-epoch statistics.
/// @noRd
#[extendr]
fn em_info(mut st: ExternalPtr<AnyState>) -> Robj {
    with_state!(st, s => {
        let running = s.epoch < s.n_epochs() && s.final_centroids.is_none();
        list!(
            step = s.step as f64,
            epoch = s.epoch as f64,
            n_epochs = s.n_epochs() as f64,
            total_steps = s.recipe.total_steps as f64,
            expected_batch = if running { s.expected_batch() as f64 } else { 0.0 },
            n_splits = s.n_splits as f64,
            pending = s.batch_n as f64,
            epoch_done_rows = s.epoch_done_rows as f64,
            finalized = s.final_centroids.is_some(),
            obj = s.stats.iter().map(|e| e.obj).collect::<Vec<f64>>(),
            imbalance = s.stats.iter().map(|e| e.imbalance).collect::<Vec<f64>>(),
            nsplit = s.stats.iter().map(|e| e.nsplit as f64).collect::<Vec<f64>>()
        ).into()
    })
}

/// Nearest-centroid assignment for new data.
/// @noRd
#[extendr]
fn em_assign(centers: RMatrix<f64>, x: RMatrix<f64>, spherical: bool, single: bool, n_threads: i32) -> Result<Robj> {
    let (k, d) = (centers.nrows(), centers.ncols());
    if x.ncols() != d {
        return err(format!("newdata has {} columns, centers have {}", x.ncols(), d));
    }
    let n = x.nrows();
    let c = to_rows::<f64>(&centers);
    let pool = build_pool(n_threads.max(1) as usize)?;
    let mut labels = vec![0i32; n];
    let mut dist = vec![0.0f64; n];
    fn run<T: Float>(c: &[f64], k: usize, d: usize, spherical: bool, x: &RMatrix<f64>,
                     labels: &mut [i32], dist: &mut [f64], pool: Option<&rayon::ThreadPool>) {
        // Centers are used as given; spherical callers pass unit directions.
        let mut snap = Snapshot::<T>::from_f64(c, k, d, false);
        snap.spherical = spherical;
        let rows = to_rows::<T>(x);
        kernel::assign(&Rows { x: &rows, d, index: None }, &snap, labels, dist, pool);
    }
    if single {
        run::<f32>(&c, k, d, spherical, &x, &mut labels, &mut dist, pool.as_ref());
    } else {
        run::<f64>(&c, k, d, spherical, &x, &mut labels, &mut dist, pool.as_ref());
    }
    if spherical {
        dist.iter_mut().for_each(|v| *v *= -0.5);
    }
    let labels: Vec<i32> = labels.iter().map(|&l| l + 1).collect();
    Ok(list!(cluster = labels, distance = dist).into())
}

/// First `take` entries of a seeded random permutation of 1..n.
/// @noRd
#[extendr]
fn em_sample(n: f64, take: f64, seed: f64) -> Vec<f64> {
    let mut r = rng::Rng::new(seed as i64 as u64);
    r.partial_permutation(n as usize, take as usize)
        .into_iter()
        .map(|v| v as f64 + 1.0)
        .collect()
}

/// Total sum of squares about the column means (two passes per column).
/// @noRd
#[extendr]
fn em_totss(x: RMatrix<f64>) -> f64 {
    let n = x.nrows();
    if n == 0 {
        return 0.0;
    }
    x.data()
        .chunks_exact(n)
        .map(|col| {
            let mean = col.iter().sum::<f64>() / n as f64;
            col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
        })
        .sum()
}

extendr_module! {
    mod emeansR;
    fn em_totss;
    fn em_data;
    fn em_new;
    fn em_train_steps;
    fn em_accumulate;
    fn em_update;
    fn em_end_epoch;
    fn em_finalize;
    fn em_centroids;
    fn em_labels;
    fn em_info;
    fn em_assign;
    fn em_sample;
}
