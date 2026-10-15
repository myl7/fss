//! Criterion benchmarks for sycret's EQ and CMP FSS keys (fixed 32-bit domain).
//!
//! Work units: EqKeygen/LeKeygen time one keygen call over a batch, so the
//! trailing number is keys per batch; EqEval/LeEval time one party's eval
//! call over a batch, so the trailing number is points per batch. Each timed
//! iteration is a single such batch call.
//!
//! There is no EvalAll benchmark: sycret exposes no full-domain evaluation
//! API, only batch point evaluation (the domain is 2^32 points per key).
//!
//! The harness pins OMP_NUM_THREADS=1 and RAYON_NUM_THREADS=1 in the
//! environment; all calls also pass n_threads = 1 to match.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sycret_bench::{check_correctness, eval_into, key_len, keygen_into, EQ_OP_ID, LE_OP_ID};

// Batch sizes: keys per batch for keygen, points per batch for eval.
const BATCH_SIZES: [usize; 3] = [1024, 16384, 65536];

fn bench_eq_keygen(c: &mut Criterion) {
    check_correctness(EQ_OP_ID);
    for &n in &BATCH_SIZES {
        let mut keys_a = vec![0u8; key_len(EQ_OP_ID) * n];
        let mut keys_b = vec![0u8; key_len(EQ_OP_ID) * n];
        c.bench_function(format!("sycret/CPU/EqKeygen/{n}").as_str(), |b| {
            b.iter(|| {
                keygen_into(black_box(&mut keys_a), black_box(&mut keys_b), black_box(n), EQ_OP_ID)
            })
        });
    }
}

fn bench_le_keygen(c: &mut Criterion) {
    check_correctness(LE_OP_ID);
    for &n in &BATCH_SIZES {
        let mut keys_a = vec![0u8; key_len(LE_OP_ID) * n];
        let mut keys_b = vec![0u8; key_len(LE_OP_ID) * n];
        c.bench_function(format!("sycret/CPU/LeKeygen/{n}").as_str(), |b| {
            b.iter(|| {
                keygen_into(black_box(&mut keys_a), black_box(&mut keys_b), black_box(n), LE_OP_ID)
            })
        });
    }
}

fn bench_eq_eval(c: &mut Criterion) {
    check_correctness(EQ_OP_ID);
    for &n in &BATCH_SIZES {
        let mut keys_a = vec![0u8; key_len(EQ_OP_ID) * n];
        let mut keys_b = vec![0u8; key_len(EQ_OP_ID) * n];
        keygen_into(&mut keys_a, &mut keys_b, n, EQ_OP_ID);
        let xs_le = probe_points(n);
        let mut results = vec![0i64; n];
        c.bench_function(format!("sycret/CPU/EqEval/{n}").as_str(), |b| {
            b.iter(|| {
                eval_into(
                    black_box(0),
                    black_box(&xs_le),
                    black_box(&keys_a),
                    black_box(&mut results),
                    EQ_OP_ID,
                )
            })
        });
    }
}

fn bench_le_eval(c: &mut Criterion) {
    check_correctness(LE_OP_ID);
    for &n in &BATCH_SIZES {
        let mut keys_a = vec![0u8; key_len(LE_OP_ID) * n];
        let mut keys_b = vec![0u8; key_len(LE_OP_ID) * n];
        keygen_into(&mut keys_a, &mut keys_b, n, LE_OP_ID);
        let xs_le = probe_points(n);
        let mut results = vec![0i64; n];
        c.bench_function(format!("sycret/CPU/LeEval/{n}").as_str(), |b| {
            b.iter(|| {
                eval_into(
                    black_box(0),
                    black_box(&xs_le),
                    black_box(&keys_a),
                    black_box(&mut results),
                    LE_OP_ID,
                )
            })
        });
    }
}

/// Evaluation cost is independent of the queried points, so a fixed
/// deterministic spread over the 32-bit domain is used.
fn probe_points(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes())
        .collect()
}

criterion_group!(benches, bench_eq_keygen, bench_le_keygen, bench_eq_eval, bench_le_eval);
criterion_main!(benches);
