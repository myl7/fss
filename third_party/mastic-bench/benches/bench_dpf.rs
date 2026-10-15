//! Criterion benchmarks for the Mastic VIDPF (upstream crate `mastic`,
//! github.com/TrustworthyComputing/mastic), following third_party/libdpf-bench.
//!
//! Benchmark IDs are `mastic/CPU/<op>/<domain bits>`:
//!   Gen/<NBITS>     one `VidpfKey::gen` producing both party keys; counts keys (1).
//!   Eval/<NBITS>    one point evaluation of one party with the chained 16-byte
//!                   proof; counts point evaluations (1).
//!   EvalAll/<NBITS> `eval_tree` over all 2^NBITS leaves for both parties,
//!                   including the blake3 digest comparison that constitutes
//!                   verification; counts domain outputs (2^NBITS).
//!
//! `beta` is a vector of `INPUT_LEN` Field128 elements, as in upstream's
//! examples/vidpf_smoke.rs; every node word is `beta || counter`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mastic::{u32_to_bits, vec_add, vidpf::VidpfKey, HASH_SIZE};
use prio::field::{Field128, FieldElement};

const INPUT_LEN: usize = 2;
const CHECK_BITS: u8 = 12;

fn beta() -> Vec<Field128> {
    vec![Field128::from(1337u128), Field128::from(42u128)]
}

fn alpha_bits(nbits: u8) -> Vec<bool> {
    u32_to_bits(nbits, 0xBEEF % (1u32 << nbits))
}

fn sum(a: &[Field128], b: &[Field128]) -> Vec<Field128> {
    let mut s = a.to_vec();
    vec_add(&mut s, b);
    s
}

/// One-off correctness gate before timing, mirroring examples/vidpf_smoke.rs at
/// CHECK_BITS: shares reconstruct, the two parties' eval_tree digests match, and
/// chained point-eval proofs agree.
fn check_correctness() {
    let nbits = CHECK_BITS;
    let domain: u32 = 1 << nbits;
    let alpha: u32 = 0xBEEF % domain;
    let alpha_bits = alpha_bits(nbits);
    let beta = beta();
    let mut expected_on_path = beta.clone();
    expected_on_path.push(Field128::one());
    let expected_off_path = vec![Field128::zero(); INPUT_LEN + 1];

    let (key_0, key_1) = VidpfKey::gen(&alpha_bits, &beta);

    // Point evaluation with the chained proof on sample points.
    let mut pi_0 = [0u8; HASH_SIZE];
    let mut pi_1 = [0u8; HASH_SIZE];
    for x in [alpha, alpha ^ 1, 0, domain - 1] {
        let bits = u32_to_bits(nbits, x);
        let (prefix_0, leaf_0) = key_0.eval(&bits, &mut pi_0, INPUT_LEN);
        let (prefix_1, leaf_1) = key_1.eval(&bits, &mut pi_1, INPUT_LEN);
        assert_eq!(pi_0, pi_1, "chained point-eval proof differs at {x}");
        let expected = if x == alpha {
            &expected_on_path
        } else {
            &expected_off_path
        };
        assert_eq!(&sum(&leaf_0, &leaf_1), expected, "leaf value at {x}");
        for (j, (p0, p1)) in prefix_0.iter().zip(prefix_1.iter()).enumerate() {
            let on_path = bits[..=j] == alpha_bits[..=j];
            assert_eq!(
                &sum(p0, p1),
                if on_path {
                    &expected_on_path
                } else {
                    &expected_off_path
                },
                "prefix {} of {x}",
                j + 1
            );
        }
    }

    // Full-domain eval_tree for both parties; verification = matching digests.
    let leaves: Vec<Vec<bool>> = (0..domain).map(|x| u32_to_bits(nbits, x)).collect();
    let mut proof_0 = blake3::Hasher::new();
    let mut proof_1 = blake3::Hasher::new();
    let (vals_0, beta_0) = key_0.eval_tree(leaves.iter(), INPUT_LEN, &mut proof_0);
    let (vals_1, beta_1) = key_1.eval_tree(leaves.iter(), INPUT_LEN, &mut proof_1);
    assert_eq!(
        proof_0.finalize(),
        proof_1.finalize(),
        "eval_tree proofs differ for honest keys"
    );
    assert_eq!(sum(&beta_0, &beta_1), beta, "beta shares do not reconstruct");
    let mut hits = 0;
    for (x, (v0, v1)) in vals_0.iter().zip(vals_1.iter()).enumerate() {
        let expected = if x as u32 == alpha {
            &expected_on_path
        } else {
            &expected_off_path
        };
        assert_eq!(&sum(v0, v1), expected, "value share at {x}");
        hits += (x as u32 == alpha) as usize;
    }
    assert_eq!(hits, 1, "point function must be hot exactly once");
}

fn bench_mastic_gen(c: &mut Criterion) {
    check_correctness();
    let beta = beta();
    for nbits in [12u8, 16, 20] {
        let alpha_bits = alpha_bits(nbits);
        c.bench_function(format!("mastic/CPU/Gen/{nbits}").as_str(), |b| {
            b.iter(|| VidpfKey::gen(black_box(&alpha_bits), black_box(&beta)))
        });
    }
}

fn bench_mastic_eval(c: &mut Criterion) {
    check_correctness();
    let nbits = 16u8;
    let alpha_bits = alpha_bits(nbits);
    let (key_0, _key_1) = VidpfKey::gen(&alpha_bits, &beta());
    // The proof `pi` chains across evaluations, as in upstream's usage.
    let mut pi = [0u8; HASH_SIZE];
    c.bench_function(format!("mastic/CPU/Eval/{nbits}").as_str(), |b| {
        b.iter(|| key_0.eval(black_box(&alpha_bits), &mut pi, black_box(INPUT_LEN)))
    });
}

fn bench_mastic_eval_all(c: &mut Criterion) {
    check_correctness();
    let beta = beta();
    for nbits in [12u8, 16, 20] {
        let domain: u32 = 1 << nbits;
        let (key_0, key_1) = VidpfKey::gen(&alpha_bits(nbits), &beta);
        let leaves: Vec<Vec<bool>> = (0..domain).map(|x| u32_to_bits(nbits, x)).collect();
        c.bench_function(format!("mastic/CPU/EvalAll/{nbits}").as_str(), |b| {
            b.iter(|| {
                let mut proof_0 = blake3::Hasher::new();
                let mut proof_1 = blake3::Hasher::new();
                let (vals_0, beta_0) =
                    key_0.eval_tree(black_box(leaves.iter()), INPUT_LEN, &mut proof_0);
                let (vals_1, beta_1) =
                    key_1.eval_tree(black_box(leaves.iter()), INPUT_LEN, &mut proof_1);
                // Verification: the two aggregators' digests must be identical.
                black_box(proof_0.finalize() == proof_1.finalize());
                black_box((vals_0, beta_0, vals_1, beta_1));
            })
        });
    }
}

criterion_group!(benches, bench_mastic_gen, bench_mastic_eval, bench_mastic_eval_all);
criterion_main!(benches);
