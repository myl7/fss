//! Criterion benchmarks for the PLASMA DPF (upstream crate `plasma`,
//! github.com/TrustworthyComputing/plasma), following third_party/libdpf-bench.
//!
//! Benchmark IDs are `plasma/CPU/<op>/<domain bits>`:
//!   Gen/<NBITS>  one `DPFKey::gen` producing both party keys; counts keys (1).
//!   Eval/<NBITS> one point evaluation of one party, updating the running
//!                12-byte VIDPF proof; counts point evaluations (1).
//!
//! There is no EvalAll benchmark: unlike mastic, plasma's DPFKey exposes no
//! full-domain evaluation entry point (no eval_tree/EvalAll equivalent). Upstream
//! only walks whole trees through collect::KeyCollection::tree_crawl, whose
//! frontier expansion maintains per-level aggregator state across many keys;
//! emulating a full-domain run by looping per-point `eval` would measure a code
//! path upstream never executes (and would re-walk all prefixes for each leaf),
//! so only Gen and point Eval are benchmarked.
//!
//! The output group is u64 modulo p = 2^63 - 25.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use plasma::consts::XOF_SIZE;
use plasma::dpf::DPFKey;
use plasma::{u32_to_bits, Group};

const CHECK_BITS: u8 = 10;

/// One payload word per tree level, exactly as upstream's DPFKey::gen_from_str.
fn values(nbits: u8) -> Vec<u64> {
    vec![<u64 as Group>::one(); nbits as usize]
}

fn alpha_bits(nbits: u8) -> Vec<bool> {
    u32_to_bits(nbits, 0xBEEF % (1u32 << nbits))
}

/// One-off correctness gate before timing, mirroring upstream's tests/smoke.rs:
/// on-path words sum to one, off-path words sum to zero, and the two parties'
/// running proofs agree. The whole-domain loop uses per-point eval because
/// plasma has no full-domain API.
fn check_correctness() {
    let nbits = CHECK_BITS;
    let domain: u32 = 1 << nbits;
    let alpha: u32 = 0xBEEF % domain;
    let alpha_bits = alpha_bits(nbits);
    let (key_0, key_1) = DPFKey::<u64>::gen(&alpha_bits, &values(nbits));

    // On-path point eval: every prefix word and the final word sum to one.
    let mut pi_0 = [0u8; XOF_SIZE];
    let mut pi_1 = [0u8; XOF_SIZE];
    let (prefix_0, last_0) = key_0.eval(&alpha_bits, &mut pi_0);
    let (prefix_1, last_1) = key_1.eval(&alpha_bits, &mut pi_1);
    assert_eq!(pi_0, pi_1, "running proofs differ at alpha");
    assert_eq!(prefix_0.len(), nbits as usize - 1);
    for (w0, w1) in prefix_0.iter().zip(prefix_1.iter()) {
        let mut word = *w0;
        word.add(w1);
        assert_eq!(word.value(), 1u64, "on-path prefix word");
    }
    let mut word = last_0;
    word.add(&last_1);
    assert_eq!(word.value(), 1u64, "on-path final word");

    // Off-path point eval: all words sum to zero, proofs still agree.
    let other = u32_to_bits(nbits, (alpha + 1) % domain);
    let mut pi_x0 = [0u8; XOF_SIZE];
    let mut pi_x1 = [0u8; XOF_SIZE];
    let (prefix_x0, last_x0) = key_0.eval(&other, &mut pi_x0);
    let (prefix_x1, last_x1) = key_1.eval(&other, &mut pi_x1);
    assert_eq!(pi_x0, pi_x1, "running proofs differ off alpha");
    for (w0, w1) in prefix_x0.iter().zip(prefix_x1.iter()) {
        let mut word = *w0;
        word.add(w1);
        assert_eq!(word.value(), 0u64, "off-path prefix word");
    }
    let mut word = last_x0;
    word.add(&last_x1);
    assert_eq!(word.value(), 0u64, "off-path final word");

    // Whole domain via per-point eval: exactly one hot leaf, proofs always agree.
    let mut hits = 0usize;
    for x in 0..domain {
        let bits = u32_to_bits(nbits, x);
        let mut p0 = [0u8; XOF_SIZE];
        let mut p1 = [0u8; XOF_SIZE];
        let (_, l0) = key_0.eval(&bits, &mut p0);
        let (_, l1) = key_1.eval(&bits, &mut p1);
        assert_eq!(p0, p1, "running proofs differ at x={x}");
        let mut word = l0;
        word.add(&l1);
        if x == alpha {
            assert_eq!(word.value(), 1u64, "value at alpha");
            hits += 1;
        } else {
            assert_eq!(word.value(), 0u64, "nonzero value at x={x}");
        }
    }
    assert_eq!(hits, 1, "point function must be hot exactly once");
}

fn bench_plasma_gen(c: &mut Criterion) {
    check_correctness();
    for nbits in [16u8, 20] {
        let alpha_bits = alpha_bits(nbits);
        let values = values(nbits);
        c.bench_function(format!("plasma/CPU/Gen/{nbits}").as_str(), |b| {
            b.iter(|| DPFKey::<u64>::gen(black_box(&alpha_bits), black_box(&values)))
        });
    }
}

fn bench_plasma_eval(c: &mut Criterion) {
    check_correctness();
    let nbits = 16u8;
    let alpha_bits = alpha_bits(nbits);
    let (key_0, _key_1) = DPFKey::<u64>::gen(&alpha_bits, &values(nbits));
    // `pi` is the running VIDPF proof, chained across evaluations as upstream does.
    let mut pi = [0u8; XOF_SIZE];
    c.bench_function(format!("plasma/CPU/Eval/{nbits}").as_str(), |b| {
        b.iter(|| key_0.eval(black_box(&alpha_bits), &mut pi))
    });
}

criterion_group!(benches, bench_plasma_gen, bench_plasma_eval);
criterion_main!(benches);
