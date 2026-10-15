//! One-off correctness checks for the PLASMA DPF wrapper, adapted from
//! upstream's tests/smoke.rs (not part of upstream).
//!
//! Run with `cargo test`. plasma has no full-domain API, so the whole-domain
//! check loops per-point `eval`, which is the only evaluation entry point on
//! DPFKey.

use plasma::consts::XOF_SIZE;
use plasma::dpf::DPFKey;
use plasma::{u32_to_bits, Group};

const NBITS: u8 = 10;
const DOMAIN: u32 = 1 << NBITS;
const ALPHA: u32 = 0xBEEF % DOMAIN;

/// One payload word per tree level, exactly as upstream's DPFKey::gen_from_str.
fn values(nbits: u8) -> Vec<u64> {
    vec![<u64 as Group>::one(); nbits as usize]
}

fn alpha_bits() -> Vec<bool> {
    u32_to_bits(NBITS, ALPHA)
}

#[test]
fn point_eval_reconstructs_and_proofs_agree() {
    let alpha_bits = alpha_bits();
    let (k0, k1) = DPFKey::<u64>::gen(&alpha_bits, &values(NBITS));

    // On-path: all prefix words and the final word sum to one.
    let mut pi_0 = [0u8; XOF_SIZE];
    let mut pi_1 = [0u8; XOF_SIZE];
    let (prefix_0, last_0) = k0.eval(&alpha_bits, &mut pi_0);
    let (prefix_1, last_1) = k1.eval(&alpha_bits, &mut pi_1);
    assert_eq!(prefix_0.len(), NBITS as usize - 1);
    for (w0, w1) in prefix_0.iter().zip(prefix_1.iter()) {
        let mut sum = *w0;
        sum.add(w1);
        assert_eq!(sum.value(), 1u64);
    }
    let mut sum = last_0;
    sum.add(&last_1);
    assert_eq!(sum.value(), 1u64);

    // Off-path: everything sums to zero.
    let other = u32_to_bits(NBITS, (ALPHA + 1) % DOMAIN);
    let mut pi_x0 = [0u8; XOF_SIZE];
    let mut pi_x1 = [0u8; XOF_SIZE];
    let (prefix_x0, last_x0) = k0.eval(&other, &mut pi_x0);
    let (prefix_x1, last_x1) = k1.eval(&other, &mut pi_x1);
    for (w0, w1) in prefix_x0.iter().zip(prefix_x1.iter()) {
        let mut sum = *w0;
        sum.add(w1);
        assert_eq!(sum.value(), 0u64);
    }
    let mut sum = last_x0;
    sum.add(&last_x1);
    assert_eq!(sum.value(), 0u64);

    // VIDPF consistency: both parties' running proofs must agree on any path.
    assert_eq!(pi_0, pi_1);
    assert_eq!(pi_x0, pi_x1);
}

#[test]
fn whole_domain_point_evals_reconstruct_unit_function() {
    let alpha_bits = alpha_bits();
    let (k0, k1) = DPFKey::<u64>::gen(&alpha_bits, &values(NBITS));
    let mut hits = 0usize;
    for x in 0..DOMAIN {
        let bits = u32_to_bits(NBITS, x);
        let mut p0 = [0u8; XOF_SIZE];
        let mut p1 = [0u8; XOF_SIZE];
        let (_, l0) = k0.eval(&bits, &mut p0);
        let (_, l1) = k1.eval(&bits, &mut p1);
        assert_eq!(p0, p1, "proof mismatch at x={x}");
        let mut sum = l0;
        sum.add(&l1);
        if x == ALPHA {
            assert_eq!(sum.value(), 1u64, "value at alpha");
            hits += 1;
        } else {
            assert_eq!(sum.value(), 0u64, "nonzero value at x={x}");
        }
    }
    assert_eq!(hits, 1);
}

#[test]
fn mismatched_key_pair_proofs_differ() {
    // Keys from two independent Gen calls (same alpha) must not verify together:
    // their running proofs diverge (collision probability 2^-96).
    let alpha_bits = alpha_bits();
    let (k0, _) = DPFKey::<u64>::gen(&alpha_bits, &values(NBITS));
    let (_, other_k1) = DPFKey::<u64>::gen(&alpha_bits, &values(NBITS));
    let mut pi_0 = [0u8; XOF_SIZE];
    let mut pi_1 = [0u8; XOF_SIZE];
    let _ = k0.eval(&alpha_bits, &mut pi_0);
    let _ = other_k1.eval(&alpha_bits, &mut pi_1);
    assert_ne!(pi_0, pi_1, "mismatched key pair was not detected");
}
