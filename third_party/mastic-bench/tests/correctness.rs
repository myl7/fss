//! One-off correctness checks for the Mastic VIDPF wrapper, adapted from
//! upstream's examples/vidpf_smoke.rs (not part of upstream).
//!
//! Run with `cargo test`.

use mastic::{u32_to_bits, vec_add, vidpf::VidpfKey, HASH_SIZE};
use prio::field::{Field128, FieldElement};

const NBITS: u8 = 12;
const DOMAIN: u32 = 1 << NBITS;
const INPUT_LEN: usize = 2;

fn sum(a: &[Field128], b: &[Field128]) -> Vec<Field128> {
    let mut s = a.to_vec();
    vec_add(&mut s, b);
    s
}

fn setup() -> (Vec<bool>, Vec<Field128>, Vec<Field128>, Vec<Field128>, VidpfKey, VidpfKey) {
    let alpha = 0xBEEF % DOMAIN;
    let alpha_bits = u32_to_bits(NBITS, alpha);
    let beta = vec![Field128::from(1337u128), Field128::from(42u128)];
    let mut expected_on_path = beta.clone();
    expected_on_path.push(Field128::one());
    let expected_off_path = vec![Field128::zero(); INPUT_LEN + 1];
    let (key_0, key_1) = VidpfKey::gen(&alpha_bits, &beta);
    assert!(!key_0.key_idx && key_1.key_idx);
    assert_eq!(key_0.domain_size(), NBITS as usize);
    (alpha_bits, beta, expected_on_path, expected_off_path, key_0, key_1)
}

#[test]
fn eval_tree_reconstructs_and_proofs_match() {
    let (_alpha_bits, beta, expected_on_path, expected_off_path, key_0, key_1) = setup();
    let alpha = 0xBEEF % DOMAIN;

    let leaves: Vec<Vec<bool>> = (0..DOMAIN).map(|x| u32_to_bits(NBITS, x)).collect();
    let mut proof_0 = blake3::Hasher::new();
    let mut proof_1 = blake3::Hasher::new();
    let (vals_0, beta_0) = key_0.eval_tree(leaves.iter(), INPUT_LEN, &mut proof_0);
    let (vals_1, beta_1) = key_1.eval_tree(leaves.iter(), INPUT_LEN, &mut proof_1);

    // Verification: the two aggregators' proof digests must be identical.
    assert_eq!(
        proof_0.finalize(),
        proof_1.finalize(),
        "eval_tree proofs differ for honest keys"
    );

    // beta shares reconstruct.
    assert_eq!(sum(&beta_0, &beta_1), beta, "beta shares do not reconstruct");

    // Point function reconstruction over the whole domain.
    assert_eq!(vals_0.len(), DOMAIN as usize);
    assert_eq!(vals_1.len(), DOMAIN as usize);
    let mut hits = 0usize;
    for (x, (v0, v1)) in vals_0.iter().zip(vals_1.iter()).enumerate() {
        let s = sum(v0, v1);
        if x as u32 == alpha {
            assert_eq!(s, expected_on_path, "wrong value at alpha");
            hits += 1;
        } else {
            assert_eq!(s, expected_off_path, "nonzero value at x={x}");
        }
    }
    assert_eq!(hits, 1);

    // A single share must not reveal beta (fails with probability 2^-128).
    assert_ne!(vals_0[alpha as usize], expected_on_path);
    assert_ne!(vals_1[alpha as usize], expected_on_path);
}

#[test]
fn point_eval_chains_proof_and_reconstructs() {
    let (alpha_bits, _beta, expected_on_path, expected_off_path, key_0, key_1) = setup();
    let alpha = 0xBEEF % DOMAIN;

    let samples: Vec<u32> = vec![
        alpha,
        alpha ^ 1,
        0,
        DOMAIN - 1,
        (alpha + 12345) % DOMAIN,
        alpha ^ (1 << (NBITS - 1)),
    ];
    let mut pi_0 = [0u8; HASH_SIZE];
    let mut pi_1 = [0u8; HASH_SIZE];
    for &x in &samples {
        let bits = u32_to_bits(NBITS, x);
        let (prefix_0, leaf_0) = key_0.eval(&bits, &mut pi_0, INPUT_LEN);
        let (prefix_1, leaf_1) = key_1.eval(&bits, &mut pi_1, INPUT_LEN);
        assert_eq!(pi_0, pi_1, "chained point-eval proof differs at x={x}");

        let expected = if x == alpha {
            &expected_on_path
        } else {
            &expected_off_path
        };
        assert_eq!(&sum(&leaf_0, &leaf_1), expected, "leaf value mismatch at x={x}");

        // `eval` also returns the shares of every proper prefix (lengths 1..NBITS-1).
        assert_eq!(prefix_0.len(), NBITS as usize - 1);
        for (j, (p0, p1)) in prefix_0.iter().zip(prefix_1.iter()).enumerate() {
            let on_path = bits[..=j] == alpha_bits[..=j];
            let expected = if on_path {
                &expected_on_path
            } else {
                &expected_off_path
            };
            assert_eq!(
                &sum(p0, p1),
                expected,
                "prefix mismatch at x={x} len={}",
                j + 1
            );
        }
    }
}

#[test]
fn tampered_and_mismatched_keys_fail_verification() {
    let (_, beta, _, _, key_0, key_1) = setup();
    let leaves: Vec<Vec<bool>> = (0..DOMAIN).map(|x| u32_to_bits(NBITS, x)).collect();
    let mut proof_0 = blake3::Hasher::new();
    let _ = key_0.eval_tree(leaves.iter(), INPUT_LEN, &mut proof_0);
    let digest_0 = proof_0.finalize();

    // Tamper with a correction seed hash `cs` of party 1: verification must fail.
    let mut bad_key_1 = key_1.clone();
    bad_key_1.cs[NBITS as usize / 2][0] ^= 0x01;
    let mut bad_proof = blake3::Hasher::new();
    let _ = bad_key_1.eval_tree(leaves.iter(), INPUT_LEN, &mut bad_proof);
    assert_ne!(bad_proof.finalize(), digest_0, "tampered cs was not detected");

    // Keys from two independent Gen calls (same alpha/beta) must not verify together.
    let alpha_bits = u32_to_bits(NBITS, 0xBEEF % DOMAIN);
    let (_, other_key_1) = VidpfKey::gen(&alpha_bits, &beta);
    let mut other_proof = blake3::Hasher::new();
    let _ = other_key_1.eval_tree(leaves.iter(), INPUT_LEN, &mut other_proof);
    assert_ne!(
        other_proof.finalize(),
        digest_0,
        "mismatched key pair was not detected"
    );
}
