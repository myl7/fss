//! Benchmark-only crate. See benches/ for benchmark code.
//!
//! The wrapper drives the sycret crate through its public C ABI batch entry
//! points, `sycret::keygen` and `sycret::eval` (src/lib.rs upstream), which are
//! the exact functions the Python cffi bindings call: they dispatch on
//! `op_id` (0 = EQ keys of width 621, 1 = CMP keys of width 920), split the
//! batch into fixed AES streams, and size the Rayon pool from `n_threads`.
//! The lower-level Rust modules (`eq`, `le`, `stream`) only expose per-row
//! key structures without a batch API, so the C ABI layer is both the
//! production path and the convenient one; the crate builds as an rlib, so
//! the functions are called directly with no `extern` declarations needed.
//!
//! The bench harness pins OMP_NUM_THREADS=1 and RAYON_NUM_THREADS=1 in the
//! environment; every call below also passes n_threads = 1 so the
//! measurements are single-threaded to match.

use sycret::eq::EqKey;
use sycret::le::LeKey;
use sycret::stream::RawKey;

/// Operation selector for the C ABI entry points: 0 = EQ keys, 1 = CMP (<=) keys.
pub const EQ_OP_ID: usize = 0;
pub const LE_OP_ID: usize = 1;

/// Raw key width in bytes per key: 621 for EQ, 920 for CMP.
pub fn key_len(op_id: usize) -> usize {
    match op_id {
        LE_OP_ID => <LeKey as RawKey>::KEY_LEN,
        _ => <EqKey as RawKey>::KEY_LEN,
    }
}

/// Share modulus of the eval outputs: rows are u32 additive shares.
const MOD32: u64 = 1 << 32;

/// Generate `n_values` key pairs in place. Buffers must be sized by key_len.
pub fn keygen_into(keys_a: &mut [u8], keys_b: &mut [u8], n_values: usize, op_id: usize) {
    assert_eq!(keys_a.len(), key_len(op_id) * n_values, "keys_a mis-sized");
    assert_eq!(keys_b.len(), key_len(op_id) * n_values, "keys_b mis-sized");
    unsafe {
        sycret::keygen(
            keys_a.as_mut_ptr(),
            keys_b.as_mut_ptr(),
            n_values,
            /* n_threads */ 1,
            op_id,
        );
    }
}

/// Evaluate one party over a batch. `xs_le` holds one little-endian u32 per
/// point, `keys` the matching party's key lines, `results` one i64 row per
/// point holding that party's u32 additive share.
pub fn eval_into(
    party_id: usize,
    xs_le: &[u8],
    keys: &[u8],
    results: &mut [i64],
    op_id: usize,
) {
    assert_eq!(xs_le.len() % 4, 0, "xs must be 4-byte little-endian values");
    let n_values = xs_le.len() / 4;
    assert_eq!(keys.len(), key_len(op_id) * n_values, "keys mis-sized");
    assert_eq!(results.len(), n_values, "results mis-sized");
    unsafe {
        sycret::eval(
            party_id,
            xs_le.as_ptr(),
            keys.as_ptr(),
            results.as_mut_ptr(),
            n_values,
            /* n_threads */ 1,
            op_id,
        );
    }
}

/// Reconstruct the secret points alpha from a key pair: alpha is additively
/// shared in the first 4 bytes of each raw key line (little-endian u32).
fn alphas(keys_a: &[u8], keys_b: &[u8], keylen: usize) -> Vec<u32> {
    keys_a
        .chunks_exact(keylen)
        .zip(keys_b.chunks_exact(keylen))
        .map(|(a, b)| {
            let share = |line: &[u8]| u32::from_le_bytes([line[0], line[1], line[2], line[3]]);
            share(a).wrapping_add(share(b))
        })
        .collect()
}

/// Keygen plus both parties' eval on probe points around alpha, checking the
/// FSS semantics: EQ keys output the bit [x == alpha], CMP keys the bit
/// [x <= alpha], reconstructed as (r_a + r_b) mod 2^32.
pub fn check_correctness(op_id: usize) {
    const N: usize = 256;
    const OFFSETS: [i32; 8] = [-1000, -2, -1, 0, 1, 2, 7, i32::MAX];

    let keylen = key_len(op_id);
    let mut keys_a = vec![0u8; keylen * N];
    let mut keys_b = vec![0u8; keylen * N];
    keygen_into(&mut keys_a, &mut keys_b, N, op_id);

    let alpha = alphas(&keys_a, &keys_b, keylen);
    let xs_le: Vec<u8> = (0..N)
        .flat_map(|i| alpha[i].wrapping_add(OFFSETS[i % OFFSETS.len()] as u32).to_le_bytes())
        .collect();

    let mut results_a = vec![0i64; N];
    let mut results_b = vec![0i64; N];
    eval_into(0, &xs_le, &keys_a, &mut results_a, op_id);
    eval_into(1, &xs_le, &keys_b, &mut results_b, op_id);

    for i in 0..N {
        let offset = OFFSETS[i % OFFSETS.len()];
        let x = alpha[i].wrapping_add(offset as u32);
        let bit = ((results_a[i] as u64 + results_b[i] as u64) % MOD32) as u32;
        // Probe offsets that wrap mod 2^32 legitimately compare as u32 values.
        let expected = match op_id {
            LE_OP_ID => (x <= alpha[i]) as u32,
            _ => (x == alpha[i]) as u32,
        };
        assert_eq!(
            bit, expected,
            "op_id {op_id} row {i}: alpha {}, x {x} reconstructs {bit}, expected {expected}",
            alpha[i]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_keygen_eval_semantics() {
        // EQ: reconstructed bit is 1 iff x == alpha.
        check_correctness(EQ_OP_ID);
    }

    #[test]
    fn le_keygen_eval_semantics() {
        // CMP: reconstructed bit is 1 iff x <= alpha.
        check_correctness(LE_OP_ID);
    }
}
