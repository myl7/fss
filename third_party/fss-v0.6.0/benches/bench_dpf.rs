use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use fss_rs::dcf::{BoundState, CmpFn, Dcf, DcfImpl};
use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::int::U128Group;
use fss_rs::group::Group;
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

const fn domain_bits() -> usize {
    let value = match option_env!("FSS_BENCH_DOMAIN_BITS") {
        Some(value) => value,
        None => "20",
    };
    let bytes = value.as_bytes();
    let mut bits = 0;
    let mut i = 0;
    while i < bytes.len() {
        assert!(bytes[i] >= b'0' && bytes[i] <= b'9', "invalid domain bits");
        bits = bits * 10 + (bytes[i] - b'0') as usize;
        i += 1;
    }
    assert!(
        bits >= 8 && bits <= 20,
        "domain bits must be between 8 and 20"
    );
    bits
}

const FILTER_BITN: usize = domain_bits();
// Keep the historical three-byte input representation across the sweep.
const IN_BLEN: usize = 3;
const OUT_BLEN: usize = 16;

// The upstream filter reads the first FILTER_BITN bits in MSB order.
fn mask_input(value: &mut [u8; IN_BLEN]) {
    for bit in FILTER_BITN..IN_BLEN * 8 {
        value[bit / 8] &= !(1 << (7 - bit % 8));
    }
}

fn check_outputs<G: Group<OUT_BLEN> + std::fmt::Debug>(
    alpha: &[u8; IN_BLEN],
    beta: &G,
    comparison: bool,
    mut eval: impl FnMut(bool, &[u8; IN_BLEN], &mut G),
) {
    let mut upper = [0xff; IN_BLEN];
    mask_input(&mut upper);
    let mut miss = *alpha;
    let bit = FILTER_BITN - 1;
    miss[bit / 8] ^= 1 << (7 - bit % 8);
    for x in [*alpha, miss, [0; IN_BLEN], upper] {
        let mut first = G::zero();
        let mut second = G::zero();
        eval(false, &x, &mut first);
        eval(true, &x, &mut second);
        let hit = if comparison { x < *alpha } else { x == *alpha };
        let expected = if hit { beta.clone() } else { G::zero() };
        let result = first + second;
        assert!(
            result == expected,
            "share reconstruction failed: group={}, comparison={}, alpha={:02x?}, query={:02x?}, result={:?}, expected={:?}",
            std::any::type_name::<G>(),
            comparison,
            alpha,
            x,
            result,
            expected
        );
    }
}

// --- DPF ByteGroup ---

fn bench_dpf_gen_bytes(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 2];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 1, 2>::new(&keys_iter);
    let dpf = DpfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };

    c.bench_function("fss-v0.6.0/CPU/DPF-bytes/Gen", |b| {
        let checked_key = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, false, |party, x, y| {
            dpf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dpf.gen(&f, [&s0s[0], &s0s[1]]))
    });
}

fn bench_dpf_eval_bytes(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 2];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 1, 2>::new(&keys_iter);
    let dpf = DpfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut x);
    mask_input(&mut x);
    let mut y = ByteGroup::zero();

    c.bench_function("fss-v0.6.0/CPU/DPF-bytes/Eval", |b| {
        let checked_key = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, false, |party, x, y| {
            dpf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dpf.eval(false, &k, &[&x], &mut [&mut y]))
    });
}

fn bench_dpf_full_eval_bytes(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 2];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 1, 2>::new(&keys_iter);
    let dpf = DpfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut ys = vec![ByteGroup::zero(); 1 << FILTER_BITN];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_function("fss-v0.6.0/CPU/DPF-bytes/FullEval", |b| {
        let checked_key = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, false, |party, x, y| {
            dpf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dpf.full_eval(false, &k, &mut ys_iter))
    });
}

// --- DPF U128Group ---

fn bench_dpf_gen_uint(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 2];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 1, 2>::new(&keys_iter);
    let dpf = DpfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let beta = U128Group(thread_rng().gen());
    let f = PointFn { alpha, beta };

    c.bench_function("fss-v0.6.0/CPU/DPF-uint/Gen", |b| {
        let checked_key = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, false, |party, x, y| {
            dpf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dpf.gen(&f, [&s0s[0], &s0s[1]]))
    });
}

fn bench_dpf_eval_uint(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 2];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 1, 2>::new(&keys_iter);
    let dpf = DpfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let beta = U128Group(thread_rng().gen());
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut x);
    mask_input(&mut x);
    let mut y = <U128Group as Group<OUT_BLEN>>::zero();

    c.bench_function("fss-v0.6.0/CPU/DPF-uint/Eval", |b| {
        let checked_key = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, false, |party, x, y| {
            dpf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dpf.eval(false, &k, &[&x], &mut [&mut y]))
    });
}

fn bench_dpf_full_eval_uint(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 2];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 1, 2>::new(&keys_iter);
    let dpf = DpfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let beta = U128Group(thread_rng().gen());
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut ys = vec![<U128Group as Group<OUT_BLEN>>::zero(); 1 << FILTER_BITN];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_function("fss-v0.6.0/CPU/DPF-uint/FullEval", |b| {
        let checked_key = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, false, |party, x, y| {
            dpf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dpf.full_eval(false, &k, &mut ys_iter))
    });
}

// --- DCF ByteGroup ---

fn bench_dcf_gen_bytes(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 4];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, 4>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };

    c.bench_function("fss-v0.6.0/CPU/DCF-bytes/Gen", |b| {
        let checked_key = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, true, |party, x, y| {
            dcf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dcf.gen(&f, [&s0s[0], &s0s[1]]))
    });
}

fn bench_dcf_eval_bytes(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 4];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, 4>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut x);
    mask_input(&mut x);
    let mut y = ByteGroup::zero();

    c.bench_function("fss-v0.6.0/CPU/DCF-bytes/Eval", |b| {
        let checked_key = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, true, |party, x, y| {
            dcf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dcf.eval(false, &k, &[&x], &mut [&mut y]))
    });
}

fn bench_dcf_full_eval_bytes(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 4];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, 4>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut ys = vec![ByteGroup::zero(); 1 << FILTER_BITN];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_function("fss-v0.6.0/CPU/DCF-bytes/FullEval", |b| {
        let checked_key = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, true, |party, x, y| {
            dcf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dcf.full_eval(false, &k, &mut ys_iter))
    });
}

// --- DCF U128Group ---

fn bench_dcf_gen_uint(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 4];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, 4>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let beta = U128Group(thread_rng().gen());
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };

    c.bench_function("fss-v0.6.0/CPU/DCF-uint/Gen", |b| {
        let checked_key = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, true, |party, x, y| {
            dcf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dcf.gen(&f, [&s0s[0], &s0s[1]]))
    });
}

fn bench_dcf_eval_uint(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 4];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, 4>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let beta = U128Group(thread_rng().gen());
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut x);
    mask_input(&mut x);
    let mut y = <U128Group as Group<OUT_BLEN>>::zero();

    c.bench_function("fss-v0.6.0/CPU/DCF-uint/Eval", |b| {
        let checked_key = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, true, |party, x, y| {
            dcf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dcf.eval(false, &k, &[&x], &mut [&mut y]))
    });
}

fn bench_dcf_full_eval_uint(c: &mut Criterion) {
    let mut keys = [[0u8; 16]; 4];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, 4>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, FILTER_BITN);

    let mut s0s = [[0u8; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    mask_input(&mut alpha);
    let beta = U128Group(thread_rng().gen());
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut ys = vec![<U128Group as Group<OUT_BLEN>>::zero(); 1 << FILTER_BITN];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_function("fss-v0.6.0/CPU/DCF-uint/FullEval", |b| {
        let checked_key = dcf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut checked_key1 = checked_key.clone();
        checked_key1.s0s = vec![checked_key.s0s[1]];
        check_outputs(&f.alpha, &f.beta, true, |party, x, y| {
            dcf.eval(
                party,
                if party { &checked_key1 } else { &checked_key },
                &[x],
                &mut [y],
            );
        });

        b.iter(|| dcf.full_eval(false, &k, &mut ys_iter))
    });
}

criterion_group!(
    benches,
    // DPF ByteGroup
    bench_dpf_gen_bytes,
    bench_dpf_eval_bytes,
    bench_dpf_full_eval_bytes,
    // DPF U128Group
    bench_dpf_gen_uint,
    bench_dpf_eval_uint,
    bench_dpf_full_eval_uint,
    // DCF ByteGroup
    bench_dcf_gen_bytes,
    bench_dcf_eval_bytes,
    bench_dcf_full_eval_bytes,
    // DCF U128Group
    bench_dcf_gen_uint,
    bench_dcf_eval_uint,
    bench_dcf_full_eval_uint,
);
criterion_main!(benches);
