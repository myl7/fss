use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use fss_rs::dcf::{BoundState, CmpFn, Dcf, DcfImpl};
use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::int::U128Group;
use fss_rs::group::Group;
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

const FILTER_BITN: usize = 20;
// IN_BLEN=3 (24 bits) is the smallest byte count that can hold 20 bits.
const IN_BLEN: usize = 3;
const OUT_BLEN: usize = 16;

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
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };

    c.bench_function("fss-v0.6.0/CPU/DPF-bytes/Gen", |b| {
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
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut x);
    let mut y = ByteGroup::zero();

    c.bench_function("fss-v0.6.0/CPU/DPF-bytes/Eval", |b| {
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
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut ys = vec![ByteGroup::zero(); 1 << FILTER_BITN];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_function("fss-v0.6.0/CPU/DPF-bytes/FullEval", |b| {
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
    let beta = U128Group(thread_rng().gen());
    let f = PointFn { alpha, beta };

    c.bench_function("fss-v0.6.0/CPU/DPF-uint/Gen", |b| {
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
    let beta = U128Group(thread_rng().gen());
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut x);
    let mut y = <U128Group as Group<OUT_BLEN>>::zero();

    c.bench_function("fss-v0.6.0/CPU/DPF-uint/Eval", |b| {
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
    let beta = U128Group(thread_rng().gen());
    let f = PointFn { alpha, beta };
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut ys = vec![<U128Group as Group<OUT_BLEN>>::zero(); 1 << FILTER_BITN];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_function("fss-v0.6.0/CPU/DPF-uint/FullEval", |b| {
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
    let mut beta_buf = [0u8; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };

    c.bench_function("fss-v0.6.0/CPU/DCF-bytes/Gen", |b| {
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
    let mut y = ByteGroup::zero();

    c.bench_function("fss-v0.6.0/CPU/DCF-bytes/Eval", |b| {
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
    let beta = U128Group(thread_rng().gen());
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };

    c.bench_function("fss-v0.6.0/CPU/DCF-uint/Gen", |b| {
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
    let beta = U128Group(thread_rng().gen());
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0u8; IN_BLEN];
    thread_rng().fill_bytes(&mut x);
    let mut y = <U128Group as Group<OUT_BLEN>>::zero();

    c.bench_function("fss-v0.6.0/CPU/DCF-uint/Eval", |b| {
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
