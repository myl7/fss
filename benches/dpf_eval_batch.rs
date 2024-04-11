// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use fss_rs::dpf::prg::Aes128MatyasMeyerOseasPrg;
use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;

const POINT_NUM: usize = 10000;

fn from_domain_range_size<const DOM_SZ: usize, const LAMBDA: usize, const CIPHER_N: usize>(
    c: &mut Criterion,
) {
    let mut keys = [[0; 16]; CIPHER_N];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<LAMBDA, CIPHER_N>::new(keys_iter);
    let dpf = DpfImpl::<DOM_SZ, LAMBDA, _>::new(prg);

    let mut s0s = [[0; LAMBDA]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0; DOM_SZ];
    thread_rng().fill_bytes(&mut alpha);
    let mut beta_buf = [0; LAMBDA];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };

    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut xs = vec![[0; DOM_SZ]; POINT_NUM];
    xs.iter_mut().for_each(|x| thread_rng().fill_bytes(x));
    let xs_iter: Vec<_> = xs.iter().collect();
    let mut ys = vec![ByteGroup::zero(); POINT_NUM];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_with_input(
        BenchmarkId::new(
            "dpf eval batch",
            format!("{} points, {}b -> {}B", POINT_NUM, DOM_SZ * 8 - 1, LAMBDA),
        ),
        &(POINT_NUM, DOM_SZ, LAMBDA),
        |b, &_| {
            b.iter(|| {
                dpf.eval(false, &k, &xs_iter, &mut ys_iter);
            });
        },
    );
}

fn bench(c: &mut Criterion) {
    from_domain_range_size::<16, 16, 2>(c);
    from_domain_range_size::<24, 16, 2>(c);
    from_domain_range_size::<32, 16, 2>(c);
    from_domain_range_size::<16, 256, 32>(c);
}

criterion_group!(benches, bench);
criterion_main!(benches);
