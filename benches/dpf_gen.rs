// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use fss_rs::dpf::prg::Aes128MatyasMeyerOseasPrg;
use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
use fss_rs::group::byte::ByteGroup;

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
    let mut beta_buf: [u8; LAMBDA] = [0; LAMBDA];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = PointFn { alpha, beta };

    c.bench_with_input(
        BenchmarkId::new("dpf gen", format!("{}b -> {}B", DOM_SZ * 8, LAMBDA)),
        &(DOM_SZ, LAMBDA),
        |b, &_| {
            b.iter(|| {
                dpf.gen(&f, [&s0s[0], &s0s[1]]);
            });
        },
    );
}

fn bench(c: &mut Criterion) {
    from_domain_range_size::<16, 16, 2>(c);
    from_domain_range_size::<24, 16, 2>(c);
    from_domain_range_size::<32, 16, 2>(c);
    from_domain_range_size::<16, 256, 32>(c);
    from_domain_range_size::<16, 16384, 2048>(c);
}

criterion_group!(benches, bench);
criterion_main!(benches);
