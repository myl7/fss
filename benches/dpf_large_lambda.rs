// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use fss_rs::dpf::prg::Aes128MatyasMeyerOseasPrg;
use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;

pub fn bench(c: &mut Criterion) {
    let mut keys = vec![[0; 16]; 2048];
    keys.iter_mut().for_each(|key| thread_rng().fill_bytes(key));
    let prg = Aes128MatyasMeyerOseasPrg::<16384, 2048>::new(std::array::from_fn(|i| &keys[i]));
    let dpf = DpfImpl::<16, 16384, _>::new(prg);
    let mut s0s = vec![[0; 16384]; 2];
    thread_rng().fill_bytes(&mut s0s[0]);
    thread_rng().fill_bytes(&mut s0s[1]);
    let mut f = Box::new(PointFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup::zero(),
    });
    thread_rng().fill_bytes(&mut f.beta.0);
    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
    const N: usize = 10_000;
    let xs: Vec<[u8; 16]> = (0..N).map(|_| thread_rng().gen()).collect();

    c.bench_function("dpf eval 10k xs with lambda 16384", |b| {
        b.iter(|| {
            let prg =
                Aes128MatyasMeyerOseasPrg::<16384, 2048>::new(std::array::from_fn(|i| &keys[i]));
            let dpf = DpfImpl::<16, 16384, _>::new(prg);
            let mut ys = vec![ByteGroup::zero(); N];
            dpf.eval(
                false,
                &k,
                &xs.iter().collect::<Vec<_>>(),
                &mut ys.iter_mut().collect::<Vec<_>>(),
            );
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench
}
criterion_main!(benches);
