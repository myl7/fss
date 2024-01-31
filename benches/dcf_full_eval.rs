// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use fss_rs::dcf::prg::Aes256HirosePrg;
use fss_rs::dcf::{BoundState, CmpFn, Dcf, DcfImpl};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;

pub fn bench(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<3, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = CmpFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup(thread_rng().gen()),
        bound: BoundState::LtBeta,
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);
    const N: usize = 2usize.pow(24);

    c.bench_function("dcf full eval 2^24 xs", |b| {
        b.iter(|| {
            let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
            let dcf = DcfImpl::<3, 16, _>::new(prg);
            let mut ys = vec![ByteGroup::zero(); N];
            dcf.full_eval(false, &k, &mut ys.iter_mut().collect::<Vec<_>>());
        })
    });

    let mut xs = vec![[0; 3]; N];
    xs.iter_mut().for_each(|x| *x = thread_rng().gen());
    c.bench_function("dcf full eval batch 2^24 xs", |b| {
        b.iter(|| {
            let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
            let dcf = DcfImpl::<3, 16, _>::new(prg);
            let mut ys = vec![ByteGroup::zero(); N];
            dcf.eval(
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
