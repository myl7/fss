// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;
use fss_rs::prg::Aes256HirosePrg;

pub fn bench_gen(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::new(&keys.iter().collect::<Vec<_>>());
    let dpf = DpfImpl::<16, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = PointFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup(thread_rng().gen()),
    };

    c.bench_function("dpf gen", |b| {
        b.iter(|| {
            dpf.gen(&f, [&s0s[0], &s0s[1]]);
        })
    });
}

pub fn bench_eval(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::new(&keys.iter().collect::<Vec<_>>());
    let dpf = DpfImpl::<16, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = PointFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup(thread_rng().gen()),
    };

    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
    let prg = Aes256HirosePrg::new(&keys.iter().collect::<Vec<_>>());
    let dpf = DpfImpl::<16, 16, _>::new(prg);
    let x: [u8; 16] = thread_rng().gen();
    let mut y = ByteGroup::zero();

    c.bench_function("dpf eval", |b| {
        b.iter(|| {
            dpf.eval(false, &k, &[&x], &mut [&mut y]);
        })
    });
}

criterion_group!(benches, bench_gen, bench_eval);
criterion_main!(benches);
