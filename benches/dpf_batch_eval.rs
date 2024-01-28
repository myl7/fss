use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;
use fss_rs::prg::Aes256HirosePrg;

pub fn bench(c: &mut Criterion) {
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
    const N: usize = 100_000;
    let mut xs = vec![[0; 16]; N];
    xs.iter_mut().for_each(|x| *x = thread_rng().gen());
    let mut ys = vec![ByteGroup::zero(); N];

    c.bench_function("dpf eval 100k xs", |b| {
        b.iter(|| {
            dpf.eval(
                false,
                &k,
                &xs.iter().collect::<Vec<_>>(),
                &mut ys.iter_mut().collect::<Vec<_>>(),
            );
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
