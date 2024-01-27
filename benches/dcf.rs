use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use fss_rs::dcf::prg::Aes256HirosePrg;
use fss_rs::dcf::{BoundState, CmpFn, Dcf, DcfImpl};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;

pub fn bench_gen(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<16, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = CmpFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup(thread_rng().gen()),
        bound: BoundState::LtBeta,
    };

    c.bench_function("dcf gen", |b| {
        b.iter(|| {
            dcf.gen(&f, [&s0s[0], &s0s[1]]);
        })
    });
}

pub fn bench_eval(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<16, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = CmpFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup(thread_rng().gen()),
        bound: BoundState::LtBeta,
    };

    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<16, 16, _>::new(prg);
    let x: [u8; 16] = thread_rng().gen();
    let mut y = ByteGroup::zero();

    c.bench_function("dcf eval", |b| {
        b.iter(|| {
            dcf.eval(false, &k, &[&x], &mut [&mut y]);
        })
    });
}

criterion_group!(benches, bench_gen, bench_eval);
criterion_main!(benches);
