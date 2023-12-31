extern crate dpf_fss as dpf;
extern crate group_math as group;

use criterion::{criterion_group, criterion_main, Criterion};
use group::byte::ByteGroup;
use group::Group;
use rand::{thread_rng, Rng};

use dpf::prg::Aes256HirosePrg;
use dpf::{Dpf, DpfImpl, PointFn};

pub fn bench_gen(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
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
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dpf = DpfImpl::<16, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = PointFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup(thread_rng().gen()),
    };

    let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
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
