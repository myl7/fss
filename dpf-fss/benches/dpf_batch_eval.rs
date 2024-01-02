extern crate dpf_fss as dpf;
extern crate group_math as group;

use criterion::{criterion_group, criterion_main, Criterion};
use group::byte::ByteGroup;
use group::Group;
use rand::{thread_rng, Rng};

use dpf::prg::Aes256HirosePrg;
use dpf::{Dpf, DpfImpl, PointFn};

pub fn bench(c: &mut Criterion) {
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
