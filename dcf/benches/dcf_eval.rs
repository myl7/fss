extern crate group_math as group;

use criterion::{criterion_group, criterion_main, Criterion};
use group::byte::ByteGroup;
use group::Group;
use rand::{thread_rng, Rng};

use dcf::prg::Aes256HirosePrg;
use dcf::{BoundState, CmpFn, Dcf, DcfImpl};

pub fn bench(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<16, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = CmpFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup(thread_rng().gen()),
    };

    let k = dcf.gen(&f, [&s0s[0], &s0s[1]], BoundState::LtBeta);
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<16, 16, _>::new(prg);
    const N: usize = 100000;
    let mut xs = vec![[0; 16]; N];
    xs.iter_mut().for_each(|x| *x = thread_rng().gen());
    let mut ys = vec![ByteGroup::zero(); N];

    c.bench_function("eval_n", |b| {
        b.iter(|| {
            dcf.eval(
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
