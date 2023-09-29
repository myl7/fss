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
    const N: usize = 100_000;
    let xs: [[u8; 16]; N] = std::array::from_fn(|_| thread_rng().gen());

    c.bench_function("xs_100k_lambda_16", |b| {
        b.iter(|| {
            let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
            let dcf = DcfImpl::<16, 16, _>::new(prg);
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
