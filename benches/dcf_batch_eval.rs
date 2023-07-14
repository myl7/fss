use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{thread_rng, Rng};

use dcf::cache::CachedPrg;
use dcf::prg::Aes256HirosePrg;
use dcf::{BoundState, CmpFn, Dcf, DcfImpl};

pub fn bench_rand(c: &mut Criterion) {
    let keys: [[u8; 32]; 2] = thread_rng().gen();
    let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<16, 16, _>::new(prg);
    let s0s: [[u8; 16]; 2] = thread_rng().gen();
    let f = CmpFn {
        alpha: thread_rng().gen(),
        beta: thread_rng().gen(),
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]], BoundState::LtBeta);
    const N: usize = 100_000;
    let xs: [[u8; 16]; N] = std::array::from_fn(|_| thread_rng().gen());

    c.bench_function("eval 100k rand", |b| {
        b.iter(|| {
            let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
            let cached_prg = CachedPrg::new(prg, HashMap::new());
            let dcf = DcfImpl::<16, 16, _>::new(cached_prg);
            xs.iter()
                .map(|x| dcf.eval(false, &k, x))
                .collect::<Vec<_>>()
        })
    });
}

criterion_group!(benches, bench_rand);
criterion_main!(benches);
