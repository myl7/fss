use criterion::{criterion_group, criterion_main, Criterion};
use rand::{thread_rng, Rng};

use dcf::prg::Aes256HirosePrg;
use dcf::{BoundState, CmpFn, Dcf, DcfImpl};

pub fn bench_gen(c: &mut Criterion) {
    c.bench_function("gen", |b| {
        b.iter(|| {
            let keys: [[u8; 32]; 2] = thread_rng().gen();
            let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
            let dcf = DcfImpl::<16, 16, _>::new(prg);
            let s0s: [[u8; 16]; 2] = thread_rng().gen();
            let f = CmpFn {
                alpha: thread_rng().gen(),
                beta: thread_rng().gen(),
            };
            dcf.gen(&f, [&s0s[0], &s0s[1]], BoundState::LtBeta);
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
        beta: thread_rng().gen(),
    };
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]], BoundState::LtBeta);

    c.bench_function("eval", |b| {
        b.iter(|| {
            let prg = Aes256HirosePrg::<16, 2>::new(std::array::from_fn(|i| &keys[i]));
            let dcf = DcfImpl::<16, 16, _>::new(prg);
            let x: [u8; 16] = thread_rng().gen();
            let mut y = [0; 16];
            dcf.eval(false, &k, &[&x], &mut [&mut y]);
        })
    });
}

criterion_group!(benches, bench_gen, bench_eval);
criterion_main!(benches);
