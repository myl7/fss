use criterion::{criterion_group, criterion_main, Criterion};
use dcf::group::{ByteGroup, Group};
use rand::{thread_rng, Rng, RngCore};

use dcf::prg::Aes256HirosePrg;
use dcf::{BoundState, CmpFn, Dcf, DcfImpl};

pub fn bench(c: &mut Criterion) {
    let mut keys = vec![[0; 32]; 2048];
    keys.iter_mut().for_each(|key| thread_rng().fill_bytes(key));
    let prg = Aes256HirosePrg::<16384, 2048>::new(std::array::from_fn(|i| &keys[i]));
    let dcf = DcfImpl::<16, 16384, _>::new(prg);
    let mut s0s = vec![[0; 16384]; 2];
    thread_rng().fill_bytes(&mut s0s[0]);
    thread_rng().fill_bytes(&mut s0s[1]);
    let mut f = Box::new(CmpFn {
        alpha: thread_rng().gen(),
        beta: ByteGroup::zero(),
    });
    thread_rng().fill_bytes(&mut f.beta.0);
    let k = dcf.gen(&f, [&s0s[0], &s0s[1]], BoundState::LtBeta);
    const N: usize = 10_000;
    let xs: Vec<[u8; 16]> = (0..N).map(|_| thread_rng().gen()).collect();

    c.bench_function("xs_10k_lambda_16384", |b| {
        b.iter(|| {
            let prg = Aes256HirosePrg::<16384, 2048>::new(std::array::from_fn(|i| &keys[i]));
            let dcf = DcfImpl::<16, 16384, _>::new(prg);
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
