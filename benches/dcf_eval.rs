// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use fss_rs::dcf::prg::Aes128MatyasMeyerOseasPrg;
use fss_rs::dcf::{BoundState, CmpFn, Dcf, DcfImpl};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;

fn from_domain_range_size<const DOM_SZ: usize, const LAMBDA: usize, const CIPHER_N: usize>(
    c: &mut Criterion,
) {
    let mut keys = [[0; 16]; CIPHER_N];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<LAMBDA, CIPHER_N>::new(keys_iter);
    let dcf = DcfImpl::<DOM_SZ, LAMBDA, _>::new(prg);

    let mut s0s = [[0; LAMBDA]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0; DOM_SZ];
    thread_rng().fill_bytes(&mut alpha);
    let mut beta_buf = [0; LAMBDA];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtBeta,
    };

    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut x = [0; DOM_SZ];
    thread_rng().fill_bytes(&mut x);
    let mut y = ByteGroup::zero();

    c.bench_with_input(
        BenchmarkId::new("dcf eval", format!("{}b -> {}B", DOM_SZ * 8, LAMBDA)),
        &(DOM_SZ, LAMBDA),
        |b, &_| {
            b.iter(|| {
                dcf.eval(false, &k, &[&x], &mut [&mut y]);
            });
        },
    );
}

fn bench(c: &mut Criterion) {
    from_domain_range_size::<16, 16, 4>(c);
    from_domain_range_size::<24, 16, 4>(c);
    from_domain_range_size::<32, 16, 4>(c);
    from_domain_range_size::<16, 256, 64>(c);
    from_domain_range_size::<16, 16384, 4096>(c);
}

criterion_group!(benches, bench);
criterion_main!(benches);
