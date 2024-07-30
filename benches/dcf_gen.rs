// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use fss_rs::dcf::{BoundState, CmpFn, Dcf, DcfImpl};
use fss_rs::group::byte::ByteGroup;
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

fn from_domain_range_size<const IN_BLEN: usize, const OUT_BLEN: usize, const CIPHER_N: usize>(
    c: &mut Criterion,
) {
    let mut keys = [[0; 16]; CIPHER_N];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, CIPHER_N>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new(prg);

    let mut s0s = [[0; OUT_BLEN]; 2];
    s0s.iter_mut().for_each(|s0| thread_rng().fill_bytes(s0));

    let mut alpha = [0; IN_BLEN];
    thread_rng().fill_bytes(&mut alpha);
    let mut beta_buf = [0; OUT_BLEN];
    thread_rng().fill_bytes(&mut beta_buf);
    let beta = ByteGroup(beta_buf);
    let f = CmpFn {
        alpha,
        beta,
        bound: BoundState::LtAlpha,
    };

    c.bench_with_input(
        BenchmarkId::new("dcf gen", format!("{}b -> {}B", IN_BLEN * 8, OUT_BLEN)),
        &(IN_BLEN, OUT_BLEN),
        |b, &_| {
            b.iter(|| {
                dcf.gen(&f, [&s0s[0], &s0s[1]]);
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
