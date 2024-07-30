// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use fss_rs::dcf::{BoundState, CmpFn, Dcf, DcfImpl};
use fss_rs::group::byte::ByteGroup;
use fss_rs::group::Group;
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

fn from_domain_range_size<const IN_BLEN: usize, const OUT_BLEN: usize, const CIPHER_N: usize>(
    c: &mut Criterion,
    filter_bitn: usize,
) {
    let mut keys = [[0; 16]; CIPHER_N];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<OUT_BLEN, 2, CIPHER_N>::new(&keys_iter);
    let dcf = DcfImpl::<IN_BLEN, OUT_BLEN, _>::new_with_filter(prg, filter_bitn);

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

    let k = dcf.gen(&f, [&s0s[0], &s0s[1]]);

    let mut ys = vec![ByteGroup::zero(); 1 << filter_bitn];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_with_input(
        BenchmarkId::new(
            "dcf full_eval",
            format!("{}b -> {}B", filter_bitn, OUT_BLEN),
        ),
        &(IN_BLEN, OUT_BLEN, filter_bitn),
        |b, &_| {
            b.iter(|| {
                dcf.full_eval(false, &k, &mut ys_iter);
            });
        },
    );
}

fn bench(c: &mut Criterion) {
    from_domain_range_size::<2, 16, 4>(c, 16);
    from_domain_range_size::<3, 16, 4>(c, 18);
    from_domain_range_size::<3, 16, 4>(c, 20);
}

criterion_group!(benches, bench);
criterion_main!(benches);
