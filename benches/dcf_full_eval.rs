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
    filter_bitn: usize,
) {
    let mut keys = [[0; 16]; CIPHER_N];
    keys.iter_mut().for_each(|k| thread_rng().fill_bytes(k));
    let keys_iter = std::array::from_fn(|i| &keys[i]);

    let prg = Aes128MatyasMeyerOseasPrg::<LAMBDA, CIPHER_N>::new(keys_iter);
    let dcf = DcfImpl::<DOM_SZ, LAMBDA, _>::new_with_filter(prg, filter_bitn);

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

    let mut ys = vec![ByteGroup::zero(); 1 << filter_bitn];
    let mut ys_iter: Vec<_> = ys.iter_mut().collect();

    c.bench_with_input(
        BenchmarkId::new("dcf full_eval", format!("{}b -> {}B", filter_bitn, LAMBDA)),
        &(DOM_SZ, LAMBDA, filter_bitn),
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
