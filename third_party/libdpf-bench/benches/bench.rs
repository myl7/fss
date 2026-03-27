use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libdpf::Dpf;

const N: u8 = 20;
const ALPHA: u64 = 12345;

fn bench_dpf_gen(c: &mut Criterion) {
    let dpf = Dpf::with_default_key();
    c.bench_function("libdpf/CPU/DPF/Gen", |b| {
        b.iter(|| dpf.gen(black_box(ALPHA), black_box(N)))
    });
}

fn bench_dpf_eval(c: &mut Criterion) {
    let dpf = Dpf::with_default_key();
    let (k0, _) = dpf.gen(ALPHA, N);
    c.bench_function("libdpf/CPU/DPF/Eval", |b| {
        b.iter(|| dpf.eval(black_box(&k0), black_box(ALPHA)))
    });
}

fn bench_dpf_eval_all(c: &mut Criterion) {
    let dpf = Dpf::with_default_key();
    let (k0, _) = dpf.gen(ALPHA, N);
    c.bench_function("libdpf/CPU/DPF/EvalAll", |b| {
        b.iter(|| dpf.eval_full(black_box(&k0)))
    });
}

criterion_group!(benches, bench_dpf_gen, bench_dpf_eval, bench_dpf_eval_all);
criterion_main!(benches);
