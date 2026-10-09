use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libdpf::Dpf;

const fn domain_bits() -> usize {
    let value = match option_env!("FSS_BENCH_DOMAIN_BITS") {
        Some(value) => value,
        None => "20",
    };
    let bytes = value.as_bytes();
    let mut bits = 0;
    let mut i = 0;
    while i < bytes.len() {
        assert!(bytes[i] >= b'0' && bytes[i] <= b'9', "invalid domain bits");
        bits = bits * 10 + (bytes[i] - b'0') as usize;
        i += 1;
    }
    assert!(
        bits >= 8 && bits <= 20,
        "domain bits must be between 8 and 20"
    );
    bits
}

const N: u8 = domain_bits() as u8;
const ALPHA: u64 = 12345 & ((1u64 << N) - 1);

fn check_correctness(dpf: &Dpf) {
    let (k0, k1) = dpf.gen(ALPHA, N);
    // eval returns the 128-bit packed leaf containing the queried point.
    for x in [ALPHA, 0, (1u64 << N) - 1, ALPHA ^ 128] {
        let result = dpf.eval(&k0, x).xor(&dpf.eval(&k1, x));
        let position = ALPHA % 128;
        let expected = if x / 128 == ALPHA / 128 {
            libdpf::Block::new(
                if position >= 64 {
                    1u64 << (position - 64)
                } else {
                    0
                },
                if position < 64 { 1u64 << position } else { 0 },
            )
        } else {
            libdpf::Block::zero()
        };
        assert!(result == expected, "share reconstruction failed");
    }
}

fn bench_dpf_gen(c: &mut Criterion) {
    let dpf = Dpf::with_default_key();
    check_correctness(&dpf);
    c.bench_function("libdpf/CPU/DPF/Gen", |b| {
        b.iter(|| dpf.gen(black_box(ALPHA), black_box(N)))
    });
}

fn bench_dpf_eval(c: &mut Criterion) {
    let dpf = Dpf::with_default_key();
    check_correctness(&dpf);
    let (k0, _) = dpf.gen(ALPHA, N);
    c.bench_function("libdpf/CPU/DPF/Eval", |b| {
        b.iter(|| dpf.eval(black_box(&k0), black_box(ALPHA)))
    });
}

fn bench_dpf_eval_all(c: &mut Criterion) {
    let dpf = Dpf::with_default_key();
    check_correctness(&dpf);
    let (k0, _) = dpf.gen(ALPHA, N);
    c.bench_function("libdpf/CPU/DPF/EvalAll", |b| {
        b.iter(|| dpf.eval_full(black_box(&k0)))
    });
}

criterion_group!(benches, bench_dpf_gen, bench_dpf_eval, bench_dpf_eval_all);
criterion_main!(benches);
