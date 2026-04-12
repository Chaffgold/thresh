use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use thresh_association::hungarian::hungarian_assignment;

fn random_cost_matrix(dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim)
        .map(|_| (0..dim).map(|_| rng.random::<f64>() * 100.0).collect())
        .collect()
}

fn hungarian_10x10(c: &mut Criterion) {
    let cost = random_cost_matrix(10, 42);
    c.bench_function("hungarian_10x10", |b| {
        b.iter(|| hungarian_assignment(black_box(&cost), black_box(1e6)))
    });
}

fn hungarian_100x100(c: &mut Criterion) {
    let cost = random_cost_matrix(100, 42);
    c.bench_function("hungarian_100x100", |b| {
        b.iter(|| hungarian_assignment(black_box(&cost), black_box(1e6)))
    });
}

fn hungarian_500x500(c: &mut Criterion) {
    let cost = random_cost_matrix(500, 42);
    c.bench_function("hungarian_500x500", |b| {
        b.iter(|| hungarian_assignment(black_box(&cost), black_box(1e6)))
    });
}

criterion_group!(
    benches,
    hungarian_10x10,
    hungarian_100x100,
    hungarian_500x500
);
criterion_main!(benches);
