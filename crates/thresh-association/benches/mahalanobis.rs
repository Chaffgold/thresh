use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use thresh_association::gating::mahalanobis_squared;

fn mahalanobis_6d(c: &mut Criterion) {
    // 6D innovation vector
    let z = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let predicted_z = DVector::from_column_slice(&[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);

    // Positive-definite covariance: diagonal + small off-diagonal terms
    let mut s = DMatrix::identity(6, 6) * 10.0;
    for i in 0..5 {
        s[(i, i + 1)] = 1.0;
        s[(i + 1, i)] = 1.0;
    }

    c.bench_function("mahalanobis_6d", |b| {
        b.iter(|| mahalanobis_squared(black_box(&z), black_box(&predicted_z), black_box(&s)))
    });
}

criterion_group!(benches, mahalanobis_6d);
criterion_main!(benches);
