use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use thresh_filter::kf::KalmanFilter;
use thresh_filter::models::cv::ConstantVelocity;
use thresh_filter::traits::{LinearModel, MotionModel};

fn make_kf_6d() -> KalmanFilter {
    let x = DVector::from_column_slice(&[100.0, 10.0, 200.0, 5.0, 50.0, 0.0]);
    let p = DMatrix::identity(6, 6) * 100.0;
    KalmanFilter::new(x, p)
}

fn observation_matrix() -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        6,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0,
        ],
    )
}

fn kf_predict_6d(c: &mut Criterion) {
    let model = ConstantVelocity::new(5.0);
    let f = model.transition_matrix(1.0);
    let q = model.process_noise(1.0);

    c.bench_function("kf_predict_6d", |b| {
        b.iter_batched(
            make_kf_6d,
            |mut kf| {
                kf.x = &f * &kf.x;
                kf.p = &f * &kf.p * f.transpose() + &q;
                black_box(&kf);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn kf_update_6d(c: &mut Criterion) {
    let h = observation_matrix();
    let r = DMatrix::identity(3, 3) * 10.0;
    let z = DVector::from_column_slice(&[110.0, 205.0, 50.0]);

    c.bench_function("kf_update_6d", |b| {
        b.iter_batched(
            make_kf_6d,
            |mut kf| {
                kf.update(black_box(&z), black_box(&h), black_box(&r));
                black_box(&kf);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, kf_predict_6d, kf_update_6d);
criterion_main!(benches);
