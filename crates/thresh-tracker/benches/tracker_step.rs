use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use thresh_tracker::tracker::MultiObjectTracker;

/// Generate N detections spread across space, with small Gaussian noise.
fn generate_detections(n: usize, seed: u64) -> Vec<DVector<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 5.0).unwrap();
    (0..n)
        .map(|i| {
            let base_x = (i as f64) * 200.0;
            let base_y = (i as f64) * 100.0;
            let base_z = 50.0;
            DVector::from_column_slice(&[
                base_x + noise.sample(&mut rng),
                base_y + noise.sample(&mut rng),
                base_z + noise.sample(&mut rng),
            ])
        })
        .collect()
}

/// Create a tracker and warm it up for 5 frames so tracks are confirmed.
fn warm_tracker(n_targets: usize) -> (MultiObjectTracker, Vec<DVector<f64>>) {
    let mut tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);
    let detections = generate_detections(n_targets, 42);

    // Run 5 warm-up steps to confirm tracks
    for _ in 0..5 {
        tracker.step(&detections, 1.0);
    }

    (tracker, detections)
}

fn tracker_step_10(c: &mut Criterion) {
    c.bench_function("tracker_step_10", |b| {
        b.iter_batched(
            || warm_tracker(10),
            |(mut tracker, dets)| {
                tracker.step(black_box(&dets), black_box(1.0));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn tracker_step_50(c: &mut Criterion) {
    c.bench_function("tracker_step_50", |b| {
        b.iter_batched(
            || warm_tracker(50),
            |(mut tracker, dets)| {
                tracker.step(black_box(&dets), black_box(1.0));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn tracker_step_200(c: &mut Criterion) {
    c.bench_function("tracker_step_200", |b| {
        b.iter_batched(
            || warm_tracker(200),
            |(mut tracker, dets)| {
                tracker.step(black_box(&dets), black_box(1.0));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, tracker_step_10, tracker_step_50, tracker_step_200);
criterion_main!(benches);
