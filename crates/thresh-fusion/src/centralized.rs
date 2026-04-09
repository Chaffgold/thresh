//! Centralized measurement-level fusion.

use nalgebra::{DMatrix, DVector};

/// Stack measurements from multiple sensors into a single update.
///
/// Returns (z_stacked, H_stacked, R_stacked) where R_stacked is block-diagonal.
pub fn stack_measurements(
    measurements: &[(DVector<f64>, DMatrix<f64>, DMatrix<f64>)],
) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>) {
    let total_meas_dim: usize = measurements.iter().map(|(z, _, _)| z.len()).sum();
    let state_dim = measurements[0].1.ncols();

    let mut z_stacked = DVector::zeros(total_meas_dim);
    let mut h_stacked = DMatrix::zeros(total_meas_dim, state_dim);
    let mut r_stacked = DMatrix::zeros(total_meas_dim, total_meas_dim);

    let mut offset = 0;
    for (z, h, r) in measurements {
        let m = z.len();
        z_stacked.rows_mut(offset, m).copy_from(z);

        for row in 0..m {
            for col in 0..state_dim {
                h_stacked[(offset + row, col)] = h[(row, col)];
            }
        }

        for row in 0..m {
            for col in 0..m {
                r_stacked[(offset + row, offset + col)] = r[(row, col)];
            }
        }

        offset += m;
    }

    (z_stacked, h_stacked, r_stacked)
}

/// Perform a single centralized KF update using stacked sensor measurements.
///
/// Returns updated (x, P).
pub fn centralized_update(
    x: &DVector<f64>,
    p: &DMatrix<f64>,
    z: &DVector<f64>,
    h: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let y = z - h * x;
    let s = h * p * h.transpose() + r;
    let s_inv = s.try_inverse().expect("S singular in centralized update");
    let k = p * h.transpose() * &s_inv;

    let x_new = x + &k * &y;
    let n = x.len();
    let i_kh = DMatrix::identity(n, n) - &k * h;
    let p_new = &i_kh * p * i_kh.transpose() + &k * r * k.transpose();

    (x_new, p_new)
}

/// Apply asynchronous sequential sensor updates.
///
/// Each sensor's measurement is applied independently, updating state in sequence.
pub fn sequential_update(
    x: &DVector<f64>,
    p: &DMatrix<f64>,
    measurements: &[(DVector<f64>, DMatrix<f64>, DMatrix<f64>)],
) -> (DVector<f64>, DMatrix<f64>) {
    let mut x_cur = x.clone();
    let mut p_cur = p.clone();

    for (z, h, r) in measurements {
        let (x_new, p_new) = centralized_update(&x_cur, &p_cur, z, h, r);
        x_cur = x_new;
        p_cur = p_new;
    }

    (x_cur, p_cur)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centralized_matches_sequential() {
        let n = 4; // state dim
        let x = DVector::from_column_slice(&[10.0, 1.0, 20.0, 2.0]);
        let p = DMatrix::identity(n, n) * 100.0;

        // Two position sensors
        let h1 = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let r1 = DMatrix::identity(2, 2) * 5.0;
        let z1 = DVector::from_column_slice(&[12.0, 22.0]);

        let h2 = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let r2 = DMatrix::identity(2, 2) * 10.0;
        let z2 = DVector::from_column_slice(&[11.0, 21.0]);

        // Centralized (stacked)
        let (z_s, h_s, r_s) = stack_measurements(&[
            (z1.clone(), h1.clone(), r1.clone()),
            (z2.clone(), h2.clone(), r2.clone()),
        ]);
        let (x_cent, _p_cent) = centralized_update(&x, &p, &z_s, &h_s, &r_s);

        // Sequential
        let (x_seq, _p_seq) = sequential_update(&x, &p, &[(z1, h1, r1), (z2, h2, r2)]);

        // Results should be close (not identical due to sequential vs batch, but same
        // for linear systems with independent measurements)
        let x_diff = (&x_cent - &x_seq).norm();
        assert!(
            x_diff < 0.5,
            "Centralized vs sequential state diff: {x_diff}"
        );
    }
}
