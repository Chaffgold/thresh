//! Information filter (dual of Kalman filter).

use nalgebra::{DMatrix, DVector};

/// Information filter state: (Y = P⁻¹, y = P⁻¹ x).
#[derive(Debug, Clone)]
pub struct InformationState {
    /// Information matrix Y = P⁻¹.
    pub info_matrix: DMatrix<f64>,
    /// Information state y = P⁻¹ x.
    pub info_state: DVector<f64>,
}

impl InformationState {
    /// Create from standard KF state (x, P).
    pub fn from_covariance(x: &DVector<f64>, p: &DMatrix<f64>) -> Self {
        let y_mat = p.clone().try_inverse().expect("P is singular");
        let y_state = &y_mat * x;
        Self {
            info_matrix: y_mat,
            info_state: y_state,
        }
    }

    /// Convert back to covariance form (x, P).
    pub fn to_covariance(&self) -> (DVector<f64>, DMatrix<f64>) {
        let p = self
            .info_matrix
            .clone()
            .try_inverse()
            .expect("Information matrix is singular");
        let x = &p * &self.info_state;
        (x, p)
    }

    /// Additive measurement update: Y += Hᵀ R⁻¹ H, y += Hᵀ R⁻¹ z.
    pub fn update(&mut self, z: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
        let r_inv = r.clone().try_inverse().expect("R is singular");
        let ht_rinv = h.transpose() * &r_inv;
        self.info_matrix += &ht_rinv * h;
        self.info_state += &ht_rinv * z;
    }

    /// Fuse multiple sensor contributions additively.
    pub fn fuse_sensors(&mut self, contributions: &[(DMatrix<f64>, DVector<f64>)]) {
        for (info_contrib, state_contrib) in contributions {
            self.info_matrix += info_contrib;
            self.info_state += state_contrib;
        }
    }

    /// Compute a sensor's information contribution: (Hᵀ R⁻¹ H, Hᵀ R⁻¹ z).
    pub fn sensor_contribution(
        z: &DVector<f64>,
        h: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let r_inv = r.clone().try_inverse().expect("R is singular");
        let ht_rinv = h.transpose() * &r_inv;
        let info_mat = &ht_rinv * h;
        let info_state = &ht_rinv * z;
        (info_mat, info_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_filter_matches_kf() {
        let x = DVector::from_column_slice(&[10.0, 1.0, 20.0, 2.0]);
        let p = DMatrix::identity(4, 4) * 100.0;

        let h = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let r = DMatrix::identity(2, 2) * 5.0;
        let z = DVector::from_column_slice(&[12.0, 22.0]);

        // KF update
        let s = &h * &p * h.transpose() + &r;
        let k = &p * h.transpose() * s.try_inverse().unwrap();
        let x_kf = &x + &k * (&z - &h * &x);
        let n = 4;
        let i_kh = DMatrix::identity(n, n) - &k * &h;
        let p_kf = &i_kh * &p * i_kh.transpose() + &k * &r * k.transpose();

        // Information filter update
        let mut info = InformationState::from_covariance(&x, &p);
        info.update(&z, &h, &r);
        let (x_if, p_if) = info.to_covariance();

        let x_diff = (&x_kf - &x_if).norm();
        let p_diff = (&p_kf - &p_if).norm();

        assert!(x_diff < 1e-8, "State diff: {x_diff}");
        assert!(p_diff < 1e-6, "Covariance diff: {p_diff}");
    }

    #[test]
    fn roundtrip_covariance() {
        let x = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let p = DMatrix::from_diagonal(&DVector::from_column_slice(&[10.0, 20.0, 30.0]));

        let info = InformationState::from_covariance(&x, &p);
        let (x_back, p_back) = info.to_covariance();

        assert!((&x - &x_back).norm() < 1e-10);
        assert!((&p - &p_back).norm() < 1e-10);
    }
}
