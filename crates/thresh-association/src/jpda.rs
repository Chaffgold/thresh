//! Joint Probabilistic Data Association (JPDA).
//!
//! Computes soft association probabilities between tracks and detections,
//! allowing ambiguous detections to contribute to multiple tracks weighted
//! by their likelihood. This is superior to hard (Hungarian) assignment in
//! dense, cluttered environments with closely-spaced targets.
//!
//! # Algorithm overview
//!
//! 1. For each track, compute the Gaussian likelihood of each gated detection.
//! 2. Form unnormalized association weights: `w_ij = p_d * L_ij`.
//! 3. Missed-detection weight: `w_i0 = (1 - p_d) * clutter_density`.
//! 4. Normalize per track: `beta_ij = w_ij / (w_i0 + sum_j w_ij)`.
//!
//! # References
//!
//! Bar-Shalom, Y. & Li, X.-R. (2009). *Multitarget-Multisensor Tracking*.

use nalgebra::{DMatrix, DVector};

use crate::gating::mahalanobis_squared;

/// Per-track predicted measurement and innovation covariance.
#[derive(Debug, Clone)]
pub struct JpdaTrack {
    /// Predicted measurement z_hat = H * x_predicted.
    pub predicted_measurement: DVector<f64>,
    /// Innovation covariance S = H * P * H^T + R.
    pub innovation_covariance: DMatrix<f64>,
}

/// Result of JPDA association probability computation.
#[derive(Debug, Clone)]
pub struct JpdaResult {
    /// `beta[i][j]` = probability that track `i` is associated with detection `j`.
    /// `beta[i][n_dets]` = probability that track `i` has no detection (missed).
    pub beta: Vec<Vec<f64>>,
    /// Number of tracks.
    pub n_tracks: usize,
    /// Number of detections.
    pub n_dets: usize,
}

/// Compute JPDA association probabilities.
///
/// Uses the approximate (per-track normalization) JPDA formula for
/// tractability. Given a set of tracks and detections, computes the
/// marginal association probability for each track-detection pair.
///
/// # Arguments
///
/// * `tracks` — predicted measurement and innovation covariance per track.
/// * `detections` — measurement vectors.
/// * `gate` — Mahalanobis distance squared threshold for gating.
/// * `p_detection` — probability that a true target generates a detection.
/// * `clutter_density` — spatial density of false alarms (per unit volume).
///
/// # Returns
///
/// A [`JpdaResult`] where `beta[i]` has length `n_dets + 1`. The last
/// element is the missed-detection probability for track `i`.
pub fn jpda_probabilities(
    tracks: &[JpdaTrack],
    detections: &[DVector<f64>],
    gate: f64,
    p_detection: f64,
    clutter_density: f64,
) -> JpdaResult {
    let n_tracks = tracks.len();
    let n_dets = detections.len();
    let mut beta = Vec::with_capacity(n_tracks);

    for track in tracks {
        let mut weights = vec![0.0_f64; n_dets + 1];

        // Missed-detection weight (index n_dets)
        weights[n_dets] = (1.0 - p_detection) * clutter_density.max(f64::MIN_POSITIVE);

        for (j, det) in detections.iter().enumerate() {
            let d2 = mahalanobis_squared(
                det,
                &track.predicted_measurement,
                &track.innovation_covariance,
            );
            if d2 > gate {
                // Outside gate — zero weight
                continue;
            }
            let likelihood = gaussian_likelihood(
                det,
                &track.predicted_measurement,
                &track.innovation_covariance,
                d2,
            );
            weights[j] = p_detection * likelihood;
        }

        // Normalize
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        } else {
            // All zero — assign full probability to missed detection
            weights[n_dets] = 1.0;
        }

        beta.push(weights);
    }

    JpdaResult {
        beta,
        n_tracks,
        n_dets,
    }
}

/// Compute the JPDA combined (merged) innovation for a single track.
///
/// `z_combined = sum_j(beta_j * (z_j - z_hat))`
///
/// # Arguments
///
/// * `track` — the track's predicted measurement.
/// * `detections` — all detections (only those with nonzero beta contribute).
/// * `betas` — association probabilities for this track, length `n_dets + 1`.
///   The last element is the missed-detection probability (unused here).
pub fn jpda_combined_innovation(
    track: &JpdaTrack,
    detections: &[DVector<f64>],
    betas: &[f64],
) -> DVector<f64> {
    let m = track.predicted_measurement.nrows();
    let mut combined = DVector::zeros(m);
    for (j, det) in detections.iter().enumerate() {
        if betas[j] > 0.0 {
            let innov = det - &track.predicted_measurement;
            combined += betas[j] * innov;
        }
    }
    combined
}

/// Compute the JPDA spread-of-innovations covariance correction.
///
/// `P_spread = sum_j(beta_j * v_j * v_j^T) - v_combined * v_combined^T`
///
/// This term inflates the covariance to account for association uncertainty.
pub fn jpda_covariance_correction(
    track: &JpdaTrack,
    detections: &[DVector<f64>],
    betas: &[f64],
    combined_innovation: &DVector<f64>,
) -> DMatrix<f64> {
    let m = track.predicted_measurement.nrows();
    let mut spread = DMatrix::zeros(m, m);

    for (j, det) in detections.iter().enumerate() {
        if betas[j] > 0.0 {
            let innov = det - &track.predicted_measurement;
            spread += betas[j] * (&innov * innov.transpose());
        }
    }

    spread -= combined_innovation * combined_innovation.transpose();
    spread
}

/// Evaluate the multivariate Gaussian PDF: N(z; mu, S).
///
/// Uses log-space internally to avoid underflow/overflow.
fn gaussian_likelihood(
    z: &DVector<f64>,
    _mu: &DVector<f64>,
    s: &DMatrix<f64>,
    mahalanobis_sq: f64,
) -> f64 {
    let m = z.nrows() as f64;
    let det = s.determinant();
    if det <= 0.0 {
        return 0.0;
    }
    let log_norm = -0.5 * (m * (2.0 * std::f64::consts::PI).ln() + det.ln());
    let log_exp = -0.5 * mahalanobis_sq;
    (log_norm + log_exp).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_track(pred: &[f64], s_diag: f64) -> JpdaTrack {
        let m = pred.len();
        JpdaTrack {
            predicted_measurement: DVector::from_column_slice(pred),
            innovation_covariance: DMatrix::identity(m, m) * s_diag,
        }
    }

    #[test]
    fn test_jpda_single_track_single_det() {
        let track = make_track(&[0.0, 0.0], 1.0);
        let det = DVector::from_column_slice(&[0.1, 0.1]);
        let p_d = 0.9;
        let clutter = 1e-6;
        let gate = 100.0;

        let result = jpda_probabilities(&[track], &[det], gate, p_d, clutter);

        // With one close detection and low clutter, beta for detection should be near 1.0
        assert_eq!(result.beta.len(), 1);
        assert_eq!(result.beta[0].len(), 2); // 1 det + 1 miss
        let beta_det = result.beta[0][0];
        let beta_miss = result.beta[0][1];
        assert!(
            beta_det > 0.99,
            "detection beta should be near 1.0, got {beta_det}"
        );
        assert!(
            beta_miss < 0.01,
            "miss beta should be near 0.0, got {beta_miss}"
        );
    }

    #[test]
    fn test_jpda_single_track_no_det_in_gate() {
        let track = make_track(&[0.0, 0.0], 1.0);
        // Detection very far away — outside gate
        let det = DVector::from_column_slice(&[100.0, 100.0]);
        let p_d = 0.9;
        let clutter = 1e-3;
        let gate = 10.0; // tight gate

        let result = jpda_probabilities(&[track], &[det], gate, p_d, clutter);

        let beta_miss = result.beta[0][1];
        assert!(
            (beta_miss - 1.0).abs() < 1e-10,
            "miss beta should be 1.0 when no dets in gate, got {beta_miss}"
        );
    }

    #[test]
    fn test_jpda_two_tracks_two_dets() {
        // Two tracks near two detections — should produce a reasonable split
        let tracks = vec![make_track(&[0.0, 0.0], 1.0), make_track(&[5.0, 0.0], 1.0)];
        let dets = vec![
            DVector::from_column_slice(&[0.1, 0.0]),
            DVector::from_column_slice(&[4.9, 0.0]),
        ];
        let p_d = 0.9;
        let clutter = 1e-6;
        let gate = 100.0;

        let result = jpda_probabilities(&tracks, &dets, gate, p_d, clutter);

        // Track 0 should prefer detection 0
        assert!(
            result.beta[0][0] > result.beta[0][1],
            "track 0 should prefer det 0: beta[0][0]={}, beta[0][1]={}",
            result.beta[0][0],
            result.beta[0][1]
        );
        // Track 1 should prefer detection 1
        assert!(
            result.beta[1][1] > result.beta[1][0],
            "track 1 should prefer det 1: beta[1][0]={}, beta[1][1]={}",
            result.beta[1][0],
            result.beta[1][1]
        );
    }

    #[test]
    fn test_jpda_probabilities_sum_to_one() {
        let tracks = vec![
            make_track(&[0.0, 0.0], 1.0),
            make_track(&[2.0, 0.0], 1.0),
            make_track(&[4.0, 0.0], 1.0),
        ];
        let dets = vec![
            DVector::from_column_slice(&[0.5, 0.0]),
            DVector::from_column_slice(&[1.8, 0.0]),
            DVector::from_column_slice(&[3.5, 0.0]),
            DVector::from_column_slice(&[10.0, 10.0]), // far away clutter
        ];
        let p_d = 0.9;
        let clutter = 1e-4;
        let gate = 50.0;

        let result = jpda_probabilities(&tracks, &dets, gate, p_d, clutter);

        for (i, betas) in result.beta.iter().enumerate() {
            let sum: f64 = betas.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "track {i}: betas sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_jpda_combined_innovation() {
        let track = make_track(&[0.0, 0.0], 1.0);
        let dets = vec![
            DVector::from_column_slice(&[1.0, 0.0]),
            DVector::from_column_slice(&[0.0, 1.0]),
        ];
        // beta: 0.5 for det 0, 0.3 for det 1, 0.2 for miss
        let betas = vec![0.5, 0.3, 0.2];

        let combined = jpda_combined_innovation(&track, &dets, &betas);

        // Expected: 0.5 * [1,0] + 0.3 * [0,1] = [0.5, 0.3]
        assert!((combined[0] - 0.5).abs() < 1e-10);
        assert!((combined[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_jpda_covariance_correction_positive_semidefinite() {
        let track = make_track(&[0.0, 0.0], 1.0);
        let dets = vec![
            DVector::from_column_slice(&[1.0, 0.0]),
            DVector::from_column_slice(&[0.0, 1.0]),
        ];
        let betas = vec![0.4, 0.4, 0.2];
        let combined = jpda_combined_innovation(&track, &dets, &betas);
        let correction = jpda_covariance_correction(&track, &dets, &betas, &combined);

        // Spread of innovations should be PSD
        let eigenvalues = correction.symmetric_eigen().eigenvalues;
        for (i, &ev) in eigenvalues.iter().enumerate() {
            assert!(ev > -1e-10, "eigenvalue[{i}] = {ev} should be non-negative");
        }
    }
}
