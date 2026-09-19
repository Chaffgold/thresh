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

use std::collections::HashMap;

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

/// Result of JPDA state and covariance update for a single track.
#[derive(Debug, Clone)]
pub struct JpdaUpdateResult {
    /// Updated state estimate.
    pub state: DVector<f64>,
    /// Updated covariance estimate.
    pub covariance: DMatrix<f64>,
    /// Missed-detection probability (beta_0).
    pub miss_probability: f64,
}

/// Perform the full JPDA state update for a single track.
///
/// Computes `x+ = x- + K * v_combined` where `K` is the standard Kalman gain
/// and `v_combined` is the merged innovation.
///
/// # Arguments
///
/// * `state` — prior state estimate `x-`.
/// * `covariance` — prior covariance `P-`.
/// * `track` — predicted measurement and innovation covariance.
/// * `detections` — measurement vectors.
/// * `betas` — association probabilities (length `n_dets + 1`, last is miss).
/// * `h` — observation matrix.
pub fn jpda_state_update(
    state: &DVector<f64>,
    covariance: &DMatrix<f64>,
    track: &JpdaTrack,
    detections: &[DVector<f64>],
    betas: &[f64],
    h: &DMatrix<f64>,
) -> JpdaUpdateResult {
    let combined = jpda_combined_innovation(track, detections, betas);
    let spread = jpda_covariance_correction(track, detections, betas, &combined);

    // Kalman gain: K = P * H^T * S^{-1}
    let s_inv = track
        .innovation_covariance
        .clone()
        .try_inverse()
        .expect("Innovation covariance S is singular in JPDA update");
    let k = covariance * h.transpose() * &s_inv;

    // State update: x+ = x- + K * v_combined
    let state_updated = state + &k * &combined;

    // Covariance update with spread-of-innovations:
    //   P+ = beta_0 * P- + (1 - beta_0) * P_kf + K * P_spread * K^T
    // where P_kf = P- - K * S * K^T (standard Kalman covariance update)
    let beta_0 = *betas.last().unwrap_or(&0.0);
    let p_kf = covariance - &k * &track.innovation_covariance * k.transpose();
    let cov_updated = beta_0 * covariance + (1.0 - beta_0) * &p_kf + &k * &spread * k.transpose();

    JpdaUpdateResult {
        state: state_updated,
        covariance: cov_updated,
        miss_probability: beta_0,
    }
}

/// Run JPDA association and state update for a set of tracks.
///
/// This is the main public API combining probability computation and
/// Kalman-based state/covariance update.
///
/// # Arguments
///
/// * `tracks` — predicted measurement info per track.
/// * `states` — prior state estimates, one per track.
/// * `covariances` — prior covariance matrices, one per track.
/// * `detections` — measurement vectors.
/// * `observation_matrices` — observation matrix per track, sized to that
///   track's state dimension (pad unobserved trailing components with zero
///   columns when tracks of different dimensions coexist).
/// * `gate` — Mahalanobis distance squared gating threshold.
/// * `p_detection` — probability of detection.
/// * `clutter_density` — false alarm spatial density.
#[allow(clippy::too_many_arguments)]
pub fn jpda_associate_and_update(
    tracks: &[JpdaTrack],
    states: &[DVector<f64>],
    covariances: &[DMatrix<f64>],
    detections: &[DVector<f64>],
    observation_matrices: &[DMatrix<f64>],
    gate: f64,
    p_detection: f64,
    clutter_density: f64,
) -> Vec<JpdaUpdateResult> {
    debug_assert_eq!(
        observation_matrices.len(),
        tracks.len(),
        "one observation matrix per track"
    );
    let result = jpda_probabilities(tracks, detections, gate, p_detection, clutter_density);

    result
        .beta
        .iter()
        .enumerate()
        .map(|(i, betas)| {
            jpda_state_update(
                &states[i],
                &covariances[i],
                &tracks[i],
                detections,
                betas,
                &observation_matrices[i],
            )
        })
        .collect()
}

/// Partition tracks into independent clusters based on shared gated detections.
///
/// Two tracks are in the same cluster if they gate on at least one common detection.
/// This is equivalent to finding connected components of the bipartite gating graph.
///
/// # Arguments
///
/// * `n_tracks` — number of tracks.
/// * `n_dets` — number of detections.
/// * `gated` — `gated[i][j]` is `true` if track `i` gates detection `j`.
///
/// # Returns
///
/// A list of clusters, each containing `(track_indices, detection_indices)`.
pub fn cluster_tracks(
    n_tracks: usize,
    n_dets: usize,
    gated: &[Vec<bool>],
) -> Vec<(Vec<usize>, Vec<usize>)> {
    // Union-find on tracks: merge tracks that share a detection.
    let mut parent: Vec<usize> = (0..n_tracks).collect();
    union_tracks_sharing_detections(&mut parent, n_tracks, n_dets, gated);

    let cluster_map = group_tracks_by_root(&mut parent, n_tracks);
    collect_cluster_detections(cluster_map, n_dets, gated)
}

/// Find the root of `x` in the union-find forest, compressing the path.
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path compression
        x = parent[x];
    }
    x
}

/// Merge the sets holding `a` and `b`; the root of `a` becomes the joint root.
fn union_sets(parent: &mut [usize], a: usize, b: usize) {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

/// For each detection, find all tracks that gate on it and union them.
///
/// `parent` must hold one entry per track (`n_tracks` of them); rows of
/// `gated` beyond `n_tracks` are ignored.
fn union_tracks_sharing_detections(
    parent: &mut [usize],
    n_tracks: usize,
    n_dets: usize,
    gated: &[Vec<bool>],
) {
    for j in 0..n_dets {
        let mut first_track: Option<usize> = None;
        for (i, gated_row) in gated.iter().enumerate().take(n_tracks) {
            if gated_row[j] {
                if let Some(ft) = first_track {
                    union_sets(parent, ft, i);
                } else {
                    first_track = Some(i);
                }
            }
        }
    }
}

/// Group track indices by their union-find root, ascending within a group.
fn group_tracks_by_root(parent: &mut [usize], n_tracks: usize) -> HashMap<usize, Vec<usize>> {
    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n_tracks {
        let root = find_root(parent, i);
        cluster_map.entry(root).or_default().push(i);
    }
    cluster_map
}

/// For each cluster, collect the detections gated by any of its tracks.
fn collect_cluster_detections(
    cluster_map: HashMap<usize, Vec<usize>>,
    n_dets: usize,
    gated: &[Vec<bool>],
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut clusters: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for (_, track_indices) in cluster_map {
        let det_indices: Vec<usize> = (0..n_dets)
            .filter(|&j| track_indices.iter().any(|&ti| gated[ti][j]))
            .collect();
        clusters.push((track_indices, det_indices));
    }

    clusters
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
    fn test_jpda_state_update_single_detection() {
        // Single track, single detection very close: update should move state toward detection.
        let state = DVector::from_column_slice(&[0.0, 0.0, 0.0, 0.0]);
        let covariance = DMatrix::identity(4, 4) * 10.0;
        let h = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let track = JpdaTrack {
            predicted_measurement: DVector::from_column_slice(&[0.0, 0.0]),
            innovation_covariance: &h * &covariance * h.transpose() + DMatrix::identity(2, 2),
        };
        let det = DVector::from_column_slice(&[5.0, 3.0]);
        // High beta for the detection, low for miss
        let betas = vec![0.99, 0.01];

        let result = jpda_state_update(&state, &covariance, &track, &[det], &betas, &h);

        // State should move toward [5, 0, 3, 0]
        assert!(result.state[0] > 0.0, "x should move toward detection");
        assert!(result.state[2] > 0.0, "y should move toward detection");
        // Covariance should be reduced
        assert!(
            result.covariance[(0, 0)] < covariance[(0, 0)],
            "covariance should decrease"
        );
    }

    #[test]
    fn test_jpda_state_update_missed_detection() {
        // If beta_0 = 1.0 (all miss), state should not change.
        let state = DVector::from_column_slice(&[1.0, 2.0]);
        let covariance = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let track = JpdaTrack {
            predicted_measurement: DVector::from_column_slice(&[1.0, 2.0]),
            innovation_covariance: DMatrix::identity(2, 2) * 2.0,
        };
        let det = DVector::from_column_slice(&[10.0, 10.0]);
        let betas = vec![0.0, 1.0]; // all miss

        let result = jpda_state_update(&state, &covariance, &track, &[det], &betas, &h);

        // With beta_0 = 1.0 and beta_det = 0.0, combined innovation is zero
        assert!(
            (result.state[0] - 1.0).abs() < 1e-10,
            "state x should not change"
        );
        assert!(
            (result.state[1] - 2.0).abs() < 1e-10,
            "state y should not change"
        );
    }

    #[test]
    fn test_jpda_associate_and_update() {
        let h = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let p = DMatrix::identity(4, 4) * 5.0;
        let r = DMatrix::identity(2, 2);

        let states = vec![
            DVector::from_column_slice(&[0.0, 0.0, 0.0, 0.0]),
            DVector::from_column_slice(&[10.0, 0.0, 0.0, 0.0]),
        ];
        let covariances = vec![p.clone(), p.clone()];
        let tracks: Vec<JpdaTrack> = states
            .iter()
            .map(|s| {
                let pred = &h * s;
                let s_mat = &h * &p * h.transpose() + &r;
                JpdaTrack {
                    predicted_measurement: pred,
                    innovation_covariance: s_mat,
                }
            })
            .collect();

        let detections = vec![
            DVector::from_column_slice(&[0.5, 0.0]),
            DVector::from_column_slice(&[9.5, 0.0]),
        ];

        let results = jpda_associate_and_update(
            &tracks,
            &states,
            &covariances,
            &detections,
            &[h.clone(), h.clone()],
            100.0,
            0.9,
            1e-6,
        );

        assert_eq!(results.len(), 2);
        // Track 0 should move toward det 0
        assert!(results[0].state[0] > 0.0);
        // Track 1 should move toward det 1
        assert!(results[1].state[0] < 10.0);
    }

    /// Tracks of different state dimensions update side by side when each
    /// carries its own (padded) observation matrix — a 5D track whose extra
    /// component is unobserved must not panic next to a 4D one.
    #[test]
    fn test_jpda_mixed_state_dimensions() {
        let h4 = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut h5 = DMatrix::zeros(2, 5);
        h5.view_mut((0, 0), (2, 4)).copy_from(&h4);
        let r = DMatrix::identity(2, 2);

        let states = vec![
            DVector::from_column_slice(&[0.0, 0.0, 0.0, 0.0]),
            DVector::from_column_slice(&[10.0, 0.0, 0.0, 0.0, 2.0]),
        ];
        let covariances = vec![DMatrix::identity(4, 4) * 5.0, DMatrix::identity(5, 5) * 5.0];
        let tracks: Vec<JpdaTrack> = states
            .iter()
            .zip([&h4, &h5])
            .map(|(s, h)| JpdaTrack {
                predicted_measurement: h * s,
                innovation_covariance: {
                    let p = DMatrix::identity(h.ncols(), h.ncols()) * 5.0;
                    h * &p * h.transpose() + &r
                },
            })
            .collect();

        let detections = vec![
            DVector::from_column_slice(&[0.5, 0.0]),
            DVector::from_column_slice(&[9.5, 0.0]),
        ];

        let results = jpda_associate_and_update(
            &tracks,
            &states,
            &covariances,
            &detections,
            &[h4, h5],
            100.0,
            0.9,
            1e-6,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].state.len(), 4);
        assert_eq!(results[1].state.len(), 5);
        assert!(results[0].state[0] > 0.0, "4D track moves toward det 0");
        assert!(results[1].state[0] < 10.0, "5D track moves toward det 1");
        assert!(
            results[1].covariance[(4, 4)].is_finite(),
            "unobserved component keeps a finite variance"
        );
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

    #[test]
    fn test_cluster_tracks_independent() {
        // 3 tracks, 3 detections. Track 0 gates det 0, track 1 gates det 1,
        // track 2 gates det 2. No shared detections -> 3 clusters.
        let gated = vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ];
        let clusters = cluster_tracks(3, 3, &gated);
        assert_eq!(clusters.len(), 3, "should have 3 independent clusters");
    }

    #[test]
    fn test_cluster_tracks_shared_detection() {
        // 2 tracks, 2 detections. Both tracks gate detection 0.
        // They should be in the same cluster.
        let gated = vec![vec![true, false], vec![true, true]];
        let clusters = cluster_tracks(2, 2, &gated);
        assert_eq!(
            clusters.len(),
            1,
            "shared detection should merge into 1 cluster"
        );
        let (ref tracks, ref dets) = clusters[0];
        assert_eq!(tracks.len(), 2);
        assert!(dets.contains(&0));
        assert!(dets.contains(&1));
    }

    #[test]
    fn test_cluster_tracks_two_clusters() {
        // 4 tracks, 4 detections.
        // Tracks 0,1 share det 0; tracks 2,3 share det 3. -> 2 clusters.
        let gated = vec![
            vec![true, true, false, false],
            vec![true, false, false, false],
            vec![false, false, true, true],
            vec![false, false, false, true],
        ];
        let clusters = cluster_tracks(4, 4, &gated);
        assert_eq!(clusters.len(), 2, "should have 2 clusters");
    }

    /// `cluster_tracks` groups through a `HashMap`, so the order of the
    /// returned clusters varies from run to run. Order them by their lowest
    /// track index so the whole result can be compared exactly.
    fn sorted_clusters(
        mut clusters: Vec<(Vec<usize>, Vec<usize>)>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        clusters.sort_by_key(|(tracks, _)| tracks[0]);
        clusters
    }

    /// Pins the complete output on a fixed case: a transitive chain
    /// (0-2 share det 1, 2-4 share det 3), a pair (1-5 share det 0), a track
    /// that gates nothing (singleton with no detections) and a detection
    /// nobody gates (det 2, in no cluster). Track and detection indices are
    /// ascending within each cluster.
    #[test]
    fn test_cluster_tracks_pins_full_output() {
        let t = true;
        let f = false;
        let gated = vec![
            vec![f, t, f, f, f, f],
            vec![t, f, f, f, f, f],
            vec![f, t, f, t, f, f],
            vec![f, f, f, f, f, f],
            vec![f, f, f, t, t, f],
            vec![t, f, f, f, f, t],
        ];

        let clusters = sorted_clusters(cluster_tracks(6, 6, &gated));

        assert_eq!(
            clusters,
            vec![
                (vec![0, 2, 4], vec![1, 3, 4]),
                (vec![1, 5], vec![0, 5]),
                (vec![3], vec![]),
            ]
        );
    }

    /// Rows of `gated` beyond `n_tracks` are ignored: here the extra row is
    /// the only link between tracks 0 and 1, and they stay separate.
    #[test]
    fn test_cluster_tracks_ignores_rows_beyond_n_tracks() {
        let gated = vec![vec![true, false], vec![false, true], vec![true, true]];

        let clusters = sorted_clusters(cluster_tracks(2, 2, &gated));

        assert_eq!(clusters, vec![(vec![0], vec![0]), (vec![1], vec![1])]);
    }

    /// A non-square case (3 tracks, 2 detections) whose union phase leaves a
    /// forest two levels deep: det 0 joins tracks 1 and 2, then det 1 roots
    /// track 1 under track 0, so track 2 reaches its root only through
    /// `find_root`. Swapping the two counts, or grouping by `parent[i]`
    /// instead of the root, both fail here and in no square, flat case.
    #[test]
    fn test_cluster_tracks_non_square_two_level_forest() {
        let gated = vec![vec![false, true], vec![true, true], vec![true, false]];

        let clusters = sorted_clusters(cluster_tracks(3, 2, &gated));

        assert_eq!(clusters, vec![(vec![0, 1, 2], vec![0, 1])]);
    }

    #[test]
    fn test_cluster_tracks_no_tracks() {
        assert!(cluster_tracks(0, 3, &[]).is_empty());
    }

    #[test]
    fn test_union_sets_roots_under_first_argument() {
        let mut parent = vec![0, 1, 2];

        union_sets(&mut parent, 0, 1);
        assert_eq!(parent, vec![0, 0, 2]);

        // Track 1's root is 0, so it is root 0 that goes under root 2.
        union_sets(&mut parent, 2, 1);
        assert_eq!(parent, vec![2, 0, 2]);

        // Lookup compresses the path from 1 straight to the root.
        assert_eq!(find_root(&mut parent, 1), 2);
        assert_eq!(parent, vec![2, 2, 2]);
    }

    /// Same gating matrix as `test_cluster_tracks_pins_full_output`.
    #[test]
    fn test_union_tracks_sharing_detections_pins_forest() {
        let t = true;
        let f = false;
        let gated = vec![
            vec![f, t, f, f, f, f],
            vec![t, f, f, f, f, f],
            vec![f, t, f, t, f, f],
            vec![f, f, f, f, f, f],
            vec![f, f, f, t, t, f],
            vec![t, f, f, f, f, t],
        ];
        let mut parent: Vec<usize> = (0..6).collect();

        union_tracks_sharing_detections(&mut parent, 6, 6, &gated);

        assert_eq!(parent, vec![0, 1, 0, 3, 0, 1]);
    }

    #[test]
    fn test_group_tracks_by_root() {
        let mut parent = vec![0, 1, 0, 3, 0, 1];

        let groups = group_tracks_by_root(&mut parent, 6);

        let expected: HashMap<usize, Vec<usize>> =
            HashMap::from([(0, vec![0, 2, 4]), (1, vec![1, 5]), (3, vec![3])]);
        assert_eq!(groups, expected);
    }

    #[test]
    fn test_collect_cluster_detections() {
        let gated = vec![vec![true, false, true], vec![false, false, true]];
        let cluster_map = HashMap::from([(0, vec![0, 1])]);

        let clusters = collect_cluster_detections(cluster_map, 3, &gated);

        assert_eq!(clusters, vec![(vec![0, 1], vec![0, 2])]);
    }
}
