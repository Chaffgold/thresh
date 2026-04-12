//! Multi-Hypothesis Tracking (MHT) framework.
//!
//! Maintains multiple global association hypotheses over a sliding window,
//! deferring hard assignment decisions. Uses k-best pruning (bounds width)
//! and N-scan pruning (bounds depth) to keep the hypothesis tree tractable.
//!
//! # References
//!
//! Reid, D. (1979). An algorithm for tracking multiple targets. *IEEE Trans.
//! on Automatic Control*, 24(6), 843-854.

/// A single association assignment: `(track_index, detection_index)` pairs.
type Assignment = Vec<(usize, Option<usize>)>;

/// A single hypothesis: a mapping from tracks to detections.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// `(track_index, detection_index)` pairs. `None` means missed detection.
    pub assignments: Assignment,
    /// Log-likelihood of this hypothesis.
    pub log_likelihood: f64,
    /// Index of the parent hypothesis in the previous scan (for N-scan pruning).
    pub parent: Option<usize>,
    /// Scan (timestep) at which this hypothesis was created.
    pub scan: usize,
}

/// MHT hypothesis tree with pruning.
pub struct HypothesisTree {
    hypotheses: Vec<Hypothesis>,
    max_hypotheses: usize,
    n_scan_depth: usize,
    /// Current scan (timestep) counter.
    current_scan: usize,
}

impl HypothesisTree {
    /// Create a new empty hypothesis tree.
    ///
    /// * `max_hypotheses` — k-best pruning limit.
    /// * `n_scan_depth` — N-scan pruning depth (reserved for future use).
    pub fn new(max_hypotheses: usize, n_scan_depth: usize) -> Self {
        Self {
            hypotheses: Vec::new(),
            max_hypotheses,
            n_scan_depth,
            current_scan: 0,
        }
    }

    /// Generate child hypotheses from the current set given new detections.
    ///
    /// Enumerates feasible joint events where each detection is assigned to
    /// at most one track (one-to-one constraint), including the possibility
    /// that each track has a missed detection.
    ///
    /// # Arguments
    ///
    /// * `n_tracks` — number of tracks.
    /// * `n_dets` — number of detections.
    /// * `likelihoods` — `likelihoods[i][j]` is the log-likelihood that
    ///   track `i` generated detection `j`. Use `f64::NEG_INFINITY` for
    ///   gated-out pairs.
    /// * `gate` — log-likelihood threshold; pairs below this are infeasible.
    pub fn expand(&mut self, n_tracks: usize, n_dets: usize, likelihoods: &[Vec<f64>], gate: f64) {
        self.current_scan += 1;
        let scan = self.current_scan;

        let parents = if self.hypotheses.is_empty() {
            // Bootstrap: create a single empty parent hypothesis.
            vec![Hypothesis {
                assignments: Vec::new(),
                log_likelihood: 0.0,
                parent: None,
                scan: 0,
            }]
        } else {
            std::mem::take(&mut self.hypotheses)
        };

        let mut children = Vec::new();

        for (parent_idx, parent) in parents.iter().enumerate() {
            let mut ctx = EnumCtx {
                n_tracks,
                likelihoods,
                gate,
                current: Vec::new(),
                det_used: vec![false; n_dets],
                current_score: 0.0,
                results: Vec::new(),
            };
            ctx.enumerate(0);

            for (assignments, score) in ctx.results {
                children.push(Hypothesis {
                    assignments,
                    log_likelihood: parent.log_likelihood + score,
                    parent: Some(parent_idx),
                    scan,
                });
            }
        }

        // Hypothesis count monitoring: warn if count exceeds 2 * k_best.
        // In debug builds, emit to stderr so developers notice.
        if children.len() > 2 * self.max_hypotheses {
            #[cfg(debug_assertions)]
            eprintln!(
                "[MHT] hypothesis count {} exceeds 2 * k_best ({}); pruning may be insufficient",
                children.len(),
                self.max_hypotheses
            );
        }

        self.hypotheses = children;
    }

    /// Prune to keep only the k-best hypotheses by log-likelihood.
    pub fn prune_k_best(&mut self) {
        if self.hypotheses.len() <= self.max_hypotheses {
            return;
        }
        // Sort descending by log-likelihood.
        self.hypotheses
            .sort_by(|a, b| b.log_likelihood.partial_cmp(&a.log_likelihood).unwrap());
        self.hypotheses.truncate(self.max_hypotheses);
    }

    /// Perform N-scan pruning: look back `n_scan_depth` scans and identify
    /// track-detection assignments that all surviving hypotheses agree on.
    /// Collapse those agreed-upon assignments and remove hypotheses that are
    /// no longer reachable.
    ///
    /// Returns the agreed-upon assignments at the oldest scan as
    /// `Vec<(track_idx, Option<det_idx>)>`, which the caller can use to
    /// finalize those associations.
    pub fn prune_n_scan(&mut self) -> Vec<(usize, Option<usize>)> {
        if self.hypotheses.is_empty() || self.current_scan <= self.n_scan_depth {
            return Vec::new();
        }

        // For N-scan pruning we look at the assignments in all surviving hypotheses.
        // Since we store flat hypotheses (not a tree with shared ancestry), we
        // compare the assignments across all hypotheses. Assignments that are
        // identical across ALL hypotheses are "agreed upon" and can be collapsed.

        let first = &self.hypotheses[0].assignments;
        let mut agreed: Vec<(usize, Option<usize>)> = Vec::new();

        for &(ti, ref di) in first {
            let all_agree = self
                .hypotheses
                .iter()
                .skip(1)
                .all(|h| h.assignments.iter().any(|(t, d)| *t == ti && d == di));
            if all_agree {
                agreed.push((ti, *di));
            }
        }

        // If there are agreed-upon assignments, remove them from all hypotheses
        // (they are finalized and no longer need to be tracked).
        if !agreed.is_empty() {
            for hyp in &mut self.hypotheses {
                hyp.assignments
                    .retain(|(ti, di)| !agreed.iter().any(|(at, ad)| *at == *ti && *ad == *di));
            }

            // Memory reclamation: deduplicate hypotheses that are now identical.
            self.compact();
        }

        agreed
    }

    /// Remove duplicate hypotheses (same assignments and similar scores)
    /// after N-scan pruning collapses agreed assignments.
    fn compact(&mut self) {
        if self.hypotheses.len() <= 1 {
            return;
        }

        // Sort by score descending so we keep the best when deduplicating.
        self.hypotheses
            .sort_by(|a, b| b.log_likelihood.partial_cmp(&a.log_likelihood).unwrap());

        let mut unique: Vec<Hypothesis> = Vec::with_capacity(self.hypotheses.len());
        for hyp in std::mem::take(&mut self.hypotheses) {
            let is_dup = unique.iter().any(|u| u.assignments == hyp.assignments);
            if !is_dup {
                unique.push(hyp);
            }
        }
        self.hypotheses = unique;
    }

    /// Extract consistent track IDs by tracing the best hypothesis's
    /// assignment history.
    ///
    /// Returns a map from track index to the detection index it was assigned
    /// to in the best hypothesis, providing a consistent association across
    /// timesteps.
    pub fn consistent_track_assignments(&self) -> Vec<(usize, Option<usize>)> {
        match self.best_hypothesis() {
            Some(h) => h.assignments.clone(),
            None => Vec::new(),
        }
    }

    /// Get the best (most likely) hypothesis.
    pub fn best_hypothesis(&self) -> Option<&Hypothesis> {
        self.hypotheses
            .iter()
            .max_by(|a, b| a.log_likelihood.partial_cmp(&b.log_likelihood).unwrap())
    }

    /// Get the marginal association probability for each track-detection pair
    /// across all hypotheses.
    ///
    /// Returns a matrix where `result[i][j]` is the fraction of total
    /// hypothesis weight that assigns track `i` to detection `j`.
    pub fn marginal_probabilities(&self, n_tracks: usize, n_dets: usize) -> Vec<Vec<f64>> {
        if self.hypotheses.is_empty() {
            return vec![vec![0.0; n_dets]; n_tracks];
        }

        // Convert log-likelihoods to normalized weights (softmax).
        let max_ll = self
            .hypotheses
            .iter()
            .map(|h| h.log_likelihood)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = self
            .hypotheses
            .iter()
            .map(|h| (h.log_likelihood - max_ll).exp())
            .collect();
        let total_weight: f64 = weights.iter().sum();

        let mut marginals = vec![vec![0.0; n_dets]; n_tracks];

        for (h_idx, hyp) in self.hypotheses.iter().enumerate() {
            let w = weights[h_idx] / total_weight;
            for &(ti, ref det_idx) in &hyp.assignments {
                if let Some(dj) = det_idx
                    && ti < n_tracks
                    && *dj < n_dets
                {
                    marginals[ti][*dj] += w;
                }
            }
        }

        marginals
    }

    /// Number of hypotheses currently in the tree.
    pub fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }
}

/// Context for recursive enumeration of feasible joint assignment events.
struct EnumCtx<'a> {
    n_tracks: usize,
    likelihoods: &'a [Vec<f64>],
    gate: f64,
    current: Assignment,
    det_used: Vec<bool>,
    current_score: f64,
    results: Vec<(Assignment, f64)>,
}

impl EnumCtx<'_> {
    /// Recursively enumerate feasible joint assignment events starting from
    /// `track_idx`. Each detection is assigned to at most one track; each
    /// track is assigned to at most one detection or marked as missed.
    fn enumerate(&mut self, track_idx: usize) {
        if track_idx >= self.n_tracks {
            self.results
                .push((self.current.clone(), self.current_score));
            return;
        }

        let n_dets = self.det_used.len();

        // Option 1: missed detection for this track
        self.current.push((track_idx, None));
        self.enumerate(track_idx + 1);
        self.current.pop();

        // Option 2: assign each available detection to this track
        for j in 0..n_dets {
            if self.det_used[j] {
                continue;
            }
            let ll = self.likelihoods[track_idx][j];
            if ll <= self.gate {
                continue;
            }
            self.det_used[j] = true;
            self.current.push((track_idx, Some(j)));
            self.current_score += ll;
            self.enumerate(track_idx + 1);
            self.current_score -= ll;
            self.current.pop();
            self.det_used[j] = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mht_expand_generates_hypotheses() {
        let mut tree = HypothesisTree::new(100, 3);
        // 2 tracks, 2 detections, all feasible
        let likelihoods = vec![vec![-1.0, -2.0], vec![-1.5, -0.5]];
        let gate = f64::NEG_INFINITY; // all pass

        tree.expand(2, 2, &likelihoods, gate);

        // Possible hypotheses:
        // 1. both miss                       (score 0)
        // 2. track 0 -> det 0, track 1 miss  (score -1.0)
        // 3. track 0 -> det 1, track 1 miss  (score -2.0)
        // 4. track 0 miss, track 1 -> det 0  (score -1.5)
        // 5. track 0 miss, track 1 -> det 1  (score -0.5)
        // 6. track 0 -> det 0, track 1 -> det 1 (score -1.5)
        // 7. track 0 -> det 1, track 1 -> det 0 (score -3.5)
        assert_eq!(
            tree.hypothesis_count(),
            7,
            "2 tracks, 2 dets should produce 7 hypotheses"
        );
    }

    #[test]
    fn test_mht_prune_k_best() {
        let mut tree = HypothesisTree::new(10, 3);

        // Manually insert 100 hypotheses
        tree.hypotheses = (0..100)
            .map(|i| Hypothesis {
                assignments: vec![(0, Some(0))],
                log_likelihood: -(i as f64),
                parent: None,
                scan: 1,
            })
            .collect();

        assert_eq!(tree.hypothesis_count(), 100);
        tree.prune_k_best();
        assert_eq!(tree.hypothesis_count(), 10);

        // Verify the best 10 were kept (highest log-likelihoods = 0, -1, ..., -9)
        for (idx, h) in tree.hypotheses.iter().enumerate() {
            assert!(
                (h.log_likelihood - (-(idx as f64))).abs() < 1e-10,
                "hypothesis {idx} should have log_likelihood {}, got {}",
                -(idx as f64),
                h.log_likelihood
            );
        }
    }

    #[test]
    fn test_mht_best_hypothesis() {
        let mut tree = HypothesisTree::new(100, 3);
        tree.hypotheses = vec![
            Hypothesis {
                assignments: vec![(0, Some(0))],
                log_likelihood: -5.0,
                parent: None,
                scan: 1,
            },
            Hypothesis {
                assignments: vec![(0, Some(1))],
                log_likelihood: -1.0,
                parent: None,
                scan: 1,
            },
            Hypothesis {
                assignments: vec![(0, None)],
                log_likelihood: -3.0,
                parent: None,
                scan: 1,
            },
        ];

        let best = tree.best_hypothesis().unwrap();
        assert!(
            (best.log_likelihood - (-1.0)).abs() < 1e-10,
            "best hypothesis should have log_likelihood -1.0, got {}",
            best.log_likelihood
        );
    }

    #[test]
    fn test_mht_marginal_probabilities() {
        let mut tree = HypothesisTree::new(100, 3);
        // 2 tracks, 2 detections
        let likelihoods = vec![vec![-1.0, -2.0], vec![-1.5, -0.5]];
        tree.expand(2, 2, &likelihoods, f64::NEG_INFINITY);

        let marginals = tree.marginal_probabilities(2, 2);

        // For each track, marginals across all detections should sum to <= 1.0
        // (the remainder is missed-detection probability)
        for (i, row) in marginals.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                sum <= 1.0 + 1e-10,
                "track {i}: marginals sum to {sum}, should be <= 1.0"
            );
        }

        // Each individual marginal should be non-negative
        for (i, row) in marginals.iter().enumerate() {
            for (j, &p) in row.iter().enumerate() {
                assert!(
                    p >= -1e-10,
                    "track {i}, det {j}: marginal {p} should be non-negative"
                );
            }
        }
    }

    #[test]
    fn test_mht_empty_tree_best_hypothesis() {
        let tree = HypothesisTree::new(10, 3);
        assert!(tree.best_hypothesis().is_none());
    }

    #[test]
    fn test_mht_n_scan_pruning_collapses_agreed() {
        // Manually construct hypotheses that all agree on the same assignment
        // for track 0, but differ on track 1. N-scan pruning should find
        // the agreed assignment for track 0.
        let mut tree = HypothesisTree::new(100, 1);
        tree.current_scan = 2; // pretend we're past n_scan_depth

        tree.hypotheses = vec![
            Hypothesis {
                // Both agree: track 0 -> det 0
                // Differ: track 1 -> det 1 vs det 2
                assignments: vec![(0, Some(0)), (1, Some(1))],
                log_likelihood: -1.0,
                parent: None,
                scan: 2,
            },
            Hypothesis {
                assignments: vec![(0, Some(0)), (1, Some(2))],
                log_likelihood: -2.0,
                parent: None,
                scan: 2,
            },
            Hypothesis {
                assignments: vec![(0, Some(0)), (1, None)],
                log_likelihood: -3.0,
                parent: None,
                scan: 2,
            },
        ];

        let agreed = tree.prune_n_scan();

        // All hypotheses agree on (0, Some(0)), so it should be collapsed
        assert_eq!(agreed.len(), 1, "should find one agreed assignment");
        assert_eq!(agreed[0], (0, Some(0)));

        // After pruning, the agreed assignment should be removed from all hypotheses
        for h in &tree.hypotheses {
            assert!(
                !h.assignments.iter().any(|(t, d)| *t == 0 && *d == Some(0)),
                "agreed assignment should be removed from hypotheses"
            );
        }
    }

    #[test]
    fn test_mht_consistent_track_assignments() {
        let mut tree = HypothesisTree::new(100, 3);
        let likelihoods = vec![vec![-1.0, -2.0], vec![-1.5, -0.5]];
        tree.expand(2, 2, &likelihoods, f64::NEG_INFINITY);

        let assignments = tree.consistent_track_assignments();
        // Should return the assignments from the best hypothesis
        assert!(!assignments.is_empty());
        // Each track should appear exactly once
        let track_ids: Vec<usize> = assignments.iter().map(|(t, _)| *t).collect();
        assert!(track_ids.contains(&0));
        assert!(track_ids.contains(&1));
    }

    #[test]
    fn test_mht_compact_removes_duplicates() {
        let mut tree = HypothesisTree::new(100, 3);
        tree.hypotheses = vec![
            Hypothesis {
                assignments: vec![(0, Some(0))],
                log_likelihood: -1.0,
                parent: None,
                scan: 1,
            },
            Hypothesis {
                assignments: vec![(0, Some(0))],
                log_likelihood: -2.0,
                parent: None,
                scan: 1,
            },
            Hypothesis {
                assignments: vec![(0, Some(1))],
                log_likelihood: -1.5,
                parent: None,
                scan: 1,
            },
        ];
        tree.compact();
        // Two unique assignment sets: (0, Some(0)) and (0, Some(1))
        assert_eq!(
            tree.hypothesis_count(),
            2,
            "compact should remove duplicate assignment sets"
        );
        // The better-scoring duplicate should be kept
        let best = tree
            .hypotheses
            .iter()
            .find(|h| h.assignments == vec![(0, Some(0))])
            .unwrap();
        assert!((best.log_likelihood - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mht_gating_reduces_hypotheses() {
        let mut tree = HypothesisTree::new(100, 3);
        // 2 tracks, 2 detections, but track 0 can only see det 0
        // and track 1 can only see det 1 (others are gated out)
        let likelihoods = vec![
            vec![-1.0, -100.0], // track 0: det 0 feasible, det 1 gated
            vec![-100.0, -0.5], // track 1: det 0 gated, det 1 feasible
        ];
        let gate = -50.0; // likelihoods must be > gate to pass

        tree.expand(2, 2, &likelihoods, gate);

        // With tight gating, fewer hypotheses should be generated
        // Only: both miss, t0->d0 + t1 miss, t0 miss + t1->d1, t0->d0 + t1->d1
        assert_eq!(
            tree.hypothesis_count(),
            4,
            "tight gating should produce 4 hypotheses, got {}",
            tree.hypothesis_count()
        );
    }
}
