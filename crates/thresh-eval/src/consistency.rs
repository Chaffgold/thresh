//! Filter consistency metrics: NEES / NIS with two-sided chi-squared bounds.
//!
//! Implementation of the `eval-consistency-metrics` OpenSpec change,
//! sections 2 (chi-squared bounds, NEES/NIS core, accumulators) and 3
//! (`EstimateFrame` truth alignment via [`sequence_anees`]).
//!
//! The formulation follows Bar-Shalom, *Estimation with Applications to
//! Tracking and Navigation*, §5.4: for a consistent filter each NEES/NIS
//! sample is chi-squared distributed with `dof` degrees of freedom, so the
//! sum of `n` samples is judged against two-sided chi-squared acceptance
//! bounds on `n · dof` degrees of freedom (the time-average test). Both
//! tails are failures: above the upper bound the reported covariance is too
//! small ([`ConsistencyVerdict::Overconfident`]); below the lower bound it
//! is inflated ([`ConsistencyVerdict::Underconfident`]).

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::matching::{FrameData, match_frame};

/// Chi-squared quantile math: Acklam inverse-normal, Wilson–Hilferty seed,
/// and Newton refinement on the regularized lower incomplete gamma.
///
/// Pure `f64`, no external dependencies. All iterations have fixed caps and
/// tolerances so evaluation is deterministic: identical inputs produce
/// bitwise-identical outputs.
pub mod chi2 {
    /// Default two-sided significance level (95% confidence interval).
    pub const DEFAULT_ALPHA: f64 = 0.05;

    /// Maximum Newton refinement iterations for [`quantile`].
    const NEWTON_MAX_ITER: usize = 64;
    /// Relative step tolerance terminating the Newton refinement.
    const NEWTON_REL_TOL: f64 = 1e-12;
    /// Maximum series / continued-fraction iterations for the regularized
    /// incomplete gamma. Sized for the slowest case (dof of order 1e5, where
    /// the series needs O(sqrt(dof)) terms near its convergence seam); each
    /// iteration is a few flops, so the worst case is still microseconds.
    const GAMMA_MAX_ITER: usize = 10_000;
    /// Convergence tolerance for the incomplete-gamma series / fraction.
    const GAMMA_EPS: f64 = 1e-15;

    /// Two-sided chi-squared acceptance bounds for `dof` degrees of freedom
    /// at significance `alpha` (confidence `1 - alpha`; use
    /// [`DEFAULT_ALPHA`] for the conventional 95% interval).
    ///
    /// Returns `(lower, upper)` = `(quantile(dof, alpha/2),
    /// quantile(dof, 1 - alpha/2))`.
    ///
    /// # Panics
    ///
    /// Panics if `dof == 0` or `alpha` is not strictly inside `(0, 1)`.
    pub fn two_sided_bounds(dof: usize, alpha: f64) -> (f64, f64) {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "significance level must be in (0, 1), got {alpha}"
        );
        (quantile(dof, 0.5 * alpha), quantile(dof, 1.0 - 0.5 * alpha))
    }

    /// Chi-squared quantile function: the `x` with `P(chi2_dof <= x) = p`.
    ///
    /// A Wilson–Hilferty seed (via Acklam's inverse-normal approximation) is
    /// refined by Newton iterations on the regularized lower incomplete
    /// gamma `P(dof/2, x/2)`. The refined quantile is accurate to near
    /// machine precision across the full dof range; the raw seed alone is
    /// not (its small-dof lower-tail error reaches −48% at dof 2).
    ///
    /// # Panics
    ///
    /// Panics if `dof == 0` or `p` is not strictly inside `(0, 1)`.
    pub fn quantile(dof: usize, p: f64) -> f64 {
        assert!(dof > 0, "chi-squared dof must be positive");
        assert!(
            p > 0.0 && p < 1.0,
            "quantile probability must be in (0, 1), got {p}"
        );
        let k = dof as f64;
        let x = positive_seed(k, p);
        newton_refine(k, p, x)
    }

    /// Wilson–Hilferty seed clamped to a positive value.
    ///
    /// When the cube in the Wilson–Hilferty transform goes non-positive
    /// (deep lower tail at very small dof), falls back to inverting the
    /// small-`x` leading term of the lower incomplete gamma.
    fn positive_seed(k: f64, p: f64) -> f64 {
        let seed = wilson_hilferty(k, p);
        if seed.is_finite() && seed > 0.0 {
            return seed;
        }
        let a = 0.5 * k;
        // P(a, x/2) ~ (x/2)^a / (a * Gamma(a)) for small x.
        let fallback = 2.0 * (p * a * ln_gamma(a).exp()).powf(1.0 / a);
        if fallback.is_finite() && fallback > 0.0 {
            fallback
        } else {
            1e-8
        }
    }

    /// Wilson–Hilferty approximation of the chi-squared quantile:
    /// `k * (1 - 2/(9k) + z_p * sqrt(2/(9k)))^3`.
    fn wilson_hilferty(k: f64, p: f64) -> f64 {
        let z = inv_norm_cdf(p);
        let t = 2.0 / (9.0 * k);
        k * (1.0 - t + z * t.sqrt()).powi(3)
    }

    /// Newton refinement of `x` toward `P(k/2, x/2) = p`, using the
    /// chi-squared pdf as the exact derivative. Fixed iteration cap and
    /// relative tolerance; the iterate is kept strictly positive by halving
    /// instead of overshooting past zero.
    fn newton_refine(k: f64, p: f64, seed: f64) -> f64 {
        let a = 0.5 * k;
        let mut x = seed;
        for _ in 0..NEWTON_MAX_ITER {
            let f = reg_lower_gamma(a, 0.5 * x) - p;
            let deriv = chi2_pdf(k, x);
            if deriv <= 0.0 || !deriv.is_finite() {
                break;
            }
            let mut dx = f / deriv;
            if dx >= x {
                // A full step would land at or below zero; halve instead.
                dx = 0.5 * x;
            }
            x -= dx;
            if dx.abs() <= NEWTON_REL_TOL * x {
                break;
            }
        }
        x
    }

    /// Chi-squared probability density with `k` degrees of freedom at `x`.
    fn chi2_pdf(k: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let a = 0.5 * k;
        ((a - 1.0) * x.ln() - 0.5 * x - a * std::f64::consts::LN_2 - ln_gamma(a)).exp()
    }

    /// Regularized lower incomplete gamma `P(a, x)`, via the standard
    /// series / continued-fraction split (*Numerical Recipes* `gammp`):
    /// series for `x < a + 1`, continued fraction for the complement
    /// otherwise.
    fn reg_lower_gamma(a: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x < a + 1.0 {
            lower_series(a, x)
        } else {
            1.0 - upper_continued_fraction(a, x)
        }
    }

    /// Series representation of `P(a, x)`, convergent for `x < a + 1`.
    fn lower_series(a: f64, x: f64) -> f64 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..GAMMA_MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * GAMMA_EPS {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    }

    /// Continued-fraction representation of `Q(a, x) = 1 - P(a, x)`,
    /// convergent for `x >= a + 1` (modified Lentz's method).
    fn upper_continued_fraction(a: f64, x: f64) -> f64 {
        const FPMIN: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=GAMMA_MAX_ITER {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < GAMMA_EPS {
                break;
            }
        }
        h * (-x + a * x.ln() - ln_gamma(a)).exp()
    }

    /// Natural log of the gamma function (Lanczos approximation, accurate to
    /// about 2e-10 relative for positive arguments).
    fn ln_gamma(x: f64) -> f64 {
        const COF: [f64; 6] = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ];
        let tmp = x + 5.5;
        let tmp = tmp - (x + 0.5) * tmp.ln();
        let mut y = x;
        let mut ser = 1.000000000190015;
        for c in COF {
            y += 1.0;
            ser += c / y;
        }
        -tmp + (2.5066282746310005 * ser / x).ln()
    }

    /// Acklam's rational approximation of the inverse standard-normal CDF
    /// (relative error below 1.15e-9 over the full open interval).
    fn inv_norm_cdf(p: f64) -> f64 {
        const P_LOW: f64 = 0.02425;
        if p < P_LOW {
            let q = (-2.0 * p.ln()).sqrt();
            tail_expansion(q)
        } else if p > 1.0 - P_LOW {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -tail_expansion(q)
        } else {
            central_expansion(p)
        }
    }

    /// Acklam tail branch: returns the (negative) lower-tail value for
    /// `q = sqrt(-2 ln p)`.
    fn tail_expansion(q: f64) -> f64 {
        const C: [f64; 6] = [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838,
            -2.549732539343734,
            4.374664141464968,
            2.938163982698783,
        ];
        const D: [f64; 4] = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996,
            3.754408661907416,
        ];
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    }

    /// Acklam central branch, valid for `p` in `[0.02425, 0.97575]`.
    fn central_expansion(p: f64) -> f64 {
        const A: [f64; 6] = [
            -3.969683028665376e+1,
            2.209460984245205e+2,
            -2.759285104469687e+2,
            1.38357751867269e+2,
            -3.066479806614716e+1,
            2.506628277459239,
        ];
        const B: [f64; 5] = [
            -5.447609879822406e+1,
            1.615858368580409e+2,
            -1.556989798598866e+2,
            6.680131188771972e+1,
            -1.328068155288572e+1,
        ];
        let q = p - 0.5;
        let r = q * q;
        let num = ((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5];
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num * q / den
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn rel_err(actual: f64, expected: f64) -> f64 {
            ((actual - expected) / expected).abs()
        }

        #[test]
        fn inverse_normal_matches_reference() {
            // Reference z-quantiles of the standard normal.
            let cases = [
                (0.025, -1.9599639845400545),
                (0.975, 1.9599639845400545),
                (0.005, -2.5758293035489004),
                (0.995, 2.5758293035489004),
                (0.5, 0.0),
                (0.16, -0.9944578832097532),
            ];
            for (p, z) in cases {
                let got = inv_norm_cdf(p);
                assert!(
                    (got - z).abs() < 1e-7,
                    "inv_norm_cdf({p}) = {got}, expected {z}"
                );
            }
        }

        /// Spec: "Bounds match reference chi-squared quantiles" — dof 2, 4,
        /// 100, 400, both tails of the 95% interval, within 1% relative.
        /// Reference values computed offline with mpmath (30 decimal digits).
        #[test]
        fn reference_quantiles_95_both_tails() {
            let cases = [
                (2, 0.0506356159686, 7.37775890823),
                (4, 0.484418557088, 11.1432867819),
                (100, 74.2219274749, 129.561197186),
                (400, 346.481765363, 457.305481966),
            ];
            for (dof, lo_ref, hi_ref) in cases {
                let (lo, hi) = two_sided_bounds(dof, 0.05);
                // Spec tolerance: 1% relative.
                assert!(
                    rel_err(lo, lo_ref) < 0.01,
                    "dof {dof} lower {lo} vs {lo_ref}"
                );
                assert!(
                    rel_err(hi, hi_ref) < 0.01,
                    "dof {dof} upper {hi} vs {hi_ref}"
                );
                // The Newton refinement actually clears the spec tolerance by
                // orders of magnitude; pin that margin too.
                assert!(rel_err(lo, lo_ref) < 1e-8, "dof {dof} lower margin");
                assert!(rel_err(hi, hi_ref) < 1e-8, "dof {dof} upper margin");
            }
        }

        /// Spec: "Configurable confidence level" — the 99% interval strictly
        /// contains the 95% interval and matches its own reference
        /// quantiles at the same tolerance.
        #[test]
        fn reference_quantiles_99_widen_and_match() {
            let cases = [
                (2, 0.0100250836471, 10.5966347331),
                (4, 0.206989093496, 14.8602590006),
                (100, 67.3275633055, 140.169489442),
                (400, 330.902750344, 476.60642674),
            ];
            for (dof, lo_ref, hi_ref) in cases {
                let (lo95, hi95) = two_sided_bounds(dof, 0.05);
                let (lo99, hi99) = two_sided_bounds(dof, 0.01);
                assert!(lo99 < lo95, "dof {dof}: 99% lower must sit below 95% lower");
                assert!(hi99 > hi95, "dof {dof}: 99% upper must sit above 95% upper");
                assert!(
                    rel_err(lo99, lo_ref) < 0.01,
                    "dof {dof} lower {lo99} vs {lo_ref}"
                );
                assert!(
                    rel_err(hi99, hi_ref) < 0.01,
                    "dof {dof} upper {hi99} vs {hi_ref}"
                );
                assert!(rel_err(lo99, lo_ref) < 1e-8, "dof {dof} 99% lower margin");
                assert!(rel_err(hi99, hi_ref) < 1e-8, "dof {dof} 99% upper margin");
            }
        }

        /// The small-dof lower-tail quantiles that the raw Wilson–Hilferty
        /// approximation misses (design Decision 2). Seed-vs-refined error at
        /// the reference chi2_2(0.025) = 0.0506356:
        /// - Wilson–Hilferty seed: 0.02614 (relative error ≈ −48%)
        /// - Newton-refined: matches the reference to ~1e-12 relative.
        ///
        /// At dof 4 the seed error is ≈ −7%; still an order of magnitude
        /// outside the 1% spec tolerance, and also repaired by refinement.
        #[test]
        fn small_dof_lower_tail_seed_vs_refined() {
            // dof 2, p = 0.025
            let seed2 = wilson_hilferty(2.0, 0.025);
            let refined2 = quantile(2, 0.025);
            assert!(
                rel_err(seed2, 0.0506356159686) > 0.40,
                "expected the raw Wilson–Hilferty seed to be ~48% off, got {seed2}"
            );
            assert!(rel_err(refined2, 0.0506356159686) < 1e-8);
            // dof 4, p = 0.025
            let seed4 = wilson_hilferty(4.0, 0.025);
            let refined4 = quantile(4, 0.025);
            assert!(
                rel_err(seed4, 0.484418557088) > 0.05,
                "expected the raw Wilson–Hilferty seed to be ~7% off, got {seed4}"
            );
            assert!(rel_err(refined4, 0.484418557088) < 1e-8);
        }

        /// Self-consistency across the dof range, including the large
        /// aggregate dof (N·d) an ANEES gate over a long run produces: the
        /// CDF evaluated at the quantile must recover the probability.
        #[test]
        fn round_trip_cdf_of_quantile() {
            for dof in [1usize, 2, 3, 7, 100, 400, 3_000, 30_000] {
                for p in [0.005, 0.025, 0.5, 0.975, 0.995] {
                    let q = quantile(dof, p);
                    let back = reg_lower_gamma(0.5 * dof as f64, 0.5 * q);
                    assert!(
                        (back - p).abs() < 1e-9,
                        "dof {dof}, p {p}: cdf(quantile) = {back}"
                    );
                }
                let (lo, hi) = two_sided_bounds(dof, DEFAULT_ALPHA);
                assert!(lo < dof as f64 && (dof as f64) < hi);
            }
        }

        #[test]
        fn default_alpha_is_the_95_percent_interval() {
            let (lo, hi) = two_sided_bounds(4, DEFAULT_ALPHA);
            assert_eq!(lo.to_bits(), quantile(4, 0.025).to_bits());
            assert_eq!(hi.to_bits(), quantile(4, 0.975).to_bits());
        }

        #[test]
        #[should_panic(expected = "dof must be positive")]
        fn zero_dof_panics() {
            let _ = quantile(0, 0.5);
        }

        #[test]
        #[should_panic(expected = "significance level")]
        fn out_of_range_alpha_panics() {
            let _ = two_sided_bounds(2, 1.0);
        }
    }
}

/// Normalized estimation error squared: `e^T P^{-1} e` for estimation error
/// `e = x_truth - x_hat` and estimate covariance `P` (Bar-Shalom NEES).
///
/// The degrees of freedom of the resulting chi-squared sample is
/// `error.len()`; feed samples into a [`ConsistencyAccumulator`] constructed
/// with that dof. Position-marginal NEES is obtained by passing the position
/// error and the corresponding marginal covariance block (see
/// [`sequence_anees`]).
///
/// Returns `None` — never `NaN`, infinity, or a panic — when the covariance
/// is singular / numerically non-invertible, when dimensions are
/// inconsistent, or when the input is empty.
pub fn nees(error: &DVector<f64>, p: &DMatrix<f64>) -> Option<f64> {
    quadratic_form(error, p)
}

/// Normalized innovation squared: `nu^T S^{-1} nu` for innovation `nu` and
/// innovation covariance `S` from a filter update (Bar-Shalom NIS).
///
/// NIS needs no ground truth: the inputs come from filter update diagnostics
/// alone, so NIS-based consistency checking works on live/real data. The
/// degrees of freedom is the measurement dimension `innovation.len()`.
///
/// Returns `None` — never `NaN`, infinity, or a panic — when `S` is singular
/// / numerically non-invertible, when dimensions are inconsistent, or when
/// the input is empty.
pub fn nis(innovation: &DVector<f64>, s: &DMatrix<f64>) -> Option<f64> {
    quadratic_form(innovation, s)
}

/// Shared quadratic form `v^T M^{-1} v` with explicit failure on singular or
/// dimension-mismatched inputs and a finiteness guard on the result.
fn quadratic_form(v: &DVector<f64>, m: &DMatrix<f64>) -> Option<f64> {
    if v.is_empty() || m.nrows() != v.len() || m.ncols() != v.len() {
        return None;
    }
    let inv = m.clone().try_inverse()?;
    let q = (inv * v).dot(v);
    q.is_finite().then_some(q)
}

/// Three-way verdict of the two-sided chi-squared consistency test.
///
/// Serializable so report sections (e.g. the consistency section of
/// `EvalReport`) can carry verdicts verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyVerdict {
    /// The averaged statistic lies inside the two-sided acceptance interval.
    Consistent,
    /// Above the upper bound: normalized errors are larger than the reported
    /// covariance claims, i.e. the covariance is too small (optimistic).
    Overconfident,
    /// Below the lower bound: the reported covariance is inflated
    /// (pessimistic), e.g. process noise pumped up to game association.
    Underconfident,
}

/// Running ANEES/ANIS accumulator (Bar-Shalom time-average test).
///
/// Push per-sample NEES or NIS values; [`mean`](Self::mean) is the
/// ANEES/ANIS, and [`verdict`](Self::verdict) judges `n · mean` against
/// two-sided chi-squared bounds on `n · dof` degrees of freedom. Per-track
/// aggregation is one accumulator per track ID ([`PerTrackConsistency`]);
/// time-averaged aggregation is a single accumulator over everything — same
/// type, no parallel code paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsistencyAccumulator {
    /// Sum of the samples pushed so far.
    pub sum: f64,
    /// Number of samples pushed so far.
    pub n: usize,
    /// Chi-squared degrees of freedom of each individual sample (state
    /// dimension for NEES, measurement dimension for NIS).
    pub dof: usize,
}

impl ConsistencyAccumulator {
    /// Create an empty accumulator for samples with `dof` degrees of freedom.
    pub fn new(dof: usize) -> Self {
        Self {
            sum: 0.0,
            n: 0,
            dof,
        }
    }

    /// Add one NEES/NIS sample.
    pub fn push(&mut self, sample: f64) {
        self.sum += sample;
        self.n += 1;
    }

    /// The averaged statistic (ANEES/ANIS); `None` while no samples have
    /// been pushed.
    pub fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    /// Two-sided acceptance interval for [`mean`](Self::mean) at
    /// significance `alpha`: `chi2` bounds on `n · dof` degrees of freedom,
    /// divided by `n`. `None` while no samples have been pushed.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not strictly inside `(0, 1)` or `dof == 0`.
    pub fn bounds(&self, alpha: f64) -> Option<(f64, f64)> {
        (self.n > 0).then(|| {
            let (lo, hi) = chi2::two_sided_bounds(self.n * self.dof, alpha);
            (lo / self.n as f64, hi / self.n as f64)
        })
    }

    /// Judge the averaged statistic against [`bounds`](Self::bounds):
    /// above the upper bound is [`ConsistencyVerdict::Overconfident`], below
    /// the lower bound is [`ConsistencyVerdict::Underconfident`], inside
    /// (bounds inclusive) is [`ConsistencyVerdict::Consistent`]. `None`
    /// while no samples have been pushed.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not strictly inside `(0, 1)` or `dof == 0`.
    pub fn verdict(&self, alpha: f64) -> Option<ConsistencyVerdict> {
        let mean = self.mean()?;
        let (lo, hi) = self.bounds(alpha)?;
        Some(if mean > hi {
            ConsistencyVerdict::Overconfident
        } else if mean < lo {
            ConsistencyVerdict::Underconfident
        } else {
            ConsistencyVerdict::Consistent
        })
    }
}

/// Per-track ANEES/ANIS aggregation: one [`ConsistencyAccumulator`] per
/// track ID, so an inconsistent track can be singled out from an otherwise
/// consistent set.
///
/// All read paths iterate track IDs in sorted order, so results carry no
/// hash-order dependence (deterministic-evaluation requirement).
#[derive(Debug, Clone)]
pub struct PerTrackConsistency {
    dof: usize,
    tracks: HashMap<u64, ConsistencyAccumulator>,
}

impl PerTrackConsistency {
    /// Create an empty per-track aggregation for samples with `dof` degrees
    /// of freedom.
    pub fn new(dof: usize) -> Self {
        Self {
            dof,
            tracks: HashMap::new(),
        }
    }

    /// Add one sample for `track_id`, creating its accumulator on first use.
    pub fn push(&mut self, track_id: u64, sample: f64) {
        self.tracks
            .entry(track_id)
            .or_insert_with(|| ConsistencyAccumulator::new(self.dof))
            .push(sample);
    }

    /// The accumulator for `track_id`, if any samples have been pushed.
    pub fn accumulator(&self, track_id: u64) -> Option<&ConsistencyAccumulator> {
        self.tracks.get(&track_id)
    }

    /// Per-track verdicts at significance `alpha`, sorted by track ID.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not strictly inside `(0, 1)` or `dof == 0`.
    pub fn verdicts(&self, alpha: f64) -> Vec<(u64, ConsistencyVerdict)> {
        let mut ids: Vec<u64> = self.tracks.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .filter_map(|id| Some((id, self.tracks[&id].verdict(alpha)?)))
            .collect()
    }

    /// Number of tracks that have received at least one sample.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether no track has received a sample yet.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

/// Position indices of the tracker's interleaved state layout
/// `[x, vx, y, vy, z, vz, ...]`: elements 0, 2, 4 are the position
/// components. The 7D ballistic state shares the same first-six layout.
pub const INTERLEAVED_POSITION_INDICES: [usize; 3] = [0, 2, 4];

/// A track's full state estimate and covariance, as needed for NEES.
///
/// Sibling of the position-only track entries in
/// [`FrameData`]; carries the full filter state
/// and `P` so position-marginal NEES can extract the covariance block.
#[derive(Debug, Clone)]
pub struct TrackEstimate {
    /// Track identifier (same ID space as the MOT `FrameData` tracks).
    pub id: u64,
    /// Full filter state vector.
    pub state: DVector<f64>,
    /// Full state covariance `P`.
    pub covariance: DMatrix<f64>,
}

/// One frame of ground truth and full track estimates for NEES evaluation.
///
/// `gt` has the same shape as [`FrameData::gt`](crate::matching::FrameData);
/// `tracks` carry full states and covariances instead of bare positions.
#[derive(Debug, Clone)]
pub struct EstimateFrame {
    /// Ground-truth positions `[(id, [x, y, z])]`.
    pub gt: Vec<(u64, [f64; 3])>,
    /// Track estimates with full state and covariance.
    pub tracks: Vec<TrackEstimate>,
}

/// Position-marginal ANEES over a sequence of frames (dof = 3).
///
/// Each frame's estimates are projected to positions via `pos_idx`
/// (the tracker's interleaved convention is
/// [`INTERLEAVED_POSITION_INDICES`]), matched to ground truth by Euclidean
/// distance through the same [`match_frame`]
/// the MOT metrics use, at the caller's `dist_threshold`. **Population
/// identity:** when called with the same threshold a run uses for MOTA, the
/// NEES sample population is exactly the MOTA true-positive pair
/// population. Unmatched tracks and unmatched truths contribute zero
/// samples — cardinality errors are MOTA's and GOSPA's job, not NEES's.
///
/// Per matched pair, the position error `e = gt - est` is scored as
/// `e^T P_pos^{-1} e` where `P_pos` is the 3×3 **marginal** position block
/// of the full covariance extracted at `pos_idx` (the correct marginal
/// covariance of the position components — not a block of `P^{-1}`).
/// Matched pairs whose position block is not invertible are skipped and
/// contribute no sample.
///
/// # Panics
///
/// Panics if any index in `pos_idx` is out of range for a track's state or
/// covariance (a caller configuration error, not a data property).
pub fn sequence_anees(
    frames: &[EstimateFrame],
    pos_idx: [usize; 3],
    dist_threshold: f64,
) -> ConsistencyAccumulator {
    let mut acc = ConsistencyAccumulator::new(3);
    for frame in frames {
        accumulate_frame_nees(frame, pos_idx, dist_threshold, &mut acc);
    }
    acc
}

/// Match one frame and push the NEES of every matched pair into `acc`.
fn accumulate_frame_nees(
    frame: &EstimateFrame,
    pos_idx: [usize; 3],
    dist_threshold: f64,
    acc: &mut ConsistencyAccumulator,
) {
    let projected = project_frame(frame, pos_idx);
    let matched = match_frame(&projected, dist_threshold);
    for (gt_id, track_id, _) in &matched.matches {
        if let Some(sample) = matched_pair_nees(frame, *gt_id, *track_id, pos_idx) {
            acc.push(sample);
        }
    }
}

/// Project an [`EstimateFrame`] to the position-only [`FrameData`] shape the
/// MOT matcher consumes.
fn project_frame(frame: &EstimateFrame, pos_idx: [usize; 3]) -> FrameData {
    FrameData {
        gt: frame.gt.clone(),
        tracks: frame
            .tracks
            .iter()
            .map(|t| (t.id, project_position(&t.state, pos_idx)))
            .collect(),
    }
}

/// Extract the position components of a full state vector.
fn project_position(state: &DVector<f64>, pos_idx: [usize; 3]) -> [f64; 3] {
    [state[pos_idx[0]], state[pos_idx[1]], state[pos_idx[2]]]
}

/// Position-marginal NEES of one matched (ground truth, track) pair.
fn matched_pair_nees(
    frame: &EstimateFrame,
    gt_id: u64,
    track_id: u64,
    pos_idx: [usize; 3],
) -> Option<f64> {
    let (_, gt_pos) = frame.gt.iter().find(|(id, _)| *id == gt_id)?;
    let track = frame.tracks.iter().find(|t| t.id == track_id)?;
    let est_pos = project_position(&track.state, pos_idx);
    let error = DVector::from_column_slice(&[
        gt_pos[0] - est_pos[0],
        gt_pos[1] - est_pos[1],
        gt_pos[2] - est_pos[2],
    ]);
    let p_pos = position_block(&track.covariance, pos_idx);
    nees(&error, &p_pos)
}

/// Extract the 3×3 marginal position block of a full covariance at
/// `pos_idx` (rows and columns of the position components).
fn position_block(cov: &DMatrix<f64>, pos_idx: [usize; 3]) -> DMatrix<f64> {
    DMatrix::from_fn(3, 3, |i, j| cov[(pos_idx[i], pos_idx[j])])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 6-dim interleaved-state estimate at the given position with
    /// the given full covariance.
    fn interleaved_estimate(id: u64, pos: [f64; 3], covariance: DMatrix<f64>) -> TrackEstimate {
        TrackEstimate {
            id,
            state: DVector::from_column_slice(&[pos[0], 0.0, pos[1], 0.0, pos[2], 0.0]),
            covariance,
        }
    }

    /// 6×6 covariance with the given variances on the interleaved position
    /// diagonal entries and 9.0 on the velocity entries (to prove velocity
    /// terms do not leak into position-marginal NEES).
    fn diag_position_cov(px: f64, py: f64, pz: f64) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_column_slice(&[px, 9.0, py, 9.0, pz, 9.0]))
    }

    // ---- 2.2 scalars ----------------------------------------------------

    /// Spec: "NEES matches the closed-form value" — e = [1, 2],
    /// P = diag(1, 4) → 1 + 4/4 = 2, with reported dof 2.
    #[test]
    fn nees_matches_closed_form() {
        let e = DVector::from_column_slice(&[1.0, 2.0]);
        let p = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 4.0]));
        let value = nees(&e, &p).expect("nonsingular P");
        assert!((value - 2.0).abs() < 1e-12, "NEES = {value}, expected 2.0");
        // Degrees of freedom are reported alongside via the accumulator.
        let mut acc = ConsistencyAccumulator::new(e.len());
        acc.push(value);
        assert_eq!(acc.dof, 2);
        assert_eq!(acc.n, 1);
    }

    /// Spec: "NIS matches the closed-form value" — nu = [2], S = [[4]] → 1.
    #[test]
    fn nis_matches_closed_form() {
        let nu = DVector::from_column_slice(&[2.0]);
        let s = DMatrix::from_column_slice(1, 1, &[4.0]);
        let value = nis(&nu, &s).expect("nonsingular S");
        assert!((value - 1.0).abs() < 1e-12, "NIS = {value}, expected 1.0");
    }

    /// Spec: "Singular covariance is rejected" — explicit `None`, never
    /// NaN, infinity, or a panic.
    #[test]
    fn singular_covariance_rejected() {
        let e = DVector::from_column_slice(&[1.0, 2.0]);
        // Exactly singular (rank 1).
        let p_singular = DMatrix::from_column_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(nees(&e, &p_singular), None);
        // Zero innovation covariance.
        let nu = DVector::from_column_slice(&[2.0]);
        let s_zero = DMatrix::from_column_slice(1, 1, &[0.0]);
        assert_eq!(nis(&nu, &s_zero), None);
        // Dimension mismatch is also an explicit error, not a panic.
        let p_wrong = DMatrix::identity(3, 3);
        assert_eq!(nees(&e, &p_wrong), None);
        // Empty input.
        assert_eq!(nees(&DVector::zeros(0), &DMatrix::zeros(0, 0)), None);
    }

    // ---- 2.3 accumulator + per-track ------------------------------------

    /// Spec: "ANEES aggregation over a track lifetime" — K = 100 samples at
    /// n = 4, judged against chi2(400)/100 bounds.
    #[test]
    fn anees_aggregation_over_track_lifetime() {
        let samples: Vec<f64> = (0..100).map(|i| 3.0 + f64::from(i % 7) * 0.3).collect();
        let mut acc = ConsistencyAccumulator::new(4);
        for &s in &samples {
            acc.push(s);
        }
        // ANEES equals the sample mean of the per-step values.
        let expected_mean = samples.iter().sum::<f64>() / 100.0;
        assert_eq!(acc.mean().unwrap().to_bits(), expected_mean.to_bits());
        // The interval is exactly the chi2(400) bounds divided by K = 100.
        let (lo, hi) = acc.bounds(0.05).unwrap();
        let (lo400, hi400) = chi2::two_sided_bounds(400, 0.05);
        assert_eq!(lo.to_bits(), (lo400 / 100.0).to_bits());
        assert_eq!(hi.to_bits(), (hi400 / 100.0).to_bits());
        // Against offline references: chi2_400(0.025)/100, chi2_400(0.975)/100.
        assert!(((lo - 3.46481765363) / 3.46481765363).abs() < 1e-8);
        assert!(((hi - 4.57305481966) / 4.57305481966).abs() < 1e-8);
        // Mean 3.9 sits inside [3.4648, 4.5731].
        assert_eq!(acc.verdict(0.05), Some(ConsistencyVerdict::Consistent));
    }

    /// Verdict directions per Bar-Shalom: above the upper bound the
    /// covariance is too small (Overconfident); below the lower bound it is
    /// inflated (Underconfident).
    #[test]
    fn verdict_directions_match_bar_shalom() {
        // dof 4, K = 100: mean interval is [3.4648, 4.5731].
        let mut high = ConsistencyAccumulator::new(4);
        let mut low = ConsistencyAccumulator::new(4);
        for _ in 0..100 {
            high.push(4.6); // mean 4.6 > 4.5731 → errors too large → P too small
            low.push(3.4); // mean 3.4 < 3.4648 → errors too small → P inflated
        }
        assert_eq!(high.verdict(0.05), Some(ConsistencyVerdict::Overconfident));
        assert_eq!(low.verdict(0.05), Some(ConsistencyVerdict::Underconfident));
    }

    #[test]
    fn empty_accumulator_reports_none() {
        let acc = ConsistencyAccumulator::new(3);
        assert_eq!(acc.mean(), None);
        assert_eq!(acc.bounds(0.05), None);
        assert_eq!(acc.verdict(0.05), None);
    }

    /// Spec: "Per-track verdicts isolate the inconsistent track" — honest
    /// track A vs track B whose covariances are scaled down by 100
    /// (covariance × 0.01 → NEES × 100).
    #[test]
    fn per_track_verdicts_isolate_overconfident_track() {
        let p_honest = DMatrix::<f64>::identity(3, 3);
        let p_shrunk = &p_honest * 0.01;
        let mut per_track = PerTrackConsistency::new(3);
        // 50 deterministic errors whose squared norms alternate 2.5 / 3.5:
        // honest ANEES = 3.0, inside chi2(150)/50 = [2.3597, 3.7160].
        for i in 0..50 {
            let e = if i % 2 == 0 {
                DVector::from_column_slice(&[2.5f64.sqrt(), 0.0, 0.0])
            } else {
                DVector::from_column_slice(&[0.0, 3.5f64.sqrt(), 0.0])
            };
            per_track.push(1, nees(&e, &p_honest).unwrap());
            per_track.push(2, nees(&e, &p_shrunk).unwrap());
        }
        let verdicts = per_track.verdicts(0.05);
        assert_eq!(
            verdicts,
            vec![
                (1, ConsistencyVerdict::Consistent),
                (2, ConsistencyVerdict::Overconfident),
            ]
        );
        // Track B's ANEES is 100× track A's (≈ 300, far above the bound).
        let anees_b = per_track.accumulator(2).unwrap().mean().unwrap();
        assert!((anees_b - 300.0).abs() < 1e-9, "ANEES B = {anees_b}");
        assert_eq!(per_track.len(), 2);
        assert!(!per_track.is_empty());
    }

    // ---- 3.1 / 3.2 sequence_anees ----------------------------------------

    /// Hand-built two-frame sequence: frame 1 has e = [1,0,0] against
    /// P_pos = I (NEES 1), frame 2 has e = [0,2,0] against
    /// P_pos = diag(1,2,1) (NEES 4/2 = 2) → ANEES 1.5 over 2 samples.
    #[test]
    fn sequence_anees_matches_hand_computed_value() {
        let frames = vec![
            EstimateFrame {
                gt: vec![(1, [1.0, 0.0, 0.0])],
                tracks: vec![interleaved_estimate(
                    10,
                    [0.0, 0.0, 0.0],
                    diag_position_cov(1.0, 1.0, 1.0),
                )],
            },
            EstimateFrame {
                gt: vec![(1, [0.0, 2.0, 0.0])],
                tracks: vec![interleaved_estimate(
                    10,
                    [0.0, 0.0, 0.0],
                    diag_position_cov(1.0, 2.0, 1.0),
                )],
            },
        ];
        let acc = sequence_anees(&frames, INTERLEAVED_POSITION_INDICES, 5.0);
        assert_eq!(acc.n, 2);
        assert_eq!(acc.dof, 3);
        let anees = acc.mean().unwrap();
        assert!((anees - 1.5).abs() < 1e-12, "ANEES = {anees}, expected 1.5");
    }

    /// Matched-pairs-only: unmatched tracks and unmatched truths contribute
    /// zero samples — cardinality errors are MOTA's and GOSPA's job.
    #[test]
    fn sequence_anees_counts_matched_pairs_only() {
        let cov = diag_position_cov(1.0, 1.0, 1.0);
        // No pair within the threshold: zero samples.
        let unmatched = vec![EstimateFrame {
            gt: vec![(1, [0.0, 0.0, 0.0]), (2, [50.0, 0.0, 0.0])],
            tracks: vec![interleaved_estimate(7, [100.0, 0.0, 0.0], cov.clone())],
        }];
        let acc = sequence_anees(&unmatched, INTERLEAVED_POSITION_INDICES, 5.0);
        assert_eq!(acc.n, 0);
        assert_eq!(acc.mean(), None);
        // One matched pair plus an unmatched truth and an unmatched track:
        // exactly one sample.
        let mixed = vec![EstimateFrame {
            gt: vec![(1, [0.0, 0.0, 0.0]), (2, [50.0, 0.0, 0.0])],
            tracks: vec![
                interleaved_estimate(7, [1.0, 0.0, 0.0], cov.clone()),
                interleaved_estimate(8, [200.0, 0.0, 0.0], cov),
            ],
        }];
        let acc = sequence_anees(&mixed, INTERLEAVED_POSITION_INDICES, 5.0);
        assert_eq!(acc.n, 1);
        // The one sample is e = [1,0,0] against P_pos = I → NEES 1.
        assert!((acc.mean().unwrap() - 1.0).abs() < 1e-12);
    }

    /// Position-marginal NEES must come from inverting the extracted 3×3
    /// marginal block of P, not from the position submatrix of P^{-1}.
    /// With per-axis covariance [[2, 1], [1, 2]] over (pos, vel), the
    /// marginal position variance is 2 (→ NEES 0.5 for e = [1,0,0]) while
    /// the (pos,pos) entry of the full inverse is 2/3 (→ wrong value 2/3).
    #[test]
    fn sequence_anees_uses_marginal_position_block() {
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for axis in 0..3 {
            let (i, j) = (2 * axis, 2 * axis + 1);
            cov[(i, i)] = 2.0;
            cov[(j, j)] = 2.0;
            cov[(i, j)] = 1.0;
            cov[(j, i)] = 1.0;
        }
        // Sanity: the wrong method (submatrix of the full inverse) would
        // score e = [1,0,0] as 2/3, not 0.5.
        let full_inv = cov.clone().try_inverse().unwrap();
        assert!((full_inv[(0, 0)] - 2.0 / 3.0).abs() < 1e-12);
        let frames = vec![EstimateFrame {
            gt: vec![(1, [1.0, 0.0, 0.0])],
            tracks: vec![interleaved_estimate(10, [0.0, 0.0, 0.0], cov)],
        }];
        let acc = sequence_anees(&frames, INTERLEAVED_POSITION_INDICES, 5.0);
        assert_eq!(acc.n, 1);
        let value = acc.mean().unwrap();
        assert!(
            (value - 0.5).abs() < 1e-12,
            "expected marginal-block NEES 0.5, got {value}"
        );
        assert!(
            (value - 2.0 / 3.0).abs() > 0.1,
            "value must not be the inverse-of-full-P submatrix result"
        );
    }

    /// A matched pair whose position block is singular is skipped rather
    /// than producing NaN or a panic.
    #[test]
    fn sequence_anees_skips_singular_position_block() {
        let frames = vec![EstimateFrame {
            gt: vec![(1, [1.0, 0.0, 0.0])],
            tracks: vec![interleaved_estimate(
                10,
                [0.0, 0.0, 0.0],
                diag_position_cov(0.0, 1.0, 1.0),
            )],
        }];
        let acc = sequence_anees(&frames, INTERLEAVED_POSITION_INDICES, 5.0);
        assert_eq!(acc.n, 0);
    }

    // ---- 2.5 determinism --------------------------------------------------

    /// Recorded per-track NEES error stream: `(track_id, error)` pairs.
    type ErrorStream = Vec<(u64, DVector<f64>)>;
    /// Recorded NIS innovation stream: `(innovation, S)` pairs.
    type InnovationStream = Vec<(DVector<f64>, DMatrix<f64>)>;

    /// Fixed recorded error/innovation streams for the determinism test.
    fn recorded_streams() -> (ErrorStream, InnovationStream) {
        let track_ids = [11u64, 3, 7, 42, 19, 5, 28, 1];
        let mut errors = Vec::new();
        for (t, &id) in track_ids.iter().enumerate() {
            for k in 0..20 {
                let a = ((t * 31 + k * 17) % 13) as f64 / 13.0;
                let b = ((t * 7 + k * 29) % 11) as f64 / 11.0;
                errors.push((
                    id,
                    DVector::from_column_slice(&[1.5 * a - 0.7, 2.0 * b - 1.0, a * b]),
                ));
            }
        }
        let innovations = (0..50)
            .map(|k| {
                let v = f64::from(k % 9) * 0.25 - 1.0;
                (
                    DVector::from_column_slice(&[v, -0.5 * v]),
                    DMatrix::from_column_slice(2, 2, &[1.0, 0.2, 0.2, 0.8]),
                )
            })
            .collect();
        (errors, innovations)
    }

    /// One full evaluation pass over the recorded streams; `reverse_tracks`
    /// permutes the per-track insertion order (per-track sample order
    /// preserved) to exercise hash-map insertion-order independence.
    #[allow(clippy::type_complexity)]
    fn evaluate_recorded(
        reverse_tracks: bool,
    ) -> (
        u64,
        u64,
        u64,
        Option<ConsistencyVerdict>,
        Vec<(u64, ConsistencyVerdict)>,
        u64,
    ) {
        let (errors, innovations) = recorded_streams();
        let p = DMatrix::<f64>::identity(3, 3);
        // Global NEES accumulator: fixed recorded order.
        let mut global = ConsistencyAccumulator::new(3);
        for (_, e) in &errors {
            global.push(nees(e, &p).unwrap());
        }
        // Per-track path, with optionally permuted track interleaving.
        let mut per_track = PerTrackConsistency::new(3);
        let ordered: Vec<&(u64, DVector<f64>)> = if reverse_tracks {
            let mut v: Vec<_> = errors.iter().collect();
            v.sort_by_key(|(id, _)| std::cmp::Reverse(*id));
            v
        } else {
            errors.iter().collect()
        };
        for (id, e) in ordered {
            per_track.push(*id, nees(e, &p).unwrap());
        }
        // NIS stream.
        let mut nis_acc = ConsistencyAccumulator::new(2);
        for (nu, s) in &innovations {
            nis_acc.push(nis(nu, s).unwrap());
        }
        let (lo, hi) = global.bounds(chi2::DEFAULT_ALPHA).unwrap();
        (
            global.mean().unwrap().to_bits(),
            lo.to_bits(),
            hi.to_bits(),
            global.verdict(chi2::DEFAULT_ALPHA),
            per_track.verdicts(chi2::DEFAULT_ALPHA),
            nis_acc.mean().unwrap().to_bits(),
        )
    }

    /// Spec: "Repeated evaluation is bitwise identical" — the same recorded
    /// sequences evaluated twice in-process produce bitwise-identical
    /// statistics, bounds, and verdicts; the per-track path is additionally
    /// independent of hash-map insertion order (sorted-ID iteration).
    #[test]
    fn repeated_evaluation_is_bitwise_identical() {
        let first = evaluate_recorded(false);
        let second = evaluate_recorded(false);
        assert_eq!(first, second);
        // Per-track results survive a permuted insertion order: each track's
        // own sample order is preserved, so sums are bit-identical and the
        // sorted-ID iteration removes any hash-order dependence.
        let permuted = evaluate_recorded(true);
        assert_eq!(first.4, permuted.4);
        // sequence_anees is bitwise stable too.
        let frames = vec![EstimateFrame {
            gt: vec![(1, [1.0, 0.0, 0.0]), (2, [4.0, 4.0, 0.0])],
            tracks: vec![
                interleaved_estimate(10, [0.0, 0.0, 0.0], diag_position_cov(1.0, 2.0, 3.0)),
                interleaved_estimate(20, [4.0, 3.0, 0.0], diag_position_cov(2.0, 1.0, 1.0)),
            ],
        }];
        let a = sequence_anees(&frames, INTERLEAVED_POSITION_INDICES, 5.0);
        let b = sequence_anees(&frames, INTERLEAVED_POSITION_INDICES, 5.0);
        assert_eq!(a.n, b.n);
        assert_eq!(a.sum.to_bits(), b.sum.to_bits());
        assert_eq!(
            a.verdict(chi2::DEFAULT_ALPHA),
            b.verdict(chi2::DEFAULT_ALPHA)
        );
    }

    // ---- 2.4 Monte-Carlo falsifiability bracket ---------------------------

    /// Task 2.4: seeded Monte-Carlo falsifiability bracket on a
    /// linear-Gaussian reference problem — the true-Q/R filter passes both
    /// the ANEES and average-NIS gates; a covariance lie fails in the
    /// correct direction. The NIS legs consume [`crate::consistency::nis`]
    /// inputs from `UpdateOutcome` streams alone, through KF, UKF, and CKF.
    mod monte_carlo {
        use nalgebra::{DMatrix, DVector};
        use rand::prelude::*;
        use thresh_filter::ckf::CubatureKalmanFilter;
        use thresh_filter::kf::KalmanFilter;
        use thresh_filter::traits::{LinearModel, MotionModel};
        use thresh_filter::ukf::UnscentedKalmanFilter;

        use crate::consistency::{ConsistencyAccumulator, ConsistencyVerdict, chi2, nees};

        const DT: f64 = 1.0;
        /// True white-acceleration process-noise intensity (accel variance).
        const Q_TRUE: f64 = 0.25;
        /// True measurement-noise variance.
        const R_TRUE: f64 = 1.0;
        /// Monte-Carlo runs (spec: "50+ runs or equivalent length").
        const RUNS: usize = 50;
        /// Steps per run.
        const STEPS: usize = 60;
        /// NEES is recorded every `NEES_STRIDE` steps. Posterior errors are
        /// time-correlated (unlike innovations, which are provably white for
        /// a matched linear-Gaussian filter), so decimating within each run
        /// restores the near-independence the chi-squared time-average test
        /// assumes; across runs samples are independent by construction.
        const NEES_STRIDE: usize = 20;
        /// Fixed seed: the bracket is deterministic, not statistical luck.
        /// Observed at this seed (all comfortably inside / outside their
        /// gates — bounds for ANEES are chi2(300)/150 = [1.6927, 2.3325],
        /// for average NIS chi2(3000)/3000 = [0.9500, 1.0512]):
        /// honest ANEES ≈ 2.0970, honest average NIS ≈ 1.0127 (KF/UKF/CKF
        /// agree to ~1e-9), Q × 0.01 ANEES ≈ 95.2, Q × 100 ANEES ≈ 1.2475.
        const SEED: u64 = 7;

        /// 1D nearly-constant-velocity model `[pos, vel]` with
        /// piecewise-constant white-acceleration process noise
        /// `Q = q · G Gᵀ`, `G = [dt²/2, dt]ᵀ` — the textbook Bar-Shalom
        /// reference problem. `q` is the *filter's assumed* acceleration
        /// variance; truth is always generated with [`Q_TRUE`].
        struct Cv1d {
            q: f64,
        }

        fn f_matrix(dt: f64) -> DMatrix<f64> {
            DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0])
        }

        fn g_vector(dt: f64) -> DVector<f64> {
            DVector::from_column_slice(&[0.5 * dt * dt, dt])
        }

        impl MotionModel for Cv1d {
            fn state_dim(&self) -> usize {
                2
            }
            fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
                f_matrix(dt) * state
            }
            fn jacobian(&self, _state: &DVector<f64>, dt: f64) -> DMatrix<f64> {
                f_matrix(dt)
            }
            fn process_noise(&self, dt: f64) -> DMatrix<f64> {
                let g = g_vector(dt);
                self.q * &g * g.transpose()
            }
        }

        impl LinearModel for Cv1d {
            fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
                f_matrix(dt)
            }
        }

        fn h_matrix() -> DMatrix<f64> {
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0])
        }

        fn r_matrix() -> DMatrix<f64> {
            DMatrix::from_element(1, 1, R_TRUE)
        }

        /// Standard normal via Box–Muller on the seeded uniform stream
        /// (`1 - u` keeps the log argument in `(0, 1]`).
        fn gauss(rng: &mut StdRng) -> f64 {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            (-2.0 * (1.0 - u1).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }

        /// Initial estimate sampled from `N(truth0, P0)` with
        /// `P0 = diag(1, 0.25)`, so NEES is consistent from the first step.
        fn sampled_init(rng: &mut StdRng) -> (DVector<f64>, DMatrix<f64>) {
            let p0 = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 0.25]));
            let x0 = DVector::from_column_slice(&[gauss(rng), 1.0 + 0.5 * gauss(rng)]);
            (x0, p0)
        }

        /// One seeded linear-Gaussian trajectory from truth `[0, 1]`: the
        /// truth state after each step and the corresponding position
        /// measurement, generated with the true `Q`/`R` (process noise is
        /// `G·a`, `a ~ N(0, Q_TRUE)`, whose covariance is exactly
        /// `Cv1d::process_noise` at `q = Q_TRUE`).
        fn simulate_run(rng: &mut StdRng) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
            let f = f_matrix(DT);
            let g = g_vector(DT);
            let mut x = DVector::from_column_slice(&[0.0, 1.0]);
            let mut truths = Vec::with_capacity(STEPS);
            let mut zs = Vec::with_capacity(STEPS);
            for _ in 0..STEPS {
                x = &f * &x + &g * (Q_TRUE.sqrt() * gauss(rng));
                zs.push(DVector::from_element(1, x[0] + R_TRUE.sqrt() * gauss(rng)));
                truths.push(x.clone());
            }
            (truths, zs)
        }

        /// ANEES leg: run [`RUNS`] seeded trajectories through a linear KF
        /// whose assumed process noise is `q_scale ×` the true `Q`,
        /// recording full-state posterior NEES (dof 2) every
        /// [`NEES_STRIDE`] steps.
        fn kf_anees(q_scale: f64) -> ConsistencyAccumulator {
            let mut rng = StdRng::seed_from_u64(SEED);
            let model = Cv1d {
                q: Q_TRUE * q_scale,
            };
            let mut acc = ConsistencyAccumulator::new(2);
            for _ in 0..RUNS {
                let (x0, p0) = sampled_init(&mut rng);
                let (truths, zs) = simulate_run(&mut rng);
                let mut kf = KalmanFilter::new(x0, p0);
                for (k, (truth, z)) in truths.iter().zip(&zs).enumerate() {
                    kf.predict_dt(&model, DT);
                    kf.update(z, &h_matrix(), &r_matrix());
                    if (k + 1) % NEES_STRIDE == 0 {
                        acc.push(nees(&(truth - &kf.x), &kf.p).expect("P nonsingular"));
                    }
                }
            }
            acc
        }

        /// Average-NIS leg over the same seeded measurement streams.
        ///
        /// `run_filter` receives only the sampled initial estimate and the
        /// measurement sequence and returns the per-update NIS taken from
        /// each `UpdateOutcome` — the truth states never cross this call
        /// boundary, so **no ground-truth argument exists anywhere in the
        /// NIS path** (spec: "NIS from filter diagnostics only").
        fn average_nis<Run>(mut run_filter: Run) -> ConsistencyAccumulator
        where
            Run: FnMut(DVector<f64>, DMatrix<f64>, &[DVector<f64>]) -> Vec<f64>,
        {
            let mut rng = StdRng::seed_from_u64(SEED);
            let mut acc = ConsistencyAccumulator::new(1);
            for _ in 0..RUNS {
                let (x0, p0) = sampled_init(&mut rng);
                let (_truths, zs) = simulate_run(&mut rng);
                for sample in run_filter(x0, p0, &zs) {
                    acc.push(sample);
                }
            }
            acc
        }

        /// KF NIS stream: per-update `UpdateOutcome::nis` only.
        fn kf_nis_run(x0: DVector<f64>, p0: DMatrix<f64>, zs: &[DVector<f64>]) -> Vec<f64> {
            let model = Cv1d { q: Q_TRUE };
            let mut kf = KalmanFilter::new(x0, p0);
            zs.iter()
                .map(|z| {
                    kf.predict_dt(&model, DT);
                    kf.update(z, &h_matrix(), &r_matrix()).nis
                })
                .collect()
        }

        /// UKF NIS stream (spec: "Diagnostics feed NIS directly").
        fn ukf_nis_run(x0: DVector<f64>, p0: DMatrix<f64>, zs: &[DVector<f64>]) -> Vec<f64> {
            let model = Cv1d { q: Q_TRUE };
            let mut ukf = UnscentedKalmanFilter::with_defaults(x0, p0);
            zs.iter()
                .map(|z| {
                    ukf.predict(&model, DT);
                    ukf.update_linear(z, &h_matrix(), &r_matrix()).nis
                })
                .collect()
        }

        /// CKF NIS stream (spec: "CKF diagnostics feed NIS").
        fn ckf_nis_run(x0: DVector<f64>, p0: DMatrix<f64>, zs: &[DVector<f64>]) -> Vec<f64> {
            let model = Cv1d { q: Q_TRUE };
            let mut ckf = CubatureKalmanFilter::new(x0, p0);
            zs.iter()
                .map(|z| {
                    ckf.predict(&model, DT);
                    ckf.update_linear(z, &h_matrix(), &r_matrix()).nis
                })
                .collect()
        }

        /// Spec: "Consistent filter passes" — with the true `Q`/`R` the
        /// ANEES and the average NIS both sit inside the two-sided 95%
        /// bounds. The NIS legs run through KF, UKF, and CKF on
        /// `UpdateOutcome` streams alone (specs: "NIS from filter
        /// diagnostics only", "Diagnostics feed NIS directly", "CKF
        /// diagnostics feed NIS").
        #[test]
        fn consistent_filter_passes() {
            let anees = kf_anees(1.0);
            assert_eq!(anees.n, RUNS * (STEPS / NEES_STRIDE));
            assert_eq!(
                anees.verdict(chi2::DEFAULT_ALPHA),
                Some(ConsistencyVerdict::Consistent),
                "ANEES {:?} outside bounds {:?}",
                anees.mean(),
                anees.bounds(chi2::DEFAULT_ALPHA)
            );

            for (name, acc) in [
                ("KF", average_nis(kf_nis_run)),
                ("UKF", average_nis(ukf_nis_run)),
                ("CKF", average_nis(ckf_nis_run)),
            ] {
                assert_eq!(acc.n, RUNS * STEPS, "{name} sample count");
                assert_eq!(acc.dof, 1, "{name} measurement dof");
                assert_eq!(
                    acc.verdict(chi2::DEFAULT_ALPHA),
                    Some(ConsistencyVerdict::Consistent),
                    "{name} average NIS {:?} outside bounds {:?}",
                    acc.mean(),
                    acc.bounds(chi2::DEFAULT_ALPHA)
                );
            }
        }

        /// Spec: "Overconfident filter fails high" — `Q × 0.01` understates
        /// the true error, so ANEES exceeds the upper bound on identical
        /// data (a covariance lie MOT metrics cannot see).
        #[test]
        fn overconfident_filter_fails_high() {
            let anees = kf_anees(0.01);
            let (_, hi) = anees.bounds(chi2::DEFAULT_ALPHA).unwrap();
            let mean = anees.mean().unwrap();
            assert!(mean > hi, "ANEES {mean} must exceed the upper bound {hi}");
            assert_eq!(
                anees.verdict(chi2::DEFAULT_ALPHA),
                Some(ConsistencyVerdict::Overconfident)
            );
        }

        /// Spec: "Underconfident filter fails low" — `Q × 100` inflates the
        /// reported covariance, so ANEES falls below the lower bound.
        #[test]
        fn underconfident_filter_fails_low() {
            let anees = kf_anees(100.0);
            let (lo, _) = anees.bounds(chi2::DEFAULT_ALPHA).unwrap();
            let mean = anees.mean().unwrap();
            assert!(
                mean < lo,
                "ANEES {mean} must fall below the lower bound {lo}"
            );
            assert_eq!(
                anees.verdict(chi2::DEFAULT_ALPHA),
                Some(ConsistencyVerdict::Underconfident)
            );
        }
    }
}
