//! Class-specific track heads: target class -> motion model + noise params + policies.

use thresh_core::orbital::GravityModel;
use thresh_core::track::TargetClass;
use thresh_filter::models::ballistic_reentry::BallisticReentry;
use thresh_filter::models::cv::ConstantVelocity;
use thresh_filter::models::kepler_j2::KeplerJ2;
use thresh_filter::traits::MotionModel;

use crate::lifecycle::{ConfirmationPolicy, DeletionPolicy};

/// Motion-model selector for a track head (design Decision 6 of the
/// `orbital-ballistic-filter-models` change).
///
/// A head *implies* a motion model; this enum makes the selection explicit
/// so the tracker can dispatch per track instead of hardcoding a single
/// constant-velocity model. The enum keeps [`TrackHead`] a plain data record
/// (`Clone + Debug`-friendly) — the model itself is built on demand by
/// [`TrackHead::build_model`], which combines the variant with the head's
/// `process_noise_sigma`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeadModel {
    /// Constant-velocity linear model on the 6D interleaved state
    /// (existing behavior; the tracker keeps a linear fast path for it).
    Cv,
    /// 7D ballistic reentry model `[x, vx, y, vy, z, vz, β]`
    /// ([`BallisticReentry`]) with an estimated ballistic coefficient.
    Reentry {
        /// β random-walk intensity (kg/m² per √s).
        sigma_beta: f64,
        /// β value new tracks are born with (kg/m²).
        beta_init: f64,
    },
    /// 6D Kepler+J2 orbital model ([`KeplerJ2`]) on the interleaved ECI
    /// state.
    KeplerJ2 {
        /// RK4 sub-step ceiling (s) for the orbital propagation.
        max_step_s: f64,
    },
}

/// Configuration for a class-specific tracking head.
#[derive(Debug, Clone)]
pub struct TrackHead {
    /// Target class this head handles.
    pub class: TargetClass,
    /// State dimension for this class.
    pub state_dim: usize,
    /// Process noise spectral density.
    pub process_noise_sigma: f64,
    /// Initial covariance diagonal values.
    pub initial_covariance: Vec<f64>,
    /// Confirmation policy.
    pub confirmation: ConfirmationPolicy,
    /// Deletion policy.
    pub deletion: DeletionPolicy,
    /// Motion model this head's tracks run (default [`HeadModel::Cv`]).
    pub model: HeadModel,
}

impl TrackHead {
    /// Default aircraft tracking head (CV model, 6D state).
    pub fn aircraft() -> Self {
        Self {
            class: TargetClass::Aircraft,
            state_dim: 6,
            process_noise_sigma: 5.0, // 5 m/s² acceleration noise
            initial_covariance: vec![1000.0, 100.0, 1000.0, 100.0, 1000.0, 100.0],
            confirmation: ConfirmationPolicy::new(3, 5),
            deletion: DeletionPolicy::new(5),
            model: HeadModel::Cv,
        }
    }

    /// Default ballistic missile tracking head: the 7D ballistic reentry
    /// model `[x, vx, y, vy, z, vz, β]` (design Decision 6, **BREAKING**:
    /// formerly a 9D constant-acceleration configuration).
    ///
    /// Tuning: `process_noise_sigma` is the reentry model's
    /// unmodeled-acceleration PSD (lift/attitude effects, Decision 4 default
    /// 5 m/s²). β is born at 1000 kg/m² under a wide prior (variance 1e7,
    /// σ ≈ 3200 kg/m² — β is a priori unknown within ~[50, 10000] kg/m²).
    /// `sigma_beta = 50 kg/m²·s^-½` keeps β variance growth over a full
    /// ~20 min midcourse (50²·1200 s = 3e6) within the initial prior's order
    /// of magnitude, so exo-atmospheric unobservability re-inflates rather
    /// than explodes the β uncertainty. Confirmation stays 2-of-3 /
    /// delete-3: reentry passes are short.
    ///
    /// Birth priors (calibrated on the `ballistic-mrbm` benchmark, task
    /// 6.5): position (1 km)² — ballistic tracks are born from long-range
    /// radar detections whose angle-noise-induced error is km-scale — and
    /// velocity 1e7 m²/s², the unknown-direction prior `v_max²/3` at
    /// `v_max ≈ 5.5 km/s` (MRBM burnout ≈ 3.5 km/s through ICBM-RV entry
    /// ≈ 7 km/s), mirroring the orbital head's circular-orbit prior
    /// rationale. A tight velocity prior is fatal here: a zero-velocity
    /// birth with σ_v ≈ 32 m/s gives the first re-association a velocity
    /// gain of ~1e-3, the track freezes at its birth position while the
    /// target moves km/scan, and the 2-of-3 window never completes.
    pub fn ballistic() -> Self {
        Self {
            class: TargetClass::Ballistic,
            state_dim: 7,
            process_noise_sigma: 5.0,
            initial_covariance: vec![1.0e6, 1.0e7, 1.0e6, 1.0e7, 1.0e6, 1.0e7, 1.0e7],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(3),
            model: HeadModel::Reentry {
                sigma_beta: 50.0,
                beta_init: 1000.0,
            },
        }
    }

    /// Orbital target head: 6D Kepler+J2 model on the interleaved ECI state
    /// (design Decision 6 of `orbital-ballistic-filter-models`).
    ///
    /// Priors: (1 km)² position. The velocity entries are the static
    /// circular-orbit prior for LEO — a birth position fixes only 3 of 6
    /// states, and the unknown velocity has magnitude ≈ √(μ/r) ≈ 7.8 km/s
    /// in an unknown direction, giving per-axis variance v_c²/3 ≈ 2.0e7
    /// m²/s² (task 5.5 resolution; `birth_track` refines this from the
    /// measured radius when the detection is a plausible ECI position).
    /// `process_noise_sigma` is the Kepler+J2 model's unmodeled-acceleration
    /// PSD (drag/SRP/higher harmonics, Decision 3 default 1e-3 m/s²).
    /// Confirmation 2-of-3 with a longer deletion window (5): pass-edge
    /// dropouts are routine at sparse orbital sampling.
    pub fn orbital() -> Self {
        Self {
            class: TargetClass::Orbital,
            state_dim: 6,
            process_noise_sigma: 1e-3,
            initial_covariance: vec![1.0e6, 2.0e7, 1.0e6, 2.0e7, 1.0e6, 2.0e7],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(5),
            model: HeadModel::KeplerJ2 { max_step_s: 10.0 },
        }
    }

    /// Default UAV tracking head (CTRV model, 5D state).
    pub fn uav() -> Self {
        Self {
            class: TargetClass::Uav,
            state_dim: 5,
            process_noise_sigma: 3.0,
            initial_covariance: vec![100.0, 100.0, 1.0, 50.0, 0.5],
            confirmation: ConfirmationPolicy::new(3, 5),
            deletion: DeletionPolicy::new(10),
            model: HeadModel::Cv,
        }
    }

    /// Default unknown target head (CV model, conservative).
    pub fn unknown() -> Self {
        Self {
            class: TargetClass::Unknown,
            state_dim: 6,
            process_noise_sigma: 10.0,
            initial_covariance: vec![10000.0, 1000.0, 10000.0, 1000.0, 10000.0, 1000.0],
            confirmation: ConfirmationPolicy::new(3, 5),
            deletion: DeletionPolicy::new(5),
            model: HeadModel::Cv,
        }
    }

    /// Build the motion model this head's tracks run, combining the
    /// [`HeadModel`] selector with the head's `process_noise_sigma`
    /// (design Decision 6). Earth-orbit/reentry heads use
    /// [`GravityModel::EARTH_WGS84`].
    pub fn build_model(&self) -> Box<dyn MotionModel> {
        match self.model {
            HeadModel::Cv => Box::new(ConstantVelocity::new(self.process_noise_sigma)),
            HeadModel::Reentry { sigma_beta, .. } => {
                let mut model = BallisticReentry::new(GravityModel::EARTH_WGS84, sigma_beta);
                model.sigma_accel = self.process_noise_sigma;
                Box::new(model)
            }
            HeadModel::KeplerJ2 { max_step_s } => {
                let mut model = KeplerJ2::new(GravityModel::EARTH_WGS84);
                model.max_step_s = max_step_s;
                model.sigma_accel = self.process_noise_sigma;
                Box::new(model)
            }
        }
    }

    // --- Automotive heads (automotive-tracking-pipeline, task 2.5) ---------
    //
    // All automotive heads keep `state_dim = 6` ([x,vx,y,vy,z,vz]) so they are
    // compatible with the position tracker's 3x6 observation matrix. The
    // class-specific motion priors are expressed through:
    // - `process_noise_sigma` (m/s² agility: motorcycle > car > truck/bus >
    //   bicycle/pedestrian),
    // - the velocity entries of `initial_covariance` (speed prior: ~40 m/s for
    //   car/motorcycle -> var 225; ~30 m/s truck/bus -> 100; ~8 m/s bicycle ->
    //   25; ~6 m/s pedestrian -> 4), and
    // - confirmation/deletion (pedestrians appear/occlude quickly, so they
    //   confirm fast and delete fast).
    // Position priors assume automotive-grade detections (~2 m std -> var 4);
    // the vertical channel is tight (ground vehicles stay near the road).

    /// Passenger car head.
    pub fn car() -> Self {
        Self {
            class: TargetClass::Car,
            state_dim: 6,
            process_noise_sigma: 2.0,
            initial_covariance: vec![4.0, 225.0, 4.0, 225.0, 1.0, 1.0],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(5),
            model: HeadModel::Cv,
        }
    }

    /// Truck head (heavier, less agile than a car).
    pub fn truck() -> Self {
        Self {
            class: TargetClass::Truck,
            state_dim: 6,
            process_noise_sigma: 1.5,
            initial_covariance: vec![4.0, 100.0, 4.0, 100.0, 1.0, 1.0],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(5),
            model: HeadModel::Cv,
        }
    }

    /// Bus head (largest, least agile road vehicle).
    pub fn bus() -> Self {
        Self {
            class: TargetClass::Bus,
            state_dim: 6,
            process_noise_sigma: 1.0,
            initial_covariance: vec![4.0, 100.0, 4.0, 100.0, 1.0, 1.0],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(5),
            model: HeadModel::Cv,
        }
    }

    /// Motorcycle head (small and highly agile).
    pub fn motorcycle() -> Self {
        Self {
            class: TargetClass::Motorcycle,
            state_dim: 6,
            process_noise_sigma: 3.0,
            initial_covariance: vec![4.0, 225.0, 4.0, 225.0, 1.0, 1.0],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(4),
            model: HeadModel::Cv,
        }
    }

    /// Bicycle head (slow, moderately agile).
    pub fn bicycle() -> Self {
        Self {
            class: TargetClass::Bicycle,
            state_dim: 6,
            process_noise_sigma: 1.5,
            initial_covariance: vec![4.0, 25.0, 4.0, 25.0, 1.0, 1.0],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(4),
            model: HeadModel::Cv,
        }
    }

    /// Pedestrian head (slowest; confirms and deletes quickly because
    /// pedestrians enter, occlude, and leave the scene rapidly).
    pub fn pedestrian() -> Self {
        Self {
            class: TargetClass::Pedestrian,
            state_dim: 6,
            process_noise_sigma: 1.0,
            initial_covariance: vec![4.0, 4.0, 4.0, 4.0, 1.0, 1.0],
            confirmation: ConfirmationPolicy::new(2, 2),
            deletion: DeletionPolicy::new(3),
            model: HeadModel::Cv,
        }
    }
}

/// Registry of class-specific track heads.
#[derive(Debug, Clone)]
pub struct HeadRegistry {
    pub heads: Vec<TrackHead>,
}

impl Default for HeadRegistry {
    fn default() -> Self {
        Self {
            heads: vec![
                TrackHead::aircraft(),
                TrackHead::ballistic(),
                TrackHead::orbital(),
                TrackHead::uav(),
                TrackHead::car(),
                TrackHead::truck(),
                TrackHead::bus(),
                TrackHead::motorcycle(),
                TrackHead::bicycle(),
                TrackHead::pedestrian(),
                TrackHead::unknown(),
            ],
        }
    }
}

impl HeadRegistry {
    /// Look up the track head for a given class.
    pub fn get(&self, class: TargetClass) -> &TrackHead {
        self.heads
            .iter()
            .find(|h| h.class == class)
            .unwrap_or_else(|| {
                self.heads
                    .iter()
                    .find(|h| h.class == TargetClass::Unknown)
                    .expect("No Unknown head in registry")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Velocity-prior variance of a 6D head (x,vx,y,vy,z,vz → index 1).
    fn vel_prior(head: &TrackHead) -> f64 {
        head.initial_covariance[1]
    }

    #[test]
    fn default_registry_contains_all_automotive_classes() {
        let reg = HeadRegistry::default();
        for class in [
            TargetClass::Car,
            TargetClass::Truck,
            TargetClass::Bus,
            TargetClass::Motorcycle,
            TargetClass::Bicycle,
            TargetClass::Pedestrian,
        ] {
            assert_eq!(reg.get(class).class, class, "missing head for {class:?}");
        }
    }

    #[test]
    fn aerospace_heads_still_present() {
        let reg = HeadRegistry::default();
        for class in [
            TargetClass::Aircraft,
            TargetClass::Ballistic,
            TargetClass::Uav,
            TargetClass::Unknown,
        ] {
            assert_eq!(reg.get(class).class, class);
        }
    }

    #[test]
    fn automotive_heads_are_position_tracker_compatible() {
        // The ENU position tracker births tracks against a 3x6 observation
        // matrix, so every automotive head MUST be 6-dimensional.
        for head in [
            TrackHead::car(),
            TrackHead::truck(),
            TrackHead::bus(),
            TrackHead::motorcycle(),
            TrackHead::bicycle(),
            TrackHead::pedestrian(),
        ] {
            assert_eq!(head.state_dim, 6, "{:?} head must be 6D", head.class);
            assert_eq!(head.initial_covariance.len(), 6);
        }
    }

    #[test]
    fn speed_priors_are_ordered_by_class() {
        // pedestrian < bicycle < truck/bus < car/motorcycle
        assert!(vel_prior(&TrackHead::pedestrian()) < vel_prior(&TrackHead::bicycle()));
        assert!(vel_prior(&TrackHead::bicycle()) < vel_prior(&TrackHead::truck()));
        assert!(vel_prior(&TrackHead::truck()) < vel_prior(&TrackHead::car()));
        assert_eq!(
            vel_prior(&TrackHead::car()),
            vel_prior(&TrackHead::motorcycle())
        );
    }

    #[test]
    fn agility_priors_are_ordered_by_class() {
        // Process noise (agility): motorcycle > car > truck > bus/pedestrian;
        // every road user is steadier than the aerospace defaults.
        assert!(TrackHead::motorcycle().process_noise_sigma > TrackHead::car().process_noise_sigma);
        assert!(TrackHead::car().process_noise_sigma > TrackHead::truck().process_noise_sigma);
        assert!(TrackHead::truck().process_noise_sigma > TrackHead::bus().process_noise_sigma);
        assert!(TrackHead::car().process_noise_sigma < TrackHead::aircraft().process_noise_sigma);
    }

    #[test]
    fn unmapped_class_falls_back_to_unknown_head() {
        let reg = HeadRegistry {
            heads: vec![TrackHead::car(), TrackHead::unknown()],
        };
        assert_eq!(reg.get(TargetClass::Bus).class, TargetClass::Unknown);
    }

    // --- Orbital & ballistic heads (orbital-ballistic-filter-models, §5) ---

    /// Task 5.2 regression: `TargetClass::Orbital` resolves to the dedicated
    /// orbital head, not the `Unknown` fallback it silently hit before.
    #[test]
    fn orbital_class_resolves_to_orbital_head_not_unknown() {
        let reg = HeadRegistry::default();
        let head = reg.get(TargetClass::Orbital);
        assert_eq!(head.class, TargetClass::Orbital);
        assert_eq!(head.model, HeadModel::KeplerJ2 { max_step_s: 10.0 });
        assert_eq!(head.state_dim, 6);
    }

    /// Task 5.1: every pre-existing head defaults to the CV model, so the
    /// dispatch change is behavior-preserving for aircraft/UAV/automotive.
    #[test]
    fn preexisting_heads_default_to_cv_model() {
        for head in [
            TrackHead::aircraft(),
            TrackHead::uav(),
            TrackHead::unknown(),
            TrackHead::car(),
            TrackHead::truck(),
            TrackHead::bus(),
            TrackHead::motorcycle(),
            TrackHead::bicycle(),
            TrackHead::pedestrian(),
        ] {
            assert_eq!(head.model, HeadModel::Cv, "{:?}", head.class);
        }
    }

    /// Task 5.3: the ballistic head is the 7D reentry configuration with a
    /// wide β prior and unchanged 2-of-3 / delete-3 lifecycle policies.
    #[test]
    fn ballistic_head_is_7d_reentry_with_wide_beta_prior() {
        let head = TrackHead::ballistic();
        assert_eq!(head.state_dim, 7);
        assert_eq!(head.initial_covariance.len(), 7);
        assert!(
            head.initial_covariance[6] >= 1e6,
            "β prior {} not wide",
            head.initial_covariance[6]
        );
        assert!(matches!(
            head.model,
            HeadModel::Reentry { sigma_beta, beta_init }
                if sigma_beta > 0.0 && beta_init > 0.0
        ));
        assert_eq!((head.confirmation.m, head.confirmation.n), (2, 3));
        assert_eq!(head.deletion.max_coast_age, 3);
        // The built model really is the 7D reentry dynamics.
        assert_eq!(head.build_model().state_dim(), 7);
    }

    /// Task 5.1: `build_model` combines the selector with the head's
    /// `process_noise_sigma` — the CV fast path and the nonlinear heads all
    /// carry the head's noise into the model's Q.
    #[test]
    fn build_model_wires_head_process_noise() {
        let dt = 1.0;
        // CV: Q velocity diagonal is σ²·dt² (DWNA block).
        let aircraft = TrackHead::aircraft();
        let q_cv = aircraft.build_model().process_noise(dt);
        let sigma2 = aircraft.process_noise_sigma * aircraft.process_noise_sigma;
        assert!((q_cv[(1, 1)] - sigma2 * dt * dt).abs() < 1e-12);

        // KeplerJ2: CWNA block gives Q[1,1] = σ²·dt.
        let orbital = TrackHead::orbital();
        let q_orb = orbital.build_model().process_noise(dt);
        let sigma2 = orbital.process_noise_sigma * orbital.process_noise_sigma;
        assert!((q_orb[(1, 1)] - sigma2 * dt).abs() < 1e-18);

        // Reentry: β random walk on the 7th diagonal.
        let ballistic = TrackHead::ballistic();
        let q_re = ballistic.build_model().process_noise(dt);
        let HeadModel::Reentry { sigma_beta, .. } = ballistic.model else {
            panic!("ballistic head must be a Reentry model");
        };
        assert!((q_re[(6, 6)] - sigma_beta * sigma_beta * dt).abs() < 1e-9);
    }

    /// Task 5.4 (spec "Orbital head tracks a satellite", model level): the
    /// orbital head's model curves with gravity across a measurement gap —
    /// it stays on the circular orbit while straight-line extrapolation of
    /// the same state diverges by hundreds of km.
    #[test]
    fn orbital_head_model_curves_with_gravity_across_gap() {
        let earth = GravityModel::EARTH_WGS84;
        let r = earth.equatorial_radius + 400_000.0;
        let v = (earth.mu / r).sqrt();
        // Circular equatorial LEO, interleaved [x, vx, y, vy, z, vz].
        let x0 = nalgebra::DVector::from_column_slice(&[r, 0.0, 0.0, v, 0.0, 0.0]);

        let model = TrackHead::orbital().build_model();
        let gap_s = 300.0;
        let predicted = model.predict(&x0, gap_s);

        // Straight-line extrapolation of the same state.
        let straight = [x0[0] + x0[1] * gap_s, x0[2] + x0[3] * gap_s, 0.0];
        let divergence =
            ((predicted[0] - straight[0]).powi(2) + (predicted[2] - straight[1]).powi(2)).sqrt();
        assert!(
            divergence > 100_000.0,
            "gravity curvature only {divergence} m over {gap_s} s"
        );

        // And the curved prediction is the physical one: it stays on the
        // circular orbit's radius (straight-line leaves it by ~divergence).
        let pred_radius = (predicted[0].powi(2) + predicted[2].powi(2)).sqrt();
        assert!(
            (pred_radius - r).abs() < 5_000.0,
            "orbital prediction left the circular radius by {} m",
            (pred_radius - r).abs()
        );
    }
}
