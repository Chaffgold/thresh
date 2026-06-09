//! Class-specific track heads: target class -> motion model + noise params + policies.

use thresh_core::track::TargetClass;

use crate::lifecycle::{ConfirmationPolicy, DeletionPolicy};

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
        }
    }

    /// Default ballistic missile tracking head (CA model, 9D state).
    pub fn ballistic() -> Self {
        Self {
            class: TargetClass::Ballistic,
            state_dim: 9,
            process_noise_sigma: 20.0, // high acceleration uncertainty
            initial_covariance: vec![
                10000.0, 1000.0, 100.0, 10000.0, 1000.0, 100.0, 10000.0, 1000.0, 100.0,
            ],
            confirmation: ConfirmationPolicy::new(2, 3),
            deletion: DeletionPolicy::new(3),
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
}
