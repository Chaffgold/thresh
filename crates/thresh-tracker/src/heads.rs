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
