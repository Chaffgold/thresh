//! MOT evaluation metrics: MOTA, IDF1, HOTA, AMOTA with per-class breakdowns.

pub mod builder;
pub mod hota;
pub mod matching;
pub mod metrics;
pub mod report;

pub use builder::{MotMetrics, MotMetricsBuilder};
