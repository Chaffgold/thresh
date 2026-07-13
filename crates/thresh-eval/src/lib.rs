//! MOT evaluation metrics: MOTA, IDF1, HOTA, AMOTA with per-class breakdowns.

pub mod builder;
pub mod consistency;
pub mod gospa;
pub mod hota;
pub mod matching;
pub mod metrics;
pub mod per_class;
pub mod report;

pub use builder::{MotMetrics, MotMetricsBuilder};
pub use consistency::{
    ConsistencyAccumulator, ConsistencyVerdict, EstimateFrame, INTERLEAVED_POSITION_INDICES,
    PerTrackConsistency, TrackEstimate, nees, nis, sequence_anees,
};
pub use gospa::{GospaParams, GospaResult, GospaSummary, gospa_frame, gospa_sequence};
pub use report::{ClassReport, ConsistencyReportSection, EvalReport, GospaReportSection};
