//! Report output: JSON and human-readable table.

use serde::{Deserialize, Serialize};

use crate::consistency::{ConsistencyAccumulator, ConsistencyVerdict};
use crate::gospa::{GospaParams, GospaSummary};

/// Complete evaluation report.
///
/// The `gospa` / `consistency` sections are optional and additive
/// (eval-consistency-metrics, design Decision 6): reports serialized before
/// those metrics existed deserialize unchanged (`serde(default)`), and a
/// report that did not compute them serializes byte-identically to the
/// pre-change output (`skip_serializing_if`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub mota: f64,
    pub motp: f64,
    pub idf1: f64,
    pub hota: f64,
    pub id_switches: usize,
    pub total_gt: usize,
    pub total_tracks: usize,
    pub per_class: Vec<ClassReport>,
    /// GOSPA sequence summary; `None` when GOSPA was not computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gospa: Option<GospaReportSection>,
    /// NEES/NIS consistency statistics; `None` when not computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consistency: Option<ConsistencyReportSection>,
}

/// Per-class metric breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassReport {
    pub class_name: String,
    pub mota: f64,
    pub hota: f64,
    pub count: usize,
}

/// GOSPA section of an [`EvalReport`]: the sequence-level headline values
/// (per-frame results are summarized, not embedded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GospaReportSection {
    /// Parameters (cutoff `c`, order `p`) the sequence was scored with.
    pub params: GospaParams,
    /// Order-p mean of the per-frame GOSPA totals.
    pub mean_gospa: f64,
    /// Summed localization components over all frames.
    pub localization: f64,
    /// Summed missed-target components over all frames.
    pub missed: f64,
    /// Summed false-track components over all frames.
    pub false_tracks: f64,
}

impl GospaReportSection {
    /// Build the report section from a [`GospaSummary`] and the parameters
    /// it was computed with.
    pub fn from_summary(summary: &GospaSummary, params: GospaParams) -> Self {
        Self {
            params,
            mean_gospa: summary.mean_gospa,
            localization: summary.localization,
            missed: summary.missed,
            false_tracks: summary.false_tracks,
        }
    }
}

/// NEES/NIS consistency section of an [`EvalReport`].
///
/// Absent statistics are explicitly `None` (zero samples), never `0.0` or
/// `NaN` — absence and zero must stay distinguishable for the fail-loud
/// benchmark gates (design Decision 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsistencyReportSection {
    /// Significance level the bounds and verdicts were computed at
    /// (confidence `1 - alpha`).
    pub alpha: f64,
    /// Average NEES over matched truth/track pairs; `None` without samples.
    pub anees: Option<f64>,
    /// Number of NEES samples behind `anees`.
    pub anees_samples: usize,
    /// Two-sided acceptance interval for the ANEES, when samples exist.
    pub anees_bounds: Option<(f64, f64)>,
    /// Verdict of the ANEES against its bounds, when samples exist.
    pub anees_verdict: Option<ConsistencyVerdict>,
    /// Average NIS from filter update diagnostics; `None` without samples.
    pub anis: Option<f64>,
    /// Number of NIS samples behind `anis`.
    pub anis_samples: usize,
    /// Two-sided acceptance interval for the ANIS, when samples exist.
    pub anis_bounds: Option<(f64, f64)>,
    /// Verdict of the ANIS against its bounds, when samples exist.
    pub anis_verdict: Option<ConsistencyVerdict>,
}

impl ConsistencyReportSection {
    /// Build the section from ANEES and ANIS accumulators at significance
    /// `alpha` (use [`crate::consistency::chi2::DEFAULT_ALPHA`] for the
    /// conventional 95% interval). Empty accumulators yield `None`
    /// statistics with their sample counts at zero.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not strictly inside `(0, 1)` or a non-empty
    /// accumulator has `dof == 0` (propagated from the chi-squared bounds).
    pub fn from_accumulators(
        anees: &ConsistencyAccumulator,
        anis: &ConsistencyAccumulator,
        alpha: f64,
    ) -> Self {
        Self {
            alpha,
            anees: anees.mean(),
            anees_samples: anees.n,
            anees_bounds: anees.bounds(alpha),
            anees_verdict: anees.verdict(alpha),
            anis: anis.mean(),
            anis_samples: anis.n,
            anis_bounds: anis.bounds(alpha),
            anis_verdict: anis.verdict(alpha),
        }
    }
}

impl EvalReport {
    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    /// Format as a human-readable table.
    ///
    /// The GOSPA and consistency sections appear only when present, so
    /// reports without those metrics render exactly as before.
    pub fn to_table(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{:<12} {:<10} {:<10} {:<10} {:<10} {:<10}\n",
            "Metric", "MOTA", "MOTP", "IDF1", "HOTA", "IDSW"
        ));
        s.push_str(&format!(
            "{:<12} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:<10}\n",
            "Overall", self.mota, self.motp, self.idf1, self.hota, self.id_switches
        ));

        if !self.per_class.is_empty() {
            s.push_str("\nPer-class:\n");
            s.push_str(&format!(
                "{:<15} {:<10} {:<10} {:<10}\n",
                "Class", "MOTA", "HOTA", "Count"
            ));
            for c in &self.per_class {
                s.push_str(&format!(
                    "{:<15} {:<10.4} {:<10.4} {:<10}\n",
                    c.class_name, c.mota, c.hota, c.count
                ));
            }
        }

        if let Some(gospa) = &self.gospa {
            append_gospa_section(&mut s, gospa);
        }
        if let Some(consistency) = &self.consistency {
            append_consistency_section(&mut s, consistency);
        }
        s
    }
}

/// Append the GOSPA table rows (phase helper of `to_table`).
fn append_gospa_section(s: &mut String, g: &GospaReportSection) {
    s.push_str(&format!(
        "\nGOSPA (c = {}, p = {}):\n",
        g.params.c, g.params.p
    ));
    s.push_str(&format!(
        "{:<12} {:<14} {:<14} {:<14}\n",
        "Mean", "Localization", "Missed", "False"
    ));
    s.push_str(&format!(
        "{:<12.4} {:<14.4} {:<14.4} {:<14.4}\n",
        g.mean_gospa, g.localization, g.missed, g.false_tracks
    ));
}

/// Append the NEES/NIS consistency table rows (phase helper of `to_table`).
fn append_consistency_section(s: &mut String, c: &ConsistencyReportSection) {
    s.push_str(&format!("\nConsistency (alpha = {}):\n", c.alpha));
    s.push_str(&format!(
        "{:<8} {:<12} {:<10} {:<12} {:<12} {:<14}\n",
        "Stat", "Value", "Samples", "Lower", "Upper", "Verdict"
    ));
    s.push_str(&consistency_row(
        "ANEES",
        c.anees,
        c.anees_samples,
        c.anees_bounds,
        c.anees_verdict,
    ));
    s.push_str(&consistency_row(
        "ANIS",
        c.anis,
        c.anis_samples,
        c.anis_bounds,
        c.anis_verdict,
    ));
}

/// One consistency table row; absent statistics render as `-`.
fn consistency_row(
    name: &str,
    value: Option<f64>,
    samples: usize,
    bounds: Option<(f64, f64)>,
    verdict: Option<ConsistencyVerdict>,
) -> String {
    let num = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.4}"));
    let (lo, hi) = bounds.map_or((None, None), |(l, h)| (Some(l), Some(h)));
    let verdict = verdict.map_or_else(|| "-".to_string(), |v| format!("{v:?}"));
    format!(
        "{:<8} {:<12} {:<10} {:<12} {:<12} {:<14}\n",
        name,
        num(value),
        samples,
        num(lo),
        num(hi),
        verdict
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consistency::chi2::DEFAULT_ALPHA;

    fn base_report() -> EvalReport {
        EvalReport {
            mota: 0.85,
            motp: 2.5,
            idf1: 0.78,
            hota: 0.72,
            id_switches: 5,
            total_gt: 100,
            total_tracks: 95,
            per_class: vec![ClassReport {
                class_name: "Aircraft".into(),
                mota: 0.9,
                hota: 0.8,
                count: 50,
            }],
            gospa: None,
            consistency: None,
        }
    }

    fn populated_sections() -> (GospaReportSection, ConsistencyReportSection) {
        let gospa = GospaReportSection {
            params: GospaParams::new(10.0),
            mean_gospa: 7.14,
            localization: 12.5,
            missed: 100.0,
            false_tracks: 50.0,
        };
        let mut anees_acc = ConsistencyAccumulator::new(3);
        let mut anis_acc = ConsistencyAccumulator::new(3);
        for _ in 0..100 {
            anees_acc.push(3.1);
            anis_acc.push(9.0); // far above the upper bound
        }
        let consistency =
            ConsistencyReportSection::from_accumulators(&anees_acc, &anis_acc, DEFAULT_ALPHA);
        (gospa, consistency)
    }

    #[test]
    fn report_json_roundtrip() {
        let report = base_report();
        let json = report.to_json();
        let restored: EvalReport = serde_json::from_str(&json).unwrap();
        assert!((restored.mota - 0.85).abs() < 1e-10);
    }

    #[test]
    fn report_table_output() {
        let report = EvalReport {
            mota: 1.0,
            motp: 0.0,
            idf1: 1.0,
            hota: 1.0,
            id_switches: 0,
            total_gt: 20,
            total_tracks: 20,
            per_class: vec![],
            gospa: None,
            consistency: None,
        };
        let table = report.to_table();
        assert!(table.contains("MOTA"));
        assert!(table.contains("1.0000"));
    }

    /// Exactly the pre-change `EvalReport` shape (same fields, same order):
    /// its pretty-JSON output is the byte-level contract the new struct must
    /// preserve when the optional sections are absent.
    #[derive(Serialize)]
    struct LegacyReport {
        mota: f64,
        motp: f64,
        idf1: f64,
        hota: f64,
        id_switches: usize,
        total_gt: usize,
        total_tracks: usize,
        per_class: Vec<ClassReport>,
    }

    fn legacy_json() -> String {
        serde_json::to_string_pretty(&LegacyReport {
            mota: 0.85,
            motp: 2.5,
            idf1: 0.78,
            hota: 0.72,
            id_switches: 5,
            total_gt: 100,
            total_tracks: 95,
            per_class: vec![ClassReport {
                class_name: "Aircraft".into(),
                mota: 0.9,
                hota: 0.8,
                count: 50,
            }],
        })
        .unwrap()
    }

    /// Task 5.1: an old serialized report (fixture without the new fields)
    /// deserializes unchanged, with both optional sections absent.
    #[test]
    fn old_fixture_deserializes_unchanged() {
        let fixture = legacy_json();
        let restored: EvalReport = serde_json::from_str(&fixture).unwrap();
        assert!((restored.mota - 0.85).abs() < 1e-12);
        assert!((restored.motp - 2.5).abs() < 1e-12);
        assert!((restored.idf1 - 0.78).abs() < 1e-12);
        assert!((restored.hota - 0.72).abs() < 1e-12);
        assert_eq!(restored.id_switches, 5);
        assert_eq!(restored.total_gt, 100);
        assert_eq!(restored.total_tracks, 95);
        assert_eq!(restored.per_class.len(), 1);
        assert_eq!(restored.per_class[0].class_name, "Aircraft");
        assert!(restored.gospa.is_none());
        assert!(restored.consistency.is_none());
    }

    /// Task 5.1: a new report without the metrics serializes byte-identically
    /// to today's (pre-change) output — the optional fields leave no trace.
    #[test]
    fn report_without_metrics_serializes_byte_identically() {
        let report = base_report();
        assert_eq!(report.to_json(), legacy_json());
    }

    /// Populated sections survive a JSON round trip, including the
    /// serialized verdicts and bounds.
    #[test]
    fn populated_sections_roundtrip() {
        let (gospa, consistency) = populated_sections();
        let mut report = base_report();
        report.gospa = Some(gospa.clone());
        report.consistency = Some(consistency.clone());
        let restored: EvalReport = serde_json::from_str(&report.to_json()).unwrap();
        assert_eq!(restored.gospa, Some(gospa));
        let restored_consistency = restored.consistency.expect("section present");
        assert_eq!(
            restored_consistency.anees_verdict,
            Some(ConsistencyVerdict::Consistent)
        );
        assert_eq!(
            restored_consistency.anis_verdict,
            Some(ConsistencyVerdict::Overconfident)
        );
        assert_eq!(restored_consistency, consistency);
    }

    /// `from_accumulators` on empty accumulators keeps every statistic
    /// explicitly absent (absent ≠ zero ≠ NaN).
    #[test]
    fn empty_accumulators_yield_absent_statistics() {
        let section = ConsistencyReportSection::from_accumulators(
            &ConsistencyAccumulator::new(3),
            &ConsistencyAccumulator::new(1),
            DEFAULT_ALPHA,
        );
        assert_eq!(section.anees, None);
        assert_eq!(section.anees_samples, 0);
        assert_eq!(section.anees_bounds, None);
        assert_eq!(section.anees_verdict, None);
        assert_eq!(section.anis, None);
        assert_eq!(section.anis_samples, 0);
        assert_eq!(section.anis_bounds, None);
        assert_eq!(section.anis_verdict, None);
    }

    /// Task 5.1: `to_table` rows appear only when the sections are present.
    #[test]
    fn table_rows_only_when_sections_present() {
        let bare = base_report();
        let bare_table = bare.to_table();
        assert!(!bare_table.contains("GOSPA"));
        assert!(!bare_table.contains("Consistency"));

        let (gospa, consistency) = populated_sections();
        let mut full = base_report();
        full.gospa = Some(gospa);
        full.consistency = Some(consistency);
        let full_table = full.to_table();
        assert!(full_table.contains("GOSPA (c = 10, p = 2):"));
        assert!(full_table.contains("ANEES"));
        assert!(full_table.contains("ANIS"));
        assert!(full_table.contains("Consistent"));
        assert!(full_table.contains("Overconfident"));
        // The MOT half of the table is unchanged by the added sections.
        assert!(full_table.starts_with(&bare_table));
    }
}
