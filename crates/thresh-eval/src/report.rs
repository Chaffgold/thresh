//! Report output: JSON and human-readable table.

use serde::{Deserialize, Serialize};

/// Complete evaluation report.
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
}

/// Per-class metric breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassReport {
    pub class_name: String,
    pub mota: f64,
    pub hota: f64,
    pub count: usize,
}

impl EvalReport {
    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    /// Format as a human-readable table.
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
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_json_roundtrip() {
        let report = EvalReport {
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
        };
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
        };
        let table = report.to_table();
        assert!(table.contains("MOTA"));
        assert!(table.contains("1.0000"));
    }
}
