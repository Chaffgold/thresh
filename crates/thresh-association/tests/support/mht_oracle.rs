//! Exhaustive multi-scan MHT reference, enumerated by per-track recursion.
//!
//! This oracle is written from the textbook hypothesis-oriented MHT
//! formulation that the fixture model below declares (design.md Decision 6 of
//! the `resolve-stonesoup-mht-imm-claims` change). It uses `std` only and
//! shares no code with the production hypothesis tree. The `mht_reference`
//! target reads this file with `include_str!` and fails if it names a
//! workspace crate or imports anything outside `std`.
//!
//! # Fixture model
//!
//! - Each track is a scalar random-walk Kalman filter: predict `P⁻ = P + q`;
//!   innovation variance `S = P⁻ + r`; gain `K = P⁻ / S`; update
//!   `m⁺ = m + K (z − m)` and `P⁺ = (1 − K) P⁻`.
//! - Within a scan every existing track is either missed or assigned exactly
//!   one detection. Every detection is assigned to at most one track, or
//!   labelled a false alarm, or labelled a new track when `λ_B > 0`. Tracks
//!   do not die inside the window.
//! - Gate: `(z − m)² / S ≤ g`. An ungated fixture accepts every pair.
//! - Per-scan log scores, normalised by the scan-common `λ_FA^M` so that
//!   rankings and score differences are unchanged:
//!   - assigned: `ln(P_D · N(z; m, S) / λ_FA)`;
//!   - missed: `ln(1 − P_D)`;
//!   - false alarm: `0`;
//!   - new track: `ln(λ_B / λ_FA)`, and the new track starts at `(z, r)`.
//!
//! # Enumeration order
//!
//! `predict_tracks → enumerate_track_events → label_remaining_detections →
//! score_event → extend_history`. Track events are enumerated first, by
//! recursion over the tracks: each track chooses a miss or one gated
//! detection that no earlier track has taken. Only then are the detections
//! left over labelled.
//!
//! # History notation
//!
//! A history key renders each scan as its track events in track order
//! (`A z0`, `B miss`) followed by its unassigned detections in index order
//! (`z0 FA`, `z1 new`), joined by `, `. Scans are joined by ` / `. A track
//! born from detection `zj` of scan `s` (1-based) is named `N{s}z{j}`.
//! Tracks are ordered as declared, then by birth scan and detection index.

use std::f64::consts::PI;

/// A prior track declared by a fixture.
#[derive(Clone, Copy, Debug)]
pub struct Prior {
    pub name: &'static str,
    pub mean: f64,
    pub variance: f64,
}

/// A fixture: priors, scans and the declared score, birth and gate models.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub priors: Vec<Prior>,
    pub scans: Vec<Vec<f64>>,
    pub detection_probability: f64,
    pub process_variance: f64,
    pub measurement_variance: f64,
    pub false_alarm_density: f64,
    /// Birth density `λ_B`. Zero disables births.
    pub birth_density: f64,
    /// Gate on the squared normalised innovation. `None` is ungated.
    pub gate: Option<f64>,
}

impl Fixture {
    /// Whether a leftover detection may start a new track.
    pub fn births_enabled(&self) -> bool {
        self.birth_density > 0.0
    }
}

/// A scan being expanded: its 1-based number and its detections.
#[derive(Clone, Copy, Debug)]
pub struct Scan<'a> {
    pub number: usize,
    pub detections: &'a [f64],
}

/// A named scalar track estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackState {
    pub name: String,
    pub mean: f64,
    pub variance: f64,
}

/// What one existing track does in one scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackEvent {
    Missed,
    Detected(usize),
}

/// The label of a detection that no track took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionLabel {
    FalseAlarm,
    NewTrack,
}

/// One feasible global association event for one scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanEvent {
    /// One event per existing track, in track order.
    pub track_events: Vec<TrackEvent>,
    /// Labels for the detections no track took, in index order.
    pub labels: Vec<(usize, DetectionLabel)>,
}

/// The recorded events of one scan of a history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanRecord {
    /// Number of detections in the scan.
    pub detections: usize,
    /// Each existing track's name and event, in track order.
    pub track_events: Vec<(String, TrackEvent)>,
    /// Each unassigned detection's index and label, in index order.
    pub unassigned: Vec<(usize, DetectionLabel)>,
}

impl ScanRecord {
    /// Renders the scan in the history notation of the module docs.
    pub fn render(&self) -> String {
        let tracks = self.track_events.iter().map(|(name, event)| match event {
            TrackEvent::Missed => format!("{name} miss"),
            TrackEvent::Detected(j) => format!("{name} z{j}"),
        });
        let unassigned = self.unassigned.iter().map(|(j, label)| match label {
            DetectionLabel::FalseAlarm => format!("z{j} FA"),
            DetectionLabel::NewTrack => format!("z{j} new"),
        });
        tracks.chain(unassigned).collect::<Vec<_>>().join(", ")
    }
}

/// A global hypothesis: a complete history through the scans so far.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    /// Posterior track estimates after the last scan.
    pub tracks: Vec<TrackState>,
    /// One record per scan.
    pub scans: Vec<ScanRecord>,
    /// One log score per scan.
    pub scan_scores: Vec<f64>,
    /// Cumulative log score, accumulated scan by scan.
    pub score: f64,
}

impl Hypothesis {
    /// The history key: every scan rendered and joined by ` / `.
    pub fn history_key(&self) -> String {
        self.scans
            .iter()
            .map(ScanRecord::render)
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

/// Enumerates every global hypothesis of the fixture, without pruning.
pub fn enumerate(fixture: &Fixture) -> Vec<Hypothesis> {
    let mut leaves = vec![root_hypothesis(fixture)];
    for (index, detections) in fixture.scans.iter().enumerate() {
        let scan = Scan {
            number: index + 1,
            detections,
        };
        leaves = leaves
            .iter()
            .flat_map(|parent| expand_hypothesis(parent, &scan, fixture))
            .collect();
    }
    leaves
}

fn root_hypothesis(fixture: &Fixture) -> Hypothesis {
    let tracks = fixture
        .priors
        .iter()
        .map(|prior| TrackState {
            name: prior.name.to_string(),
            mean: prior.mean,
            variance: prior.variance,
        })
        .collect();
    Hypothesis {
        tracks,
        scans: Vec::new(),
        scan_scores: Vec::new(),
        score: 0.0,
    }
}

fn expand_hypothesis(parent: &Hypothesis, scan: &Scan<'_>, fixture: &Fixture) -> Vec<Hypothesis> {
    let predicted = predict_tracks(&parent.tracks, fixture.process_variance);
    let births = fixture.births_enabled();
    let mut children = Vec::new();
    for track_events in enumerate_track_events(&predicted, scan.detections, fixture) {
        for labels in label_remaining_detections(&track_events, scan.detections.len(), births) {
            let event = ScanEvent {
                track_events: track_events.clone(),
                labels,
            };
            let score = score_event(&predicted, &event, scan.detections, fixture);
            children.push(extend_history(
                parent, &predicted, event, score, scan, fixture,
            ));
        }
    }
    children
}

/// Phase 1: random-walk prediction, `P⁻ = P + q`, means unchanged.
pub fn predict_tracks(tracks: &[TrackState], process_variance: f64) -> Vec<TrackState> {
    tracks
        .iter()
        .map(|track| TrackState {
            name: track.name.clone(),
            mean: track.mean,
            variance: track.variance + process_variance,
        })
        .collect()
}

/// Phase 2: every exclusive, gated choice of one event per track, by
/// recursion over the tracks in order.
pub fn enumerate_track_events(
    predicted: &[TrackState],
    detections: &[f64],
    fixture: &Fixture,
) -> Vec<Vec<TrackEvent>> {
    let mut out = Vec::new();
    let mut chosen = Vec::with_capacity(predicted.len());
    recurse_tracks(predicted, detections, fixture, &mut chosen, &mut out);
    out
}

fn recurse_tracks(
    remaining: &[TrackState],
    detections: &[f64],
    fixture: &Fixture,
    chosen: &mut Vec<TrackEvent>,
    out: &mut Vec<Vec<TrackEvent>>,
) {
    let Some((track, rest)) = remaining.split_first() else {
        out.push(chosen.clone());
        return;
    };
    for event in track_options(track, detections, chosen, fixture) {
        chosen.push(event);
        recurse_tracks(rest, detections, fixture, chosen, out);
        chosen.pop();
    }
}

/// A miss, plus every gated detection that no earlier track has taken.
fn track_options(
    track: &TrackState,
    detections: &[f64],
    taken: &[TrackEvent],
    fixture: &Fixture,
) -> Vec<TrackEvent> {
    let free_and_gated = detections.iter().enumerate().filter(|&(j, &z)| {
        !taken.contains(&TrackEvent::Detected(j)) && within_gate(track, z, fixture)
    });
    std::iter::once(TrackEvent::Missed)
        .chain(free_and_gated.map(|(j, _)| TrackEvent::Detected(j)))
        .collect()
}

fn innovation_variance(predicted: &TrackState, measurement_variance: f64) -> f64 {
    predicted.variance + measurement_variance
}

fn within_gate(predicted: &TrackState, z: f64, fixture: &Fixture) -> bool {
    fixture.gate.is_none_or(|gate| {
        let s = innovation_variance(predicted, fixture.measurement_variance);
        (z - predicted.mean).powi(2) / s <= gate
    })
}

/// Phase 3: every labelling of the detections that no track took, as a
/// false alarm or, when births are enabled, as a new track.
pub fn label_remaining_detections(
    track_events: &[TrackEvent],
    detections: usize,
    births: bool,
) -> Vec<Vec<(usize, DetectionLabel)>> {
    let remaining: Vec<usize> = (0..detections)
        .filter(|&j| !track_events.contains(&TrackEvent::Detected(j)))
        .collect();
    let mut out = Vec::new();
    recurse_labels(&remaining, births, &mut Vec::new(), &mut out);
    out
}

fn recurse_labels(
    remaining: &[usize],
    births: bool,
    chosen: &mut Vec<(usize, DetectionLabel)>,
    out: &mut Vec<Vec<(usize, DetectionLabel)>>,
) {
    let Some((&detection, rest)) = remaining.split_first() else {
        out.push(chosen.clone());
        return;
    };
    for label in label_options(births) {
        chosen.push((detection, label));
        recurse_labels(rest, births, chosen, out);
        chosen.pop();
    }
}

fn label_options(births: bool) -> Vec<DetectionLabel> {
    let mut options = vec![DetectionLabel::FalseAlarm];
    if births {
        options.push(DetectionLabel::NewTrack);
    }
    options
}

/// Phase 4: the scan's log score, track terms in track order and then the
/// labels of the unassigned detections.
pub fn score_event(
    predicted: &[TrackState],
    event: &ScanEvent,
    detections: &[f64],
    fixture: &Fixture,
) -> f64 {
    let track_terms: f64 = predicted
        .iter()
        .zip(&event.track_events)
        .map(|(track, &track_event)| track_event_score(track, track_event, detections, fixture))
        .sum();
    let label_terms: f64 = event
        .labels
        .iter()
        .map(|&(_, label)| label_score(label, fixture))
        .sum();
    track_terms + label_terms
}

fn track_event_score(
    predicted: &TrackState,
    event: TrackEvent,
    detections: &[f64],
    fixture: &Fixture,
) -> f64 {
    let p_d = fixture.detection_probability;
    match event {
        TrackEvent::Missed => (1.0 - p_d).ln(),
        TrackEvent::Detected(j) => {
            let s = innovation_variance(predicted, fixture.measurement_variance);
            let likelihood = gaussian_density(detections[j], predicted.mean, s);
            (p_d * likelihood / fixture.false_alarm_density).ln()
        }
    }
}

fn label_score(label: DetectionLabel, fixture: &Fixture) -> f64 {
    match label {
        DetectionLabel::FalseAlarm => 0.0,
        DetectionLabel::NewTrack => (fixture.birth_density / fixture.false_alarm_density).ln(),
    }
}

/// `N(x; mean, variance)` for a scalar Gaussian.
fn gaussian_density(x: f64, mean: f64, variance: f64) -> f64 {
    let residual = x - mean;
    (-residual * residual / (2.0 * variance)).exp() / (2.0 * PI * variance).sqrt()
}

/// Phase 5: the child history. Assigned tracks take the Kalman update,
/// missed tracks keep their prediction, and each new-track label appends a
/// track starting at `(z, r)`.
pub fn extend_history(
    parent: &Hypothesis,
    predicted: &[TrackState],
    event: ScanEvent,
    score: f64,
    scan: &Scan<'_>,
    fixture: &Fixture,
) -> Hypothesis {
    let r = fixture.measurement_variance;
    let mut tracks: Vec<TrackState> = predicted
        .iter()
        .zip(&event.track_events)
        .map(|(track, &track_event)| update_track(track, track_event, scan.detections, r))
        .collect();
    tracks.extend(born_tracks(&event.labels, scan, r));
    let record = ScanRecord {
        detections: scan.detections.len(),
        track_events: predicted
            .iter()
            .map(|track| track.name.clone())
            .zip(event.track_events)
            .collect(),
        unassigned: event.labels,
    };
    let mut scans = parent.scans.clone();
    scans.push(record);
    let mut scan_scores = parent.scan_scores.clone();
    scan_scores.push(score);
    Hypothesis {
        tracks,
        scans,
        scan_scores,
        score: parent.score + score,
    }
}

fn update_track(
    predicted: &TrackState,
    event: TrackEvent,
    detections: &[f64],
    measurement_variance: f64,
) -> TrackState {
    let TrackEvent::Detected(j) = event else {
        return predicted.clone();
    };
    let s = innovation_variance(predicted, measurement_variance);
    let gain = predicted.variance / s;
    TrackState {
        name: predicted.name.clone(),
        mean: predicted.mean + gain * (detections[j] - predicted.mean),
        variance: (1.0 - gain) * predicted.variance,
    }
}

fn born_tracks(
    labels: &[(usize, DetectionLabel)],
    scan: &Scan<'_>,
    measurement_variance: f64,
) -> Vec<TrackState> {
    labels
        .iter()
        .filter(|&&(_, label)| label == DetectionLabel::NewTrack)
        .map(|&(j, _)| TrackState {
            name: format!("N{}z{}", scan.number, j),
            mean: scan.detections[j],
            variance: measurement_variance,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(gate: Option<f64>, birth_density: f64) -> Fixture {
        Fixture {
            name: "unit",
            priors: vec![
                Prior {
                    name: "A",
                    mean: 0.0,
                    variance: 1.0,
                },
                Prior {
                    name: "B",
                    mean: 4.0,
                    variance: 1.0,
                },
            ],
            scans: Vec::new(),
            detection_probability: 0.9,
            process_variance: 0.5,
            measurement_variance: 0.5,
            false_alarm_density: 0.05,
            birth_density,
            gate,
        }
    }

    fn track(name: &str, mean: f64, variance: f64) -> TrackState {
        TrackState {
            name: name.to_string(),
            mean,
            variance,
        }
    }

    fn assert_near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-14,
            "actual {actual:?}, expected {expected:?}"
        );
    }

    #[test]
    fn predict_tracks_adds_process_variance_and_keeps_means() {
        let predicted = predict_tracks(&[track("A", 0.0, 1.0), track("B", 4.0, 0.375)], 0.5);
        assert_eq!(
            predicted,
            vec![track("A", 0.0, 1.5), track("B", 4.0, 0.875)]
        );
    }

    #[test]
    fn enumerate_track_events_gives_each_detection_to_at_most_one_track() {
        use super::TrackEvent::{Detected, Missed};
        let predicted = [track("A", 0.0, 1.5), track("B", 4.0, 1.5)];
        let events = enumerate_track_events(&predicted, &[1.0, 3.0], &model(None, 0.0));
        // Three choices per track, minus the two that share a detection.
        let expected = vec![
            vec![Missed, Missed],
            vec![Missed, Detected(0)],
            vec![Missed, Detected(1)],
            vec![Detected(0), Missed],
            vec![Detected(0), Detected(1)],
            vec![Detected(1), Missed],
            vec![Detected(1), Detected(0)],
        ];
        assert_eq!(events, expected);
    }

    #[test]
    fn enumerate_track_events_excludes_detections_outside_the_gate() {
        use super::TrackEvent::{Detected, Missed};
        // S = 2 for both tracks. z = 5: A gives 12.5 > 9, B gives 0.5 <= 9.
        // z = 0: A gives 0 <= 9, B gives 8 <= 9.
        let predicted = [track("A", 0.0, 1.5), track("B", 4.0, 1.5)];
        let events = enumerate_track_events(&predicted, &[5.0, 0.0], &model(Some(9.0), 0.0));
        let expected = vec![
            vec![Missed, Missed],
            vec![Missed, Detected(0)],
            vec![Missed, Detected(1)],
            vec![Detected(1), Missed],
            vec![Detected(1), Detected(0)],
        ];
        assert_eq!(events, expected);
    }

    #[test]
    fn label_remaining_detections_offers_new_track_only_with_births() {
        use super::DetectionLabel::{FalseAlarm, NewTrack};
        let taken = [TrackEvent::Missed, TrackEvent::Detected(1)];
        assert_eq!(
            label_remaining_detections(&taken, 3, false),
            vec![vec![(0, FalseAlarm), (2, FalseAlarm)]]
        );
        assert_eq!(
            label_remaining_detections(&taken, 3, true),
            vec![
                vec![(0, FalseAlarm), (2, FalseAlarm)],
                vec![(0, FalseAlarm), (2, NewTrack)],
                vec![(0, NewTrack), (2, FalseAlarm)],
                vec![(0, NewTrack), (2, NewTrack)],
            ]
        );
        assert_eq!(
            label_remaining_detections(&[TrackEvent::Missed], 0, true),
            vec![Vec::new()]
        );
    }

    #[test]
    fn score_event_sums_textbook_terms() {
        let fixture = model(None, 0.02);
        let predicted = [track("A", 0.0, 1.5), track("B", 4.0, 1.5)];
        let event = ScanEvent {
            track_events: vec![TrackEvent::Detected(0), TrackEvent::Missed],
            labels: vec![
                (1, DetectionLabel::NewTrack),
                (2, DetectionLabel::FalseAlarm),
            ],
        };
        // A takes z = 2.1 with S = 2: ln(0.9 N(2.1; 0, 2) / 0.05).
        let density = (-2.1_f64 * 2.1 / 4.0).exp() / (4.0 * PI).sqrt();
        let expected = (0.9 * density / 0.05).ln() + 0.1_f64.ln() + (0.02_f64 / 0.05).ln() + 0.0;
        assert_near(
            score_event(&predicted, &event, &[2.1, 7.0, -3.0], &fixture),
            expected,
        );
    }

    #[test]
    fn extend_history_updates_tracks_appends_births_and_accumulates_score() {
        let fixture = model(None, 0.02);
        let parent = Hypothesis {
            tracks: vec![track("A", 0.0, 1.0), track("B", 4.0, 1.0)],
            scans: Vec::new(),
            scan_scores: vec![-1.0],
            score: -1.0,
        };
        let predicted = predict_tracks(&parent.tracks, 0.5);
        let event = ScanEvent {
            track_events: vec![TrackEvent::Detected(1), TrackEvent::Missed],
            labels: vec![(0, DetectionLabel::NewTrack)],
        };
        let scan = Scan {
            number: 2,
            detections: &[7.0, 2.1],
        };
        let child = extend_history(&parent, &predicted, event, -0.25, &scan, &fixture);
        // A: K = 1.5 / 2 = 0.75, m = 0.75 * 2.1, P = 0.25 * 1.5.
        assert_eq!(child.tracks[0].name, "A");
        assert_near(child.tracks[0].mean, 1.575);
        assert_near(child.tracks[0].variance, 0.375);
        assert_eq!(child.tracks[1], track("B", 4.0, 1.5));
        assert_eq!(child.tracks[2], track("N2z0", 7.0, 0.5));
        assert_eq!(child.scan_scores, vec![-1.0, -0.25]);
        assert_near(child.score, -1.25);
        assert_eq!(child.history_key(), "A z1, B miss, z0 new");
    }

    #[test]
    fn history_key_renders_tracks_then_unassigned_detections() {
        let hypothesis = Hypothesis {
            tracks: Vec::new(),
            scans: vec![
                ScanRecord {
                    detections: 2,
                    track_events: vec![("A".to_string(), TrackEvent::Detected(0))],
                    unassigned: vec![(1, DetectionLabel::NewTrack)],
                },
                ScanRecord {
                    detections: 1,
                    track_events: vec![
                        ("A".to_string(), TrackEvent::Missed),
                        ("N1z1".to_string(), TrackEvent::Missed),
                    ],
                    unassigned: vec![(0, DetectionLabel::FalseAlarm)],
                },
            ],
            scan_scores: Vec::new(),
            score: 0.0,
        };
        assert_eq!(
            hypothesis.history_key(),
            "A z0, z1 new / A miss, N1z1 miss, z0 FA"
        );
    }

    #[test]
    fn enumerate_expands_every_scan_without_pruning() {
        let mut fixture = model(None, 0.0);
        fixture.scans = vec![vec![2.1], vec![3.0]];
        // Per scan: both miss (z0 false alarm), A takes z0, or B takes z0.
        let leaves = enumerate(&fixture);
        assert_eq!(leaves.len(), 9);
        assert!(leaves.iter().all(|leaf| leaf.scans.len() == 2));
        assert_eq!(
            leaves[0].history_key(),
            "A miss, B miss, z0 FA / A miss, B miss, z0 FA"
        );
    }
}
