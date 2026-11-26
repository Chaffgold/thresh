//! Exhaustive MHT reference for the committed C1–C5 fixtures.
//!
//! The fixtures, their declared models and the expected values come from
//! design.md Decision 6 of the `resolve-stonesoup-mht-imm-claims` change.
//! The declared models define only these fixtures' expected values; they do
//! not specify the production tracker's scoring, birth initialisation or
//! gating, and nothing here exercises or asserts native output.
//!
//! Two enumerations run over every fixture:
//!
//! - the per-track-recursion oracle in `support/mht_oracle.rs`, which is
//!   `std`-only and names no workspace crate (checked below with
//!   `include_str!`);
//! - a per-detection label product with a feasibility filter, written in
//!   this file: `label_product → filter_feasible → score_labels →
//!   key_history`. It updates tracks in information form and scores in log
//!   space, so it shares neither enumeration order nor arithmetic with the
//!   oracle.
//!
//! # Provenance
//!
//! The committed constants were produced during the P0-C review by two
//! independently written Python enumerators, one labelling each detection and
//! one recursing per track. A third per-detection enumeration (2026-11-25)
//! matched every leaf count, best score, best history, gap and tie structure.
//! A fourth (2026-11-25, artifact review) matched every leaf count, best score
//! and best history, and produced the runner-up histories and scores. The
//! scratch scripts are not committed; this target is the reproduction. Here
//! the two Rust enumerations above produce identical history sets with scores
//! that agree within 1e-12, both reproduce every constant at atol 1e-10 plus
//! rtol 1e-9, and the C1 anchors are re-derived a third time in closed form.
//!
//! # Contract neutrality
//!
//! Assertions cover only hypothesis sets and counts, cumulative scores,
//! within-scan exclusivity, ranking as ordered tie groups (|Δ| ≤ 1e-9,
//! membership-only within a group, no asserted prefix ending inside a group)
//! and the unpruned maximum. Nothing here asserts an N-scan rule, a retained
//! set, a committed prefix, a track-ID policy, a tie-break order or native
//! output.
//!
//! History keys use the notation of the oracle module docs: per scan, track
//! events in track order (`A z0`, `B miss`), then unassigned detections in
//! index order (`z0 FA`, `z1 new`). A track born from detection `zj` of scan
//! `s` is named `N{s}z{j}`. These names are the reference's own labels for
//! its fixture tracks, used only to key histories; they are not track IDs.

mod support;

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use support::mht_oracle::{
    self, DetectionLabel, Fixture, Hypothesis, Prior, ScanRecord, TrackEvent,
};

/// Absolute tolerance for committed values.
const ATOL: f64 = 1e-10;
/// Relative tolerance for committed values.
const RTOL: f64 = 1e-9;
/// Agreement required between the two enumeration orders.
const ORDER_TOLERANCE: f64 = 1e-12;
/// Scores within this distance of their neighbour share a tie group.
const TIE_TOLERANCE: f64 = 1e-9;
/// The spec's cap on global hypotheses per committed fixture.
const HYPOTHESIS_BOUND: usize = 100_000;

const ORACLE_SOURCE: &str = include_str!("support/mht_oracle.rs");

// ---------------------------------------------------------------------------
// Fixtures (design.md Decision 6)
// ---------------------------------------------------------------------------

const DETECTION_PROBABILITY: f64 = 0.9;
const PROCESS_VARIANCE: f64 = 0.5;
const MEASUREMENT_VARIANCE: f64 = 0.5;
const FALSE_ALARM_DENSITY: f64 = 0.05;
const GATE: f64 = 9.0;

const TRACK_A: Prior = Prior {
    name: "A",
    mean: 0.0,
    variance: 1.0,
};
const TRACK_B: Prior = Prior {
    name: "B",
    mean: 4.0,
    variance: 1.0,
};

fn two_track_fixture(name: &'static str, scans: &[&[f64]], gate: Option<f64>) -> Fixture {
    Fixture {
        name,
        priors: vec![TRACK_A, TRACK_B],
        scans: scans.iter().map(|scan| scan.to_vec()).collect(),
        detection_probability: DETECTION_PROBABILITY,
        process_variance: PROCESS_VARIANCE,
        measurement_variance: MEASUREMENT_VARIANCE,
        false_alarm_density: FALSE_ALARM_DENSITY,
        birth_density: 0.0,
        gate,
    }
}

fn c1_deferred_resolution() -> Fixture {
    two_track_fixture("C1 deferred resolution", &[&[2.1], &[3.0], &[5.0]], None)
}

fn c2_two_target_confirmation() -> Fixture {
    two_track_fixture(
        "C2 two-target confirmation",
        &[&[2.5], &[0.5], &[2.5, 5.0]],
        Some(GATE),
    )
}

fn c3_recent_agreement() -> Fixture {
    two_track_fixture(
        "C3 recent agreement",
        &[&[2.5], &[5.0], &[-2.0]],
        Some(GATE),
    )
}

fn c4_birth_across_a_miss() -> Fixture {
    Fixture {
        name: "C4 birth across a miss",
        priors: vec![Prior {
            name: "A",
            mean: 0.0,
            variance: 0.25,
        }],
        scans: vec![vec![0.1, 5.0], vec![-0.1], vec![0.0, 5.2]],
        detection_probability: DETECTION_PROBABILITY,
        process_variance: PROCESS_VARIANCE,
        measurement_variance: MEASUREMENT_VARIANCE,
        false_alarm_density: 0.01,
        birth_density: 0.02,
        gate: Some(GATE),
    }
}

fn c5_empty_final_scan() -> Fixture {
    two_track_fixture("C5 empty final scan", &[&[2.1], &[3.0], &[]], None)
}

/// A committed history and its cumulative score.
struct Ranked {
    history: &'static str,
    score: f64,
}

/// Committed tie-group structure at |Δ| ≤ 1e-9.
#[derive(Debug, PartialEq, Eq)]
struct TieStructure {
    groups: usize,
    tied_groups: usize,
    largest: usize,
    adjacent_tied_pairs: usize,
}

const UNTIED_27: TieStructure = untied(27);
const UNTIED_13: TieStructure = untied(13);
const UNTIED_9: TieStructure = untied(9);

const fn untied(leaves: usize) -> TieStructure {
    TieStructure {
        groups: leaves,
        tied_groups: 0,
        largest: 1,
        adjacent_tied_pairs: 0,
    }
}

/// A fixture with its committed expected values.
struct Reference {
    fixture: Fixture,
    leaves: usize,
    best: Ranked,
    runner_up: Ranked,
    gap: f64,
    ties: TieStructure,
}

fn references() -> Vec<Reference> {
    vec![
        Reference {
            fixture: c1_deferred_resolution(),
            leaves: 27,
            best: Ranked {
                history: "A z0, B miss / A z0, B miss / A miss, B z0",
                score: -4.056137962656715,
            },
            runner_up: Ranked {
                history: "A miss, B z0 / A miss, B z0 / A miss, B z0",
                score: -4.36634826630236,
            },
            gap: 0.31021030364564517,
            ties: UNTIED_27,
        },
        Reference {
            fixture: c2_two_target_confirmation(),
            leaves: 51,
            best: Ranked {
                history: "A z0, B miss / A z0, B miss / A z0, B z1",
                score: -1.1827855182382336,
            },
            runner_up: Ranked {
                history: "A miss, B z0 / A z0, B miss / A z0, B z1",
                score: -1.3983633581276353,
            },
            gap: 0.2155778398894017,
            ties: TieStructure {
                groups: 50,
                tied_groups: 1,
                largest: 2,
                adjacent_tied_pairs: 1,
            },
        },
        Reference {
            fixture: c3_recent_agreement(),
            leaves: 13,
            best: Ranked {
                history: "A miss, B z0 / A miss, B z0 / A z0, B miss",
                score: -4.919774326293078,
            },
            runner_up: Ranked {
                history: "A miss, B miss, z0 FA / A miss, B z0 / A z0, B miss",
                score: -7.141592099531,
            },
            gap: 2.2218177732379196,
            ties: UNTIED_13,
        },
        Reference {
            fixture: c4_birth_across_a_miss(),
            leaves: 185,
            best: Ranked {
                history: "A z0, z1 new / A z0, N1z1 miss / A z0, N1z1 z1",
                score: 11.96617537989422,
            },
            runner_up: Ranked {
                history: "A z0, z1 FA / A z0 / A z0, z1 new",
                score: 11.044462926042648,
            },
            gap: 0.9217124538515726,
            ties: TieStructure {
                groups: 148,
                tied_groups: 31,
                largest: 4,
                adjacent_tied_pairs: 37,
            },
        },
        Reference {
            fixture: c5_empty_final_scan(),
            leaves: 9,
            best: Ranked {
                history: "A miss, B z0 / A miss, B z0 / A miss, B miss",
                score: -6.741456196614258,
            },
            runner_up: Ranked {
                history: "A z0, B miss / A miss, B z0 / A miss, B miss",
                score: -7.37469287881025,
            },
            gap: 0.6332366821959923,
            ties: UNTIED_9,
        },
    ]
}

// C1 hand anchors, derived in closed form (see
// `c1_hand_anchors_follow_from_closed_form`).
const C1_SCAN1_A_TAKES: f64 = -1.7802254585825268;
const C1_SCAN1_B_TAKES: f64 = -1.5802254585825266;
const C1_SCAN2_BEST: f64 = -1.228787824770912;
const C1_SCAN3_BEST: f64 = -1.0471246793032758;
const C1_TOTAL_BEST: f64 = -4.056137962656715;

// C2: ranks 10 and 11 form the only tie group.
const C2_TIE_RANK: usize = 10;
const C2_TIE_SCORE: f64 = -4.74083102492305;

// Every runner-up is unique: rank 3 trails rank 2 by at least this much in
// every fixture (design.md Decision 6; the smallest is C1's 0.059705…).
const RUNNER_UP_MARGIN: f64 = 0.0597;

// ---------------------------------------------------------------------------
// Second enumeration: per-detection label product with a feasibility filter
// ---------------------------------------------------------------------------

/// The label of one detection in the label product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    /// Assigned to the existing track at this index.
    Track(usize),
    FalseAlarm,
    Birth,
}

/// One scan of a branch: the tracks that existed and each detection's label.
#[derive(Clone, Debug)]
struct ScanLabels {
    names: Vec<String>,
    labels: Vec<Label>,
}

/// A partial history of the label-product enumeration.
#[derive(Clone, Debug)]
struct Branch {
    names: Vec<String>,
    means: Vec<f64>,
    variances: Vec<f64>,
    scans: Vec<ScanLabels>,
    scan_scores: Vec<f64>,
}

/// A predicted scan as seen by every child of one branch.
struct ScanContext<'a> {
    number: usize,
    detections: &'a [f64],
    means: &'a [f64],
    /// Predicted variances `P⁻`.
    variances: &'a [f64],
    fixture: &'a Fixture,
}

impl ScanContext<'_> {
    fn innovation_variance(&self, track: usize) -> f64 {
        self.variances[track] + self.fixture.measurement_variance
    }
}

/// A complete history from either enumeration.
#[derive(Clone, Debug)]
struct Leaf {
    key: String,
    scan_scores: Vec<f64>,
    score: f64,
}

fn enumerate_by_detection_labels(fixture: &Fixture) -> Vec<Leaf> {
    let root = Branch {
        names: fixture.priors.iter().map(|p| p.name.to_string()).collect(),
        means: fixture.priors.iter().map(|p| p.mean).collect(),
        variances: fixture.priors.iter().map(|p| p.variance).collect(),
        scans: Vec::new(),
        scan_scores: Vec::new(),
    };
    let mut branches = vec![root];
    for (index, detections) in fixture.scans.iter().enumerate() {
        branches = branches
            .iter()
            .flat_map(|branch| advance_branch(branch, index + 1, detections, fixture))
            .collect();
    }
    branches.iter().map(finish_branch).collect()
}

fn advance_branch(
    branch: &Branch,
    number: usize,
    detections: &[f64],
    fixture: &Fixture,
) -> Vec<Branch> {
    let predicted: Vec<f64> = branch
        .variances
        .iter()
        .map(|variance| variance + fixture.process_variance)
        .collect();
    let scan = ScanContext {
        number,
        detections,
        means: &branch.means,
        variances: &predicted,
        fixture,
    };
    let births = fixture.birth_density > 0.0;
    let candidates = label_product(branch.names.len(), detections.len(), births);
    filter_feasible(candidates, &scan)
        .into_iter()
        .map(|labels| child_branch(branch, labels, &scan))
        .collect()
}

/// Phase 1: every vector assigning each detection one of `Track(0..tracks)`,
/// `FalseAlarm` or, with births, `Birth`.
fn label_product(tracks: usize, detections: usize, births: bool) -> Vec<Vec<Label>> {
    let mut alphabet: Vec<Label> = (0..tracks).map(Label::Track).collect();
    alphabet.push(Label::FalseAlarm);
    if births {
        alphabet.push(Label::Birth);
    }
    let mut product = vec![Vec::new()];
    for _ in 0..detections {
        product = product
            .iter()
            .flat_map(|prefix: &Vec<Label>| {
                alphabet
                    .iter()
                    .map(move |&label| [prefix.as_slice(), &[label]].concat())
            })
            .collect();
    }
    product
}

/// Phase 2: keep label vectors in which no track takes two detections and
/// every assignment lies inside the gate.
fn filter_feasible(candidates: Vec<Vec<Label>>, scan: &ScanContext<'_>) -> Vec<Vec<Label>> {
    candidates
        .into_iter()
        .filter(|labels| tracks_are_exclusive(labels) && assignments_are_gated(labels, scan))
        .collect()
}

fn tracks_are_exclusive(labels: &[Label]) -> bool {
    let assigned: Vec<usize> = labels
        .iter()
        .filter_map(|label| match label {
            Label::Track(track) => Some(*track),
            _ => None,
        })
        .collect();
    let distinct: BTreeSet<usize> = assigned.iter().copied().collect();
    distinct.len() == assigned.len()
}

fn assignments_are_gated(labels: &[Label], scan: &ScanContext<'_>) -> bool {
    let Some(gate) = scan.fixture.gate else {
        return true;
    };
    labels
        .iter()
        .zip(scan.detections)
        .all(|(label, &z)| match *label {
            Label::Track(t) => (z - scan.means[t]).powi(2) / scan.innovation_variance(t) <= gate,
            _ => true,
        })
}

/// Phase 3: the scan's log score, summed per detection first and then over
/// the missed tracks, with every density evaluated in log space.
fn score_labels(labels: &[Label], scan: &ScanContext<'_>) -> f64 {
    let detection_terms: f64 = labels
        .iter()
        .zip(scan.detections)
        .map(|(&label, &z)| detection_term(label, z, scan))
        .sum();
    let missed = (0..scan.means.len())
        .filter(|&track| !labels.contains(&Label::Track(track)))
        .count();
    detection_terms + missed as f64 * (1.0 - scan.fixture.detection_probability).ln()
}

fn detection_term(label: Label, z: f64, scan: &ScanContext<'_>) -> f64 {
    let fixture = scan.fixture;
    match label {
        Label::Track(track) => {
            fixture.detection_probability.ln()
                + log_gaussian(z - scan.means[track], scan.innovation_variance(track))
                - fixture.false_alarm_density.ln()
        }
        Label::FalseAlarm => 0.0,
        Label::Birth => fixture.birth_density.ln() - fixture.false_alarm_density.ln(),
    }
}

fn log_gaussian(residual: f64, variance: f64) -> f64 {
    -0.5 * (residual * residual / variance + (2.0 * PI * variance).ln())
}

fn child_branch(parent: &Branch, labels: Vec<Label>, scan: &ScanContext<'_>) -> Branch {
    let score = score_labels(&labels, scan);
    let mut child = parent.clone();
    child.variances = scan.variances.to_vec();
    apply_labels(&mut child, &labels, scan);
    child.scans.push(ScanLabels {
        names: parent.names.clone(),
        labels,
    });
    child.scan_scores.push(score);
    child
}

/// Information-form update for assignments; births start at `(z, r)`.
fn apply_labels(child: &mut Branch, labels: &[Label], scan: &ScanContext<'_>) {
    let r = scan.fixture.measurement_variance;
    for (detection, (&label, &z)) in labels.iter().zip(scan.detections).enumerate() {
        match label {
            Label::Track(track) => information_update(child, track, z, r),
            Label::Birth => {
                child.names.push(format!("N{}z{}", scan.number, detection));
                child.means.push(z);
                child.variances.push(r);
            }
            Label::FalseAlarm => {}
        }
    }
}

fn information_update(branch: &mut Branch, track: usize, z: f64, r: f64) {
    let prior_variance = branch.variances[track];
    let variance = 1.0 / (1.0 / prior_variance + 1.0 / r);
    branch.means[track] = variance * (branch.means[track] / prior_variance + z / r);
    branch.variances[track] = variance;
}

fn finish_branch(branch: &Branch) -> Leaf {
    Leaf {
        key: key_history(&branch.scans),
        scan_scores: branch.scan_scores.clone(),
        // Summed last scan first: a different order from the oracle's.
        score: branch.scan_scores.iter().rev().sum(),
    }
}

/// Phase 4: the history key, in the notation of the oracle module docs.
fn key_history(scans: &[ScanLabels]) -> String {
    scans.iter().map(key_scan).collect::<Vec<_>>().join(" / ")
}

fn key_scan(scan: &ScanLabels) -> String {
    let track_events = scan.names.iter().enumerate().map(|(track, name)| {
        match scan
            .labels
            .iter()
            .position(|&label| label == Label::Track(track))
        {
            Some(detection) => format!("{name} z{detection}"),
            None => format!("{name} miss"),
        }
    });
    let unassigned = scan
        .labels
        .iter()
        .enumerate()
        .filter_map(|(detection, label)| match label {
            Label::FalseAlarm => Some(format!("z{detection} FA")),
            Label::Birth => Some(format!("z{detection} new")),
            Label::Track(_) => None,
        });
    track_events
        .chain(unassigned)
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Shared assertion helpers
// ---------------------------------------------------------------------------

fn assert_close(actual: f64, expected: f64, what: &str) {
    let tolerance = ATOL + RTOL * expected.abs();
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{what}: actual {actual:?}, expected {expected:?}, |Δ| {error:e} > {tolerance:e}"
    );
}

fn oracle_leaves(fixture: &Fixture) -> Vec<Leaf> {
    mht_oracle::enumerate(fixture)
        .iter()
        .map(|hypothesis| Leaf {
            key: hypothesis.history_key(),
            scan_scores: hypothesis.scan_scores.clone(),
            score: hypothesis.score,
        })
        .collect()
}

/// Leaves sorted by descending score. The order inside a tie group is
/// arbitrary and never asserted.
fn ranked(mut leaves: Vec<Leaf>) -> Vec<Leaf> {
    leaves.sort_by(|a, b| b.score.total_cmp(&a.score));
    leaves
}

/// Both enumerations of one fixture, each ranked, labelled for messages.
fn both_rankings(fixture: &Fixture) -> [(&'static str, Vec<Leaf>); 2] {
    [
        ("per-track recursion", ranked(oracle_leaves(fixture))),
        (
            "per-detection label product",
            ranked(enumerate_by_detection_labels(fixture)),
        ),
    ]
}

/// Scores whose neighbours lie within `TIE_TOLERANCE`, compared by membership.
#[derive(Debug)]
struct TieGroup {
    highest: f64,
    lowest: f64,
    keys: BTreeSet<String>,
}

fn tie_groups(ranked: &[Leaf]) -> Vec<TieGroup> {
    let mut groups: Vec<TieGroup> = Vec::new();
    for leaf in ranked {
        match groups.last_mut() {
            Some(group) if group.lowest - leaf.score <= TIE_TOLERANCE => {
                group.lowest = leaf.score;
                group.keys.insert(leaf.key.clone());
            }
            _ => groups.push(TieGroup {
                highest: leaf.score,
                lowest: leaf.score,
                keys: BTreeSet::from([leaf.key.clone()]),
            }),
        }
    }
    groups
}

fn tie_structure(groups: &[TieGroup]) -> TieStructure {
    let sizes: Vec<usize> = groups.iter().map(|group| group.keys.len()).collect();
    TieStructure {
        groups: sizes.len(),
        tied_groups: sizes.iter().filter(|&&size| size > 1).count(),
        largest: sizes.iter().copied().max().unwrap_or(0),
        adjacent_tied_pairs: sizes.iter().map(|size| size - 1).sum(),
    }
}

/// The number of whole groups that make up the first `ranks` ranks, or
/// `None` if a prefix of that length would end inside a tie group.
fn prefix_boundary(groups: &[TieGroup], ranks: usize) -> Option<usize> {
    let mut covered = 0;
    for (index, group) in groups.iter().enumerate() {
        if covered == ranks {
            return Some(index);
        }
        covered += group.keys.len();
    }
    (covered == ranks).then_some(groups.len())
}

/// The groups covering the first `ranks` ranks. Panics if the prefix would
/// end inside a tie group, so no assertion can depend on a tie-break.
fn ranked_prefix(groups: &[TieGroup], ranks: usize) -> &[TieGroup] {
    let whole = prefix_boundary(groups, ranks)
        .unwrap_or_else(|| panic!("a prefix of {ranks} ranks ends inside a tie group"));
    &groups[..whole]
}

/// Asserts the single history of a singleton group against a committed one.
fn assert_singleton(group: &TieGroup, expected: &Ranked, what: &str) {
    let keys: Vec<&str> = group.keys.iter().map(String::as_str).collect();
    assert_eq!(keys, [expected.history], "{what}: history");
    assert_close(group.highest, expected.score, what);
}

fn births_in(record: &ScanRecord, number: usize) -> Vec<String> {
    record
        .unassigned
        .iter()
        .filter(|&&(_, label)| label == DetectionLabel::NewTrack)
        .map(|&(detection, _)| format!("N{number}z{detection}"))
        .collect()
}

fn sorted(mut names: Vec<String>) -> Vec<String> {
    names.sort();
    names
}

fn assert_one_event_per_existing_track(hypothesis: &Hypothesis, fixture: &Fixture) {
    let mut existing: Vec<String> = fixture.priors.iter().map(|p| p.name.to_string()).collect();
    for (index, record) in hypothesis.scans.iter().enumerate() {
        let named = record
            .track_events
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        assert_eq!(
            sorted(named),
            sorted(existing.clone()),
            "{}: scan {} of {}",
            fixture.name,
            index + 1,
            hypothesis.history_key()
        );
        existing.extend(births_in(record, index + 1));
    }
}

fn detection_use_counts(record: &ScanRecord) -> Vec<usize> {
    let mut counts = vec![0; record.detections];
    let by_tracks = record
        .track_events
        .iter()
        .filter_map(|(_, event)| match event {
            TrackEvent::Detected(detection) => Some(*detection),
            TrackEvent::Missed => None,
        });
    let by_labels = record.unassigned.iter().map(|&(detection, _)| detection);
    for detection in by_tracks.chain(by_labels) {
        counts[detection] += 1;
    }
    counts
}

/// Largest per-scan or cumulative residual between the two enumerations,
/// after asserting that they produce the same set of histories.
fn order_residual(oracle: &[Leaf], labelled: &[Leaf], what: &str) -> f64 {
    let by_key: BTreeMap<&str, &Leaf> = labelled.iter().map(|l| (l.key.as_str(), l)).collect();
    let oracle_keys: BTreeSet<&str> = oracle.iter().map(|l| l.key.as_str()).collect();
    assert_eq!(
        by_key.len(),
        labelled.len(),
        "{what}: duplicate label-product keys"
    );
    assert_eq!(
        oracle_keys.len(),
        oracle.len(),
        "{what}: duplicate oracle keys"
    );
    assert!(
        by_key.keys().copied().eq(oracle_keys.iter().copied()),
        "{what}: the two enumerations produce different history sets"
    );
    oracle
        .iter()
        .map(|leaf| leaf_residual(leaf, by_key[leaf.key.as_str()]))
        .fold(0.0, f64::max)
}

fn leaf_residual(a: &Leaf, b: &Leaf) -> f64 {
    assert_eq!(a.scan_scores.len(), b.scan_scores.len(), "{}", a.key);
    // `f64::max` drops a NaN operand, so reject non-finite scores first.
    let all_scores = a.scan_scores.iter().chain(&b.scan_scores);
    for score in all_scores.chain([&a.score, &b.score]) {
        assert!(score.is_finite(), "{}: non-finite score {score}", a.key);
    }
    a.scan_scores
        .iter()
        .zip(&b.scan_scores)
        .map(|(x, y)| (x - y).abs())
        .fold((a.score - b.score).abs(), f64::max)
}

// ---------------------------------------------------------------------------
// Independence
// ---------------------------------------------------------------------------

#[test]
fn oracle_source_names_no_thresh_crate() {
    assert!(!ORACLE_SOURCE.is_empty());
    assert!(
        !ORACLE_SOURCE.contains("thresh_"),
        "the oracle must not name a workspace crate"
    );
}

#[test]
fn oracle_source_imports_only_std() {
    let imports: Vec<&str> = ORACLE_SOURCE
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("use "))
        .collect();
    assert!(!imports.is_empty());
    for import in imports {
        assert!(
            import.starts_with("use std::") || import.starts_with("use super::"),
            "non-std import in the oracle: {import}"
        );
    }
    assert!(!ORACLE_SOURCE.contains("crate::"));
    assert!(!ORACLE_SOURCE.contains("extern crate"));
}

// ---------------------------------------------------------------------------
// Counts and agreement between the two enumeration orders
// ---------------------------------------------------------------------------

#[test]
fn leaf_counts_match_committed_values_within_hypothesis_bound() {
    for reference in references() {
        let name = reference.fixture.name;
        assert!(reference.leaves <= HYPOTHESIS_BOUND, "{name}");
        for (order, leaves) in both_rankings(&reference.fixture) {
            assert_eq!(leaves.len(), reference.leaves, "{name}: {order}");
        }
    }
}

#[test]
fn enumeration_orders_agree_on_histories_and_scores() {
    let mut worst: f64 = 0.0;
    for reference in references() {
        let fixture = &reference.fixture;
        let oracle = oracle_leaves(fixture);
        let labelled = enumerate_by_detection_labels(fixture);
        let residual = order_residual(&oracle, &labelled, fixture.name);
        assert!(
            residual <= ORDER_TOLERANCE,
            "{}: enumeration orders differ by {residual:e}",
            fixture.name
        );
        worst = worst.max(residual);
    }
    println!("maximum residual between enumeration orders: {worst:e}");
}

// ---------------------------------------------------------------------------
// Structural assertions on the reference
// ---------------------------------------------------------------------------

#[test]
fn no_detection_is_reused_within_a_scan() {
    for reference in references() {
        for hypothesis in mht_oracle::enumerate(&reference.fixture) {
            for record in &hypothesis.scans {
                let counts = detection_use_counts(record);
                assert!(
                    counts.iter().all(|&count| count == 1),
                    "{}: {} uses detections {counts:?} times",
                    reference.fixture.name,
                    hypothesis.history_key()
                );
            }
        }
    }
}

#[test]
fn every_existing_track_has_one_event_per_scan() {
    for reference in references() {
        let fixture = &reference.fixture;
        for hypothesis in mht_oracle::enumerate(fixture) {
            assert_eq!(hypothesis.scans.len(), fixture.scans.len());
            assert_one_event_per_existing_track(&hypothesis, fixture);
        }
    }
}

#[test]
fn cumulative_scores_equal_sum_of_scan_scores() {
    for reference in references() {
        let fixture = &reference.fixture;
        for hypothesis in mht_oracle::enumerate(fixture) {
            assert_eq!(hypothesis.scan_scores.len(), fixture.scans.len());
            let sum: f64 = hypothesis.scan_scores.iter().sum();
            assert_close(hypothesis.score, sum, fixture.name);
        }
    }
}

// ---------------------------------------------------------------------------
// Committed values
// ---------------------------------------------------------------------------

#[test]
fn best_histories_and_scores_match_committed_values() {
    for reference in references() {
        for (order, leaves) in both_rankings(&reference.fixture) {
            let what = format!("{} best ({order})", reference.fixture.name);
            let groups = tie_groups(&leaves);
            let prefix = ranked_prefix(&groups, 1);
            assert_singleton(&prefix[0], &reference.best, &what);
        }
    }
}

#[test]
fn runner_up_histories_scores_and_gaps_match_committed_values() {
    for reference in references() {
        for (order, leaves) in both_rankings(&reference.fixture) {
            let what = format!("{} runner-up ({order})", reference.fixture.name);
            let groups = tie_groups(&leaves);
            let prefix = ranked_prefix(&groups, 2);
            assert_singleton(&prefix[1], &reference.runner_up, &what);
            assert_close(prefix[0].highest - prefix[1].highest, reference.gap, &what);
            let rank_three = groups[2].highest;
            assert!(prefix[1].lowest - rank_three >= RUNNER_UP_MARGIN, "{what}");
        }
    }
}

#[test]
fn c1_per_scan_scores_match_hand_anchors() {
    let references = references();
    let c1 = &references[0];
    for (order, leaves) in both_rankings(&c1.fixture) {
        let what = format!("C1 ({order})");
        // Ranks 1 and 2 are whole tie groups, so reading them is order-free.
        ranked_prefix(&tie_groups(&leaves), 2);
        let best = &leaves[0];
        let runner_up = &leaves[1];
        assert_eq!(best.key, c1.best.history, "{what}");
        assert_eq!(runner_up.key, c1.runner_up.history, "{what}");
        assert_close(best.scan_scores[0], C1_SCAN1_A_TAKES, &what);
        assert_close(best.scan_scores[1], C1_SCAN2_BEST, &what);
        assert_close(best.scan_scores[2], C1_SCAN3_BEST, &what);
        assert_close(best.score, C1_TOTAL_BEST, &what);
        assert_close(runner_up.scan_scores[0], C1_SCAN1_B_TAKES, &what);
    }
}

/// `ln(P_D N(d; 0, S) / λ_FA) + ln(1 − P_D)` for one assignment and one miss.
fn closed_form_scan(residual: f64, innovation_variance: f64) -> f64 {
    (0.9_f64 / 0.05).ln()
        - residual * residual / (2.0 * innovation_variance)
        - 0.5 * (2.0 * PI * innovation_variance).ln()
        + 0.1_f64.ln()
}

#[test]
fn c1_hand_anchors_follow_from_closed_form() {
    // Scan 1: either prior predicts to P⁻ = 1.5, so S = 2. A sees residual
    // 2.1, B sees −1.9.
    assert_close(
        closed_form_scan(2.1, 2.0),
        C1_SCAN1_A_TAKES,
        "scan 1, A takes",
    );
    assert_close(
        closed_form_scan(-1.9, 2.0),
        C1_SCAN1_B_TAKES,
        "scan 1, B takes",
    );
    // Scan 2 after A took 2.1: K = 0.75, m = 1.575, P = 0.375, so P⁻ = 0.875,
    // S = 1.375 and the residual to 3.0 is 1.425. B is missed.
    assert_close(closed_form_scan(1.425, 1.375), C1_SCAN2_BEST, "scan 2");
    // Scan 3: B missed twice, so P⁻ = 1 + 3 × 0.5 = 2.5, S = 3 and the
    // residual to 5.0 is 1. A is missed.
    assert_close(closed_form_scan(1.0, 3.0), C1_SCAN3_BEST, "scan 3");
    assert_close(
        C1_SCAN1_A_TAKES + C1_SCAN2_BEST + C1_SCAN3_BEST,
        C1_TOTAL_BEST,
        "total",
    );
}

#[test]
fn c4_best_history_carries_a_birth_across_a_miss() {
    let fixture = c4_birth_across_a_miss();
    assert_eq!(fixture.scans[0][1], 5.0);
    assert_eq!(fixture.scans[2][1], 5.2);
    let best = mht_oracle::enumerate(&fixture)
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("C4 has hypotheses");
    let born = (1, DetectionLabel::NewTrack);
    assert!(
        best.scans[0].unassigned.contains(&born),
        "born from 5.0 at scan 1"
    );
    let newborn = |record: &ScanRecord| {
        record
            .track_events
            .iter()
            .find(|(name, _)| name == "N1z1")
            .map(|&(_, event)| event)
    };
    assert_eq!(
        newborn(&best.scans[1]),
        Some(TrackEvent::Missed),
        "missed at scan 2"
    );
    assert_eq!(
        newborn(&best.scans[2]),
        Some(TrackEvent::Detected(1)),
        "5.2 at scan 3"
    );
}

// ---------------------------------------------------------------------------
// Ranking as ordered tie groups
// ---------------------------------------------------------------------------

#[test]
fn ranking_tie_groups_agree_between_enumeration_orders() {
    for reference in references() {
        let [(_, oracle), (_, labelled)] = both_rankings(&reference.fixture);
        let (oracle, labelled) = (tie_groups(&oracle), tie_groups(&labelled));
        let name = reference.fixture.name;
        assert_eq!(oracle.len(), labelled.len(), "{name}: group count");
        for (rank, (a, b)) in oracle.iter().zip(&labelled).enumerate() {
            assert_eq!(a.keys, b.keys, "{name}: group {}", rank + 1);
            assert!((a.highest - b.highest).abs() <= ORDER_TOLERANCE, "{name}");
            assert!((a.lowest - b.lowest).abs() <= ORDER_TOLERANCE, "{name}");
        }
    }
}

#[test]
fn tie_group_structure_matches_committed_values() {
    for reference in references() {
        for (order, leaves) in both_rankings(&reference.fixture) {
            let groups = tie_groups(&leaves);
            assert_eq!(
                tie_structure(&groups),
                reference.ties,
                "{} ({order})",
                reference.fixture.name
            );
        }
    }
}

#[test]
fn c2_ranks_ten_and_eleven_form_the_only_tie_group() {
    for (order, leaves) in both_rankings(&c2_two_target_confirmation()) {
        let groups = tie_groups(&leaves);
        let through_tie = ranked_prefix(&groups, C2_TIE_RANK + 1);
        assert_eq!(through_tie.len(), C2_TIE_RANK, "{order}");
        let tie = &through_tie[C2_TIE_RANK - 1];
        assert_eq!(tie.keys.len(), 2, "{order}");
        assert_close(tie.highest, C2_TIE_SCORE, order);
        assert_close(tie.lowest, C2_TIE_SCORE, order);
        let mut others = groups
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != C2_TIE_RANK - 1);
        assert!(others.all(|(_, group)| group.keys.len() == 1), "{order}");
    }
}

#[test]
fn asserted_prefixes_end_on_tie_group_boundaries() {
    for reference in references() {
        let groups = tie_groups(&ranked(oracle_leaves(&reference.fixture)));
        let name = reference.fixture.name;
        // The best and runner-up prefixes each end on a group boundary.
        assert_eq!(prefix_boundary(&groups, 1), Some(1), "{name}");
        assert_eq!(prefix_boundary(&groups, 2), Some(2), "{name}");
        assert_eq!(
            prefix_boundary(&groups, reference.leaves),
            Some(groups.len())
        );
    }
    // A prefix ending between C2 ranks 10 and 11 would split the tie.
    let groups = tie_groups(&ranked(oracle_leaves(&c2_two_target_confirmation())));
    assert_eq!(
        prefix_boundary(&groups, C2_TIE_RANK - 1),
        Some(C2_TIE_RANK - 1)
    );
    assert_eq!(prefix_boundary(&groups, C2_TIE_RANK), None);
    assert_eq!(prefix_boundary(&groups, C2_TIE_RANK + 1), Some(C2_TIE_RANK));
}

#[test]
fn tie_groups_are_separated_far_above_tie_tolerance() {
    let mut smallest = f64::INFINITY;
    for reference in references() {
        let groups = tie_groups(&ranked(oracle_leaves(&reference.fixture)));
        for pair in groups.windows(2) {
            smallest = smallest.min(pair[0].lowest - pair[1].highest);
        }
        for group in &groups {
            assert!(group.highest - group.lowest <= TIE_TOLERANCE);
        }
    }
    // design.md records 0.0046 (C4) as the smallest separation.
    assert!(smallest > 1e-3, "smallest group separation {smallest:e}");
}

#[test]
fn map_without_pruning_matches_committed_best_history() {
    for reference in references() {
        let fixture = &reference.fixture;
        let enumerations = [
            oracle_leaves(fixture),
            enumerate_by_detection_labels(fixture),
        ];
        for leaves in enumerations {
            let map = leaves
                .iter()
                .max_by(|a, b| a.score.total_cmp(&b.score))
                .expect("every fixture has hypotheses");
            let at_map = leaves
                .iter()
                .filter(|leaf| map.score - leaf.score <= TIE_TOLERANCE)
                .count();
            assert_eq!(at_map, 1, "{}: the MAP is unique", fixture.name);
            assert_eq!(map.key, reference.best.history, "{}", fixture.name);
            assert_close(map.score, reference.best.score, fixture.name);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase tests for the label-product enumeration
// ---------------------------------------------------------------------------

fn unit_scan<'a>(
    detections: &'a [f64],
    means: &'a [f64],
    variances: &'a [f64],
    fixture: &'a Fixture,
) -> ScanContext<'a> {
    ScanContext {
        number: 1,
        detections,
        means,
        variances,
        fixture,
    }
}

#[test]
fn label_product_enumerates_every_label_vector() {
    use Label::{Birth, FalseAlarm, Track};
    let product = label_product(2, 2, false);
    assert_eq!(product.len(), 9);
    assert_eq!(product[0], [Track(0), Track(0)]);
    assert_eq!(product[8], [FalseAlarm, FalseAlarm]);
    let distinct: BTreeSet<String> = product.iter().map(|v| format!("{v:?}")).collect();
    assert_eq!(distinct.len(), 9);
    assert_eq!(label_product(2, 2, true).len(), 16);
    assert!(label_product(1, 1, true).contains(&vec![Birth]));
    assert_eq!(label_product(3, 0, true), vec![Vec::<Label>::new()]);
}

#[test]
fn filter_feasible_rejects_shared_tracks_and_gated_detections() {
    use Label::{FalseAlarm, Track};
    let fixture = c3_recent_agreement();
    // S = 2 for both tracks. z = 5: A gives 12.5 > 9, B gives 0.5 <= 9.
    let scan = unit_scan(&[5.0, 0.0], &[0.0, 4.0], &[1.5, 1.5], &fixture);
    let kept = filter_feasible(label_product(2, 2, false), &scan);
    let expected = vec![
        vec![Track(1), Track(0)],
        vec![Track(1), FalseAlarm],
        vec![FalseAlarm, Track(0)],
        vec![FalseAlarm, Track(1)],
        vec![FalseAlarm, FalseAlarm],
    ];
    assert_eq!(kept, expected);
}

#[test]
fn score_labels_sums_detection_terms_and_misses() {
    use Label::{Birth, FalseAlarm, Track};
    let fixture = c4_birth_across_a_miss();
    // One track at 0 with P⁻ = 0.75, so S = 1.25.
    let scan = unit_scan(&[0.5, 9.0, -4.0], &[0.0], &[0.75], &fixture);
    let assigned = (0.9_f64 / 0.01).ln() - 0.25 / 2.5 - 0.5 * (2.5 * PI).ln();
    let birth = 2.0_f64.ln();
    let score = score_labels(&[Track(0), Birth, FalseAlarm], &scan);
    assert!((score - (assigned + birth)).abs() <= 1e-14, "{score:?}");
    let all_false = score_labels(&[FalseAlarm, FalseAlarm, FalseAlarm], &scan);
    assert!((all_false - 0.1_f64.ln()).abs() <= 1e-14, "{all_false:?}");
}

#[test]
fn key_history_renders_tracks_then_unassigned_detections() {
    use Label::{Birth, FalseAlarm, Track};
    let scans = [
        ScanLabels {
            names: vec!["A".to_string()],
            labels: vec![Track(0), Birth],
        },
        ScanLabels {
            names: vec!["A".to_string(), "N1z1".to_string()],
            labels: vec![FalseAlarm, Track(1)],
        },
        ScanLabels {
            names: vec!["A".to_string(), "N1z1".to_string()],
            labels: Vec::new(),
        },
    ];
    assert_eq!(
        key_history(&scans),
        "A z0, z1 new / A miss, N1z1 z1, z0 FA / A miss, N1z1 miss"
    );
}
