//! Integration tests for the nuScenes bridge.
//!
//! These tests require the `nuscenes` feature, the `nuscenes-devkit` Python
//! package installed in the active interpreter, and the `v1.0-mini` split
//! available under `/data/nuscenes`. They are `#[ignore]`d so that CI does
//! not fail when that environment is unavailable. Run locally with:
//!
//! ```sh
//! cargo test -p thresh-data --features nuscenes -- --ignored
//! ```

#![cfg(feature = "nuscenes")]

use thresh_data::dataset::Dataset;
use thresh_data::nuscenes::{NuScenesBridge, NuScenesDataset};

const DATAROOT: &str = "/data/nuscenes";
const VERSION: &str = "v1.0-mini";

#[test]
#[ignore]
fn load_mini_split_scene_count() {
    let nusc = NuScenesBridge::new(VERSION, DATAROOT).expect("load");
    let count = nusc.scene_count().expect("scene count");
    assert!(count > 0, "expected at least one scene in mini split");
}

#[test]
#[ignore]
fn load_first_scene_samples() {
    let nusc = NuScenesBridge::new(VERSION, DATAROOT).expect("load");
    let tokens = nusc.scene_tokens().expect("scene tokens");
    let first = tokens.first().expect("at least one scene");
    let samples = nusc.iter_samples(first).expect("samples");
    assert!(!samples.is_empty(), "scene should contain samples");
}

#[test]
#[ignore]
fn scene_annotations_are_nonempty() {
    let nusc = NuScenesBridge::new(VERSION, DATAROOT).expect("load");
    let tokens = nusc.scene_tokens().expect("scene tokens");
    let first = tokens.first().expect("at least one scene");
    let samples = nusc.iter_samples(first).expect("samples");
    let first_sample = samples.first().expect("at least one sample");
    let anns = nusc
        .sample_annotations(&first_sample.token)
        .expect("annotations");
    // Keyframes in the mini split should have at least one annotation.
    assert!(!anns.is_empty());
}

#[test]
#[ignore]
fn dataset_frames_match_samples() {
    let bridge = NuScenesBridge::new(VERSION, DATAROOT).expect("load");
    let tokens = bridge.scene_tokens().expect("scene tokens");
    let first = tokens.first().expect("at least one scene");
    let samples = bridge.iter_samples(first).expect("samples");
    let dataset = NuScenesDataset::load(VERSION, DATAROOT, first).expect("dataset");
    let frame_count = dataset.frames().count();
    assert_eq!(frame_count, samples.len());
}
