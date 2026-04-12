#![cfg(feature = "rcs-compute")]

use thresh_synth::rcs_compute::*;

#[test]
#[ignore] // requires pofacets Python package and a test STL file
fn sphere_load_and_sweep() {
    let bridge = RcsComputeBridge::new("/tmp/sphere.stl");
    let config = RcsSweepConfig::default();
    let _result = bridge.sweep_hemisphere(&config).expect("sweep");
}
