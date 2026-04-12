//! Pre-processing utilities for detection pipelines.
//!
//! Provides voxelization for point-cloud inputs and image normalization for
//! camera-based inputs.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Point-cloud voxelization
// ---------------------------------------------------------------------------

/// Group 3D points into voxel buckets based on `floor(point / voxel_size)`.
///
/// Each bucket is capped at `max_per_voxel` points. Returns a `Vec` of voxel
/// buckets, each containing the points that fell into that voxel.
pub fn voxelize(
    points: &[[f64; 3]],
    voxel_size: [f64; 3],
    max_per_voxel: usize,
) -> Vec<Vec<[f64; 3]>> {
    let mut buckets: HashMap<[i64; 3], Vec<[f64; 3]>> = HashMap::new();

    for &pt in points {
        let key = [
            (pt[0] / voxel_size[0]).floor() as i64,
            (pt[1] / voxel_size[1]).floor() as i64,
            (pt[2] / voxel_size[2]).floor() as i64,
        ];
        let bucket = buckets.entry(key).or_default();
        if bucket.len() < max_per_voxel {
            bucket.push(pt);
        }
    }

    buckets.into_values().collect()
}

// ---------------------------------------------------------------------------
// Image normalization
// ---------------------------------------------------------------------------

/// Per-channel normalization: `(pixel - mean) / std`.
///
/// Assumes the data is laid out in CHW order (channel-first) with `channels`
/// channels. The total length must be divisible by `channels`.
pub fn normalize_image(data: &[f32], mean: [f32; 3], std: [f32; 3], channels: usize) -> Vec<f32> {
    assert!(channels <= 3, "normalize_image supports at most 3 channels");
    let pixels_per_channel = data.len() / channels;
    assert_eq!(
        data.len(),
        pixels_per_channel * channels,
        "data length must be divisible by channels"
    );

    let mut out = vec![0.0f32; data.len()];
    for c in 0..channels {
        let offset = c * pixels_per_channel;
        for i in 0..pixels_per_channel {
            out[offset + i] = (data[offset + i] - mean[c]) / std[c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxelize_groups_points() {
        let points = vec![
            [0.05, 0.05, 0.05], // voxel (0,0,0)
            [0.06, 0.03, 0.09], // voxel (0,0,0)
            [1.05, 0.05, 0.05], // voxel (10,0,0) with voxel_size 0.1
        ];
        let voxels = voxelize(&points, [0.1, 0.1, 0.1], 10);
        assert_eq!(voxels.len(), 2, "should have 2 voxel buckets");

        let total: usize = voxels.iter().map(|v| v.len()).sum();
        assert_eq!(total, 3, "all 3 points should be placed");
    }

    #[test]
    fn test_voxelize_caps_per_voxel() {
        // All points in the same voxel, but cap at 2
        let points = vec![
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03],
            [0.04, 0.04, 0.04],
        ];
        let voxels = voxelize(&points, [0.1, 0.1, 0.1], 2);
        assert_eq!(voxels.len(), 1);
        assert_eq!(voxels[0].len(), 2, "should cap at max_per_voxel=2");
    }

    #[test]
    fn test_voxelize_empty_input() {
        let voxels = voxelize(&[], [0.1, 0.1, 0.1], 10);
        assert!(voxels.is_empty());
    }

    #[test]
    fn test_voxelize_single_voxel() {
        let points = vec![[0.05, 0.05, 0.05]];
        let voxels = voxelize(&points, [1.0, 1.0, 1.0], 100);
        assert_eq!(voxels.len(), 1);
        assert_eq!(voxels[0].len(), 1);
    }

    #[test]
    fn test_normalize_image_basic() {
        // 3 channels, 2 pixels each => length 6
        let data = vec![0.5, 0.6, 0.3, 0.4, 0.7, 0.8];
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let result = normalize_image(&data, mean, std, 3);
        assert_eq!(result.len(), 6);

        // Check first pixel of first channel: (0.5 - 0.485) / 0.229
        let expected = (0.5 - 0.485) / 0.229;
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            result[0]
        );
    }

    #[test]
    fn test_normalize_image_identity() {
        // mean=0, std=1 should return original data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = normalize_image(&data, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3);
        for (a, b) in data.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
