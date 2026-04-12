//! Multi-path disambiguation for over-the-horizon radar.
//!
//! OTHR signals can reach a target via E-layer or F-layer reflection (and
//! multi-hop paths). This module provides tools to compute the different
//! apparent ground ranges and to determine the most likely propagation mode.

use thresh_core::measurement::PropagationMode;

use crate::ionosphere::{self, IonosphereParams, Layer};

/// Compute apparent ground ranges for E-layer and F-layer single-hop paths.
///
/// Returns `(e_layer_range_km, f_layer_range_km)`. The F-layer path travels
/// through a higher virtual height, so its group delay (and hence apparent
/// ground range) is larger than the E-layer path for the same true target range.
pub fn multipath_ranges(target_range_km: f64, _params: &IonosphereParams) -> (f64, f64) {
    let vh_e = ionosphere::virtual_height_km(target_range_km, Layer::E);
    let vh_f = ionosphere::virtual_height_km(target_range_km, Layer::F);

    // Slant range via each layer
    let half = target_range_km / 2.0;
    let slant_e = (half.powi(2) + vh_e.powi(2)).sqrt();
    let slant_f = (half.powi(2) + vh_f.powi(2)).sqrt();

    // Apparent ground range is derived from slant range assuming flat geometry
    // The actual ground range plus an excess from the higher path
    let apparent_e = 2.0 * (slant_e.powi(2) - vh_e.powi(2)).sqrt();
    let apparent_f_excess = 2.0 * slant_f - 2.0 * slant_e;
    let apparent_f = target_range_km + apparent_f_excess;

    (apparent_e, apparent_f)
}

/// Disambiguate an observed range to the most likely propagation layer.
///
/// If the operating frequency exceeds the E-layer MUF for this range the
/// signal cannot have reflected off the E-layer, so it must be F-layer.
/// Otherwise the E-layer (shorter path, lower virtual height) is preferred.
pub fn disambiguate(
    observed_range_km: f64,
    params: &IonosphereParams,
    freq_mhz: f64,
) -> PropagationMode {
    // E-layer MUF at this range
    let vh_e = ionosphere::virtual_height_km(observed_range_km, Layer::E);
    let e_muf = ionosphere::muf(params.fo_e_mhz, observed_range_km, vh_e);

    if freq_mhz > e_muf {
        // E-layer cannot support this frequency — must be F-layer or multi-hop
        let vh_f = ionosphere::virtual_height_km(observed_range_km, Layer::F);
        let f_muf = ionosphere::muf(params.fo_f2_mhz, observed_range_km, vh_f);
        if freq_mhz > f_muf {
            // Single-hop F-layer can't support it either — multi-hop
            PropagationMode::MultiHop(2)
        } else {
            PropagationMode::FLayer
        }
    } else {
        PropagationMode::ELayer
    }
}

/// Check whether a multi-hop (2-hop) F-layer path is viable at the given
/// range and frequency.
///
/// A 2-hop path covers roughly twice the single-hop range. We check that the
/// frequency is below the MUF for the single-hop segment (half the total range).
pub fn is_multihop_viable(range_km: f64, params: &IonosphereParams, freq_mhz: f64) -> bool {
    let single_hop_range = range_km / 2.0;
    let vh = ionosphere::virtual_height_km(single_hop_range, Layer::F);
    let segment_muf = ionosphere::muf(params.fo_f2_mhz, single_hop_range, vh);
    freq_mhz <= segment_muf
}

// =========================================================================
// Tests — Tasks 5.4-5.5
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use thresh_core::measurement::PropagationMode;

    fn default_params() -> IonosphereParams {
        IonosphereParams {
            fo_f2_mhz: 8.0,
            hm_f2_km: 300.0,
            scale_height_km: 80.0,
            fo_e_mhz: 3.5,
            hm_e_km: 110.0,
        }
    }

    // Task 5.4 — Disambiguation selects F when E-layer MUF is exceeded
    #[test]
    fn disambiguate_selects_f_when_e_muf_exceeded() {
        let params = default_params();
        let range = 1000.0;

        // Compute the E-layer and F-layer MUFs at this range
        let vh_e = ionosphere::virtual_height_km(range, Layer::E);
        let e_muf = ionosphere::muf(params.fo_e_mhz, range, vh_e);
        let vh_f = ionosphere::virtual_height_km(range, Layer::F);
        let f_muf = ionosphere::muf(params.fo_f2_mhz, range, vh_f);

        // Pick a frequency between E-layer MUF and F-layer MUF
        let freq = (e_muf + f_muf) / 2.0;
        assert!(
            freq > e_muf,
            "Test freq {freq} should exceed E-layer MUF {e_muf}"
        );
        assert!(
            freq < f_muf,
            "Test freq {freq} should be below F-layer MUF {f_muf}"
        );

        let mode = disambiguate(range, &params, freq);
        assert_eq!(mode, PropagationMode::FLayer);
    }

    #[test]
    fn disambiguate_selects_e_when_possible() {
        let params = default_params();
        // Use a very low frequency that E-layer can support
        let freq = 2.0;
        let range = 500.0;
        let mode = disambiguate(range, &params, freq);
        assert_eq!(mode, PropagationMode::ELayer);
    }

    // Task 5.5 — Multi-hop viable at ~2x single-hop range
    #[test]
    fn multihop_viable_at_double_range() {
        let params = default_params();
        let freq = 7.0;

        // Find a single-hop range where F-layer works
        let single_hop_range = 1000.0;
        let vh = ionosphere::virtual_height_km(single_hop_range, Layer::F);
        let f_muf = ionosphere::muf(params.fo_f2_mhz, single_hop_range, vh);
        assert!(
            freq < f_muf,
            "Frequency should be below single-hop MUF for test"
        );

        // Multi-hop should be viable at 2x that range
        let double_range = 2.0 * single_hop_range;
        assert!(
            is_multihop_viable(double_range, &params, freq),
            "Multi-hop should be viable at 2x single-hop range"
        );
    }

    #[test]
    fn multihop_not_viable_above_muf() {
        let params = default_params();
        // Frequency way above F-layer MUF
        let freq = 50.0;
        let range = 2000.0;
        assert!(
            !is_multihop_viable(range, &params, freq),
            "Multi-hop should not be viable above MUF"
        );
    }

    #[test]
    fn multipath_ranges_f_exceeds_e() {
        let params = default_params();
        let (e_range, f_range) = multipath_ranges(1000.0, &params);
        assert!(
            f_range > e_range,
            "F-layer apparent range ({f_range}) should exceed E-layer ({e_range})"
        );
    }
}
