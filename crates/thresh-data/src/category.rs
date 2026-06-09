//! Object-category mapping for dataset ingestion.
//!
//! Pure string→[`TargetClass`] logic with no dataset-runtime dependency, so it
//! lives outside the (PyO3-gated) `nuscenes` module and is always compiled and
//! unit-tested in the default build.

use thresh_core::track::TargetClass;

/// Map a nuScenes category name to a thresh [`TargetClass`].
///
/// Maps each nuScenes object category to a native automotive class. The mapping
/// is **lossless** in the sense the `automotive-tracking-pipeline` change
/// requires: no automotive category (`vehicle.*` / `human.pedestrian.*`)
/// resolves to an aerospace class. Only genuinely non-automotive categories
/// (`animal`, `movable_object.*`, `static_object.*`) and unrecognized strings
/// fall back to [`TargetClass::Unknown`].
///
/// A few coarse nuScenes types are folded into the nearest concrete class for
/// now — `vehicle.trailer` / `vehicle.construction` → [`TargetClass::Truck`],
/// `vehicle.emergency.*` → [`TargetClass::Car`]. When finer classes are added
/// (e.g. `Trailer`, `ConstructionVehicle`, `EmergencyVehicle`) these arms
/// should split out; matching is prefix-based and ordered most-specific-first.
pub fn map_category(nuscenes_category: &str) -> TargetClass {
    let c = nuscenes_category;
    if c.starts_with("human.pedestrian") {
        TargetClass::Pedestrian
    } else if c.starts_with("vehicle.bicycle") {
        TargetClass::Bicycle
    } else if c.starts_with("vehicle.motorcycle") {
        TargetClass::Motorcycle
    } else if c.starts_with("vehicle.bus") {
        TargetClass::Bus
    } else if c.starts_with("vehicle.truck")
        || c.starts_with("vehicle.trailer")
        || c.starts_with("vehicle.construction")
    {
        TargetClass::Truck
    } else if c.starts_with("vehicle.car") || c.starts_with("vehicle.emergency") {
        TargetClass::Car
    } else {
        TargetClass::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_category_non_automotive_is_unknown() {
        // Genuinely non-automotive categories fall back to Unknown.
        assert_eq!(
            map_category("static_object.bicycle_rack"),
            TargetClass::Unknown
        );
        assert_eq!(map_category("movable_object.barrier"), TargetClass::Unknown);
        assert_eq!(map_category("movable_object.debris"), TargetClass::Unknown);
        assert_eq!(map_category("animal"), TargetClass::Unknown);
        assert_eq!(map_category("flying.saucer"), TargetClass::Unknown);
    }

    #[test]
    fn map_category_road_vehicles() {
        assert_eq!(map_category("vehicle.car"), TargetClass::Car);
        assert_eq!(map_category("vehicle.truck"), TargetClass::Truck);
        assert_eq!(map_category("vehicle.bus.rigid"), TargetClass::Bus);
        assert_eq!(map_category("vehicle.bus.bendy"), TargetClass::Bus);
        // Coarse types folded into the nearest concrete class for now.
        assert_eq!(map_category("vehicle.trailer"), TargetClass::Truck);
        assert_eq!(map_category("vehicle.construction"), TargetClass::Truck);
        assert_eq!(map_category("vehicle.emergency.police"), TargetClass::Car);
        assert_eq!(
            map_category("vehicle.emergency.ambulance"),
            TargetClass::Car
        );
    }

    #[test]
    fn map_category_vulnerable_road_users() {
        assert_eq!(
            map_category("human.pedestrian.adult"),
            TargetClass::Pedestrian
        );
        assert_eq!(
            map_category("human.pedestrian.child"),
            TargetClass::Pedestrian
        );
        assert_eq!(
            map_category("human.pedestrian.construction_worker"),
            TargetClass::Pedestrian
        );
        assert_eq!(map_category("vehicle.motorcycle"), TargetClass::Motorcycle);
        assert_eq!(map_category("vehicle.bicycle"), TargetClass::Bicycle);
    }

    /// The core guarantee: no automotive category maps to an aerospace class.
    #[test]
    fn map_category_no_automotive_maps_to_aerospace() {
        // The full nuScenes detection category list.
        let automotive = [
            "vehicle.car",
            "vehicle.truck",
            "vehicle.bus.bendy",
            "vehicle.bus.rigid",
            "vehicle.trailer",
            "vehicle.construction",
            "vehicle.emergency.ambulance",
            "vehicle.emergency.police",
            "vehicle.motorcycle",
            "vehicle.bicycle",
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.wheelchair",
            "human.pedestrian.stroller",
            "human.pedestrian.personal_mobility",
            "human.pedestrian.police_officer",
            "human.pedestrian.construction_worker",
        ];
        let aerospace = [
            TargetClass::Aircraft,
            TargetClass::Ballistic,
            TargetClass::Uav,
            TargetClass::Orbital,
        ];
        for cat in automotive {
            let mapped = map_category(cat);
            assert!(
                !aerospace.contains(&mapped),
                "automotive category {cat} wrongly mapped to aerospace class {mapped:?}"
            );
            assert_ne!(
                mapped,
                TargetClass::Unknown,
                "automotive category {cat} should map to a concrete automotive class, not Unknown"
            );
        }
    }
}
