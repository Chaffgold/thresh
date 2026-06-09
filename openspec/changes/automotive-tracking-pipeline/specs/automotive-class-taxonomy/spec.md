## Capability: Automotive Class Taxonomy

### Overview

Native automotive object classes for the tracker's `TargetClass`, plus a
lossless nuScenes category mapping that replaces the current best-effort
workaround (which types road vehicles as `Aircraft` and small agents as `Uav`).
Aerospace classes are retained so existing tracking is unaffected.

## ADDED Requirements

### Requirement: Automotive target classes

`TargetClass` MUST include native automotive variants covering at least cars,
trucks, buses, motorcycles, bicycles, and pedestrians, while retaining the
existing aerospace variants and the type's serialization.

#### Scenario: Track carries a native automotive class

**WHEN** a track is created for a road vehicle or vulnerable road user

**THEN** its `TargetClass` is a native automotive variant (e.g. `Car`, `Truck`,
`Pedestrian`) rather than an aerospace stand-in.

**SHALL** round-trip through serialization to the same variant.

#### Scenario: Aerospace classes remain available

**WHEN** existing aerospace tracking constructs a track with `Aircraft`,
`Ballistic`, `Uav`, or `Orbital`

**THEN** those variants still exist and behave exactly as before this change.

### Requirement: Lossless nuScenes class mapping

The nuScenes category mapping MUST translate each nuScenes category to a
semantically faithful `TargetClass`. An automotive nuScenes category (vehicle or
vulnerable-road-user) MUST NOT resolve to an aerospace class. "Lossless" here
means every automotive category maps to a distinct automotive class; only
genuinely non-automotive / unrecognized categories use the `Unknown` fallback.

#### Scenario: Road-vehicle categories map to vehicle classes

**WHEN** a nuScenes object with category `vehicle.car`, `vehicle.truck`, or
`vehicle.bus` is ingested

**THEN** it maps to the corresponding automotive `TargetClass` (e.g. `Car`,
`Truck`, `Bus`)

**SHALL** never map to `Aircraft` or any other aerospace class.

#### Scenario: Vulnerable-road-user categories map faithfully

**WHEN** a nuScenes object with category `human.pedestrian.*`,
`vehicle.motorcycle`, or `vehicle.bicycle` is ingested

**THEN** it maps to `Pedestrian`, `Motorcycle`, or `Bicycle` respectively.

#### Scenario: Genuinely non-automotive categories fall back explicitly

**WHEN** a nuScenes object has a category that is not an automotive vehicle or
vulnerable-road-user (e.g. `static_object.*`, `movable_object.debris`) or is
otherwise unrecognized

**THEN** it maps to `Unknown` and ingestion continues without error.
