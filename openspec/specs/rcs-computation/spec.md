# rcs-computation Specification

## Purpose
TBD - created by archiving change hifi-sensor-simulation. Update Purpose after archive.
## Requirements
### Requirement: Feature-gated RCS computation module
The system MUST gate all RCS computation functionality behind the `rcs-compute` Cargo feature flag. When the feature is not enabled, no Python or PyO3 dependencies MUST be compiled or required. When enabled, the module MUST link against PyO3 and require a Python environment with PyPOFacets or Open RCS installed.

#### Scenario: Build without rcs-compute feature
- **GIVEN** the thresh crate is compiled with default features (no `rcs-compute`)
- **WHEN** the build completes
- **THEN** the binary MUST compile successfully without any Python installation and MUST NOT contain RCS computation entry points

#### Scenario: Build with rcs-compute feature
- **GIVEN** the thresh crate is compiled with `--features rcs-compute`
- **WHEN** a Python 3.9+ environment with PyPOFacets is available
- **THEN** the build MUST succeed and the RCS computation API MUST be accessible from Rust code

### Requirement: Load 3D target geometry
The system MUST load 3D target geometry from STL files (binary and ASCII) and faceted model formats. The loader MUST parse the mesh into a triangle list with vertex coordinates and face normals suitable for physical optics computation. The loader MUST reject degenerate meshes (zero-area triangles, non-manifold edges) with descriptive error messages.

#### Scenario: Load a valid STL aircraft model
- **GIVEN** a binary STL file representing a simplified fighter aircraft with 5000 triangular facets
- **WHEN** the geometry is loaded via the RCS computation module
- **THEN** the system MUST parse all 5000 facets with correct vertex positions and outward-facing normals, and report the total surface area and bounding box dimensions

#### Scenario: Reject degenerate mesh
- **GIVEN** an STL file containing 10 valid triangles and 2 degenerate zero-area triangles
- **WHEN** the geometry loader processes the file
- **THEN** the system MUST return an error identifying the degenerate facets by index and MUST NOT proceed with RCS computation

### Requirement: Compute monostatic RCS vs azimuth and elevation
The system MUST compute monostatic RCS over a user-specified grid of azimuth and elevation angles at a given frequency. For each (azimuth, elevation) pair, the system MUST evaluate the physical optics integral over all illuminated facets to produce the RCS in dBsm. The computation MUST support frequencies from 1 GHz to 100 GHz and angular resolution down to 0.5 degrees.

#### Scenario: Compute RCS pattern for a flat plate
- **GIVEN** a 1 m x 1 m flat plate geometry oriented in the XY plane and a frequency of 10 GHz
- **WHEN** monostatic RCS is computed at normal incidence (azimuth=0, elevation=90 deg)
- **THEN** the peak RCS MUST be approximately 4*pi*A^2/lambda^2 = 4*pi*1/(0.03)^2 ~ 44 dBsm, consistent with the physical optics flat plate result

#### Scenario: Full hemisphere RCS sweep
- **GIVEN** an STL model of a cruise missile and frequency 10 GHz
- **WHEN** RCS is computed over azimuth [0, 360) at 2-degree steps and elevation [-90, 90] at 5-degree steps
- **THEN** the output MUST contain 180 x 37 = 6660 RCS values in dBsm, with nose-on and tail-on values lower than broadside values for a typical low-observable shape

### Requirement: Physical optics approximation
The system MUST use the physical optics (PO) approximation for RCS computation, which is valid for electrically large targets (target dimensions >> wavelength). The PO method MUST compute scattered fields by integrating the induced surface currents (J = 2 * n_hat x H_inc) over illuminated facets. Shadowed facets (those facing away from the radar) MUST be excluded from the integration.

#### Scenario: Shadowing of rear facets
- **GIVEN** a cube target with 6 faces, illuminated from the +X direction
- **WHEN** RCS is computed at azimuth=0 (looking along +X axis)
- **THEN** only the 3 front-facing facets (with normals having positive X-component toward the radar) MUST contribute to the PO integral; the 3 rear-facing facets MUST be excluded

### Requirement: Export RCS table as JSON
The system MUST export computed RCS data as a JSON file conforming to the RCS lookup table schema used by the swerling-rcs module. The JSON MUST contain fields for frequency (Hz), azimuth grid (degrees), elevation grid (degrees), RCS matrix (dBsm), target description, and computation metadata (date, method, mesh file hash). The exported file MUST be directly loadable by the runtime RCS lookup table.

#### Scenario: Round-trip compute and load
- **GIVEN** an RCS computation is performed on an STL model at 10 GHz with 1-degree resolution
- **WHEN** the result is exported to JSON and then loaded by the swerling-rcs module's RCS lookup table
- **THEN** the loaded table MUST reproduce the computed RCS values exactly (within floating-point representation limits) for all grid points

