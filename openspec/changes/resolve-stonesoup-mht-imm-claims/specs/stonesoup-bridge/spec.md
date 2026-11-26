# Stone Soup Reference Algorithm Access Delta

## REMOVED Requirements

### Requirement: Stone Soup algorithm access
**Reason**: The pinned Stone Soup 1.9.1 runtime supplies neither promised reference. It has no Gaussian IMM predictor: the wrapped `stonesoup.predictor.interacting` module does not exist, and the open upstream multiple-model proposal adds differently named modules. Its multi-target mixture tracker is a per-scan mixture tracker: it cannot be constructed with the wrapper's arguments and keeps no sliding-window hypothesis tree. Its only sliding-window associator (MFA) handles a fixed target set without births or k-best output. In an independent evaluation, MFA returned a suboptimal window solution in 3.5-10.1% of steps. No wrapper can satisfy either scenario, so both are withdrawn rather than weakened. JPDA and GM-PHD continue unchanged under "Stone Soup reference algorithm access", which renames this requirement without the two unmet scenarios. The maintainer approved this capability amendment on 2026-11-23.
**Migration**: The bridge MHT and IMM wrappers are removed with no bridge replacement.
- **MHT for ambiguous associations:** the deferred-decision obligation moves to the native `mht-tracking` requirements "Hypothesis tree maintenance" and "N-scan pruning". "Exhaustive reference verification" provides the independent oracle against which the follow-up's native conformance tests will run. **Interim gap:** until the follow-up change `rebuild-native-mht-hypothesis-history` lands, no native implementation meets that contract. Native hypothesis expansion ignores per-hypothesis history, N-scan pruning is never invoked, and the tracker's MHT path does not associate, so the tracker's MHT strategy is documented as not fit for use.
- **IMM filter for maneuvering target:** the IMM mathematics moves to the new native `imm-estimation` capability, and the reference role moves to its requirement "Independent IMM reference verification" (hand-derived exact rationals and an independent oracle). The maneuvering behaviour moves to the follow-up change `correct-native-imm-mode-policies`, which adopts the criterion preregistered by this change. **Interim gap:** native IMM does not meet the maneuvering scenario in any probed configuration except planar flight with the two-mode constant-velocity/turn bank at a turn rate of 0.15 rad/s. At altitude (probed at 1 km and 10 km) with three-dimensional measurements, turn modes stay at the mode-probability floor and combined altitude is biased. In planar flight, the four-mode preset indicates a turn before onset, and a gentle 0.05 rad/s turn is not reliably detected. The scenario stays unmet until that change lands.

## ADDED Requirements

### Requirement: Stone Soup reference algorithm access
The system SHALL expose the following Stone Soup algorithms via the bridge: Joint Probabilistic Data Association (JPDA) and Gaussian Mixture PHD filter. These serve as reference implementations and operational fallbacks until Rust-native versions exist. The bridge SHALL NOT expose, or claim to provide, a Stone Soup Multi-Hypothesis Tracking or Interacting Multiple Model reference. Native capabilities own those contracts: `mht-tracking` and `imm-estimation`.

#### Scenario: Supported reference algorithms
- **WHEN** the optional bridge is enabled and the host supplies supported linear-Gaussian components
- **THEN** executable JPDA association and GM-PHD scan references are available through the bridge, with behaviour defined by the JPDA and GM-PHD reference requirements

#### Scenario: No bridge MHT or IMM reference
- **WHEN** a caller needs multi-hypothesis tracking or interacting-multiple-model estimation
- **THEN** the bridge offers no Stone Soup MHT or IMM entry point and makes no MHT or IMM reference claim; the native MHT tracking and IMM estimation capabilities define those contracts
