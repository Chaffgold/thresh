# Bounded Stone Soup algorithm references

The optional bridge supports explicit JPDA and GM-PHD compositions against the
locked Stone Soup 1.9.1 runtime. Both require compatible caller-selected
predictor and updater objects. The [runtime example](bridge-runtime.md) shows
construction; [bridge-testing.md](bridge-testing.md) documents the exact test
environment. The bridge never initializes Python, invents dynamics/noise, or
acquires provider data.

## JPDA migration

Use `JpdaAssociator::new(&config, &predictor, &updater)` instead of the former
config-only constructor. Configuration maps explicitly to gate probability
(`prob_gate`), detection probability (`prob_detect`, default 0.85), and clutter
spatial density. Every scalar must be finite; gate probability is strictly
between zero and one, detection probability is in [0,1], and density is positive.
Invalid Rust configuration produces `AlgorithmError` before package imports.

Gate probability is dimensionless, not a squared-distance threshold. Upstream
compares squared Mahalanobis distance with the measurement-dimension-dependent
chi-squared quantile, including equality. Clutter density has inverse
measurement-volume units matching the likelihood. Use one compatible measurement
space per call; a single density cannot silently mix radians and metres.

`associate(&tracks, &detections, &timestamp)` retains Python object identity and
the missed-detection entry. It returns exact exclusive-event marginals, not hard
assignments. Empty tracks give an empty mapping; empty or wholly gated-out
detections give missed probability one. Detected predictions use each detection's
observation timestamp; missed predictions use the call timestamp. There is no
arrival-time substitution, missing-time fallback, or new backward-time rejection
policy. Supported fixtures use explicit timezone-aware UTC observation times.

Tiny fixture expectations come from independent Gaussian arithmetic and joint
event enumeration. The native Rust JPDA approximation is not an interchangeable
oracle and is not asserted equal to this reference.

## GM-PHD migration

Keep `PhdFilter::new(py, config, &predictor, &updater)` and replace the unsupported
two-argument step with `filter(&prior, &birth_components, &detections, &timestamp)`.
The step predicts, builds detection-ordered hypotheses, updates intensity, then
reduces the posterior. It returns a Gaussian mixture update with hypothesis
provenance, not stable target IDs or extracted tracks.

- Priors and births are tagged weighted Gaussian components with finite,
  nonnegative intensity weights. Weights can exceed one and do not sum to one.
- Every birth carries tag `birth`; no prior carries it. Births are supplied
  explicitly each scan, never inferred from detections.
- Each prior/birth entry must be a distinct component object. Repeating the same
  object returns `AlgorithmError` before copying or graph dispatch; the pinned
  hypothesiser would otherwise silently collapse its intensity. Distinct objects
  may have equal values or tags and may share numeric arrays or ancestry.
- Required times are timezone-aware UTC. Priors share a time no later than the
  scan; births and detections have exactly the scan time. Invalid type, tag,
  numeric shape/value, or temporal context is an `AlgorithmError` before
  hypothesis generation.
- Prior/birth components are deep-copied together before upstream calls, so
  retagging, shared numeric arrays, and prediction ancestry cannot alter caller
  inputs. A copy failure returns its captured Python cause, with no shallow-copy
  fallback. Predictor/updater objects are not copied.
- Any input collection can be empty. Detections alone cannot create intensity.
  Empty outputs have no component-derived timestamp; no dummy component is added.
- Inputs must be reusable collections, such as lists, tuples, sets, or Gaussian
  mixtures. Generators and other one-shot iterators are rejected before any
  collection is read, because consuming them would change caller-owned state.

The caller remains responsible for compatible predictor/updater models. Input
checks cover component/measurement shapes and explicit per-detection model
dimensions; they do not introspect the updater's fallback model when a detection
omits its own model. An incompatible fallback model can therefore fail during
hypothesis generation with a structured Python error.

Survival and detection probabilities are finite in [0,1]. `clutter_intensity`
maps to positive finite clutter spatial density in the likelihood's units.
`gating_threshold_squared: None` means unbounded measurement gating; `Some(x)`
requires finite positive squared Mahalanobis distance. The measurement gate uses
a strict comparison. This setting is distinct from `merge_threshold`, whose
existing default 4.0 already denotes squared Mahalanobis distance (do not square
it again). Prune/merge thresholds are finite and nonnegative; maximum component
count is positive.

### Pinned intensity and reduction behavior

For an existing component of weight `w`, missed weight is `(1-Pd) Ps w`.
Detection weight is `Pd Ps w q / (kappa + sum_j Pd Ps w_j q_j)`, where `q` is the
Gaussian likelihood and the denominator includes eligible components for that
detection. The pinned hypothesiser retags a detected birth before updating, so
detected births also receive the survival factor. Undetected births contribute
no missed component. Exact zero updater weights become float64 epsilon before
reduction; pruning can subsequently remove or redistribute them.

Pruning redistributes removed weight equally over survivors (or yields empty
when all are pruned). Merging uses the candidate covariance, strict squared
distance comparison, and pairwise weight clipping at one. Truncation retains
the largest weights and redistributes removed mass equally. Thus reduced total
intensity is **not universally conserved**. Generated tags and output ordering
are not stable identifiers.

Independent numerical fixtures use prune=0, merge=0, and a large enough component
cap to neutralize reduction, then check reducer behavior separately. These
documented upstream semantics are compatibility behavior, not claims that they
are the desired production tracking policy.

## What remains outside these checks

The tests establish small linear-Gaussian reference behavior, boundary contracts,
and structured failures. They do not establish native/reference parity,
sequence-level tracking accuracy, nonlinear/multi-sensor algorithm correctness,
or deployment without a linked Python library. The bridge provides JPDA and
GM-PHD references only; there is no bridge MHT or IMM reference. Those contracts
are native (`mht-tracking`, `imm-estimation`), with interim gaps.
Data/model rights, the strict no-regression release gate, and deferred crates.io
publishing remain unchanged.
