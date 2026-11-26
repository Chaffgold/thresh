# IMM Estimation Delta

## Purpose

Specifies the policy-independent interacting-multiple-model (IMM) mathematics of the native filter bank and its independent verification: predicted mode probabilities and mixing, model-conditioned Gaussian log-likelihoods, the analytic mode-probability posterior, combined output moments, and configuration validation. Mode policies that are not yet specified are listed explicitly, so that current implementation choices are not mistaken for contract.

## ADDED Requirements

### Requirement: Common-state mixing
Mixing SHALL produce, for each mode, mixed moments of the common state [x, vx, y, vy, z, vz], in metres and metres per second, computed from each mode's estimate expressed in the common state. A mode whose state contains the common-state components SHALL contribute the marginal of those components. An implementation that also mixes components outside the common state SHALL still produce these common-state moments.

Let Π be the transition matrix in effect for the cycle, whose entry π_ij is the probability of switching from mode i to mode j during that cycle; for a cycle of 1 s, Π SHALL be the configured transition matrix. Let μ be the current mode probabilities. Then:
- **Predicted mode probability:** c_j = Σ_i π_ij μ_i for each mode j.
- **Conditional mixing weight:** μ_i|j = π_ij μ_i / c_j for every mode j with c_j > 1e-12.
- **Mixed mean:** x0_j = Σ_i μ_i|j x_i for every such mode j.
- **Mixed covariance:** P0_j = Σ_i μ_i|j [P_i + (x_i − x0_j)(x_i − x0_j)ᵀ] for every such mode j, including the between-mode spread term.

The mixed estimate of a mode whose predicted probability is at most 1e-12 is not specified by this requirement.

#### Scenario: Asymmetric two-mode mixing
- **WHEN** the inputs are:
  - Π = [[0.9, 0.1], [0.2, 0.8]] and μ = [0.6, 0.4];
  - mode 1's common estimate x = [0, 10, 0, 5, 100, 0] with P = diag(4, 1, 4, 1, 4, 1);
  - mode 2's common estimate x = [2, 12, −1, 4, 101, 0] with P = diag(9, 4, 9, 4, 9, 4), plus a covariance of 1 between x and vx.
- **THEN** c = [31/50, 19/50] and, with rows i and columns j, μ_i|j = [[27/31, 3/19], [4/31, 16/19]]
- **AND** x0_1 = [8, 318, −4, 151, 3104, 0]/31 and x0_2 = [32, 222, −16, 79, 1916, 0]/19
- **AND** each spread term equals w₁w₂ΔΔᵀ, with Δ = [2, 2, −1, −1, 1, 0] and w₁w₂ = 108/961 for mode 1 and 48/361 for mode 2
- **AND** P0_1[x,x] = 4896/961, P0_1[x,vx] = 556/961, P0_2[x,x] = 3156/361 and P0_2[x,vx] = 496/361

#### Scenario: Identity and symmetric transition matrices
- **WHEN** the same estimates and μ are mixed with Π = I, or with Π = [[0.95, 0.05], [0.05, 0.95]]
- **THEN** the identity matrix gives c = [3/5, 2/5], with μ_i|j equal to the identity and each mixed estimate equal to that mode's own estimate with zero spread
- **AND** the symmetric matrix gives c = [59/100, 41/100] and μ_i|j = [[57/59, 3/41], [2/59, 38/41]]

#### Scenario: Mixing invariants
- **WHEN** a valid Π and μ mix finite common estimates, and every predicted probability c_j exceeds 1e-12
- **THEN** each column of μ_i|j sums to one within 1e-12
- **AND** Σ_j c_j x0_j equals Σ_i (Σ_j π_ij) μ_i x_i componentwise, within an absolute tolerance of 1e-10 plus 1e-9 times Σ_i (Σ_j π_ij) μ_i |x_i|, so that the bound scales with the summed terms rather than with a result that may cancel; when every row of Π sums to exactly one, the right-hand side is Σ_i μ_i x_i
- **AND** every spread term is positive semidefinite within the verification tolerance

### Requirement: Model-conditioned log-likelihood
For each mode, let x̂ and P be the mean and covariance of its predicted estimate, expressed in a state in which the measurement is linear with observation matrix H; for position measurements, that state is the common state. For a measurement z of dimension m with measurement covariance R:
- **Innovation:** y = z − H x̂.
- **Innovation covariance:** S = H P Hᵀ + R.
- **Log-likelihood:** for positive-definite S whose determinant is finite and exceeds 1e-300, the mode's log-likelihood SHALL be ℓ = −½ (m ln 2π + ln det S + yᵀ S⁻¹ y). It SHALL be evaluated in log space, without forming the linear likelihood.

#### Scenario: Worked constant-velocity mode
- **WHEN** mode 1's mixed estimate from the asymmetric mixing scenario is predicted over 1 s by a constant-velocity model, using a discrete white-noise acceleration covariance of σ_a² [[dt⁴/4, dt³/2], [dt³/2, dt²]] per axis with σ_a = 1
- **AND** it is updated with z = [11, 6, 100.5], observing x, y and z with R = 4I
- **THEN** the predicted mean is [326, 318, 147, 151, 3104, 0]/31 and the innovation is [15/31, 39/31, 23/62]
- **AND** det S = 80701246125/59105344 and ℓ = −6.461143273565262

#### Scenario: Elapsed-time intervals
- **WHEN** the same mixed estimate is predicted over 0.1 s, 1 s and 2.3 s by a mode model with no projection between state spaces
- **THEN** each mode's log-likelihood agrees with an independently derived value for that interval, within the verification tolerance

### Requirement: Analytic mode-probability posterior
With no probability floor applied (floor zero), whenever every mode's predicted probability exceeds 1e-12 and every log-likelihood is finite with magnitude at most 1e6, the analytic mode-probability update SHALL return the exact Bayes posterior μ⁺_j = c_j exp(ℓ_j) / Σ_k c_k exp(ℓ_k) within the verification tolerance. The update SHALL be evaluated in log space by log-sum-exp, so that every finite set of log-likelihoods yields finite probabilities that sum to one within 1e-12.
- **Zero predicted probability:** with floor zero and every log-likelihood finite with magnitude at most 1e6, the update SHALL also return the exact posterior within the verification tolerance, which gives zero to each mode whose predicted probability is zero, when every other mode's predicted probability exceeds 1e-12 and no zero-probability mode's log-likelihood, computed from whatever estimate that mode is assigned, exceeds the largest log-likelihood among the other modes. Other cases with a predicted probability of at most 1e-12 are not specified.
- **Underflow:** when linear likelihoods underflow, the update SHALL NOT produce NaN, and SHALL NOT fall back to a uniform distribution or to the predicted probabilities in place of the posterior. A uniform result, or one equal to the predicted probabilities, is correct only when it is the Bayes posterior, for example when every log-likelihood is equal.
- **Learned classifier:** when an opt-in learned mode classifier replaces the analytic posterior, the requirements that define that classifier govern the replacement probabilities. The combination requirement still applies to the replaced probabilities.

#### Scenario: Worked posterior
- **WHEN** c = [0.62, 0.38] and ℓ = [−6.461143273565262, −8.519306179549595], with floor zero
- **THEN** μ⁺ = [0.9274195958363679, 0.07258040416363172]

#### Scenario: Log-space underflow
- **WHEN** c = [0.62, 0.38] and ℓ = [−163428.31075105123, −44784.431767829774], with floor zero, so that both linear likelihoods underflow to zero in float64
- **THEN** μ⁺ = [0, 1], with no NaN and no reversion to c

#### Scenario: Zero-probability destination
- **WHEN** Π = [[1, 0], [1, 0]] and μ = [0.6, 0.4] mix the asymmetric-mixing estimates, with floor zero
- **AND** mode 1 uses the worked constant-velocity model and measurement, and mode 2's log-likelihood does not exceed mode 1's
- **THEN** c = [1, 0] and μ⁺ = [1, 0] within the verification tolerance
- **AND** the combined mean equals mode 1's posterior mean [1689797/152447, 1612044/152447, 207755578/37959303, 63716985/12653101, 2541809541/25306202, 494019/12653101], approximately [11.0844883796, 10.5744553845, 5.4731136133, 5.0356813717, 100.4421580528, 0.0390433144], within the verification tolerance

### Requirement: Combined output moments
After a measurement update, the combined estimate SHALL be the probability-weighted moment match of the modes' posterior common-state estimates, using whichever mode probabilities are in effect:
- x = Σ_j μ⁺_j x_j;
- P = Σ_j μ⁺_j [P_j + (x_j − x)(x_j − x)ᵀ].

#### Scenario: Worked combination
- **WHEN** μ⁺ = [0.9274195958363679, 0.07258040416363172] combines two mode posterior estimates:
  - x components 10.812892420810806 and 11.206007753504345;
  - x variances 2.6933736334545353 and 3.6409686692927807.
- **THEN** the combined x is 10.84142489054062 and the combined x variance is 2.7725529136406477, which includes the between-mode spread

#### Scenario: Supplied mode probabilities
- **WHEN** the mode probabilities in effect are replaced by a supplied distribution, as an opt-in learned classifier does
- **THEN** the combined estimate is the moment match of the modes' estimates under the supplied distribution, within the verification tolerance

### Requirement: IMM configuration validation
An IMM configuration SHALL be rejected when any of the following holds:
- it has no modes;
- its per-mode models, state mappings, transition matrix and initial mode probabilities disagree in count or dimension;
- any transition-matrix entry is non-finite or negative;
- any transition-matrix row does not sum to one within 1e-6;
- any initial mode probability is non-finite or negative;
- the initial mode probabilities do not sum to one within 1e-6.

Validation, and every fallible construction path, SHALL report a rejection as an error rather than a panic. No construction path SHALL build a filter bank from a rejected configuration. Validation SHALL accept every configuration to which none of these conditions applies, including each of the library's preset banks.

#### Scenario: Non-finite transition entry
- **WHEN** a transition-matrix entry is NaN or infinite
- **THEN** validation returns an error that identifies the transition matrix
- **AND** a fallible constructor, including the learned-IMM tracker constructor, returns an error without panicking

#### Scenario: Negative entries that sum to one
- **WHEN** a transition-matrix row is [1.2, −0.2]
- **THEN** validation returns an error, even though the row sums to one

#### Scenario: Invalid initial mode probabilities
- **WHEN** the initial mode probabilities contain NaN, an infinity, or a negative entry, as in [1.5, −0.5]
- **THEN** validation returns an error, even when the entries sum to one

#### Scenario: Library presets pass validation
- **WHEN** each of the library's two-mode and four-mode preset banks is validated
- **THEN** validation succeeds

### Requirement: Explicitly unspecified mode policies
This capability SHALL NOT be read as specifying the following behaviours. They remain implementation choices until a requirement of this capability specifies them:
- the default mode-probability floor and the renormalisation after flooring;
- the treatment of a mode whose predicted probability is at most 1e-12, including zero: the mixed estimate it is assigned, its likelihood, its weight in the mode-probability update, and any threshold below which mixing or the mode update treats a predicted probability as zero, beyond the zero-probability case in "Analytic mode-probability posterior";
- the weighting of the predicted (pre-update) combined estimate used for gating;
- mode-probability propagation when a prediction is not followed by a measurement update;
- the reconstruction of a mode's own state from mixed or updated common-state moments, including components outside the common state such as acceleration and turn rate;
- the common-state representation of modes whose state does not contain the common-state components, such as speed, heading and turn-rate modes, including their vertical channel;
- how the transition matrix in effect for a cycle depends on that cycle's elapsed time, other than for a 1 s cycle;
- behaviour when an innovation covariance is not positive definite or is numerically singular: a determinant that is not finite or is at most 1e-300, or a matrix that cannot be inverted;
- the precision of the mode-probability posterior when a log-likelihood's magnitude exceeds 1e6.

Tests that pin these behaviours SHALL be labelled as current-policy characterisation and kept separate from tests of specified behaviour. They SHALL NOT be cited as evidence for this capability's requirements.

#### Scenario: Changing an unspecified policy
- **WHEN** a listed policy changes, such as the default floor or the behaviour on a missed update
- **THEN** no test cited as verification of this capability changes, and every specified mixing, likelihood, posterior, combination and validation requirement remains satisfied

#### Scenario: Characterisation is not specification
- **WHEN** a test pins a listed policy, such as the default floor's effect on a zero-probability destination or stale mode probabilities after a missed update
- **THEN** that test's name and documentation identify it as current-policy characterisation, and it is not counted as verification of this capability

### Requirement: Independent IMM reference verification
The specified IMM requirements SHALL be verified against both of the following:
- hand-derived exact-rational cases;
- an independently authored reference that shares no code with the production IMM bank, its motion models, state mappings or leaf filters, and uses only a general-purpose linear-algebra library.

Verification SHALL cover:
- identity, symmetric and asymmetric transition matrices;
- elapsed times of 0.1 s, 1 s and 2.3 s;
- log-space underflow;
- zero-probability destinations with floor zero;
- combination under supplied mode probabilities;
- the mixing invariants.

Numerical comparisons SHALL use an absolute tolerance of 1e-10 plus a relative tolerance of 1e-9, and probability sums SHALL hold within 1e-12. The required tests SHALL run in the native test lane without Python. That lane SHALL fail when a required test is missing, renamed without a manifest update, ignored, or failing, or when its target executes zero tests.

#### Scenario: Independent numerical evidence
- **WHEN** the IMM reference suite runs
- **THEN** the reference itself reproduces the hand-derived values
- **AND** for every configuration whose outputs depend on no listed unspecified policy, the production bank's mixing, posterior and combined outputs agree with the hand-derived values and with the reference

#### Scenario: Required tests cannot disappear
- **WHEN** a required IMM reference test is removed, renamed without a manifest update, marked ignored, filtered out, or fails, or its target runs zero tests
- **THEN** the native lane fails rather than reporting the IMM as verified
