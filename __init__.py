"""certabstain -- certified runtime abstention for learned robot policies.

    false-alarm side : conformal calibration on nominal rollouts only
                       (distribution-free, finite-sample, shift-inflatable)
    miss side        : structural soundness of the score
                       (deterministic, needs no failure data, shift-immune)
    enforcement      : unforgeable single-use certificates gating actuation

Quick start::

    from certabstain import build_monitor, CertifiedModelErrorWitness

    monitor = build_monitor(
        nominal_trajectories=calib_rollouts,   # per-timestep scores, successes only
        alpha=0.01,                            # false-alarm budget per rollout
        witness=CertifiedModelErrorWitness(epsilon=0.05),
        shift_budget=0.002,                    # total-variation ball radius
        safe_action=STOP,
    )
    print(monitor.claim.summary())

    decision = monitor.step(observation=obs, proposed_action=a, score=s)
    if decision.abstained:
        handoff(decision.reason)

Every failure to establish a guarantee raises. Nothing in this package returns
a weaker claim than the one it advertises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

from .conformal import (
    MondrianCalibrator,
    SplitConformalCalibrator,
    kl_worst_case,
    min_calibration_size,
    robust_level,
    rollout_scores,
)
from .errors import (
    BindingMismatch,
    CertAbstainError,
    CertificateError,
    CoverTooSmall,
    DegenerateScores,
    EnclosureError,
    EnvironmentUnsound,
    ForgedCertificate,
    GroupTooSmall,
    HorizonTooShort,
    InsufficientCalibrationData,
    MissingCertificate,
    ModeIndeterminate,
    NetworkCertificateMismatch,
    NonFiniteEnclosure,
    ReferenceMismatch,
    ReplayedCertificate,
    ShiftBudgetExceeded,
    SoundnessNotEstablished,
    TargetNotCertified,
    StaleCertificate,
)
from .gate import (
    ActionGate,
    CertificateAuthority,
    CertificateVerifier,
    GateDecision,
    SafetyCertificate,
)
from .nnbound import MLP, crown_bounds, ibp_bounds, jacobian_bounds
from .discrepancy import EpsilonCertificate, certify_epsilon, weights_hash
from .reference import CircleClearance, PusherSlider, SpringDamper2D
from .tube import TubeResult, clearance_lower_bounds, cover_contains_box, propagate_tube
from .witness2 import PredictiveTubeWitness, VerifiedDiscrepancyWitness
from .interval import (
    Interval,
    affine,
    istack,
    matvec,
    require_sound_environment,
    rounding_self_test,
)
from .soundness import (
    CertifiedModelErrorWitness,
    EmpiricalPowerWitness,
    MarginWitness,
    SoundnessWitness,
    TwoSidedClaim,
)

__version__ = "0.2.0.dev0"

__all__ = [
    "build_monitor",
    "Monitor",
    "SplitConformalCalibrator",
    "MondrianCalibrator",
    "rollout_scores",
    "robust_level",
    "kl_worst_case",
    "min_calibration_size",
    "SoundnessWitness",
    "MarginWitness",
    "CertifiedModelErrorWitness",
    "EmpiricalPowerWitness",
    "TwoSidedClaim",
    "ActionGate",
    "GateDecision",
    "SafetyCertificate",
    "CertificateAuthority",
    "CertificateVerifier",
    "CertAbstainError",
    "InsufficientCalibrationData",
    "DegenerateScores",
    "GroupTooSmall",
    "ShiftBudgetExceeded",
    "SoundnessNotEstablished",
    "CertificateError",
    "MissingCertificate",
    "ForgedCertificate",
    "ReplayedCertificate",
    "StaleCertificate",
    "BindingMismatch",
    "NetworkCertificateMismatch",
    "ReferenceMismatch",
    "VerifiedDiscrepancyWitness",
    "PredictiveTubeWitness",
    "Interval",
    "MLP",
    "SpringDamper2D",
    "PusherSlider",
    "CircleClearance",
    "EpsilonCertificate",
    "certify_epsilon",
    "weights_hash",
    "TargetNotCertified",
    "CoverTooSmall",
    "HorizonTooShort",
    "TubeResult",
    "propagate_tube",
    "cover_contains_box",
    "clearance_lower_bounds",
    "istack",
    "ibp_bounds",
    "crown_bounds",
    "jacobian_bounds",
    "matvec",
    "affine",
    "rounding_self_test",
    "require_sound_environment",
    "EnclosureError",
    "ModeIndeterminate",
    "NonFiniteEnclosure",
    "EnvironmentUnsound",
]


@dataclass(frozen=True, slots=True)
class Monitor:
    """A calibrated, gated monitor. Produced by :func:`build_monitor`."""

    gate: ActionGate
    calibrator: SplitConformalCalibrator
    claim: TwoSidedClaim | None
    calibration_level: float
    deployment_level: float
    shift_budget: float

    def step(
        self, *, observation: Any, proposed_action: Any, score: float
    ) -> GateDecision:
        return self.gate.step(
            observation=observation, proposed_action=proposed_action, score=score
        )

    @property
    def threshold(self) -> float:
        return self.calibrator.threshold

    @property
    def abstention_rate(self) -> float:
        return self.gate.abstention_rate

    def describe(self) -> str:
        lines = [
            f"certabstain monitor v{__version__}",
            f"  calibration rollouts   n = {self.calibrator.n}",
            f"  threshold                  {self.calibrator.threshold:.6g} "
            f"(order statistic {self.calibrator.order_statistic})",
            f"  calibration level      a'= {self.calibration_level:.6g}",
            f"  shift budget           rho= {self.shift_budget:.6g}",
            f"  deployment false-alarm a = {self.deployment_level:.6g}",
        ]
        if self.claim is None:
            lines.append("  miss rate                  UNBOUNDED (one-sided only)")
        else:
            lines.append(f"  miss rate              <=  {self.claim.miss_bound:.6g}")
            lines.append(f"  soundness              :   {self.claim.justification}")
        return "\n".join(lines)


def build_monitor(
    *,
    nominal_trajectories: Sequence[Sequence[float]],
    alpha: float,
    safe_action: Any,
    witness: SoundnessWitness | None = None,
    shift_budget: float = 0.0,
    divergence: str = "tv",
    require_two_sided: bool = True,
) -> Monitor:
    """Calibrate and wire a monitor, or refuse.

    Parameters
    ----------
    nominal_trajectories
        Per-timestep monitor scores from *successful* rollouts only. No failure
        data is required or used.
    alpha
        Target false-alarm probability per rollout **at deployment**, after the
        shift budget is absorbed.
    safe_action
        Emitted whenever the gate abstains or fails closed.
    witness
        Structural evidence for the miss side. Without one, only a one-sided
        certificate is available and ``require_two_sided`` must be False.
    shift_budget
        Radius of the divergence ball the claim must survive. Larger budgets
        force a tighter calibration level and may make the problem infeasible,
        in which case this raises rather than silently over-promising.
    require_two_sided
        When True (default), refuse to return a monitor whose miss rate is
        unbounded.

    Raises
    ------
    ShiftBudgetExceeded, InsufficientCalibrationData, SoundnessNotEstablished
    """
    if require_two_sided and witness is None:
        raise SoundnessNotEstablished(
            "require_two_sided=True but no soundness witness was supplied. "
            "Provide a witness (e.g. CertifiedModelErrorWitness) or pass "
            "require_two_sided=False to accept a one-sided false-alarm bound "
            "with an explicitly unbounded miss rate."
        )

    inner_alpha = robust_level(alpha, shift_budget, divergence=divergence)
    scores = rollout_scores(nominal_trajectories)
    calibrator = SplitConformalCalibrator.fit(scores, inner_alpha)

    claim: TwoSidedClaim | None = None
    if witness is not None:
        claim = TwoSidedClaim.compose(
            threshold=calibrator.threshold,
            false_alarm_bound=alpha,
            witness=witness,
        )

    gate = ActionGate(
        threshold=calibrator.threshold,
        false_alarm_bound=alpha,
        safe_action=safe_action,
        claim=claim,
    )

    return Monitor(
        gate=gate,
        calibrator=calibrator,
        claim=claim,
        calibration_level=inner_alpha,
        deployment_level=alpha,
        shift_budget=float(shift_budget),
    )
