"""Core mathematical operators for RTG-IMVC.

The functions in this module are intentionally independent of the model
architecture.  This makes the certificate, consensus teacher, and
base-objective preserving update directly testable.
"""

from __future__ import division

import torch


def topology_risk_certificate(candidate_predictions, cover_radius=None,
                              lipschitz_scale=0.0, probability_floor=1.0e-8):
    """Compute an empirical topology-transfer risk certificate.

    Args:
        candidate_predictions: Tensor with shape [K, N, C], including the
            prediction under the transferred baseline topology.
        cover_radius: Optional tensor with shape [N].  It records the largest
            normalized neighbour-set perturbation used for each sample.
        lipschitz_scale: User supplied local adjacency-to-log-probability
            sensitivity.  A positive value adds the finite-cover correction
            ``2 * lipschitz_scale * cover_radius``.
        probability_floor: Lower clipping value before taking logarithms.

    Returns:
        risk: [N] prediction diameter plus the cover correction.
        diameter: [N] maximum coordinate-wise log-probability diameter.
        cover_term: [N] finite-cover correction.

    Notes:
        The default implementation is an empirical certificate.  It becomes a
        formal cover certificate only when ``lipschitz_scale`` is a valid upper
        bound for the local model and the candidate set covers the admissible
        topology set.  The training code reports this distinction explicitly.
    """
    if candidate_predictions.ndim != 3:
        raise ValueError(
            'candidate_predictions must have shape [K, N, C], got {}'.format(
                tuple(candidate_predictions.shape)
            )
        )
    if candidate_predictions.shape[0] < 1:
        raise ValueError('at least one topology candidate is required')

    log_predictions = candidate_predictions.clamp_min(probability_floor).log()
    coordinate_diameter = (
        log_predictions.max(dim=0).values - log_predictions.min(dim=0).values
    )
    diameter = coordinate_diameter.max(dim=1).values

    if cover_radius is None:
        cover_radius = torch.zeros_like(diameter)
    else:
        cover_radius = cover_radius.to(
            device=diameter.device, dtype=diameter.dtype
        )
        if cover_radius.shape != diameter.shape:
            raise ValueError(
                'cover_radius must have shape {}, got {}'.format(
                    tuple(diameter.shape), tuple(cover_radius.shape)
                )
            )

    cover_term = 2.0 * float(lipschitz_scale) * cover_radius
    risk = (diameter + cover_term).clamp_min(0.0)
    return risk, diameter, cover_term


def certified_consensus_teacher(expert_predictions, expert_risks,
                                risk_epsilon=1.0e-4,
                                reliability_temperature=1.0,
                                teacher_temperature=1.0,
                                probability_floor=1.0e-8):
    """Build the risk-normalized logarithmic-opinion-pool teacher.

    Args:
        expert_predictions: Tensor [N, S, C] containing S probabilistic
            experts (two views and the fused prediction in RTG-IMVC).
        expert_risks: Tensor [N, S] with non-negative risk certificates.

    Returns:
        teacher: Tensor [N, C].
        weights: Tensor [N, S], detached certificate-normalized weights.
        reliability: Tensor [N], continuous guiding strength kappa.
        weighted_risk: Tensor [N], sum_s omega_s R_s.
    """
    if expert_predictions.ndim != 3:
        raise ValueError('expert_predictions must have shape [N, S, C]')
    if expert_risks.shape != expert_predictions.shape[:2]:
        raise ValueError(
            'expert_risks must have shape {}, got {}'.format(
                tuple(expert_predictions.shape[:2]), tuple(expert_risks.shape)
            )
        )
    if risk_epsilon <= 0:
        raise ValueError('risk_epsilon must be positive')
    if reliability_temperature <= 0:
        raise ValueError('reliability_temperature must be positive')
    if teacher_temperature <= 0:
        raise ValueError('teacher_temperature must be positive')

    risks = expert_risks.detach().clamp_min(0.0)
    inverse_risk = (risks + float(risk_epsilon)).reciprocal()
    weights = inverse_risk / inverse_risk.sum(dim=1, keepdim=True)

    log_experts = expert_predictions.detach().clamp_min(probability_floor).log()
    pooled_logits = (weights.unsqueeze(-1) * log_experts).sum(dim=1)
    teacher = torch.softmax(
        pooled_logits / float(teacher_temperature), dim=1
    )

    weighted_risk = (weights * risks).sum(dim=1)
    reliability = torch.exp(
        -weighted_risk / float(reliability_temperature)
    ).clamp(min=0.0, max=1.0)
    return teacher, weights, reliability, weighted_risk


def project_base_preserving_gradients(base_gradients, guide_gradients,
                                      guide_scale=1.0, margin=0.0,
                                      epsilon=1.0e-12):
    """Project the joint gradient onto a base-descent half-space.

    The returned direction is the minimum Euclidean modification of
    ``g_F + guide_scale * g_G`` satisfying

        <g_F, d> >= margin * ||g_F||^2.

    ``None`` gradients are supported so that parameters used only by the
    fusion/guiding branch can still receive their auxiliary gradient.
    """
    if len(base_gradients) != len(guide_gradients):
        raise ValueError('base_gradients and guide_gradients must align')
    if not 0.0 <= float(margin) <= 1.0:
        raise ValueError('margin must lie in [0, 1]')

    reference = next(
        (g for g in list(base_gradients) + list(guide_gradients)
         if g is not None),
        None,
    )
    if reference is None:
        return [None for _ in base_gradients], {
            'base_grad_norm': 0.0,
            'guide_grad_norm': 0.0,
            'gradient_cosine': 0.0,
            'joint_base_inner': 0.0,
            'required_inner': 0.0,
            'projection_coefficient': 0.0,
            'qp_active': 0.0,
        }

    base_norm_sq = reference.new_zeros(())
    guide_norm_sq = reference.new_zeros(())
    base_guide_inner = reference.new_zeros(())
    joint_base_inner = reference.new_zeros(())

    joint_gradients = []
    scale = float(guide_scale)
    for base_gradient, guide_gradient in zip(base_gradients, guide_gradients):
        if base_gradient is None and guide_gradient is None:
            joint_gradients.append(None)
            continue

        if base_gradient is None:
            joint = scale * guide_gradient
        elif guide_gradient is None:
            joint = base_gradient
        else:
            joint = base_gradient + scale * guide_gradient
        joint_gradients.append(joint)

        if base_gradient is not None:
            base_norm_sq = base_norm_sq + (base_gradient * base_gradient).sum()
            joint_base_inner = joint_base_inner + (base_gradient * joint).sum()
        if guide_gradient is not None:
            guide_norm_sq = guide_norm_sq + (guide_gradient * guide_gradient).sum()
        if base_gradient is not None and guide_gradient is not None:
            base_guide_inner = (
                base_guide_inner + (base_gradient * guide_gradient).sum()
            )

    required_inner = float(margin) * base_norm_sq
    correction = torch.clamp(
        (required_inner - joint_base_inner) / (base_norm_sq + float(epsilon)),
        min=0.0,
    )

    projected = []
    for joint, base_gradient in zip(joint_gradients, base_gradients):
        if joint is None:
            projected.append(None)
        elif base_gradient is None:
            projected.append(joint)
        else:
            projected.append(joint + correction * base_gradient)

    cosine = base_guide_inner / (
        torch.sqrt(base_norm_sq + float(epsilon))
        * torch.sqrt(guide_norm_sq + float(epsilon))
    )
    diagnostics = {
        'base_grad_norm': float(torch.sqrt(base_norm_sq).detach().cpu()),
        'guide_grad_norm': float(torch.sqrt(guide_norm_sq).detach().cpu()),
        'gradient_cosine': float(cosine.detach().cpu()),
        'joint_base_inner': float(joint_base_inner.detach().cpu()),
        'required_inner': float(required_inner.detach().cpu()),
        'projection_coefficient': float(correction.detach().cpu()),
        'qp_active': float((correction > 0).detach().cpu()),
    }
    return projected, diagnostics


def assign_gradients(parameters, gradients):
    """Assign detached gradients before ``optimizer.step()``."""
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.detach().clone()
