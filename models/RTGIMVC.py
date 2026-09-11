"""RTG-IMVC: risk-certified topology guidance for incomplete multi-view clustering.

The protected base objective is the instance- and cluster-level
contrastive loss.  The legacy coordinate-wise hard-max pseudo target is
replaced by a topology-risk-certified consensus teacher, and a minimum-change
half-space update prevents the auxiliary guide from reversing base descent.
"""

from __future__ import division

import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from sklearn.utils import shuffle

from torch import nn
from torch.nn.functional import normalize
from sklearn.cluster import KMeans
from models.baseModels import (
    GraphEncoder,
    InstanceProject,
    ClusterProject,
    AttentionLayer,
)
from utils.evaluation import get_cluster_sols
from utils.datasets import next_batch
from utils.evaluation import evaluation
from utils.loss import (
    ClusterLoss,
    InstanceLoss,
    reduce_contrastive_batch_losses,
)
from utils.rtg import (
    assign_gradients,
    certified_consensus_teacher,
    project_base_preserving_gradients,
    topology_risk_certificate,
)


class RTGIMVC(nn.Module):
    """Topology risk estimation, consensus guidance, and base-safe optimization."""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._input_dim1 = config["Autoencoder"]["gcnEncoder1"][0]
        self._input_dim2 = config["Autoencoder"]["gcnEncoder2"][0]
        self._latent_dim = config["Autoencoder"]["gcnEncoder1"][-1]
        self._n_clusters = config["n_clusters"]
        self.gcnEncoder1 = GraphEncoder(
            config["Autoencoder"]["gcnEncoder1"], "relu", True
        )
        self.gcnEncoder2 = GraphEncoder(
            config["Autoencoder"]["gcnEncoder2"], "relu", True
        )
        self.instance_projector1 = InstanceProject(self._latent_dim)
        self.instance_projector2 = InstanceProject(self._latent_dim)
        self.cluster = ClusterProject(self._latent_dim, self._n_clusters)
        self.fusion = AttentionLayer(self._latent_dim)
        self._rtg = dict(config.get("rtg", {}))
        self._validate_rtg_config()

    def forward(self, x1, x2, adj1, adj2):
        h1 = self.gcnEncoder1(x1, adj1)
        h2 = self.gcnEncoder2(x2, adj2)
        z1 = normalize(self.instance_projector1(h1), dim=1)
        z2 = normalize(self.instance_projector2(h2), dim=1)
        y1, p1 = self.cluster(h1)
        y2, p2 = self.cluster(h2)
        return (h1, h2, z1, z2, y1, y2, p1, p2)

    def eval_acc(self, z, Y_list, accumulated_metrics, logger):
        """cal acc"""
        if self._config.get("kmeans_l2_normalize", True):
            z = normalize(z, p=2, dim=1)
        z = z.detach().cpu().numpy()
        y_pred, _ = get_cluster_sols(
            z,
            ClusterClass=KMeans,
            n_clusters=self._n_clusters,
            init_args={"n_init": 10, "random_state": 0},
        )
        scores = evaluation(
            y_pred=y_pred, y_true=Y_list[0], accumulated_metrics=accumulated_metrics
        )
        logger.info("\x1b[2;29m" + "trainingset_view1 " + str(scores) + "\x1b[0m")
        return scores

    def _label_free_checkpoint_policy(self):
        training = self._config["training"]
        mode = training.get("label_free_checkpoint", "last")
        if mode not in ("last", "max_topology_alignment"):
            raise ValueError(
                "label_free_checkpoint must be last or max_topology_alignment"
            )
        return (
            mode,
            max(1, int(training.get("checkpoint_interval", 10))),
            max(1, int(training.get("checkpoint_min_epoch", 1))),
        )

    def _snapshot_model_state(self):
        """Clone model state to CPU without retaining an autograd graph."""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
        }

    @staticmethod
    def _reference_graph_edges(adjacencies):
        """Return unique non-self edges from both observed input graphs."""
        if not adjacencies:
            raise ValueError("at least one adjacency is required")
        sample_count = int(adjacencies[0].shape[0])
        encoded = []
        for adjacency in adjacencies:
            sparse = adjacency.coalesce()
            rows, columns = sparse.indices()
            non_self = rows != columns
            encoded.append(rows[non_self] * sample_count + columns[non_self])
        unique_edges = torch.unique(torch.cat(encoded))
        if unique_edges.numel() == 0:
            raise ValueError("input graphs contain no non-self edges")
        return torch.stack(
            (
                torch.div(unique_edges, sample_count, rounding_mode="floor"),
                unique_edges.remainder(sample_count),
            )
        )

    @staticmethod
    def _topology_alignment(embedding, reference_edges):
        """Centered graph-edge cosine alignment, computed without labels.

        A constant embedding scores zero. A representation that keeps input
        graph neighbors more similar than arbitrary sample pairs is positive.
        """
        normalized = normalize(embedding, p=2, dim=1)
        rows, columns = reference_edges
        edge_mean = (normalized[rows] * normalized[columns]).sum(dim=1).mean()
        sample_count = int(normalized.shape[0])
        if sample_count < 2:
            raise ValueError("topology alignment requires at least 2 samples")
        summed = normalized.sum(dim=0)
        off_diagonal_sum = torch.dot(summed, summed) - normalized.square().sum()
        global_mean = off_diagonal_sum / (sample_count * (sample_count - 1))
        return edge_mean - global_mean

    def _validate_rtg_config(self):
        required = {
            "attention",
            "candidate_count",
            "replacement_ratio",
            "candidate_seed",
            "certificate_refresh_interval",
            "cover_lipschitz",
            "risk_epsilon",
            "probability_floor",
            "risk_temperature",
            "teacher_temperature",
            "guide_weight",
            "guide_warmup_epochs",
            "guide_scope",
            "reliability_threshold",
            "reliability_gate_temperature",
            "reliability_aware_fusion",
            "availability_aware_fusion",
            "contrastive_pair_scope",
            "instance_batch_reduction",
            "bootstrap_teacher_epochs",
            "teacher_transition_epochs",
            "trust_lr_decay",
            "trust_lr_thresholds",
            "trust_lr_factor",
            "trust_lr_patience",
            "trust_lr_min_epoch",
            "trust_lr_floor",
            "base_safe",
            "base_margin",
            "loss_weights",
        }
        missing = sorted(required - set(self._rtg))
        if missing:
            raise KeyError(
                "RTG configuration is incomplete; missing keys: {}".format(
                    ", ".join(missing)
                )
            )
        if int(self._rtg["candidate_count"]) < 1:
            raise ValueError("rtg.candidate_count must be at least one")
        if int(self._rtg["certificate_refresh_interval"]) < 1:
            raise ValueError("rtg.certificate_refresh_interval must be positive")
        if float(self._rtg["risk_temperature"]) <= 0:
            raise ValueError("rtg.risk_temperature must be positive")
        if float(self._rtg["teacher_temperature"]) <= 0:
            raise ValueError("rtg.teacher_temperature must be positive")
        if self._rtg["guide_scope"] not in ("all", "incomplete"):
            raise ValueError("rtg.guide_scope must be either 'all' or 'incomplete'")
        if self._rtg["contrastive_pair_scope"] not in ("all", "complete"):
            raise ValueError("rtg.contrastive_pair_scope must be 'all' or 'complete'")
        if self._rtg["instance_batch_reduction"] not in ("sum", "mean"):
            raise ValueError("rtg.instance_batch_reduction must be 'sum' or 'mean'")
        if not 0.0 <= float(self._rtg["reliability_threshold"]) <= 1.0:
            raise ValueError("rtg.reliability_threshold must lie in [0, 1]")
        if float(self._rtg["reliability_gate_temperature"]) < 0.0:
            raise ValueError("rtg.reliability_gate_temperature must be non-negative")
        if int(self._rtg["bootstrap_teacher_epochs"]) < 0:
            raise ValueError("rtg.bootstrap_teacher_epochs must be non-negative")
        if int(self._rtg["teacher_transition_epochs"]) < 0:
            raise ValueError("rtg.teacher_transition_epochs must be non-negative")
        trust_thresholds = tuple(
            float(value) for value in self._rtg["trust_lr_thresholds"]
        )
        if not trust_thresholds:
            raise ValueError("rtg.trust_lr_thresholds must not be empty")
        if any(value < 0.0 or value > 1.0 for value in trust_thresholds):
            raise ValueError("rtg.trust_lr_thresholds must lie in [0, 1]")
        if any(
            left <= right for left, right in zip(trust_thresholds, trust_thresholds[1:])
        ):
            raise ValueError("rtg.trust_lr_thresholds must be strictly decreasing")
        if not 0.0 < float(self._rtg["trust_lr_factor"]) < 1.0:
            raise ValueError("rtg.trust_lr_factor must lie in (0, 1)")
        if int(self._rtg["trust_lr_patience"]) < 1:
            raise ValueError("rtg.trust_lr_patience must be positive")
        if int(self._rtg["trust_lr_min_epoch"]) < 0:
            raise ValueError("rtg.trust_lr_min_epoch must be non-negative")
        if float(self._rtg["trust_lr_floor"]) < 0.0:
            raise ValueError("rtg.trust_lr_floor must be non-negative")
        if not 0.0 <= float(self._rtg["base_margin"]) <= 1.0:
            raise ValueError("rtg.base_margin must lie in [0, 1]")

    def _candidate_predictions(self, x_train, topology):
        """Predict clusters under all candidate graphs without BN updates."""
        predictions = []
        modules = [self.gcnEncoder1, self.gcnEncoder2, self.cluster]
        training_states = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            with torch.no_grad():
                for view_index in range(2):
                    view_predictions = []
                    for candidate_adj in topology["candidate_adjs"][view_index]:
                        candidate_adj = candidate_adj.to(x_train[view_index].device)
                        if view_index == 0:
                            embedding = self.gcnEncoder1(
                                x_train[view_index], candidate_adj
                            )
                        else:
                            embedding = self.gcnEncoder2(
                                x_train[view_index], candidate_adj
                            )
                        probabilities, _ = self.cluster(embedding)
                        view_predictions.append(probabilities.detach())
                    predictions.append(torch.stack(view_predictions, dim=0))
        finally:
            for module, was_training in zip(modules, training_states):
                module.train(was_training)
        return predictions

    def _estimate_topology_risks(self, x_train, topology):
        candidate_predictions = self._candidate_predictions(x_train, topology)
        risks, diameters, cover_terms = [], [], []
        for view_index in range(2):
            risk, diameter, cover_term = topology_risk_certificate(
                candidate_predictions[view_index],
                cover_radius=topology["cover_radius"][view_index],
                lipschitz_scale=float(self._rtg["cover_lipschitz"]),
                probability_floor=float(self._rtg["probability_floor"]),
            )
            risks.append(risk.detach())
            diameters.append(diameter.detach())
            cover_terms.append(cover_term.detach())
        return risks, diameters, cover_terms

    def _base_loss(
        self,
        z1,
        z2,
        y1,
        y2,
        criterion_instance,
        criterion_cluster,
        batch_size,
        paired_mask=None,
    ):
        loss_weights = self._rtg["loss_weights"]
        cluster_weight = float(loss_weights["cluster"])
        instance_weight = float(loss_weights["instance"])

        cluster_loss = criterion_cluster(y1, y2)
        instance_loss = z1.new_zeros(())
        if instance_weight != 0.0:
            instance_z1, instance_z2 = z1, z2
            if self._rtg["contrastive_pair_scope"] == "complete":
                if paired_mask is None:
                    raise ValueError(
                        "complete-pair contrastive loss requires paired_mask"
                    )
                paired_mask = paired_mask.to(device=z1.device, dtype=torch.bool)
                if paired_mask.ndim != 1 or paired_mask.shape[0] != z1.shape[0]:
                    raise ValueError("paired_mask must be one value per sample")
                if int(paired_mask.sum()) < 2:
                    raise ValueError(
                        "at least two complete-view samples are required for "
                        "the contrastive objective"
                    )
                instance_z1 = z1[paired_mask]
                instance_z2 = z2[paired_mask]

            # Shuffle only availability-valid positive pairs. Missing-view
            # graph embeddings are not treated as observed cross-view facts.
            shuffled_z1, shuffled_z2 = shuffle(instance_z1, instance_z2)
            effective_batch_size = min(int(batch_size), int(shuffled_z1.shape[0]))
            instance_terms = []
            instance_weights = []
            for batch_z1, batch_z2, _ in next_batch(
                shuffled_z1, shuffled_z2, effective_batch_size
            ):
                instance_terms.append(criterion_instance(batch_z1, batch_z2))
                instance_weights.append(int(batch_z1.shape[0]))
            instance_loss = reduce_contrastive_batch_losses(
                instance_terms,
                reduction=self._rtg["instance_batch_reduction"],
                weights=instance_weights,
            )
        base_loss = cluster_weight * cluster_loss + instance_weight * instance_loss
        return base_loss, cluster_loss, instance_loss

    def _guide_loss(self, fused_prediction, teacher, reliability):
        """Reliability-normalized KL used by the residual topology guide.

        Normalizing by the reliable mass keeps the conditional KL scale
        stable.  The mean reliable mass is applied exactly once later through
        ``guide_scale``; the previous implementation applied it both here and
        there, unintentionally attenuating the topology gradient by roughly
        the square of the reliability.
        """
        floor = float(self._rtg["probability_floor"])
        per_sample_kl = F.kl_div(
            fused_prediction.clamp_min(floor).log(),
            teacher.detach(),
            reduction="none",
        ).sum(dim=1)
        weights = reliability.detach().clamp_min(0.0)
        reliable_mass = weights.sum().clamp_min(floor)
        return (weights * per_sample_kl).sum() / reliable_mass

    def _selective_guide_reliability(self, reliability, topology):
        """Return sample weights for a selective, abstaining RTG teacher.

        Candidate generation is induced by alternative replacements for
        missing-view samples.  Consequently, its residual teacher is scoped
        to the union of samples with incomplete evidence by default.  A
        low-reliability teacher smoothly abstains, which makes the complete
        ICMVC loss the automatic fallback without consulting labels.
        """
        if self._rtg["guide_scope"] == "incomplete":
            missing_masks = topology["missing_mask"]
            scope_mask = missing_masks[0].bool()
            for missing_mask in missing_masks[1:]:
                scope_mask = torch.logical_or(scope_mask, missing_mask.bool())
            scope = scope_mask.to(device=reliability.device, dtype=reliability.dtype)
        else:
            scope = torch.ones_like(reliability)

        detached_reliability = reliability.detach().clamp(0.0, 1.0)
        threshold = float(self._rtg["reliability_threshold"])
        temperature = float(self._rtg["reliability_gate_temperature"])
        if temperature > 0.0:
            trust_gate = torch.sigmoid((detached_reliability - threshold) / temperature)
        else:
            trust_gate = (detached_reliability >= threshold).to(reliability.dtype)

        weights = detached_reliability * trust_gate * scope
        scope_denominator = scope.sum().clamp_min(1.0)
        effective_reliability = weights.sum() / scope_denominator
        active_fraction = (
            scope * (detached_reliability >= threshold).to(reliability.dtype)
        ).sum() / scope_denominator
        return (
            weights,
            effective_reliability,
            active_fraction,
            scope.mean(),
        )

    def _reliability_aware_fuse(
        self, h1, h2, reliability, topology, attention_fused=None
    ):
        """Use the observed view when transferred topology is unreliable.

        Complete samples retain the learned ICMVC attention. For an incomplete
        sample, the certificate reliability interpolates between attention
        fusion and the embedding from its actually observed view. The gate is
        detached: it controls fusion but is not optimized through labels or a
        surrogate path into the risk certificate.
        """
        if attention_fused is None:
            attention_fused = self.fusion(h1, h2)
        availability_aware = bool(self._rtg["availability_aware_fusion"])
        reliability_aware = bool(self._rtg["reliability_aware_fusion"])
        if not availability_aware and not reliability_aware:
            return attention_fused, attention_fused.new_zeros(())

        missing1, missing2 = topology["missing_mask"]
        observed1 = (~missing1.bool()).to(device=h1.device, dtype=h1.dtype)
        observed2 = (~missing2.bool()).to(device=h2.device, dtype=h2.dtype)
        observed_count = (observed1 + observed2).clamp_min(1.0).unsqueeze(1)
        observed_fused = (
            observed1.unsqueeze(1) * h1 + observed2.unsqueeze(1) * h2
        ) / observed_count
        incomplete = (
            torch.logical_or(missing1.bool(), missing2.bool())
            .to(device=h1.device)
            .unsqueeze(1)
        )
        if availability_aware:
            # Availability is a hard fact, unlike a learned confidence score:
            # an unavailable view must not contribute to an incomplete case.
            refined = observed_fused
        else:
            trust = reliability.detach().clamp(0.0, 1.0).unsqueeze(1)
            refined = trust * attention_fused + (1.0 - trust) * observed_fused
        fused = torch.where(incomplete, refined, attention_fused)
        correction = torch.linalg.vector_norm(fused - attention_fused, dim=1).mean()
        return fused, correction

    def _safe_step(self, base_loss, guide_loss, guide_scale, optimizer, grad_clip=None):
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        base_gradients = torch.autograd.grad(
            base_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        guide_gradients = torch.autograd.grad(
            guide_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        projected, diagnostics = project_base_preserving_gradients(
            base_gradients,
            guide_gradients,
            guide_scale=guide_scale,
            margin=float(self._rtg["base_margin"]),
        )
        # Adam is highly sensitive to tiny round-off differences near zero.
        # When the QP is inactive, use the autograd gradient of the joint loss
        # directly instead of numerically reconstructing g_F + lambda g_G.
        # This exactly preserves the baseline update path during bootstrap.
        joint_gradients = torch.autograd.grad(
            base_loss + float(guide_scale) * guide_loss,
            parameters,
            allow_unused=True,
        )
        step_gradients = projected if diagnostics["qp_active"] else joint_gradients
        optimizer.zero_grad()
        assign_gradients(parameters, step_gradients)
        if grad_clip is not None:
            # Positive norm scaling cannot reverse the projected half-space
            # direction and bounds unstable late-training updates.
            torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
        optimizer.step()
        return diagnostics

    @staticmethod
    def _is_strict_sgd(optimizer):
        if not isinstance(optimizer, torch.optim.SGD):
            return False
        return all(
            float(group.get("momentum", 0.0)) == 0.0
            and float(group.get("weight_decay", 0.0)) == 0.0
            and not bool(group.get("nesterov", False))
            for group in optimizer.param_groups
        )

    def _certificate_lr_enabled(self, optimizer):
        """Return whether certificate-triggered decay has a valid scope.

        The decay is coupled to the Euclidean base-preserving argument, which
        describes a direct SGD parameter step. Applying it to Adam confounds
        the residual guide with an unrelated aggressive learning-rate change
        and makes the comparison with conventional Adam training inequitable.
        """
        return bool(self._rtg["trust_lr_decay"]) and self._is_strict_sgd(optimizer)

    @staticmethod
    def _decay_learning_rate(optimizer, scheduler, factor, floor):
        """Apply one certificate-triggered LR decay and return old/new rates."""
        old_rates = [float(group["lr"]) for group in optimizer.param_groups]
        new_rates = [max(rate * factor, floor) for rate in old_rates]
        for group, new_rate in zip(optimizer.param_groups, new_rates):
            group["lr"] = new_rate

        # Keep an optional conventional scheduler on the same reduced scale.
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [
                max(float(rate) * factor, floor) for rate in scheduler.base_lrs
            ]
        if scheduler is not None and hasattr(scheduler, "eta_min"):
            scheduler.eta_min = max(float(scheduler.eta_min) * factor, floor)
        return old_rates, new_rates

    def run_train(
        self,
        x_train,
        Y_list,
        adj,
        optimizer,
        logger,
        accumulated_metrics,
        device,
        scheduler=None,
        topology=None,
    ):
        if topology is None:
            raise ValueError(
                "RTGIMVC requires topology candidates from get_rtg_adjacency"
            )

        epochs = int(self._config["training"]["epoch"])
        print_num = max(1, int(self._config.get("print_num", 10)))
        batch_size = min(
            int(self._config["training"]["batch_size"]), x_train[0].shape[0]
        )
        criterion_instance = InstanceLoss(batch_size, 1.0, device).to(device)
        criterion_cluster = ClusterLoss(self._n_clusters, 0.5, device).to(device)
        attention = bool(self._rtg["attention"])
        eval_mode = self._config.get("eval_mode", "cluster_head")
        batchnorm_eval = self._config.get("batchnorm_eval", "transductive")
        if batchnorm_eval not in ("transductive", "running"):
            raise ValueError("batchnorm_eval must be transductive or running")
        kmeans_eval_interval = max(1, int(self._config.get("kmeans_eval_interval", 10)))
        refresh_interval = int(self._rtg["certificate_refresh_interval"])
        warmup_epochs = max(0, int(self._rtg["guide_warmup_epochs"]))
        grad_clip = self._config["training"].get("grad_clip")
        base_safe = bool(self._rtg["base_safe"])
        strict_sgd = self._is_strict_sgd(optimizer)
        trust_lr_requested = bool(self._rtg["trust_lr_decay"])
        trust_lr_enabled = self._certificate_lr_enabled(optimizer)
        trust_lr_thresholds = tuple(
            float(value) for value in self._rtg["trust_lr_thresholds"]
        )
        trust_lr_factor = float(self._rtg["trust_lr_factor"])
        trust_lr_patience = int(self._rtg["trust_lr_patience"])
        trust_lr_min_epoch = int(self._rtg["trust_lr_min_epoch"])
        trust_lr_floor = float(self._rtg["trust_lr_floor"])
        trust_lr_stage = 0
        trust_lr_counter = 0
        if base_safe and not strict_sgd:
            logger.warning(
                "Adam/non-strict SGD selected: the base-safe projection is "
                "diagnostic; a strict parameter-step guarantee requires "
                "zero-momentum, zero-decay SGD."
            )
        if trust_lr_requested and not trust_lr_enabled:
            logger.info(
                "Certificate LR decay disabled for this optimizer to preserve "
                "the configured conventional scheduler."
            )

        history_keys = [
            "epoch",
            "loss",
            "base_loss",
            "contrastive_loss",
            "cluster_loss",
            "instance_loss",
            "guide_loss",
            "guide_scale",
            "risk_mean",
            "risk_max",
            "diameter_mean",
            "cover_term_mean",
            "kappa_mean",
            "effective_kappa_mean",
            "guide_active_fraction",
            "guide_scope_fraction",
            "complete_pair_fraction",
            "teacher_agreement_mean",
            "fusion_correction_mean",
            "certified_fraction",
            "teacher_entropy",
            "prediction_entropy",
            "active_clusters",
            "max_cluster_fraction",
            "base_grad_norm",
            "guide_grad_norm",
            "gradient_cosine",
            "qp_active",
            "projection_coefficient",
            "max_expert_influence",
            "influence_bound_slack",
            "learning_rate",
            "trust_lr_event",
            "trust_lr_threshold",
            "trust_lr_stage",
            "strict_sgd",
            "epoch_seconds",
            "topology_alignment",
            "selected_checkpoint_epoch",
            "selected_acc",
            "selected_nmi",
            "selected_ari",
            "protocol_version",
            "acc",
            "nmi",
            "ari",
        ]
        history = {key: [] for key in history_keys}
        cached_risks = None
        cached_diameters = None
        cached_cover_terms = None
        last_scores = None
        checkpoint_mode, checkpoint_interval, checkpoint_min_epoch = (
            self._label_free_checkpoint_policy()
        )
        best_checkpoint_state = None
        best_checkpoint_score = -float("inf")
        best_checkpoint_epoch = 0
        reference_edges = self._reference_graph_edges(adj)
        protocol_version = self._config.get("protocol_version", "unversioned")
        complete_pair_mask = torch.logical_not(
            torch.logical_or(
                topology["missing_mask"][0].bool(),
                topology["missing_mask"][1].bool(),
            )
        )
        paired_fraction = float(complete_pair_mask.float().mean().detach().cpu())
        logger.info(
            "PAIRING    | Scope {:<8} | Complete pairs {}/{} ({:.1f}%) | "
            "Batch reduction {}".format(
                self._rtg["contrastive_pair_scope"],
                int(complete_pair_mask.sum().detach().cpu()),
                int(complete_pair_mask.numel()),
                100.0 * paired_fraction,
                self._rtg["instance_batch_reduction"],
            )
        )

        for epoch_index in range(epochs):
            epoch_start = time.time()
            self.train()
            h1, h2, z1, z2, y1, y2, _, _ = self(x_train[0], x_train[1], adj[0], adj[1])
            provisional_fused_embedding = (
                self.fusion(h1, h2) if attention else 0.5 * (h1 + h2)
            )
            provisional_fused_prediction, _ = self.cluster(provisional_fused_embedding)

            contrastive_loss, cluster_loss, instance_loss = self._base_loss(
                z1,
                z2,
                y1,
                y2,
                criterion_instance,
                criterion_cluster,
                batch_size,
                paired_mask=complete_pair_mask,
            )

            if cached_risks is None or epoch_index % refresh_interval == 0:
                cached_risks, cached_diameters, cached_cover_terms = (
                    self._estimate_topology_risks(x_train, topology)
                )

            fused_risk = torch.maximum(cached_risks[0], cached_risks[1])
            expert_risks = torch.stack(
                (cached_risks[0], cached_risks[1], fused_risk), dim=1
            )
            expert_predictions = torch.stack(
                (y1, y2, provisional_fused_prediction), dim=1
            )
            _, _, reliability, _ = certified_consensus_teacher(
                expert_predictions,
                expert_risks,
                risk_epsilon=float(self._rtg["risk_epsilon"]),
                reliability_temperature=float(self._rtg["risk_temperature"]),
                teacher_temperature=float(self._rtg["teacher_temperature"]),
                probability_floor=float(self._rtg["probability_floor"]),
            )
            if attention:
                fused_embedding, fusion_correction = self._reliability_aware_fuse(
                    h1,
                    h2,
                    reliability,
                    topology,
                    attention_fused=provisional_fused_embedding,
                )
            else:
                fused_embedding = provisional_fused_embedding
                fusion_correction = fused_embedding.new_zeros(())
            fused_prediction, _ = self.cluster(fused_embedding)
            expert_predictions = torch.stack((y1, y2, fused_prediction), dim=1)
            certified_teacher, expert_weights, reliability, weighted_risk = (
                certified_consensus_teacher(
                    expert_predictions,
                    expert_risks,
                    risk_epsilon=float(self._rtg["risk_epsilon"]),
                    reliability_temperature=float(self._rtg["risk_temperature"]),
                    teacher_temperature=float(self._rtg["teacher_temperature"]),
                    probability_floor=float(self._rtg["probability_floor"]),
                )
            )
            # Paper definition: F = L_ins + L_clu.  The original ICMVC
            # coordinate-wise hard-max teacher is intentionally not included
            # in RTG's protected objective or auxiliary path.
            base_loss = contrastive_loss
            mean_teacher = expert_predictions.detach().mean(dim=1)
            bootstrap_epochs = int(self._rtg["bootstrap_teacher_epochs"])
            transition_epochs = int(self._rtg["teacher_transition_epochs"])
            current_epoch = epoch_index + 1
            if current_epoch <= bootstrap_epochs:
                certified_fraction = 0.0
            elif transition_epochs > 0:
                certified_fraction = min(
                    1.0,
                    float(current_epoch - bootstrap_epochs) / transition_epochs,
                )
            else:
                certified_fraction = 1.0
            teacher = certified_teacher
            (
                effective_reliability,
                effective_kappa,
                guide_active_fraction,
                guide_scope_fraction,
            ) = self._selective_guide_reliability(reliability, topology)
            guide_loss = self._guide_loss(
                fused_prediction, teacher, effective_reliability
            )
            if warmup_epochs > 0:
                warmup = min(1.0, float(epoch_index + 1) / warmup_epochs)
            else:
                warmup = 1.0
            guide_scale = (
                float(self._rtg["guide_weight"])
                * certified_fraction
                * float(effective_kappa.detach().cpu())
                * warmup
            )
            trust_lr_event = 0.0
            trust_lr_threshold = -1.0
            if (
                trust_lr_enabled
                and current_epoch >= trust_lr_min_epoch
                and trust_lr_stage < len(trust_lr_thresholds)
            ):
                current_threshold = trust_lr_thresholds[trust_lr_stage]
                if float(guide_active_fraction.detach().cpu()) < current_threshold:
                    trust_lr_counter += 1
                else:
                    trust_lr_counter = 0
                if trust_lr_counter >= trust_lr_patience:
                    old_rates, new_rates = self._decay_learning_rate(
                        optimizer,
                        scheduler,
                        factor=trust_lr_factor,
                        floor=trust_lr_floor,
                    )
                    trust_lr_event = 1.0
                    trust_lr_threshold = current_threshold
                    trust_lr_stage += 1
                    trust_lr_counter = 0
                    logger.info(
                        "Certificate-aware LR decay at epoch {}: trusted "
                        "fraction remained below {:.2f}; lr {} -> {}".format(
                            current_epoch,
                            current_threshold,
                            ",".join("{:.3e}".format(rate) for rate in old_rates),
                            ",".join("{:.3e}".format(rate) for rate in new_rates),
                        )
                    )
            total_loss_value = float(
                (base_loss.detach() + guide_scale * guide_loss.detach()).cpu()
            )

            if base_safe:
                gradient_diagnostics = self._safe_step(
                    base_loss,
                    guide_loss,
                    guide_scale,
                    optimizer,
                    grad_clip=grad_clip,
                )
            else:
                optimizer.zero_grad()
                total_loss = base_loss + guide_scale * guide_loss
                total_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip))
                optimizer.step()
                gradient_diagnostics = {
                    "base_grad_norm": float("nan"),
                    "guide_grad_norm": float("nan"),
                    "gradient_cosine": float("nan"),
                    "qp_active": 0.0,
                    "projection_coefficient": 0.0,
                }
            if scheduler is not None:
                scheduler.step()

            # Match the original transductive ICMVC evaluation protocol by
            # default.  Strict running-statistics inference is opt-in because
            # its BatchNorm buffers are poorly calibrated in early epochs and
            # is not directly comparable with the baseline implementation.
            if batchnorm_eval == "running":
                self.eval()
            else:
                self.train()
            with torch.no_grad():
                h1_eval, h2_eval, _, _, _, _, _, _ = self(
                    x_train[0], x_train[1], adj[0], adj[1]
                )
                if attention:
                    fused_eval, _ = self._reliability_aware_fuse(
                        h1_eval, h2_eval, reliability, topology
                    )
                else:
                    fused_eval = 0.5 * (h1_eval + h2_eval)
                prediction, _ = self.cluster(fused_eval)
                labels = prediction.cpu().numpy().argmax(1)
                if eval_mode == "kmeans_fused":
                    if (
                        (epoch_index + 1) % kmeans_eval_interval == 0
                        or last_scores is None
                        or (epoch_index + 1) == epochs
                    ):
                        scores = self.eval_acc(fused_eval, Y_list, None, logger)
                        last_scores = scores
                    else:
                        scores = last_scores
                else:
                    scores = evaluation(
                        y_pred=labels,
                        y_true=Y_list[0],
                        accumulated_metrics=None,
                    )

                checkpoint_due = current_epoch >= checkpoint_min_epoch and (
                    current_epoch % checkpoint_interval == 0 or current_epoch == epochs
                )
                if checkpoint_due:
                    topology_alignment = float(
                        self._topology_alignment(fused_eval, reference_edges)
                        .detach()
                        .cpu()
                    )
                else:
                    topology_alignment = float("nan")
            if (
                checkpoint_mode == "max_topology_alignment"
                and checkpoint_due
                and topology_alignment > best_checkpoint_score
            ):
                best_checkpoint_score = topology_alignment
                best_checkpoint_epoch = current_epoch
                best_checkpoint_state = self._snapshot_model_state()

            risk_values = torch.cat(cached_risks)
            diameter_values = torch.cat(cached_diameters)
            cover_values = torch.cat(cached_cover_terms)
            weighted_expert_risk = expert_weights * expert_risks
            risk_epsilon = float(self._rtg["risk_epsilon"])
            teacher_temperature = float(self._rtg["teacher_temperature"])
            chi = ((expert_risks + risk_epsilon).reciprocal().sum(dim=1)).reciprocal()
            max_influence = (
                2.0 * weighted_expert_risk.max(dim=1).values / teacher_temperature
            )
            influence_slack = 2.0 * chi / teacher_temperature - max_influence
            entropy_normalizer = math.log(max(2, self._n_clusters))
            teacher_entropy = (
                -(
                    teacher.clamp_min(float(self._rtg["probability_floor"]))
                    * teacher.clamp_min(float(self._rtg["probability_floor"])).log()
                )
                .sum(dim=1)
                .mean()
                / entropy_normalizer
            )
            mean_prediction = prediction.mean(dim=0).clamp_min(
                float(self._rtg["probability_floor"])
            )
            prediction_entropy = (
                -(mean_prediction * mean_prediction.log()).sum() / entropy_normalizer
            )
            label_counts = torch.bincount(
                prediction.argmax(dim=1), minlength=self._n_clusters
            )
            active_clusters = int((label_counts > 0).sum().cpu())
            max_cluster_fraction = float(
                label_counts.max().float().cpu() / prediction.shape[0]
            )

            row = {
                "epoch": epoch_index + 1,
                "loss": total_loss_value,
                "base_loss": float(base_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "cluster_loss": float(cluster_loss.detach().cpu()),
                "instance_loss": float(instance_loss.detach().cpu()),
                "guide_loss": float(guide_loss.detach().cpu()),
                "guide_scale": guide_scale,
                "risk_mean": float(risk_values.mean().cpu()),
                "risk_max": float(risk_values.max().cpu()),
                "diameter_mean": float(diameter_values.mean().cpu()),
                "cover_term_mean": float(cover_values.mean().cpu()),
                "kappa_mean": float(reliability.mean().cpu()),
                "effective_kappa_mean": float(effective_kappa.cpu()),
                "guide_active_fraction": float(guide_active_fraction.cpu()),
                "guide_scope_fraction": float(guide_scope_fraction.cpu()),
                "complete_pair_fraction": paired_fraction,
                "teacher_agreement_mean": float(
                    (1.0 - 0.5 * torch.abs(mean_teacher - certified_teacher).sum(dim=1))
                    .clamp(0.0, 1.0)
                    .mean()
                    .cpu()
                ),
                "fusion_correction_mean": float(fusion_correction.detach().cpu()),
                "certified_fraction": certified_fraction,
                "teacher_entropy": float(teacher_entropy.cpu()),
                "prediction_entropy": float(prediction_entropy.cpu()),
                "active_clusters": active_clusters,
                "max_cluster_fraction": max_cluster_fraction,
                "base_grad_norm": gradient_diagnostics["base_grad_norm"],
                "guide_grad_norm": gradient_diagnostics["guide_grad_norm"],
                "gradient_cosine": gradient_diagnostics["gradient_cosine"],
                "qp_active": gradient_diagnostics["qp_active"],
                "projection_coefficient": gradient_diagnostics[
                    "projection_coefficient"
                ],
                "max_expert_influence": float(max_influence.max().cpu()),
                "influence_bound_slack": float(influence_slack.min().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "trust_lr_event": trust_lr_event,
                "trust_lr_threshold": trust_lr_threshold,
                "trust_lr_stage": trust_lr_stage,
                "strict_sgd": float(strict_sgd),
                "epoch_seconds": time.time() - epoch_start,
                "topology_alignment": topology_alignment,
                "selected_checkpoint_epoch": 0,
                "selected_acc": float("nan"),
                "selected_nmi": float("nan"),
                "selected_ari": float("nan"),
                "protocol_version": protocol_version,
                "acc": float(scores["accuracy"]),
                "nmi": float(scores["NMI"]),
                "ari": float(scores["ARI"]),
            }
            for key in history_keys:
                history[key].append(row[key])

            if (epoch_index + 1) % print_num == 0 or epoch_index == 0:
                if self._config.get("log_style", "compact") == "diagnostic":
                    logger.info(
                        "Epoch:{}/{} loss={:.4f} base={:.4f} "
                        "guide={:.4f} R_mean={:.4f} kappa={:.4f} "
                        "eff={:.4f} cert={:.2f} trusted={:.2f} scope={:.2f} "
                        "fuse={:.4f} head_active={}/{} lr={:.2e} stage={} "
                        "cos={:.4f} QP={:.0f}".format(
                            epoch_index + 1,
                            epochs,
                            row["loss"],
                            row["base_loss"],
                            row["guide_loss"],
                            row["risk_mean"],
                            row["kappa_mean"],
                            row["effective_kappa_mean"],
                            row["certified_fraction"],
                            row["guide_active_fraction"],
                            row["guide_scope_fraction"],
                            row["fusion_correction_mean"],
                            row["active_clusters"],
                            self._n_clusters,
                            row["learning_rate"],
                            row["trust_lr_stage"],
                            row["gradient_cosine"],
                            row["qp_active"],
                        )
                    )
                    logger.info("Scores: {}".format(scores))
                else:
                    logger.info(
                        "[Epoch {:>4}/{}] Loss {:>9.4f} | Base {:>9.4f} | "
                        "Guide {:>7.4f} | LR {:.2e}".format(
                            epoch_index + 1,
                            epochs,
                            row["loss"],
                            row["base_loss"],
                            row["guide_loss"],
                            row["learning_rate"],
                        )
                    )
                    logger.info(
                        "                 ACC {:>6.2f}% | NMI {:>6.2f}% | "
                        "ARI {:>6.2f}% | Trust {:>5.1f}% | Topo {:>7.4f} | "
                        "HeadClusters {:>2}/{}".format(
                            100.0 * row["acc"],
                            100.0 * row["nmi"],
                            100.0 * row["ari"],
                            100.0 * row["guide_active_fraction"],
                            row["topology_alignment"],
                            row["active_clusters"],
                            self._n_clusters,
                        )
                    )

        # Restore the checkpoint chosen only by graph/embedding agreement.
        # Ground-truth labels are evaluated once after the epoch is fixed.
        selected_scores = {
            "accuracy": history["acc"][-1],
            "NMI": history["nmi"][-1],
            "ARI": history["ari"][-1],
        }
        if (
            checkpoint_mode == "max_topology_alignment"
            and best_checkpoint_state is not None
        ):
            if best_checkpoint_epoch != epochs:
                self.load_state_dict(best_checkpoint_state)
            if batchnorm_eval == "running":
                self.eval()
            else:
                self.train()
            with torch.no_grad():
                (
                    h1_selected,
                    h2_selected,
                    _,
                    _,
                    y1_selected,
                    y2_selected,
                    _,
                    _,
                ) = self(x_train[0], x_train[1], adj[0], adj[1])
                provisional_selected = (
                    self.fusion(h1_selected, h2_selected)
                    if attention
                    else 0.5 * (h1_selected + h2_selected)
                )
                if attention:
                    selected_risks, _, _ = self._estimate_topology_risks(
                        x_train, topology
                    )
                    selected_fused_risk = torch.maximum(
                        selected_risks[0], selected_risks[1]
                    )
                    selected_expert_risks = torch.stack(
                        (selected_risks[0], selected_risks[1], selected_fused_risk),
                        dim=1,
                    )
                    selected_prediction, _ = self.cluster(provisional_selected)
                    selected_experts = torch.stack(
                        (y1_selected, y2_selected, selected_prediction), dim=1
                    )
                    _, _, selected_reliability, _ = certified_consensus_teacher(
                        selected_experts,
                        selected_expert_risks,
                        risk_epsilon=float(self._rtg["risk_epsilon"]),
                        reliability_temperature=float(self._rtg["risk_temperature"]),
                        teacher_temperature=float(self._rtg["teacher_temperature"]),
                        probability_floor=float(self._rtg["probability_floor"]),
                    )
                    selected_fused, _ = self._reliability_aware_fuse(
                        h1_selected,
                        h2_selected,
                        selected_reliability,
                        topology,
                        attention_fused=provisional_selected,
                    )
                else:
                    selected_fused = provisional_selected
                if eval_mode == "kmeans_fused":
                    selected_scores = self.eval_acc(
                        selected_fused, Y_list, None, logger
                    )
                else:
                    selected_prediction, _ = self.cluster(selected_fused)
                    selected_scores = evaluation(
                        y_pred=selected_prediction.cpu().numpy().argmax(1),
                        y_true=Y_list[0],
                        accumulated_metrics=None,
                    )
            logger.info(
                "LABEL-FREE CHECKPOINT | Selected epoch {} | "
                "topology alignment {:.6f}".format(
                    best_checkpoint_epoch, best_checkpoint_score
                )
            )

        selected_epoch = best_checkpoint_epoch or epochs
        history["selected_checkpoint_epoch"][-1] = selected_epoch
        history["selected_acc"][-1] = float(selected_scores["accuracy"])
        history["selected_nmi"][-1] = float(selected_scores["NMI"])
        history["selected_ari"][-1] = float(selected_scores["ARI"])

        history_path = self._config.get("history_path")
        if history_path:
            history_dir = os.path.dirname(os.path.abspath(history_path))
            os.makedirs(history_dir, exist_ok=True)
            with open(history_path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=history_keys)
                writer.writeheader()
                for row_index in range(epochs):
                    writer.writerow(
                        {key: history[key][row_index] for key in history_keys}
                    )
            logger.info("Saved RTG convergence history: " + history_path)

        return (
            float(selected_scores["accuracy"]),
            float(selected_scores["NMI"]),
            float(selected_scores["ARI"]),
        )
