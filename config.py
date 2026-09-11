import os
from copy import deepcopy


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")


def _data_path(filename):
    """Return a project-local dataset path for a portable experiment."""
    return os.path.join(DATA_ROOT, filename)


data_dict = {
    2: "handwritten",
    3: "MSRC_v1",
    4: "NoisyMNIST",
    5: "BBC",
    7: "NGs",
    11: "3V_Fashion_MV",
    12: "Caltech-5V",
    14: "3-sources",
}


# Eight-dataset benchmark; numeric IDs are retained for compatibility.
FORMAL_BENCHMARK_DATASETS = (
    "NGs",
    "NoisyMNIST",
    "MSRC_v1",
    "3V_Fashion_MV",
    "Caltech-5V",
    "BBC",
    "handwritten",
    "3-sources",
)


EXPERIMENT_DEFAULTS = dict(
    # Increment whenever histories are not comparable with an older run.
    protocol_version="theory_aligned_stable_v2",
    # Safe development defaults. They intentionally run only one condition.
    method="rtg",
    missing_rates=[0.1],
    seeds=[1],
    mask_seed=2023,
    # missing_rate denotes the fraction of samples with exactly one view.
    mask_protocol="incomplete_sample",
    device="auto",
    optimizer="adam",
    print_every=10,
    # Compact is suitable for paper experiments. Use --log-style diagnostic
    # when the full RTG certificate/projection trace is needed for debugging.
    log_style="compact",
    eval_mode="cluster_head",
    kmeans_eval_interval=10,
    # The contrastive objective is based on angular similarity. Normalizing
    # the fused rows before K-means prevents latent norm drift from changing
    # the clustering geometry late in training.
    kmeans_l2_normalize=True,
    # Evaluation uses current full-batch statistics.
    # Set this to 'running' only for strict inductive inference.
    batchnorm_eval="transductive",
)


PAPER_PROTOCOL_DEFAULTS = dict(
    # Main-table protocol. Enable it with train.py --paper-protocol.
    missing_rates=[0.1, 0.3, 0.5],
    seeds=[1, 2, 3, 4, 5],
    epochs=500,
    optimizer="adam",
)


INCOMPLETE_PROTOCOL_DATASETS = frozenset(FORMAL_BENCHMARK_DATASETS)


RTG_DEFAULTS = dict(
    attention=True,
    # Topology candidate set. candidate_count includes the transferred graph.
    candidate_count=4,
    replacement_ratio=0.3,
    candidate_seed=2023,
    # Empirical certificate. A non-zero cover_lipschitz must be a validated
    # local upper bound before the finite-cover result is called strict.
    certificate_refresh_interval=1,
    cover_lipschitz=0.0,
    risk_epsilon=1.0e-4,
    probability_floor=1.0e-8,
    # Certificate-constrained logarithmic-opinion-pool teacher.  In the RTG
    # path the protected base F is exactly instance + cluster contrastive loss;
    # auxiliary teaching is defined separately.
    risk_temperature=1.0,
    teacher_temperature=0.5,
    guide_weight=1.0,
    # Let the contrastive objective first establish a meaningful cluster
    # space before the topology residual reaches full strength.
    guide_warmup_epochs=50,
    # Candidate generation is induced by alternative replacements for
    # missing-view samples. Restrict the residual teacher to samples whose
    # evidence is incomplete instead of rewriting complete cases.
    guide_scope="incomplete",
    # The certified teacher is allowed to abstain below the neutral trust
    # level.  A soft gate keeps the residual objective continuous.
    reliability_threshold=0.5,
    reliability_gate_temperature=0.05,
    # Reliability-weighted fusion remains an explicit ablation. The main
    # availability-aware path below only prevents an actually absent view
    # from entering an incomplete sample's fused representation.
    reliability_aware_fusion=False,
    # For an incomplete sample, never average an unavailable-view embedding
    # into the final representation. Complete samples retain learned attention.
    availability_aware_fusion=True,
    # Cross-view InfoNCE positives are trustworthy only where both source
    # views are observed. The legacy all-sample behavior remains configurable.
    contrastive_pair_scope="complete",
    # Each mini-batch InstanceLoss is already a mean; averaging across batches
    # makes its effective weight independent of dataset size.
    instance_batch_reduction="mean",
    bootstrap_teacher_epochs=5,
    teacher_transition_epochs=5,
    # Label-free certificate-aware annealing for the theorem-compatible
    # zero-momentum, zero-decay SGD setting. It is intentionally disabled at
    # runtime for Adam, whose adaptive state is not covered by the Euclidean
    # half-space parameter-step argument.
    trust_lr_decay=True,
    trust_lr_thresholds=(0.5, 0.1),
    trust_lr_factor=0.1,
    trust_lr_patience=5,
    trust_lr_min_epoch=50,
    trust_lr_floor=1.0e-6,
    # Base-objective preserving half-space update.
    base_safe=True,
    base_margin=0.1,
    # Component coefficients remain one; RTG additionally applies the
    # availability-aware pairing and batch-reduction rules declared above.
    loss_weights=dict(
        cluster=1.0,
        instance=1.0,
    ),
)


def _get_dataset_config(flag=3):
    """Determine the parameter information of the network"""
    data_name = data_dict[flag]
    if data_name in ["MSRC_v1"]:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=7,
            training=dict(lr=0.001, epoch=500, batch_size=256),
            Autoencoder=dict(
                gcnEncoder1=[576, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[512, 1024, 1024, 1024, 1024 // 8],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )
    elif data_name in ["handwritten"]:
        return dict(
            dataset=data_name,
            view_indices=(1, 2),
            source_view_count=6,
            source_view_names=("X[1]-76D", "X[2]-216D"),
            normalize_views=True,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            training=dict(lr=0.001, epoch=500, batch_size=256),
            Autoencoder=dict(
                gcnEncoder1=[76, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[216, 1024, 1024, 1024, 1024 // 2],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )
    elif data_name in ["NoisyMNIST"]:
        return dict(
            dataset=data_name,
            max_samples=5000,
            sample_seed=2023,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            training=dict(lr=0.001, epoch=500, batch_size=256),
            Autoencoder=dict(
                gcnEncoder1=[784, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[784, 1024, 1024, 1024, 1024 // 8],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )
    elif data_name in ["BBC"]:
        return dict(
            dataset=data_name,
            view_indices=(0, 1),
            normalize_views=True,
            topk=10,
            missing_rate=0.0,
            n_clusters=5,
            training=dict(lr=0.001, epoch=500, batch_size=128),
            Autoencoder=dict(
                gcnEncoder1=[4659, 1024, 1024, 1024, 128],
                gcnEncoder2=[4633, 1024, 1024, 1024, 128],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )
    elif data_name in ["NGs"]:
        return dict(
            dataset=data_name,
            view_indices=(0, 1),
            normalize_views=True,
            topk=10,
            missing_rate=0.0,
            n_clusters=5,
            training=dict(lr=0.0003, epoch=500, batch_size=128),
            Autoencoder=dict(
                gcnEncoder1=[2000, 1024, 1024, 1024, 128],
                gcnEncoder2=[2000, 1024, 1024, 1024, 128],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )
    elif data_name in ["3V_Fashion_MV"]:
        return dict(
            dataset=data_name,
            source_file=_data_path("3V_Fashion_MV.mat"),
            view_keys=("X1", "X2", "X3"),
            label_key="Y",
            view_indices=(0, 1),
            source_view_count=3,
            source_view_names=("X1", "X2"),
            normalize_views=True,
            max_samples=5000,
            sample_seed=2023,
            sampling="stratified",
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            training=dict(lr=0.0003, epoch=500, batch_size=256),
            Autoencoder=dict(
                gcnEncoder1=[784, 1024, 1024, 1024, 128],
                gcnEncoder2=[784, 1024, 1024, 1024, 128],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )
    elif data_name in ["Caltech-5V"]:
        return dict(
            dataset=data_name,
            source_file=_data_path("Caltech-5V.mat"),
            view_keys=("X1", "X2", "X3", "X4", "X5"),
            label_key="Y",
            view_indices=(0, 1),
            source_view_count=5,
            source_view_names=("X1-40D", "X2-254D"),
            normalize_views=True,
            topk=10,
            missing_rate=0.0,
            n_clusters=7,
            training=dict(lr=0.0003, epoch=500, batch_size=256),
            Autoencoder=dict(
                gcnEncoder1=[40, 1024, 1024, 1024, 128],
                gcnEncoder2=[254, 1024, 1024, 1024, 128],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )
    elif data_name in ["3-sources"]:
        return dict(
            dataset=data_name,
            source_file=_data_path("3-sources.mat"),
            view_keys=("bbc", "guardian", "reuters"),
            label_key="truth",
            view_indices=(0, 1),
            source_view_count=3,
            source_view_names=("bbc", "guardian"),
            normalize_views=True,
            topk=10,
            missing_rate=0.0,
            n_clusters=6,
            training=dict(lr=0.0003, epoch=500, batch_size=64),
            Autoencoder=dict(
                gcnEncoder1=[3560, 1024, 1024, 1024, 128],
                gcnEncoder2=[3631, 1024, 1024, 1024, 128],
                activations1="relu",
                activations2="relu",
                batchnorm=True,
            ),
        )


def get_config(flag=3):
    """Return one complete, self-contained experiment configuration.

    Dataset/network settings, development defaults, the paper protocol, and
    all RTG-IMVC hyperparameters are assembled here. The command-line runner
    may override individual values but no longer owns hidden method defaults.
    """
    config = _get_dataset_config(flag)
    if config is None:
        raise ValueError("No configuration is defined for flag {}".format(flag))
    experiment_overrides = config.pop("experiment_overrides", {})
    rtg_overrides = config.pop("rtg_overrides", {})
    config["experiment"] = deepcopy(EXPERIMENT_DEFAULTS)
    config["experiment"].update(deepcopy(experiment_overrides))
    if config["dataset"] in FORMAL_BENCHMARK_DATASETS:
        # Use one clustering extractor for all eight datasets. Labels are consulted only after K-means for reporting.
        config["experiment"]["eval_mode"] = "kmeans_fused"
        config["experiment"]["kmeans_eval_interval"] = 100
        config["experiment"]["kmeans_l2_normalize"] = True
        config["formal_benchmark"] = True
    else:
        config["formal_benchmark"] = False
    config["paper_protocol"] = deepcopy(PAPER_PROTOCOL_DEFAULTS)
    if config["dataset"] in INCOMPLETE_PROTOCOL_DATASETS:
        config["paper_protocol"]["missing_rates"] = [0.1, 0.3, 0.5]
    config["rtg"] = deepcopy(RTG_DEFAULTS)
    config["rtg"].update(deepcopy(rtg_overrides))
    if config["formal_benchmark"]:
        # A single label-free stability policy is used for every formal
        # dataset. ACC/NMI/ARI never participate in checkpoint selection.
        training = config["training"]
        learning_rate = float(training["lr"])
        training.setdefault("lr_scheduler", "cosine")
        training.setdefault("min_lr", max(1.0e-6, learning_rate * 0.01))
        training.setdefault("grad_clip", 5.0)
        training.setdefault("instance_batch_reduction", "mean")
        training.setdefault("label_free_checkpoint", "max_topology_alignment")
        training.setdefault("checkpoint_interval", 10)
        training.setdefault("checkpoint_min_epoch", 100)
    config["print_num"] = int(config["experiment"]["print_every"])
    return config
