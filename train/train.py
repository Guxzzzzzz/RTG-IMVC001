"""Command-line training entry for RTG-IMVC."""

from __future__ import division, print_function
import argparse
import collections
import csv
import os
import sys
import time
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from config import data_dict, get_config
from models.RTGIMVC import RTGIMVC
from utils.datasets import load_data
from utils.graph_adjacency import get_rtg_adjacency
from utils.std_utils import cal_std
from utils.util import get_logger, get_mask, setup_seed

DATASET_TO_FLAG = {name.lower(): flag for flag, name in data_dict.items()}


def _comma_separated(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _dataset_flag(value):
    try:
        flag = int(value)
        if flag not in data_dict:
            raise ValueError
        return flag
    except ValueError:
        key = value.lower()
        if key not in DATASET_TO_FLAG:
            raise argparse.ArgumentTypeError(
                "unknown dataset {!r}; choose from {}".format(
                    value, ", ".join(data_dict.values())
                )
            )
        return DATASET_TO_FLAG[key]


def build_parser():
    parser = argparse.ArgumentParser(description="Train RTG-IMVC.")
    parser.add_argument(
        "--dataset",
        type=_dataset_flag,
        default=3,
        help="dataset name or config index; default: MSRC_v1",
    )
    parser.add_argument(
        "--missing-rates", default=None, help="comma-separated rates, e.g. 0.1,0.3,0.5"
    )
    parser.add_argument(
        "--seeds", default=None, help="comma-separated model seeds, e.g. 1,2,3,4,5"
    )
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument(
        "--mask-protocol",
        choices=("incomplete_sample", "legacy"),
        default=None,
        help="incomplete_sample means the requested fraction of samples retains exactly one view; legacy reproduces the old heuristic",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--view-normalization",
        choices=("none", "l2"),
        default=None,
        help="override dataset feature normalization",
    )
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument(
        "--log-style",
        choices=("compact", "diagnostic"),
        default=None,
        help="compact paper log or full RTG diagnostics",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--optimizer", choices=("auto", "sgd", "adam"), default=None)
    parser.add_argument(
        "--eval-mode", choices=("cluster_head", "kmeans_fused"), default=None
    )
    parser.add_argument("--kmeans-eval-interval", type=int, default=None)
    parser.add_argument(
        "--batchnorm-eval",
        choices=("transductive", "running"),
        default=None,
        help="transductive uses full-dataset batch statistics (the original ICMVC protocol); running uses stored BatchNorm statistics",
    )
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="use config.py PAPER_PROTOCOL_DEFAULTS",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip conditions whose history CSV already contains all epochs",
    )
    parser.add_argument(
        "--rtg-candidates",
        type=int,
        default=None,
        help="total candidates including baseline topology",
    )
    parser.add_argument("--replacement-ratio", type=float, default=None)
    parser.add_argument("--certificate-refresh", type=int, default=None)
    parser.add_argument(
        "--cover-lipschitz",
        type=float,
        default=None,
        help="validated local bound; 0 gives empirical diameter",
    )
    parser.add_argument("--risk-epsilon", type=float, default=None)
    parser.add_argument("--risk-temperature", type=float, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=None)
    parser.add_argument("--guide-weight", type=float, default=None)
    parser.add_argument("--guide-warmup", type=int, default=None)
    parser.add_argument(
        "--guide-scope",
        choices=("all", "incomplete"),
        default=None,
        help="samples eligible for topology residual guidance",
    )
    parser.add_argument("--reliability-threshold", type=float, default=None)
    parser.add_argument("--reliability-gate-temperature", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-epochs", type=int, default=None)
    parser.add_argument("--teacher-transition-epochs", type=int, default=None)
    parser.add_argument(
        "--trust-lr-thresholds",
        default=None,
        help="comma-separated trusted-sample fractions, e.g. 0.5,0.1",
    )
    parser.add_argument("--trust-lr-factor", type=float, default=None)
    parser.add_argument("--trust-lr-patience", type=int, default=None)
    parser.add_argument("--trust-lr-min-epoch", type=int, default=None)
    parser.add_argument("--trust-lr-floor", type=float, default=None)
    parser.add_argument(
        "--disable-trust-lr",
        action="store_true",
        default=None,
        help="disable label-free certificate-aware learning-rate decay",
    )
    fusion_group = parser.add_mutually_exclusive_group()
    fusion_group.add_argument(
        "--enable-reliability-fusion",
        action="store_true",
        default=None,
        help="enable the optional reliability-aware fusion ablation",
    )
    fusion_group.add_argument(
        "--disable-reliability-fusion",
        action="store_true",
        default=None,
        help="use attention fusion (default)",
    )
    parser.add_argument("--base-margin", type=float, default=None)
    parser.add_argument("--disable-base-safe", action="store_true", default=None)
    parser.add_argument(
        "--quick", action="store_true", help="two epochs, one seed, at most 256 samples"
    )
    return parser


def _resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _limit_data(X_list, Y_list, max_samples, seed):
    if max_samples is None or X_list[0].shape[0] <= max_samples:
        return (X_list, Y_list)
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    labels = np.asarray(Y_list[0]).reshape(-1)
    classes = np.unique(labels)
    if max_samples < classes.size:
        raise ValueError(
            "max_samples={} is smaller than the number of classes={}".format(
                max_samples, classes.size
            )
        )
    rng = np.random.RandomState(seed)
    selected = []
    base = max_samples // classes.size
    remainder = max_samples % classes.size
    for class_position, label in enumerate(classes):
        class_indices = np.flatnonzero(labels == label)
        take = min(class_indices.size, base + (1 if class_position < remainder else 0))
        selected.extend(rng.choice(class_indices, size=take, replace=False))
    selected = np.asarray(sorted(selected), dtype=np.int64)
    if selected.size < max_samples:
        remaining = np.setdiff1d(np.arange(labels.size), selected)
        extra = rng.choice(remaining, size=max_samples - selected.size, replace=False)
        selected = np.sort(np.concatenate((selected, extra)))
    return (
        [view[selected] for view in X_list],
        [labels_view[selected] for labels_view in Y_list],
    )


def _configure(args):
    config = get_config(flag=args.dataset)
    experiment = config["experiment"]
    if args.paper_protocol:
        protocol = config["paper_protocol"]
        experiment["missing_rates"] = list(protocol["missing_rates"])
        experiment["seeds"] = list(protocol["seeds"])
        experiment["optimizer"] = protocol["optimizer"]
        config["training"]["epoch"] = int(protocol["epochs"])
    if args.missing_rates is not None:
        experiment["missing_rates"] = _comma_separated(args.missing_rates, float)
    if args.seeds is not None:
        experiment["seeds"] = _comma_separated(args.seeds, int)
    if args.mask_seed is not None:
        experiment["mask_seed"] = args.mask_seed
    if args.mask_protocol is not None:
        experiment["mask_protocol"] = args.mask_protocol
    if args.device is not None:
        experiment["device"] = args.device
    if args.optimizer is not None:
        experiment["optimizer"] = args.optimizer
    if args.eval_mode is not None:
        experiment["eval_mode"] = args.eval_mode
    if args.kmeans_eval_interval is not None:
        experiment["kmeans_eval_interval"] = args.kmeans_eval_interval
    if args.batchnorm_eval is not None:
        experiment["batchnorm_eval"] = args.batchnorm_eval
    if args.print_every is not None:
        experiment["print_every"] = args.print_every
    if args.log_style is not None:
        experiment["log_style"] = args.log_style
    if args.epochs is not None:
        config["training"]["epoch"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["lr"] = args.learning_rate
    if args.topk is not None:
        config["topk"] = args.topk
    if args.view_normalization is not None:
        config["normalize_views"] = args.view_normalization == "l2"
    config["print_num"] = int(experiment["print_every"])
    config["eval_mode"] = experiment["eval_mode"]
    config["kmeans_eval_interval"] = int(experiment["kmeans_eval_interval"])
    config["kmeans_l2_normalize"] = bool(experiment.get("kmeans_l2_normalize", True))
    config["batchnorm_eval"] = experiment["batchnorm_eval"]
    config["log_style"] = experiment["log_style"]
    config["protocol_version"] = str(experiment.get("protocol_version", "unversioned"))
    rtg_overrides = {
        "candidate_count": args.rtg_candidates,
        "replacement_ratio": args.replacement_ratio,
        "certificate_refresh_interval": args.certificate_refresh,
        "cover_lipschitz": args.cover_lipschitz,
        "risk_epsilon": args.risk_epsilon,
        "risk_temperature": args.risk_temperature,
        "teacher_temperature": args.teacher_temperature,
        "guide_weight": args.guide_weight,
        "guide_warmup_epochs": args.guide_warmup,
        "guide_scope": args.guide_scope,
        "reliability_threshold": args.reliability_threshold,
        "reliability_gate_temperature": args.reliability_gate_temperature,
        "bootstrap_teacher_epochs": args.bootstrap_teacher_epochs,
        "teacher_transition_epochs": args.teacher_transition_epochs,
        "trust_lr_factor": args.trust_lr_factor,
        "trust_lr_patience": args.trust_lr_patience,
        "trust_lr_min_epoch": args.trust_lr_min_epoch,
        "trust_lr_floor": args.trust_lr_floor,
        "base_margin": args.base_margin,
    }
    for key, value in rtg_overrides.items():
        if value is not None:
            config["rtg"][key] = value
    if args.trust_lr_thresholds is not None:
        config["rtg"]["trust_lr_thresholds"] = tuple(
            _comma_separated(args.trust_lr_thresholds, float)
        )
    if args.disable_trust_lr:
        config["rtg"]["trust_lr_decay"] = False
    if args.enable_reliability_fusion:
        config["rtg"]["reliability_aware_fusion"] = True
    elif args.disable_reliability_fusion:
        config["rtg"]["reliability_aware_fusion"] = False
    if args.disable_base_safe:
        config["rtg"]["base_safe"] = False
    return config


def _build_optimizer(model, config, method, requested):
    optimizer_name = requested
    if optimizer_name == "auto":
        optimizer_name = "sgd"
    learning_rate = float(config["training"]["lr"])
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0
        )
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def _build_scheduler(optimizer, config):
    if config["training"].get("lr_scheduler") != "cosine":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epoch"]),
        eta_min=float(config["training"].get("min_lr", 1e-05)),
    )


def _move_topology(topology, device):
    for view_index in range(2):
        topology["candidate_adjs"][view_index] = [
            adjacency.to(device) for adjacency in topology["candidate_adjs"][view_index]
        ]
        topology["cover_radius"][view_index] = topology["cover_radius"][view_index].to(
            device
        )
        topology["missing_mask"][view_index] = topology["missing_mask"][view_index].to(
            device
        )
    return topology


def _close_logger(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _completed_history(path, expected_epochs, expected_protocol_version=None):
    """Return selected metrics only for a compatible complete history."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != list(range(1, int(expected_epochs) + 1)):
            return None
        final = rows[-1]
        if expected_protocol_version is not None:
            versions = {row.get("protocol_version", "") for row in rows}
            if versions != {str(expected_protocol_version)}:
                return None
        selected_keys = ("selected_acc", "selected_nmi", "selected_ari")
        if all((final.get(key, "") not in ("", "nan", "NaN") for key in selected_keys)):
            return tuple((float(final[key]) for key in selected_keys))
        return tuple((float(final[key]) for key in ("acc", "nmi", "ari")))
    except (KeyError, TypeError, ValueError, OSError):
        return None


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = _configure(args)
    if args.quick:
        config["training"]["epoch"] = 2
        config["experiment"]["seeds"] = [1]
        config["experiment"]["print_every"] = 1
        config["print_num"] = 1
        args.max_samples = 256 if args.max_samples is None else args.max_samples
    experiment = config["experiment"]
    method = experiment["method"]
    missing_rates = list(experiment["missing_rates"])
    seeds = list(experiment["seeds"])
    mask_seed = int(experiment["mask_seed"])
    optimizer_name = experiment["optimizer"]
    if not missing_rates or not seeds:
        raise ValueError("at least one missing rate and one seed are required")
    if any((rate < 0.0 or rate >= 1.0 for rate in missing_rates)):
        raise ValueError("missing rates must lie in [0, 1)")
    device = _resolve_device(experiment["device"])
    X_list, Y_list = load_data(config, train_dir=True)
    max_samples = (
        args.max_samples if args.max_samples is not None else config.get("max_samples")
    )
    X_list, Y_list = _limit_data(
        X_list, Y_list, max_samples, int(config.get("sample_seed", mask_seed))
    )
    x1_raw, x2_raw = (X_list[0], X_list[1])
    print(
        "PREPARING | Method {:<5} | Dataset {:<16} | Device {:<7} | Samples {}".format(
            method.upper(), config["dataset"], str(device), x1_raw.shape[0]
        ),
        flush=True,
    )
    for missing_rate in missing_rates:
        config["missing_rate"] = missing_rate
        np.random.seed(mask_seed)
        mask_np = get_mask(
            x1_raw.shape[0], missing_rate, protocol=experiment["mask_protocol"]
        )
        actual_incomplete = float(np.mean(mask_np.sum(axis=1) < 2))
        actual_cell_missing = float(1.0 - mask_np.mean())
        x1_missing = x1_raw * mask_np[:, 0][:, np.newaxis]
        x2_missing = x2_raw * mask_np[:, 1][:, np.newaxis]
        x_train = [
            torch.from_numpy(x1_missing).float().to(device),
            torch.from_numpy(x2_missing).float().to(device),
        ]
        mask = torch.from_numpy(mask_np).long().to(device)
        topology = None
        adj1, _, adj2, _, topology = get_rtg_adjacency(
            x_train[0],
            x_train[1],
            mask,
            x1_missing.shape[0],
            topk=int(config["topk"]),
            candidate_count=int(config["rtg"]["candidate_count"]),
            replacement_ratio=float(config["rtg"]["replacement_ratio"]),
            random_state=int(config["rtg"]["candidate_seed"]),
        )
        topology = _move_topology(topology, device)
        adj = [adj1.to(device), adj2.to(device)]
        log_dir = os.path.join(REPO_ROOT, "logs")
        history_dir = os.path.join(REPO_ROOT, "history")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        logger, _ = get_logger(config, main_dir=log_dir + os.sep)
        logger.info("=" * 72)
        logger.info(
            "EXPERIMENT | Method {:<5} | Dataset {:<16} | Missing {:.1f}".format(
                method.upper(), config["dataset"], missing_rate
            )
        )
        logger.info(
            "SETUP      | Device {:<7} | Optimizer {:<4} | Epochs {:>4} | Seeds {}".format(
                str(device),
                optimizer_name.upper(),
                int(config["training"]["epoch"]),
                ",".join((str(seed) for seed in seeds)),
            )
        )
        logger.info(
            "MASK       | Protocol {:<17} | Incomplete {:.1f}% | Missing cells {:.1f}%".format(
                experiment["mask_protocol"],
                100.0 * actual_incomplete,
                100.0 * actual_cell_missing,
            )
        )
        logger.info(
            "EVALUATION | {:<13} | K-means interval {} | L2 {}".format(
                config["eval_mode"],
                config["kmeans_eval_interval"],
                "on" if config["kmeans_l2_normalize"] else "off",
            )
        )
        logger.info(
            "STABILITY  | Scheduler {:<6} | Min LR {:.2e} | Clip {:.1f}".format(
                config["training"].get("lr_scheduler", "none"),
                float(config["training"].get("min_lr", config["training"]["lr"])),
                float(config["training"].get("grad_clip", 0.0)),
            )
        )
        logger.info(
            "CHECKPOINT | {:<22} | Grid {:>3} | Begin {:>3} | Labels no".format(
                config["training"].get("label_free_checkpoint", "last"),
                int(config["training"].get("checkpoint_interval", 1)),
                int(config["training"].get("checkpoint_min_epoch", 1)),
            )
        )
        logger.info("=" * 72)
        if topology is not None:
            logger.info(
                "Topology candidates: {}, replacement ratio: {}, certificate kind: {}".format(
                    topology["candidate_count"],
                    topology["replacement_ratio"],
                    topology["certificate_kind"],
                )
            )
        fold_acc, fold_nmi, fold_ari = ([], [], [])
        start = time.time()
        for model_seed in seeds:
            method_tag = "RTG_IMVC"
            if args.quick:
                method_tag += "_QUICK"
            rate_tag = str(missing_rate).replace(".", "_")
            config["history_path"] = os.path.join(
                history_dir,
                "{}_{}_missing{}_seed{}.csv".format(
                    method_tag, config["dataset"], rate_tag, model_seed
                ),
            )
            if args.resume:
                cached_scores = _completed_history(
                    config["history_path"],
                    config["training"]["epoch"],
                    expected_protocol_version=config["protocol_version"],
                )
                if cached_scores is not None:
                    logger.info(
                        "Seed {} skipped: complete {}-epoch history exists".format(
                            model_seed, config["training"]["epoch"]
                        )
                    )
                    fold_acc.append(cached_scores[0])
                    fold_nmi.append(cached_scores[1])
                    fold_ari.append(cached_scores[2])
                    continue
            setup_seed(model_seed)
            accumulated_metrics = collections.defaultdict(list)
            model = RTGIMVC(config)
            model.to(device)
            optimizer = _build_optimizer(model, config, method, optimizer_name)
            scheduler = _build_scheduler(optimizer, config)
            logger.info(
                "SEED {:>2}   | Optimizer {:<5} | Initial LR {:.2e}".format(
                    model_seed,
                    optimizer.__class__.__name__,
                    optimizer.param_groups[0]["lr"],
                )
            )
            acc, nmi, ari = model.run_train(
                x_train,
                Y_list,
                adj,
                optimizer,
                logger,
                accumulated_metrics,
                device,
                scheduler=scheduler,
                topology=topology,
            )
            fold_acc.append(acc)
            fold_nmi.append(nmi)
            fold_ari.append(ari)
        logger.info("--------------------Training over--------------------")
        acc_summary, nmi_summary, ari_summary = cal_std(
            logger, fold_acc, fold_nmi, fold_ari
        )
        logger.info("Elapsed seconds: {:.2f}".format(time.time() - start))
        print(
            "RESULT | Missing {:.1f} | ACC {:.2f} | NMI {:.2f} | ARI {:.2f}".format(
                missing_rate, acc_summary, nmi_summary, ari_summary
            )
        )
        _close_logger(logger)


if __name__ == "__main__":
    main()
