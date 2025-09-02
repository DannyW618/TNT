import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from config import (
    ACTIVE_DATASETS,
    ALL_BASELINE_METHODS,
    ALL_DATASET_NAMES,
    ALL_SHIFT_TYPES,
    DEFAULT_SETTINGS,
)

def parse_args() -> Tuple[List[str], Dict[str, Any]]:
    """Parse CLI args with defaults from config.py and return (datasets, config)."""
    p = argparse.ArgumentParser(description="GCN/TNTOOD Training + OOD Evaluation")

    p.add_argument(
        "--shift_type",
        nargs="*",
        default=[],
        choices=ALL_SHIFT_TYPES,
        help="OOD shifts to evaluate. 'label'/'arxiv_time' also affect training.",
    )
    p.add_argument(
        "--use_baseline",
        nargs="*",
        default=[],
        choices=ALL_BASELINE_METHODS,
        help="Baseline OOD methods to run instead of default detector.",
    )

    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(ACTIVE_DATASETS),
        help=f"Datasets to run. Available: {', '.join(ALL_DATASET_NAMES)}",
    )
    p.add_argument("--runs", type=int, default=DEFAULT_SETTINGS["runs"])
    p.add_argument("--seed", type=int, default=DEFAULT_SETTINGS["random_seed"])
    p.add_argument(
        "--verbose",
        default=DEFAULT_SETTINGS["verbose"],
        action=argparse.BooleanOptionalAction,
        help="Verbose logging",
    )

    # --- WandB ---
    p.add_argument(
        "--use_wandb",
        default=DEFAULT_SETTINGS["use_wandb"],
        action=argparse.BooleanOptionalAction,
    )

    p.add_argument("--project_prefix", default=DEFAULT_SETTINGS["project_prefix"]) 

    # --- Model Hyperparameters ---
    p.add_argument("--hidden_channels", type=int, default=DEFAULT_SETTINGS["hidden_channels"])
    p.add_argument("--dropout", type=float, default=DEFAULT_SETTINGS["dropout"])
    p.add_argument(
        "--use_linear_layer",
        default=DEFAULT_SETTINGS["use_linear_layer"],
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument("--num_layers", type=int, default=DEFAULT_SETTINGS["num_layers"])  # GNN layers

    p.add_argument(
        "--use_tntood",
        default=DEFAULT_SETTINGS.get("use_tntood", False),
        action=argparse.BooleanOptionalAction,
        help="Use TNTOODModel and TNTOODLoss",
    )
    p.add_argument("--tnt_gnn_embedding_dim", type=int, default=DEFAULT_SETTINGS.get("tnt_gnn_embedding_dim", 128))
    p.add_argument("--tnt_gnn_hidden_dim", type=int, default=DEFAULT_SETTINGS.get("tnt_gnn_hidden_dim", 128))
    p.add_argument("--tnt_num_gnn_layers", type=int, default=DEFAULT_SETTINGS.get("tnt_num_gnn_layers", 2))
    p.add_argument("--tnt_projection_hidden_dim", type=int, default=DEFAULT_SETTINGS.get("tnt_projection_hidden_dim", 128))
    p.add_argument("--tnt_projection_shared_dim", type=int, default=DEFAULT_SETTINGS.get("tnt_projection_shared_dim", 128))

    # --- Loss Parameters ---
    p.add_argument("--tnt_loss_temp", type=float, default=DEFAULT_SETTINGS.get("tnt_loss_temp", 0.01))
    p.add_argument("--contrast_w", type=float, default=DEFAULT_SETTINGS.get("contrast_w", 1.0))
    p.add_argument("--tnt_loss_id_weight", type=float, default=DEFAULT_SETTINGS.get("tnt_loss_id_weight", 1.0))
    p.add_argument("--use_batch_contrastive", default=DEFAULT_SETTINGS.get("use_batch_contrastive", False), action=argparse.BooleanOptionalAction)
    p.add_argument("--contrastive_batch_size", type=int, default=DEFAULT_SETTINGS.get("contrastive_batch_size", 256))

    # --- Detector config ---
    p.add_argument("--align_w", type=float, default=DEFAULT_SETTINGS.get("align_w", 1.0))
    p.add_argument("--tnt_ood_w_id_uncertainty", type=float, default=DEFAULT_SETTINGS.get("tnt_ood_w_id_uncertainty", 0.0))
    p.add_argument("--tnt_ood_eps", type=float, default=DEFAULT_SETTINGS.get("tnt_ood_eps", 1e-9))

    # --- Training ---
    p.add_argument("--learning_rate", type=float, default=DEFAULT_SETTINGS["learning_rate"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_SETTINGS["weight_decay"])
    p.add_argument("--epochs", type=int, default=DEFAULT_SETTINGS["epochs"])
    p.add_argument("--early_stopping_patience", type=int, default=DEFAULT_SETTINGS["early_stopping_patience"])
    p.add_argument("--clip_grad_norm", type=float, default=DEFAULT_SETTINGS.get("clip_grad_norm", 1.0))

    # --- OOD Detector (standard) ---
    p.add_argument("--T", type=float, default=DEFAULT_SETTINGS["T"], help="Temperature for GCN energy score")
    p.add_argument("--use_prop", default=DEFAULT_SETTINGS["use_prop"], action=argparse.BooleanOptionalAction)
    p.add_argument("--K", type=int, default=DEFAULT_SETTINGS["K"])  # propagation layers
    p.add_argument("--alpha", type=float, default=DEFAULT_SETTINGS["alpha"])  # propagation factor

    # --- DataLoader ---
    p.add_argument(
        "--train_loader_type",
        default=DEFAULT_SETTINGS.get("train_loader_type", "None"),
        choices=["NeighborLoader", "ClusterLoader", "None"],
        help="Type of DataLoader for training. 'None' = full-batch.",
    )
    p.add_argument("--train_loader_batch_size", type=int, default=DEFAULT_SETTINGS.get("train_loader_batch_size", 128))
    p.add_argument("--train_loader_num_neighbors", default=DEFAULT_SETTINGS.get("train_loader_num_neighbors", "15,10"))
    p.add_argument("--num_loader_workers", type=int, default=DEFAULT_SETTINGS.get("num_loader_workers", 2))

    # --- Shift params ---
    # Structure
    p.add_argument("--p_ii_factor", type=float, nargs="+", default=[DEFAULT_SETTINGS["structure_shift"]["p_ii_factor"]])
    p.add_argument("--p_ij_factor", type=float, nargs="+", default=[DEFAULT_SETTINGS["structure_shift"]["p_ij_factor"]])
    p.add_argument("--structure_noise_level", type=float, nargs="+", default=[DEFAULT_SETTINGS["structure_shift"]["noise_level"]])
    # Feature
    p.add_argument("--noise_level", type=float, nargs="+", default=[DEFAULT_SETTINGS["feature_shift"]["noise_level"]])
    # Label
    p.add_argument("--ood_class_to_leave_out", type=int, nargs="+", default=DEFAULT_SETTINGS["label_shift"]["ood_class_to_leave_out"])  # noqa: E501
    p.add_argument("--label_setting", choices=["inductive", "transductive"], default=DEFAULT_SETTINGS["label_shift"]["setting"])  # noqa: E501
    # Text
    p.add_argument("--text_augmentation_type", nargs="+", choices=["synonym", "antonym"], default=[DEFAULT_SETTINGS["text_shift"]["augmentation_type"]])  # noqa: E501
    p.add_argument("--text_noise_level", type=float, nargs="+", default=[DEFAULT_SETTINGS["text_shift"]["noise_level"]])
    # Text Swap
    p.add_argument("--swap_scope", nargs="+", default=[DEFAULT_SETTINGS["text_swap_shift"]["swap_scope"]])
    p.add_argument("--swap_ratio", type=float, nargs="+", default=[DEFAULT_SETTINGS["text_swap_shift"]["swap_ratio"]])
    # Semantic Connection
    p.add_argument("--target_density_factor", type=float, nargs="+", default=[DEFAULT_SETTINGS["semantic_connection_shift"]["target_density_factor"]])  # noqa: E501
    p.add_argument(
        "--semantic_selection_mode",
        nargs="+",
        choices=["top", "bottom", "threshold"],
        default=[DEFAULT_SETTINGS["semantic_connection_shift"]["selection_mode"]],
    )
    p.add_argument("--threshold_percentile", type=float, nargs="+", default=[DEFAULT_SETTINGS["semantic_connection_shift"]["threshold_percentile"]])  # noqa: E501
    # Arxiv Time Split
    p.add_argument("--arxiv_time_bound_train", type=int, nargs=2, default=DEFAULT_SETTINGS["arxiv_time_split"]["time_bound_train"])  # noqa: E501
    p.add_argument("--arxiv_time_bound_test", type=int, nargs="+", default=DEFAULT_SETTINGS["arxiv_time_split"]["time_bound_test"])  # noqa: E501
    p.add_argument("--arxiv_inductive", default=DEFAULT_SETTINGS["arxiv_time_split"]["inductive"], action=argparse.BooleanOptionalAction)

    args = p.parse_args()

    # Base config
    config: Dict[str, Any] = {**DEFAULT_SETTINGS, **vars(args)}

    # Consolidate shift parameters
    config["shift_configs"] = {
        "structure_shift": {
            "p_ii_factor": args.p_ii_factor,
            "p_ij_factor": args.p_ij_factor,
            "noise_level": args.structure_noise_level,
        },
        "feature_shift": {"noise_level": args.noise_level},
        "label_shift": {
            "ood_class_to_leave_out": args.ood_class_to_leave_out,
            "setting": args.label_setting,
        },
        "text_shift": {
            "augmentation_type": args.text_augmentation_type,
            "noise_level": args.text_noise_level,
            "char_edit_prob": DEFAULT_SETTINGS["text_shift"]["char_edit_prob"],
        },
        "text_swap_shift": {"swap_scope": args.swap_scope, "swap_ratio": args.swap_ratio},
        "semantic_connection_shift": {
            "target_density_factor": args.target_density_factor,
            "selection_mode": args.semantic_selection_mode,
            "threshold_percentile": args.threshold_percentile,
        },
        "arxiv_time_split": {
            "time_bound_train": args.arxiv_time_bound_train,
            "time_bound_test": args.arxiv_time_bound_test,
            "inductive": args.arxiv_inductive,
        },
    }

    # Flatten single-value lists for convenience (except label_shift classes)
    for shift_key, params in config["shift_configs"].items():
        config.setdefault(shift_key, {})
        for k, v in params.items():
            if isinstance(v, list) and len(v) == 1 and shift_key != "label_shift":
                config[shift_key][k] = v[0]
            else:
                config[shift_key][k] = v

    # Paths & flags
    config["dataset_path"] = Path(config["dataset_path"]).expanduser()
    config["embedding_path"] = Path(config["embedding_path"]).expanduser()
    config["ood_evaluation"] = bool(config["shift_type"])
    config["random_seed"] = args.seed

    # Validate datasets
    selected = [ds for ds in args.datasets if ds in ALL_DATASET_NAMES]
    if not selected:
        print(
            f"Error: No valid datasets selected from: {args.datasets}. Available: {ALL_DATASET_NAMES}",
            file=sys.stderr,
        )
        sys.exit(1)
    config["datasets"] = selected

    return selected, config