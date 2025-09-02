import gc
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data

try:
    import wandb
    WandbRun = wandb.sdk.wandb_run.Run
    os.environ.setdefault("WANDB_SILENT", "true")
except Exception:
    wandb = None
    WandbRun = Any

from baselines import run_baseline_evaluation
from config import (
    F1_DATASETS,
    SHIFTS_REQUIRING_RETRAINING,
)
from argparser import parse_args
from data_processing import DatasetProcessor, TextEncoder, load_dataset
from tntood import TNTOODDetector, TNTOODLoss, TNTOODModel
from evaluation import OODDetector
from experiment_utils import (
    NumpyEncoder,
    init_wandb,
    save_config,
    save_results,
)
from metrics import get_ood_metrics
from models import GCN
from ood_generation import create_ood_datasets
from training import Trainer

# Logging utilities
def setup_logger(log_path: Path, verbose: bool) -> None:
    """Configure root logger to write to both console and file without
    overriding sys.stdout.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

# Helpers
def generate_shift_configs(config: Dict[str, Any], shift_type: str) -> List[Dict[str, Any]]:
    """Expand a shift config into a list of concrete configs.

    - For 'label' and 'arxiv_time', return the single dict.
    - For others, if any value is a list, cycle values to produce multiple
      configurations of max list-length.
    """
    base = config.get("shift_configs", {})

    if shift_type == "arxiv_time":
        return [base.get("arxiv_time_split", {})]
    if shift_type == "label":
        return [base.get("label_shift", {})]

    specific = base.get(f"{shift_type}_shift", {})
    list_params = {k: v for k, v in specific.items() if isinstance(v, list)}
    if not list_params:
        cfg = dict(specific)
        if shift_type == "semantic_connection":
            cfg.setdefault("similarity_metric", "cosine")
        return [cfg]

    max_len = max(len(v) for v in list_params.values())
    out: List[Dict[str, Any]] = []
    for i in range(max_len):
        cfg = dict(specific)
        for k, vals in list_params.items():
            cfg[k] = vals[i % len(vals)]
        if shift_type == "semantic_connection":
            cfg.setdefault("similarity_metric", "cosine")
        out.append(cfg)
    return out


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Setup per-run
def setup_experiment(dataset_name: str, config: Dict[str, Any], run_index: int) -> Tuple[Path, Optional[WandbRun], Dict[str, Any]]:
    seed_everything(config["random_seed"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shift_str = "_".join(config["shift_type"]) if config["shift_type"] else "ID"
    baseline_str = "_".join(config["use_baseline"]) if config["use_baseline"] else ""

    model_mode = "TNTOOD" if config.get("use_tntood") else "GCN"
    parts = [model_mode]
    if baseline_str:
        parts.append(baseline_str)
    if shift_str != "ID":
        parts.append(shift_str)

    suffix = "_".join(parts)
    name = f"{dataset_name}_{suffix}_run{run_index}_seed{config['random_seed']}_{timestamp}"

    if "label" in config["shift_type"] and "label_shift" in config:
        ood_cls = config["label_shift"].get("ood_class_to_leave_out", [])
        if ood_cls:
            name = f"{dataset_name}_{suffix}_oodcls_{'_'.join(map(str, ood_cls))}_run{run_index}_seed{config['random_seed']}_{timestamp}"

    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    exp_dir = Path("results") / safe_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    setup_logger(exp_dir / "output.log", verbose=bool(config.get("verbose", True)))
    logging.info("Experiment directory: %s", exp_dir)

    # Save config snapshot
    logging.info("Configuration:")
    for k, v in sorted(config.items()):
        logging.info("  %s: %s", k, v)
    save_config(config, exp_dir)

    # WandB
    run = init_wandb(config, dataset_name, config["project_prefix"], config["random_seed"], config["shift_type"])  # type: ignore[arg-type]
    config["use_wandb"] = bool(run is not None)

    return exp_dir, run, config

# Data
def load_and_prepare_data(dataset_name: str, config: Dict[str, Any]) -> Optional[Tuple[Data, int, Dict[str, Any], Optional[TextEncoder]]]:
    """Load a dataset and (optionally) encode text features."""
    verbose = bool(config.get("verbose", True))
    text_encoder: Optional[TextEncoder] = None

    try:
        logging.info("\n--- Loading and Preparing Dataset: %s ---", dataset_name)
        dataset = load_dataset(dataset_name, config["dataset_path"], verbose=verbose)
        dataset.name = dataset_name

        processor = DatasetProcessor()
        num_classes, dataset_info = processor.prepare_dataset(dataset, verbose=verbose)

        enc_name = config.get("text_encoder_model")
        if enc_name:
            try:
                text_encoder = TextEncoder(model_name=enc_name, verbose=verbose, random_seed=config["random_seed"])  # type: ignore[arg-type]
                dataset_info["text_encoder_model"] = enc_name
            except Exception as e:
                logging.warning("Failed to initialize TextEncoder (%s). Text shifts may fail.", e)
                text_encoder = None

        needs_encoding = hasattr(dataset, "raw_texts")
        if needs_encoding:
            if text_encoder is None:
                raise ValueError("Dataset requires text encoding, but TextEncoder failed to initialize.")
            logging.info("Encoding text features...")
            text_embeddings = text_encoder.encode_texts(
                texts=dataset.raw_texts,
                dataset_name=dataset_name,
                embedding_base_path=config["embedding_path"],
                use_cache=bool(config.get("cache_embeddings", False)),
            )
            dataset.x = text_embeddings
            dataset_info["embedding_dim"] = int(text_embeddings.shape[1]) if text_embeddings.numel() > 0 else 0
            dataset_info["features_source"] = "Encoded Text"
        elif hasattr(dataset, "x") and dataset.x is not None:
            logging.info("Using pre-existing 'x' features.")
            dataset_info["embedding_dim"] = int(dataset.x.shape[1])
            dataset_info["features_source"] = "Pre-existing"
        else:
            raise ValueError("Dataset has neither 'raw_texts' nor usable 'x' features.")

        if config.get("use_wandb") and wandb and wandb.run:
            wandb.run.summary.setdefault("dataset_info", dataset_info)

        return dataset, num_classes, dataset_info, text_encoder

    except FileNotFoundError as e:
        logging.error("%s", e)
        return None
    except Exception as e:
        logging.error("Error loading/preparing dataset %s: %s", dataset_name, e)
        logging.error("%s", traceback_str())
        return None


# Model setup
def setup_model_and_components(num_node_features: int, num_classes: int, config: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, optim.Optimizer]:
    """Create model, criterion, optimizer."""
    logging.info("\n--- Setting up Model, Criterion, and Optimizer ---")

    if config.get("use_tntood", False):
        logging.info("Using TNTOODModel and TNTOODLoss")
        model = TNTOODModel(
            node_feature_dim=num_node_features,
            gnn_hidden_dim=config["hidden_channels"],
            dropout=config["dropout"],
            projection_dim=config["hidden_channels"],
            num_id_classes=num_classes,
            device=device,
        ).to(device)
        criterion = TNTOODLoss(
            temperature=config["tnt_loss_temp"],
            contrastive_weight=config["contrast_w"],
            id_loss_weight=config["tnt_loss_id_weight"],
            use_batch=config["use_batch_contrastive"],
            batch_size=config["contrastive_batch_size"],
        )
    else:
        logging.info("Using Standard GCN Model and CrossEntropyLoss")
        model = GCN(
            num_node_features=num_node_features,
            num_hidden=config["hidden_channels"],
            num_classes=num_classes,
            dropout=config["dropout"],
            use_linear=config.get("use_linear_layer", False),
            num_layers=config["num_layers"],
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])  # type: ignore[arg-type]

    logging.info("%s", model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Trainable Parameters: %s", f"{total_params:,}")

    if config.get("use_wandb") and wandb and wandb.run:
        try:
            wandb.watch(model, criterion=criterion, log="all", log_freq=100)  # type: ignore
            wandb.run.summary["total_params"] = total_params  # type: ignore[index]
        except Exception as e:  # pragma: no cover
            logging.warning("Wandb watch failed: %s", e)

    return model, criterion, optimizer

# Training
def train_model(training_graph: Data, num_classes_train: int, config: Dict[str, Any], device: torch.device, exp_dir: Path) -> Optional[Trainer]:
    try:
        training_graph = training_graph.to(device)
        model, criterion, optimizer = setup_model_and_components(training_graph.num_features, num_classes_train, config, device)
        trainer = Trainer(model, training_graph, optimizer, criterion, device, config)
        trainer.train(exp_dir)
        logging.info("--- Model Training Finished ---")
        return trainer
    except Exception as e:
        logging.error("Training error: %s", e)
        logging.error("%s", traceback_str())
        return None


# Evaluation
def perform_ood_evaluation(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    shift_config_name: str,
    ind_data_for_eval: Data,
    ood_datasets_generated: List[Data],
    experiment_dir: Path,
    original_dataset_name: str,
    training_data_for_fitting: Optional[Data] = None,
) -> Dict[str, Any]:
    """Run OOD evaluation (Baselines, TNTOOD, or Default GCN)."""
    verbose = bool(config.get("verbose", True))
    use_baselines = bool(config.get("use_baseline", []))
    use_tntood_model = bool(config.get("use_tntood", False))

    eval_results: Dict[str, Any] = {}

    if use_baselines:
        logging.info("\n--- OOD Eval (%s) | Mode: Baselines %s ---", shift_config_name.upper(), config.get("use_baseline"))
        model.to(device)
        ind_cpu = ind_data_for_eval.cpu()
        ood_cpu = [d.cpu() for d in ood_datasets_generated]
        config["shift_config_name"] = shift_config_name
        baseline_results = run_baseline_evaluation(model, ind_cpu, ood_cpu, config.get("use_baseline", []), config, device)
        eval_results = {"baselines": baseline_results}
        return eval_results

    # --- TNTOOD Detector ---
    if use_tntood_model:
        logging.info("\n--- OOD Eval (%s) | Mode: TNTOOD Detector ---", shift_config_name.upper())
        if not isinstance(model, TNTOODModel):
            logging.error("use_tntood=True but model is %s", type(model).__name__)
            return {"status": "Model Type Mismatch"}
        if training_data_for_fitting is None:
            logging.error("Missing training_data_for_fitting for TNTOODDetector.fit")
            return {"status": "Missing Training Data for Fit"}

        try:
            tnt_detector = TNTOODDetector(
                model,
                {
                    "tnt_verbose": config.get("tnt_verbose", verbose),
                    "align_w": config["align_w"],
                    "tnt_ood_w_id_uncertainty": config["tnt_ood_w_id_uncertainty"],
                    "K": config["K"],
                    "alpha": config["alpha"],
                    "tnt_ood_eps": config["tnt_ood_eps"],
                },
            )

            ind_dev = ind_data_for_eval.clone().to(device)
            if not getattr(ind_dev, "test_mask", None) is not None or int(ind_dev.test_mask.sum()) == 0:
                logging.error("ID data lacks a valid 'test_mask'.")
                return {"status": "Missing ID Test Mask"}

            t0 = time.time()
            scores_id_all = tnt_detector.compute_scores(ind_dev)
            scores_id_test = scores_id_all[ind_dev.test_mask]
            logging.info("Calculated TNTOOD scores for %d ID test nodes.", int(scores_id_test.numel()))

            all_ood_results: List[Dict[str, float]] = []
            collected_ood_scores: List[np.ndarray] = []

            if not ood_datasets_generated:
                logging.warning("No OOD datasets provided for TNTOOD evaluation.")
                eval_results = {"tntood": {"average": {"auroc": 0.0, "aupr": 0.0, "fpr95": 1.0}, "datasets": []}}
            else:
                for i, ood_cpu in enumerate(ood_datasets_generated):
                    logging.info("Processing OOD dataset %d/%d...", i + 1, len(ood_datasets_generated))
                    ood_dev = ood_cpu.to(device)
                    target_idx = getattr(ood_dev, "node_idx", None)
                    if target_idx is None or int(target_idx.numel()) == 0:
                        logging.warning("  OOD dataset %d missing 'node_idx'. Using all nodes.", i + 1)
                        target_idx = torch.arange(ood_dev.num_nodes, device=device)

                    scores_ood_all = tnt_detector.compute_scores(ood_dev)
                    valid = target_idx < scores_ood_all.shape[0]
                    valid_idx = target_idx[valid]
                    if int(valid_idx.numel()) == 0:
                        logging.warning("  No valid target OOD node indices for dataset %d. Skipping.", i + 1)
                        continue

                    scores_ood = scores_ood_all[valid_idx]
                    id_np = np.nan_to_num(scores_id_test.detach().cpu().numpy())
                    ood_np = np.nan_to_num(scores_ood.detach().cpu().numpy())
                    collected_ood_scores.append(ood_np)

                    auroc, aupr, fpr95 = get_ood_metrics(id_np, ood_np)
                    all_ood_results.append({"dataset_index": i, "auroc": auroc, "aupr": aupr, "fpr95": fpr95})

                    logging.info("OOD[%d] AUROC=%.2f AUPR=%.2f FPR95=%.2f", i, auroc * 100, aupr * 100, fpr95 * 100)

                    if config.get("use_wandb") and wandb and wandb.run:
                        wandb.log({f"ood/{shift_config_name}/ds_{i}/tnt_auroc": auroc, f"ood/{shift_config_name}/ds_{i}/tnt_aupr": aupr, f"ood/{shift_config_name}/ds_{i}/tnt_fpr95": fpr95}, commit=False)  # type: ignore

                avg = {"auroc": 0.0, "aupr": 0.0, "fpr95": 1.0}
                if all_ood_results:
                    avg = {k: float(np.mean([r[k] for r in all_ood_results])) for k in ("auroc", "aupr", "fpr95")}
                    logging.info("Average OOD: AUROC=%.2f AUPR=%.2f FPR95=%.2f", avg["auroc"] * 100, avg["aupr"] * 100, avg["fpr95"] * 100)

                eval_results = {"tntood": {"average": avg, "datasets": all_ood_results}}

                if config.get("use_wandb") and wandb and wandb.run:
                    wandb.run.summary.update({f"ood/{shift_config_name}/tnt_avg_{k}": v for k, v in avg.items()})  # type: ignore[index]
                    wandb.log({f"ood/{shift_config_name}/tnt_avg/{k}": v for k, v in avg.items()}, commit=True)  # type: ignore

            logging.info("Time taken for OOD evaluation: %.2fs", time.time() - t0)
            return eval_results

        except Exception as e:
            logging.error("TNTOOD evaluation error for %s: %s", shift_config_name, e)
            logging.error("%s", traceback_str())
            return {"status": "Error TNTOOD", "error_message": str(e)}
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    logging.info("\n--- OOD Eval (%s) | Mode: Default Detector (Energy for GCN) ---", shift_config_name.upper())
    try:
        det = OODDetector(model, config)
        id_clone = ind_data_for_eval.clone()
        ood_clone = [d.clone() for d in ood_datasets_generated]
        std_res = det.evaluate_ood(id_clone, ood_clone, shift_name=shift_config_name)
        return std_res or {"status": "Std Eval skipped/failed"}
    except Exception as e:
        logging.error("Default detector error: %s", e)
        logging.error("%s", traceback_str())
        return {"status": "Std Detector Error", "error_message": str(e)}

# Run one experiment
def run_experiment(dataset_name: str, config: Dict[str, Any], run_index: int) -> Dict[str, Any]:
    exp_dir, wandb_run, config = setup_experiment(dataset_name, config, run_index)

    mode_parts = ["TNTOOD_Model" if config.get("use_tntood", False) else "Standard_GCN"]
    if config.get("use_baseline"):
        mode_parts.append(f"Baselines_{'_'.join(config['use_baseline'])}")
    elif config.get("shift_type"):
        mode_parts.append(
            f"Eval_{'TNTOOD_Detector' if config.get('use_tntood') else 'Default_Energy_Detector'}_Shifts_{'_'.join(config['shift_type'])}"
        )
    else:
        mode_parts.append("ID_Training_Only")
    mode_str = " | ".join(mode_parts)

    logging.info("%s", "=" * 60)
    logging.info(
        "Run %d/%d | Dataset: %s | Seed: %d | %s",
        run_index + 1,
        config["runs"],
        dataset_name,
        config["random_seed"],
        mode_str,
    )
    logging.info("%s", "=" * 60)

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "run_index": run_index,
        "seed": config["random_seed"],
        "shift_type": config.get("shift_type", []),
        "use_baseline": config.get("use_baseline", []),
        "use_tntood": bool(config.get("use_tntood", False)),
        "config": json.loads(json.dumps(config, cls=NumpyEncoder, sort_keys=True)),
        "status": "started",
        "training_history": {},
        "ood_evaluations": {},
        "final_accuracies": {},
    }
    final_status = "Unknown Error"

    model: Optional[torch.nn.Module] = None
    trainer: Optional[Trainer] = None
    original_dataset: Optional[Data] = None
    device: Optional[torch.device] = None
    text_encoder: Optional[TextEncoder] = None
    training_graph: Optional[Data] = None
    ind_data_for_eval: Optional[Data] = None

    try:
        t0 = time.time()
        load_res = load_and_prepare_data(dataset_name, config)
        if load_res is None:
            raise RuntimeError(f"Failed to load/prep {dataset_name}.")
        original_dataset, num_classes_orig, _, text_encoder = load_res

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Data loaded & prepared in %.2fs. Device: %s", time.time() - t0, device)
        if config.get("use_wandb") and wandb_run:
            wandb_run.summary["device"] = str(device)

        num_classes_train = num_classes_orig
        active_shifts = config.get("shift_type", [])
        needs_retraining = any(s in SHIFTS_REQUIRING_RETRAINING for s in active_shifts)

        ood_from_train_shift: List[Data] = []
        train_shift_type: Optional[str] = None

        if needs_retraining:
            if len(active_shifts) > 1:
                logging.warning("Multiple shifts require retraining. Using the first for training data.")
            train_shift_type = next(s for s in active_shifts if s in SHIFTS_REQUIRING_RETRAINING)

            logging.info("\n--- Generating Pre-Training Data (%s) ---", train_shift_type.upper())
            t_gen = time.time()

            shift_cfgs = generate_shift_configs(config, train_shift_type)
            if len(shift_cfgs) > 1:
                logging.warning("Multiple configs for training shift '%s'. Using the first.", train_shift_type)
            current_train_shift = shift_cfgs[0]

            tmp_cfg = dict(config)
            key = "arxiv_time_split" if train_shift_type == "arxiv_time" else f"{train_shift_type}_shift"
            tmp_cfg[key] = current_train_shift

            dataset_ind_gen, _, ood_from_train_shift = create_ood_datasets(
                original_dataset.cpu(), train_shift_type, tmp_cfg, config["random_seed"], text_encoder
            )
            logging.info("Pre-training data generation took %.2fs", time.time() - t_gen)

            training_graph = dataset_ind_gen
            ind_data_for_eval = dataset_ind_gen
            if hasattr(training_graph, "y") and training_graph.y is not None:
                num_classes_train = int(torch.unique(training_graph.y.squeeze()).numel())
            else:
                num_classes_train = num_classes_orig

            logging.info(
                "Training on generated '%s' (%d classes) | Nodes=%d Edges=%d Feats=%d",
                getattr(training_graph, "name", "dataset"),
                num_classes_train,
                training_graph.num_nodes,
                training_graph.num_edges,
                training_graph.num_features,
            )
        else:
            training_graph = original_dataset
            ind_data_for_eval = original_dataset
            logging.info("Training on original '%s' (%d classes)", dataset_name, num_classes_train)

        logging.info("Training graph: %s", training_graph)

        trainer = train_model(training_graph, num_classes_train, config, device, exp_dir)
        if trainer is None:
            raise RuntimeError("Model training failed.")
        model = trainer.model
        results["training_history"] = trainer.results
        results["final_accuracies"] = trainer.final_evaluate()

        # --- OOD evaluation across shifts ---
        if active_shifts:
            all_shift_results: Dict[str, Any] = {}
            for eval_shift in active_shifts:
                shift_cfgs = generate_shift_configs(config, eval_shift)
                logging.info("\n--- Evaluating Shift: %s (%d configs) ---", eval_shift.upper(), len(shift_cfgs))

                for i_cfg, cfg_instance in enumerate(shift_cfgs):
                    desc = eval_shift if len(shift_cfgs) == 1 else f"{eval_shift}_config_{i_cfg + 1}"
                    logging.info("Processing %s...", desc)
                    if len(shift_cfgs) > 1:
                        logging.info("  Params: %s", cfg_instance)

                    tmp_cfg = dict(config)
                    key = "arxiv_time_split" if eval_shift == "arxiv_time" else f"{eval_shift}_shift"
                    tmp_cfg[key] = cfg_instance

                    ood_eval_sets: List[Data] = []
                    use_cached_from_train = needs_retraining and eval_shift == train_shift_type and i_cfg == 0
                    if use_cached_from_train:
                        ood_eval_sets = ood_from_train_shift
                        logging.info("  Using OOD data generated during training shift '%s'.", train_shift_type)
                    else:
                        t_gen = time.time()
                        logging.info("  Generating OOD Data for Eval (%s)...", desc.upper())
                        _, _, ood_eval_sets = create_ood_datasets(original_dataset.cpu(), eval_shift, tmp_cfg, config["random_seed"], text_encoder)
                        logging.info("  OOD generation took %.2fs", time.time() - t_gen)

                    # Perform evaluation
                    eval_result = perform_ood_evaluation(
                        model=model,
                        config=tmp_cfg,
                        device=device,
                        shift_config_name=desc,
                        ind_data_for_eval=ind_data_for_eval.cpu(),
                        ood_datasets_generated=ood_eval_sets,
                        experiment_dir=exp_dir,
                        original_dataset_name=original_dataset.name,
                        training_data_for_fitting=training_graph if config.get("use_tntood", False) else None,
                    )
                    all_shift_results[desc] = eval_result

                    del ood_eval_sets
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            results["ood_evaluations"] = all_shift_results
        else:
            logging.info("\nINFO: No OOD shifts specified. Only ID training performed.")
            results["ood_evaluations"] = {"status": "No OOD evaluation performed."}

        final_status = "Completed"

    except Exception as e:
        final_status = "Error"
        msg = f"Run {run_index + 1}, {dataset_name} (Shifts: {config.get('shift_type')}, TNTOOD: {config.get('use_tntood')}, Baselines: {config.get('use_baseline')}): {e}"
        logging.error("\nFATAL ERROR: %s", msg)
        logging.error("%s", traceback_str())
        results["error"] = msg
        if config.get("use_wandb", False) and wandb_run:
            try:
                wandb_run.summary["error_message"] = msg
            except Exception:
                pass

    finally:
        logging.info("\n--- Cleaning up run %d for %s ---", run_index + 1, dataset_name)
        del model, trainer, original_dataset, text_encoder, training_graph, ind_data_for_eval
        gc.collect()
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()

        results["status"] = final_status
        results_path = save_results(results, exp_dir)
        logging.info("\nFinal Status Run %d (%s): %s. Results: %s", run_index + 1, dataset_name, final_status, results_path)

        if config.get("use_wandb", False) and wandb_run is not None:
            try:
                name_raw = results_path.stem
                safe = "".join(c if c.isalnum() else "_" for c in name_raw)[:128]
                artifact = wandb.Artifact(safe, type="results") 
                artifact.add_file(str(results_path))
                wandb.log_artifact(artifact)
                wandb_run.summary["final_status"] = final_status
                wandb.finish(exit_code=0 if final_status == "Completed" else 1)
                logging.info("WandB run finished.")
            except Exception as we:
                logging.error("WandB finalization error: %s", we)
                if wandb and wandb.run is not None:
                    try:
                        wandb.finish(exit_code=1, quiet=True)
                    except Exception:
                        pass
        logging.info("-" * 60)

    return results

def calculate_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = float(sum(values)) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return float(var ** 0.5)


def traceback_str() -> str:
    import traceback as _tb

    return _tb.format_exc()

def main() -> None:
    selected_datasets, config = parse_args()
    verbose = bool(config.get("verbose", True))
    num_runs = int(config.get("runs", 1))
    base_seed = int(config.get("random_seed"))

    if not config.get("use_wandb", False) and wandb is not None:
        os.environ["WANDB_MODE"] = "disabled"
        if verbose:
            print("INFO: WandB disabled via config. WANDB_MODE=disabled.")

    all_results: List[Dict[str, Any]] = []

    for run_idx in range(num_runs):
        current_seed = base_seed + run_idx
        run_config = json.loads(json.dumps(config, cls=NumpyEncoder, sort_keys=True))
        run_config["random_seed"] = current_seed

        for ds in selected_datasets:
            ds_cfg = dict(run_config)
            ds_cfg["dataset"] = ds
            ds_cfg["use_f1_metric"] = ds.lower() in F1_DATASETS

            res = run_experiment(ds, ds_cfg, run_idx)
            all_results.append(res)

            if verbose:
                logging.info("--- Main: Finished Dataset %s | Run %d ---", ds, run_idx + 1)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    test_accs: List[float] = []
    metrics_by_cfg: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for res in all_results:
        try:
            acc = float(res.get("final_accuracies", {}).get("test_acc", 0.0))
            test_accs.append(acc)
        except Exception:
            pass

        ood_eval = res.get("ood_evaluations", {})
        if not isinstance(ood_eval, dict):
            continue

        for cfg_name, cfg_data in ood_eval.items():
            if cfg_name not in metrics_by_cfg:
                metrics_by_cfg[cfg_name] = {}

            if isinstance(cfg_data, dict) and "baselines" in cfg_data:
                for bname, bdata in cfg_data["baselines"].items():
                    metrics_by_cfg[cfg_name].setdefault(bname, {})
                    try:
                        # list-of-datasets path
                        for i, ds_metrics in enumerate(bdata.get("datasets", [])):
                            key = str(i)
                            metrics_by_cfg[cfg_name][bname].setdefault(key, {"auroc": [], "aupr": [], "fpr95": []})
                            for m in ("auroc", "aupr", "fpr95"):
                                if m in ds_metrics:
                                    metrics_by_cfg[cfg_name][bname][key][m].append(float(ds_metrics[m]))
                    except Exception:
                        # single-dataset fallback
                        key = "0"
                        metrics_by_cfg[cfg_name][bname].setdefault(key, {"auroc": [], "aupr": [], "fpr95": []})
                        for m in ("auroc", "aupr", "fpr95"):
                            if m in bdata:
                                metrics_by_cfg[cfg_name][bname][key][m].append(float(bdata[m]))

            elif isinstance(cfg_data, dict) and "tntood" in cfg_data:
                bname = "TNTOOD"
                metrics_by_cfg[cfg_name].setdefault(bname, {})
                key = "0"
                metrics_by_cfg[cfg_name][bname].setdefault(key, {"auroc": [], "aupr": [], "fpr95": []})
                ds0 = None
                try:
                    ds0 = next((d for d in cfg_data["tntood"].get("datasets", []) if d.get("dataset_index") == 0), None)
                except Exception:
                    ds0 = None
                if ds0:
                    for m in ("auroc", "aupr", "fpr95"):
                        if m in ds0:
                            metrics_by_cfg[cfg_name][bname][key][m].append(float(ds0[m]))

    # === Print summary ===
    if test_accs:
        avg_acc = sum(test_accs) / len(test_accs)
        std_acc = calculate_std(test_accs)
        print("\n=== Average Metrics Across All Runs ===")
        print(f"Average Test Accuracy: {avg_acc*100:.2f} ± {std_acc*100:.2f}")

    all_avg_saved: Dict[str, List[float]] = {}
    all_std_saved: Dict[str, List[float]] = {}

    for cfg_name, baselines in metrics_by_cfg.items():
        print(f"\nConfiguration: {cfg_name}")
        print(f"{'Baseline':<15} {'Dataset':<8} {'AUROC':<10} {'AUPR':<10} {'FPR@95':<10}")
        print("-" * 60)
        for bname, ds_map in baselines.items():
            for ds_id, metric_lists in ds_map.items():
                auroc_vals = metric_lists.get("auroc", [])
                aupr_vals = metric_lists.get("aupr", [])
                fpr_vals = metric_lists.get("fpr95", [])
                if not (auroc_vals and aupr_vals and fpr_vals):
                    continue
                avg_auroc = sum(auroc_vals) / len(auroc_vals)
                avg_aupr = sum(aupr_vals) / len(aupr_vals)
                avg_fpr = sum(fpr_vals) / len(fpr_vals)
                std_auroc = calculate_std(auroc_vals)
                std_aupr = calculate_std(aupr_vals)
                std_fpr = calculate_std(fpr_vals)

                key = f"{cfg_name}_{bname}_{ds_id}"
                all_avg_saved[key] = [avg_auroc, avg_aupr, avg_fpr]
                all_std_saved[key] = [std_auroc, std_aupr, std_fpr]

                print(f"{bname:<15} {ds_id:<8} {avg_auroc*100:.2f}     {avg_aupr*100:.2f}     {avg_fpr*100:.2f}")
                print(f"{'':<15} {'':<8} ±{std_auroc*100:.2f}    ±{std_aupr*100:.2f}    ±{std_fpr*100:.2f}")

    if all_avg_saved:
        print("\n--- All Avg Values ---")
        print(" ".join(f"{v*100:.2f}" for vals in all_avg_saved.values() for v in vals))
        print("--- All Std Values ---")
        print(" ".join(f"{v*100:.2f}" for vals in all_std_saved.values() for v in vals))

    if verbose:
        print(f"\n{'='*60}\nAll {num_runs} runs for selected datasets processed.\n{'='*60}")


if __name__ == "__main__":
    main()
