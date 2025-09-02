import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
from torch_geometric.data import Data

try:
    import wandb
    WandbRun = Optional[wandb.sdk.wandb_run.Run]
except ImportError:
    wandb = None
    WandbRun = Optional[Any]

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist() # Convert arrays to lists
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, torch.device):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def save_json(data: Dict, file_path: Path, verbose: bool = True):
    """Saves a dictionary to a JSON file with numpy handling."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        if verbose: print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
    return file_path

def save_config(config: Dict[str, Any], experiment_dir: Path) -> Path:
    """Saves the experiment configuration."""
    return save_json(config, experiment_dir / 'config.json', config.get('verbose', True))

def save_results(results: Dict[str, Any], experiment_dir: Path) -> Path:
    """Saves the experiment results."""
    return save_json(results, experiment_dir / 'results.json', True)

def init_wandb(config: Dict[str, Any], dataset_name: str,
               project_prefix: str = "GCN_OOD", run_seed: Optional[int] = None,
               shift_type: Optional[List[str]] = None) -> WandbRun:
    """Initializes a Weights & Biases run if enabled."""
    if not config.get('use_wandb', False):
        return None
    if wandb is None:
        print("Warning: wandb library not installed. Cannot initialize run.")
        return None

    verbose = config.get('verbose', True)

    try:
        timestamp = datetime.now().strftime('%y%m%d_%H%M')
        seed_suffix = f"_s{run_seed}" if run_seed is not None else ""

        model_params = f"L{config['num_layers']}_H{config['hidden_channels']}"
        opt_params = f"LR{config['learning_rate']}_D{config['dropout']}"
        base_name = f"{dataset_name}_{model_params}_{opt_params}"
        if config.get('use_linear_layer', False): base_name += "_Lin"

        run_label = "ID"
        tags = [dataset_name, "GCN", f"Seed_{run_seed}" if run_seed is not None else "NoSeed"]
        if shift_type:
            if len(shift_type) == 1:
                 run_label = shift_type[0]
                 tags.append(f"Shift_{run_label}")
                 if run_label == 'label':
                     labels = config.get('label_shift',{}).get('ood_class_to_leave_out', [])
                     run_label += f"_Rem{'_'.join(map(str, labels))}" # e.g., label_Rem0_1
                 elif run_label == 'arxiv_time':
                      run_label += f"_Ind{config.get('arxiv_time_split',{}).get('inductive', True)}"
            else:
                 run_label = "MultiShift"
                 tags.append("MultiShift_Eval")
                 tags.extend([f"Eval_{s}" for s in shift_type[:4]])
                 if len(shift_type) > 4: tags.append("Eval_Multiple")
        else:
            tags.append("ID_Training")

        run_name = f"{run_label}_{base_name}{seed_suffix}_{timestamp}"
        # Sanitize run name for wandb
        run_name = "".join(c if c.isalnum() else "_" for c in run_name)

        if verbose: print(f"Wandb Run Name: {run_name}")

        wandb_config = json.loads(json.dumps(config, cls=NumpyEncoder))

        run = wandb.init(
            project=f"{project_prefix}_{dataset_name}",
            config=wandb_config,
            name=run_name,
            tags=tags,
            reinit=True, # Allow reinitializing in the same process (e.g., for multiple runs)
            # settings=wandb.Settings(start_method="fork") # Optional: might help in some environments
        )
        if run:
            print(f"Wandb run '{run.name}' initialized (Project: {run.project}, ID: {run.id}).")
        return run

    except Exception as e:
        print(f"Error initializing Wandb: {e}. Disabling Wandb for this run.")
        config['use_wandb'] = False # Disable for the current run
        return None

def print_config(config: Dict[str, Any]):
    """Prints the configuration dictionary selectively."""
    print("\n--- Experiment Configuration ---")
    exclude_keys = {
        'verbose', 'use_wandb', 'visualize', 'vis_dir', 'dataset_path',
        'embedding_path', 'cache_embeddings', 'random_seed', 'ood_evaluation',
        'use_f1_metric', 'project_prefix', 'config', 'shift_configs', 'device',
        'm_in', 'm_out', 'lamda', 'a_in', 'a_out', 'use_reg',
        'num_layers', 'hidden_channels', 'learning_rate', 'weight_decay', 'dropout', 'use_linear_layer',
        'epochs', 'early_stopping_patience'
    }
    always_print = {'datasets', 'runs', 'shift_type', 'use_baseline', 'T', 'K', 'alpha', 'use_OE'}

    for key, value in config.items():
        if key in exclude_keys and key not in always_print:
            continue
        if isinstance(value, dict) and ("_shift" in key or "_split" in key):
            active_shifts = config.get('shift_type', [])
            is_active_shift_config = False
            if key == 'label_shift' and 'label' in active_shifts: is_active_shift_config = True
            elif key == 'arxiv_time_split' and 'arxiv_time' in active_shifts: is_active_shift_config = True
            elif key == 'structure_shift' and 'structure' in active_shifts: is_active_shift_config = True
            elif key == 'feature_shift' and 'feature' in active_shifts: is_active_shift_config = True
            elif key == 'text_shift' and 'text' in active_shifts: is_active_shift_config = True
            elif key == 'text_swap_shift' and 'text_swap' in active_shifts: is_active_shift_config = True
            elif key == 'semantic_connection_shift' and 'semantic_connection' in active_shifts: is_active_shift_config = True

            if is_active_shift_config:
                print(f"{key}:")
                if isinstance(value, dict) and any(isinstance(v, list) for v in value.values()):
                     print(f"  (Multiple configurations specified via args)")
                     for subkey, subvalue in value.items():
                         print(f"    {subkey}: {subvalue}")
                elif isinstance(value, dict):
                     for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    print("------------------------------")


def print_dataset_info(dataset: Optional[Any], name: str = "Dataset"):
    """Prints key information about a PyG Data object."""
    print(f"\n--- {name} Information ---")
    if dataset is None:
        print("  Dataset is None.")
        return

    if not isinstance(dataset, Data):
         print(f"  Object is not a PyG Data object (type: {type(dataset)}).")
         return

    try:
        info = {
            "Name": getattr(dataset, 'name', 'N/A'),
            "Nodes": getattr(dataset, 'num_nodes', 'N/A'),
            "Edges": getattr(dataset, 'num_edges', 'N/A'),
            "Features": getattr(dataset, 'num_features', 'N/A'),
            "Classes": getattr(dataset, 'num_classes', 'N/A'),
            "Isolated Nodes": dataset.has_isolated_nodes() if hasattr(dataset, 'has_isolated_nodes') else 'N/A',
            "Self-loops": dataset.has_self_loops() if hasattr(dataset, 'has_self_loops') else 'N/A',
            "Undirected": dataset.is_undirected() if hasattr(dataset, 'is_undirected') else 'N/A',
        }

        for key, val in info.items():
            print(f"  {key}: {val}")

        # Print mask counts
        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(dataset, mask_key) and dataset[mask_key] is not None:
                count = dataset[mask_key].sum().item()
                print(f"  {mask_key.replace('_', ' ').title()} Nodes: {count}")

        # Print node_idx count if present
        if hasattr(dataset, 'node_idx') and dataset.node_idx is not None:
            print(f"  Target Node Indices ('node_idx'): {len(dataset.node_idx)}")

    except Exception as e:
        print(f"  Error retrieving some dataset info: {e}")
    print("---------------------------")