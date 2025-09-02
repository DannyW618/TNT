import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Default Configuration Settings ---
DEFAULT_SETTINGS = {
    'use_wandb': False,
    'random_seed': 42,
    'verbose': True,
    'runs': 1,
    'project_prefix': "GCN_OOD",

    # --- Dataset Settings ---
    'text_encoder_model': 'all-MiniLM-L6-v2',
    'dataset_path': Path("../datasets"),
    'embedding_path': Path("../embeddings"),
    'cache_embeddings': True,

    # --- Model Architecture ---
    'hidden_channels': 128,
    'dropout': 0.0,
    'use_linear_layer': False,
    'num_layers': 2,

    # --- Training Parameters ---
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'epochs': 300,
    'early_stopping_patience': 100,
    'use_f1_metric': False,

    # --- OOD Detection Parameters (Energy/Propagation based) ---
    'use_prop': True,
    'T': 1.0,
    'K': 3,           # Propagation layers
    'alpha': 0.5,  
    'use_OE': False, 
    'use_tntood': True,

    # --- Baseline Parameters ---
    'baseline_noise': 0.0014,
    'neco_use_scaler': True,
    'neco_n_components': 50,
    'neco_feature_layer': -2,

    # --- Shift-Specific Parameters (Defaults for single value args) ---
    'structure_shift': {
        'p_ii_factor': 0.5,
        'p_ij_factor': 0.2,
        'noise_level': 0.5,
    },
    'feature_shift': {
        'noise_level': 0.9
    },
    'label_shift': {
        'ood_class_to_leave_out': [0],
        'setting': 'inductive'
    },
    'text_shift': {
        'augmentation_type': 'synonym',
        'noise_level': 1.0,
        'char_edit_prob': 1.0
    },
    'text_swap_shift': {
        'swap_scope': 'inter',
        'swap_ratio': 1.0,
    },
    'semantic_connection_shift': {
        'target_density_factor': 1.0,
        'similarity_metric': 'cosine',
        'selection_mode': 'threshold',
        'threshold_percentile': 0.95,
    },
    'arxiv_time_split': {
        'time_bound_train': [2015, 2017],
        'time_bound_test': [2017, 2018, 2019, 2020],
        'inductive': True,
    }
}

# --- Available Datasets ---
ALL_DATASET_NAMES = (
    "cora", "citeseer", "pubmed",
    "elecomp", "elephoto", "dblp", "wikics",
    "bookhis", "bookchild", "arxiv","reddit"
)

# Datasets actively used by default if --datasets is not specified
ACTIVE_DATASETS = ALL_DATASET_NAMES

# Datasets where F1 score is typically preferred over accuracy
F1_DATASETS = {'reddit'}

# --- Constants ---
SHIFTS_REQUIRING_RETRAINING = {'label', 'arxiv_time'}
ALL_SHIFT_TYPES = ['structure', 'feature', 'label', 'text', 'text_swap', 'semantic_connection', 'arxiv_time']
ALL_BASELINE_METHODS = ['msp', 'odin', 'mahalanobis', 'neco', 'energy', 'gnnsafe', 'nodesafe']