"""
TextTopoOOD Framework

This script provides a suite of functions to generate OOD
datasets from a given PyTorch Geometric `Data` object. It supports various types
of distributional shifts, including structure, feature, label, and several
text-based augmentations.

Key Features:
- Structure Shift: Modifies graph topology using a Stochastic Block Model (SBM).
- Feature Shift: Introduces noise by interpolating node features.
- Label Shift: Creates splits by holding out one or more classes.
- Text Shifts: Augments node-level text using synonyms, antonyms, character edits,
  and feature swapping.
- Semantic Connection Shift: Rewires the graph based on node feature similarity.
- ArXiv Time Shift: Splits the ogbn-arxiv dataset based on publication year.
- Caching: Automatically saves and loads generated datasets to avoid re-computation.
"""

import gc
import os
import random
import string
import warnings
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import homophily, stochastic_blockmodel_graph, subgraph, to_undirected
from tqdm import tqdm

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not found. Text augmentation features will be disabled.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "Scikit-learn not found. 'semantic_connection' shift will be disabled."
    )

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# --- Local Imports ---
class TextEncoder:
    def encode_texts(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("TextEncoder is not implemented.")

# --- Global Caches for Text Augmentation ---
synonym_cache: Dict[str, List[str]] = {}
antonym_cache: Dict[str, List[str]] = {}


# --- NLTK & Text Augmentation Utilities ---
def ensure_nltk_resources():
    """Downloads required NLTK data if it's not already present."""
    if not NLTK_AVAILABLE:
        return

    required_resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("tokenizers/punkt", "punkt"),
    ]
    try:
        for path, _ in required_resources:
            nltk.data.find(path)
    except LookupError:
        print("Downloading required NLTK data...")
        for _, pkg_id in required_resources:
            try:
                nltk.download(pkg_id, quiet=True)
            except Exception as e:
                print(f"  Failed to download {pkg_id}: {e}. Text shifts might fail.")
        print("NLTK data download complete.")


def get_wordnet_pos(tag: str) -> Optional[str]:
    """Map NLTK POS tag to WordNet POS constants."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return None


def get_word_replacements(
    word: str, pos: str, replacement_type: str = "synonym"
) -> List[str]:
    """
    Get synonyms or antonyms for a word using WordNet, with caching.
    """
    if not NLTK_AVAILABLE:
        return []

    cache = synonym_cache if replacement_type == "synonym" else antonym_cache
    cache_key = f"{word}|{pos}"
    if cache_key in cache:
        return cache[cache_key]

    if len(word) <= 2:  # Skip short words
        return []

    replacements = set()
    wordnet_pos = get_wordnet_pos(pos)
    if not wordnet_pos:
        cache[cache_key] = []
        return []

    try:
        for synset in wordnet.synsets(word, pos=wordnet_pos):
            if replacement_type == "synonym":
                for lemma in synset.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != word.lower():
                        replacements.add(name)
            elif replacement_type == "antonym":
                for lemma in synset.lemmas():
                    for antonym in lemma.antonyms():
                        name = antonym.name().replace("_", " ")
                        if name.lower() != word.lower():
                            replacements.add(name)
    except Exception as e:
        print(f"Error finding {replacement_type}s for '{word}' (POS: {pos}): {e}")

    replacement_list = sorted(list(replacements))
    cache[cache_key] = replacement_list
    return replacement_list


def build_replacement_cache(
    texts: List[str], replacement_type: str = "synonym", verbose: bool = True
):
    """Build a cache of word replacements for a given list of texts."""
    if not NLTK_AVAILABLE:
        return
    ensure_nltk_resources()
    if verbose:
        print(f"Building {replacement_type} cache...")

    unique_words = set(
        word for text in texts if isinstance(text, str) for word in word_tokenize(text)
    )
    if not unique_words:
        return

    pos_tags_to_check = ["NN", "VB", "JJ", "RB"]  # Noun, Verb, Adjective, Adverb
    for word in tqdm(
        unique_words, desc=f"Caching {replacement_type}s", disable=not verbose
    ):
        for pos in pos_tags_to_check:
            # Populates the cache via the getter function
            get_word_replacements(word, pos, replacement_type)

    if verbose:
        print(f"{replacement_type.capitalize()} cache built for {len(unique_words)} unique words.")


def apply_character_edit(word: str, seed: Optional[int] = None) -> str:
    """Apply a random character-level edit (insert, delete, replace, swap)."""
    if len(word) <= 1:
        return word
    rng = random.Random(seed) if seed is not None else random

    edit_type = rng.choice(["insert", "delete", "replace", "swap"])
    pos = rng.randint(0, len(word) - 1)
    chars = string.ascii_lowercase

    if edit_type == "insert":
        return word[:pos] + rng.choice(chars) + word[pos:]
    if edit_type == "delete":
        return word[:pos] + word[pos + 1 :]
    if edit_type == "replace":
        new_char = rng.choice(chars)
        while new_char == word[pos]:
            new_char = rng.choice(chars)
        return word[:pos] + new_char + word[pos + 1 :]
    if edit_type == "swap" and len(word) > 1:
        pos = rng.randint(0, len(word) - 2)
        return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
    return word


def shift_texts_internal(
    texts: List[str],
    replacement_type: str,
    noise_level: float,
    char_edit_prob: float,
    random_seed: int,
    verbose: bool = False,
) -> List[str]:
    """Internal helper to perform text augmentation."""
    if not NLTK_AVAILABLE or not texts:
        return texts
    ensure_nltk_resources()
    rng = random.Random(random_seed)

    shifted_texts = []
    iterator = tqdm(
        texts, desc=f"Shifting text ({replacement_type})", disable=not verbose
    )

    for i, text in enumerate(iterator):
        if not isinstance(text, str) or not text.strip():
            shifted_texts.append(text)
            continue

        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        new_words = list(words)

        num_to_modify = int(len(words) * noise_level)
        if num_to_modify == 0:
            shifted_texts.append(text)
            continue

        candidate_indices = [
            idx
            for idx, (word, tag) in enumerate(tagged_words)
            if len(word) > 2 and not tag.startswith("NNP")  # Exclude short & proper nouns
        ]
        rng.shuffle(candidate_indices)
        indices_to_modify = candidate_indices[:num_to_modify]

        for idx in indices_to_modify:
            word, tag = tagged_words[idx]
            replacements = get_word_replacements(word, tag, replacement_type)
            if replacements:
                new_words[idx] = rng.choice(replacements)

            if rng.random() < char_edit_prob:
                char_seed = random_seed + i * len(words) + idx
                new_words[idx] = apply_character_edit(new_words[idx], seed=char_seed)

        shifted_texts.append(" ".join(new_words))

    return shifted_texts


# --- Caching Decorator and Utilities ---
def cache_datasets(shift_name_template: str):
    """
    Decorator to cache and load generated datasets, avoiding re-computation.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = kwargs.get("config", {})
            dataset = kwargs.get("dataset")
            use_oe = kwargs.get("use_OE", False)
            
            if not all([config, dataset]):
                # Fallback if config or dataset is not in kwargs
                return func(*args, **kwargs)

            # Sanitize model name for file path
            encoder_model = config.get("text_encoder_model", "default")
            model_path_name = encoder_model.replace("/", "_").replace("\\", "_")
            if model_path_name == "all-MiniLM-L6-v2":
                model_path_name = ""
            else:
                model_path_name = f"_{model_path_name}"
            
            # Populate the path template with dynamic arguments
            path_template_args = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str))}
            path_template_args['dataset_name'] = getattr(dataset, 'name', 'unknown')
            path_template_args['ood_class_to_leave_out'] = config.get("label_shift", {}).get("ood_class_to_leave_out", [0])

            base_path_str = shift_name_template.format(**path_template_args)
            
            path = f"{base_path_str}_{config['random_seed']}{model_path_name}"
            embedding_path = config.get("embedding_path", ".")
            
            ind_path = os.path.join(embedding_path, f"{path}_ind.pt")
            ood_tr_path = os.path.join(embedding_path, f"{path}_ood_tr.pt")
            ood_te_path = os.path.join(embedding_path, f"{path}_ood_te.pt")

            # Check if all necessary files exist
            files_exist = os.path.exists(ind_path) and os.path.exists(ood_te_path)
            if use_oe:
                files_exist = files_exist and os.path.exists(ood_tr_path)

            if files_exist:
                print(f"Loading cached datasets for shift: {path}")
                dataset_ind = torch.load(ind_path)
                dataset_ood_te = torch.load(ood_te_path)
                dataset_ood_tr = torch.load(ood_tr_path) if use_oe else None
                return dataset_ind, dataset_ood_tr, dataset_ood_te

            print("Path:", ind_path, ood_te_path)
            # If not cached, run the generation function
            result = func(*args, **kwargs)
            dataset_ind, dataset_ood_tr, ood_test_data = result

            # Save the newly generated datasets
            print(f"Saving generated datasets to: {embedding_path}")
            torch.save(dataset_ind, ind_path)
            torch.save(ood_test_data, ood_te_path)
            
            return result

        return wrapper
    return decorator


def create_ood_datasets(
    dataset: Data,
    shift_type: str,
    config: Dict[str, Any],
    random_seed: int,
    text_encoder: Optional[TextEncoder],
) -> Tuple[Data, Optional[Data], List[Data]]:
    """
    Creates OOD datasets based on the specified shift type.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    verbose = config.get("verbose", True)
    use_OE = config.get('use_OE', False)
    if verbose:
        print(f"\n--- Creating OOD datasets for {shift_type.upper()} shift ---")

    shift_func_map = {
        "structure": create_structure_shift_datasets,
        "feature": create_feature_shift_datasets,
        "label": create_label_shift_datasets,
        "text": create_text_shift_datasets,
        "text_swap": create_text_swap_datasets,
        "semantic_connection": create_semantic_connection_datasets,
        "arxiv_time": create_arxiv_time_split_datasets,
    }

    if shift_type not in shift_func_map:
        raise ValueError(f"Shift type '{shift_type}' not recognized.")

    shift_config_key = f"{shift_type}_shift" if shift_type != "arxiv_time" else "arxiv_time_split"
    shift_config = config.get(shift_config_key, {})
    print("Shift configs:", shift_config)

    # Prepare arguments for the specific shift function
    kwargs = {
        "dataset": dataset.cpu(), # Ensure dataset is on CPU for modifications
        "config": config,
        "random_seed": random_seed,
        "verbose": verbose,
        'use_OE': use_OE,
        **shift_config,
    }

    if shift_type in ["text", "text_swap", "semantic_connection"]:
        kwargs["text_encoder"] = text_encoder

    # Call the appropriate generation function
    ind_data, ood_tr_data, ood_test_data_or_list = shift_func_map[shift_type](**kwargs)

    ood_test_datasets = (
        ood_test_data_or_list
        if isinstance(ood_test_data_or_list, list)
        else [ood_test_data_or_list]
    )

    if verbose:
        print(f"--- Finished creating {shift_type.upper()} shift datasets ---")
    return ind_data, ood_tr_data, ood_test_datasets


# --- Shift-Specific Dataset Creation Functions ---
@cache_datasets("{dataset_name}_structure_{p_ii_factor}_{p_ij_factor}_{noise_level}")
def create_structure_shift_datasets(
    dataset: Data,
    config: Dict[str, Any],
    p_ii_factor: float,
    p_ij_factor: float,
    noise_level: float,
    verbose: bool,
    random_seed: int,
    use_OE: bool,
) -> Tuple[Data, Optional[Data], Data]:
    """Generates structure shift using SBM and interpolation."""
    dataset_ind = deepcopy(dataset)
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_tr = deepcopy(dataset) if use_OE else None

    n = dataset.num_nodes
    if n <= 1:
        return dataset_ind, dataset_ood_tr, dataset_ood_te

    original_edge_index = to_undirected(dataset.edge_index, num_nodes=n)
    dataset_ind.edge_index = original_edge_index
    
    num_orig_edges = original_edge_index.size(1) // 2
    density = num_orig_edges / (n * (n - 1) / 2) if n > 1 else 0.0
    
    noise_level = max(0.0, min(1.0, noise_level))

    if verbose:
        print(f"  Structure Shift: Orig Edges={num_orig_edges}, Density={density:.4f}, Alpha={noise_level:.2f}")

    if noise_level == 0.0:
        dataset_ood_te.edge_index = original_edge_index
        if use_OE: dataset_ood_tr.edge_index = original_edge_index
        return dataset_ind, dataset_ood_tr, dataset_ood_te
    
    # Determine block sizes from labels
    if dataset.y is None:
        block_sizes = [n]
    else:
        unique_labels, counts = torch.unique(dataset.y.squeeze(), return_counts=True)
        sorted_indices = torch.argsort(unique_labels)
        block_sizes = counts[sorted_indices].tolist()

    if sum(block_sizes) != n:
        block_sizes.append(n - sum(block_sizes))
    
    num_classes = len(block_sizes)

    # Calculate SBM probabilities
    p_ii = min(max(p_ii_factor * density, 0.0), 1.0)
    p_ij = min(max(p_ij_factor * density, 0.0), 1.0)
    edge_probs = torch.full((num_classes, num_classes), p_ij)
    edge_probs.fill_diagonal_(p_ii)

    if verbose:
        print(f"  SBM Params: BlockSizes={block_sizes}, p_ii={p_ii:.4f}, p_ij={p_ij:.4f}")

    def generate_sbm(seed_offset=0):
        torch.manual_seed(random_seed + seed_offset)
        sbm_edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs, directed=False)
        return to_undirected(sbm_edge_index, num_nodes=n)

    def interpolate_edges(sbm_edges):
        if noise_level == 1.0:
            return sbm_edges
        
        num_orig_sample = int((1.0 - noise_level) * num_orig_edges)
        num_sbm_sample = int(noise_level * num_orig_edges)

        sampled_orig = sample_unique_edges(original_edge_index, num_orig_sample * 2)
        sampled_sbm = sample_unique_edges(sbm_edges, num_sbm_sample * 2)
        
        combined = torch.cat([sampled_orig, sampled_sbm], dim=1)
        combined_unique = torch.unique(combined, dim=1)
        # Remove self-loops
        combined_unique = combined_unique[:, combined_unique[0] != combined_unique[1]]
        return to_undirected(combined_unique, num_nodes=n)

    sbm_edges_te = generate_sbm(seed_offset=0)
    dataset_ood_te.edge_index = interpolate_edges(sbm_edges_te)

    if use_OE and dataset_ood_tr:
        sbm_edges_tr = generate_sbm(seed_offset=1)
        dataset_ood_tr.edge_index = interpolate_edges(sbm_edges_tr)

    if verbose and dataset.y is not None:
        h_orig = homophily(dataset_ind.edge_index, dataset.y, method='edge')
        h_te = homophily(dataset_ood_te.edge_index, dataset.y, method='edge')
        print(f"  Homophily (Edge) - Original: {h_orig:.4f}, OOD_TE: {h_te:.4f}")

    return dataset_ind, dataset_ood_tr, dataset_ood_te


@cache_datasets("{dataset_name}_feature_{noise_level}")
def create_feature_shift_datasets(
    dataset: Data,
    config: Dict[str, Any],
    noise_level: float,
    verbose: bool,
    random_seed: int,
    use_OE: bool,
) -> Tuple[Data, Optional[Data], Data]:
    """Generates feature shift by mixing node features."""
    if not hasattr(dataset, "x") or dataset.x is None:
        print("Warning: Dataset has no features ('x'). Cannot apply feature shift.")
        return deepcopy(dataset), None, deepcopy(dataset)

    dataset_ind = deepcopy(dataset)
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_tr = deepcopy(dataset) if use_OE else None

    n = dataset.num_nodes
    noise_level = max(0.0, min(1.0, noise_level))
    if verbose:
        print(f"  Feature Shift: Noise Level={noise_level:.2f}")

    if noise_level == 0.0:
        return dataset_ind, dataset_ood_tr, dataset_ood_te

    original_features = dataset.x

    def generate_mixed_features(seed):
        torch.manual_seed(seed)
        idx = torch.randint(0, n, (n, 2), device=original_features.device)
        weights = torch.rand(n, 1, device=original_features.device)
        return (
            original_features[idx[:, 0]] * weights
            + original_features[idx[:, 1]] * (1 - weights)
        )

    mixed_features_te = generate_mixed_features(random_seed)
    dataset_ood_te.x = (1 - noise_level) * original_features + noise_level * mixed_features_te

    if use_OE and dataset_ood_tr:
        mixed_features_tr = generate_mixed_features(random_seed + 1)
        dataset_ood_tr.x = (1 - noise_level) * original_features + noise_level * mixed_features_tr

    return dataset_ind, dataset_ood_tr, dataset_ood_te


@cache_datasets("{dataset_name}_label_{ood_class_to_leave_out}_{setting}")
def create_label_shift_datasets(
    dataset: Data,
    config: Dict[str, Any],
    ood_class_to_leave_out: Union[int, List[int]],
    setting: str,
    verbose: bool,
    random_seed: int,
    use_OE: bool,
) -> Tuple[Data, Optional[Data], Data]:
    """Generates label shift datasets by holding out one or more classes."""
    if not hasattr(dataset, "y") or dataset.y is None:
        raise ValueError("Label shift requires dataset labels ('y').")
    if setting not in ["inductive", "transductive"]:
        raise ValueError("Label shift setting must be 'inductive' or 'transductive'.")

    y_squeezed = dataset.y.squeeze()
    all_classes = sorted(torch.unique(y_squeezed).tolist())
    
    ood_classes = [ood_class_to_leave_out] if isinstance(ood_class_to_leave_out, int) else ood_class_to_leave_out
    id_classes = [c for c in all_classes if c not in ood_classes]
    
    if verbose:
        print(f"  Label Shift ({setting}): ID Classes={id_classes}, OOD Classes={ood_classes}")

    id_mask = torch.isin(y_squeezed, torch.tensor(id_classes))
    ood_mask = torch.isin(y_squeezed, torch.tensor(ood_classes))
    
    id_original_indices = torch.where(id_mask)[0]
    ood_original_indices = torch.where(ood_mask)[0]

    # --- Create In-Distribution Dataset ---
    dataset_ind = Data()
    dataset_ind.node_idx = id_original_indices
    id_class_map = {orig_label: new_label for new_label, orig_label in enumerate(id_classes)}

    if setting == "inductive":
        dataset_ind.num_nodes = id_original_indices.numel()
        if dataset.x is not None:
            dataset_ind.x = dataset.x[id_mask].clone()
        if hasattr(dataset, 'raw_texts'):
            dataset_ind.raw_texts = [dataset.raw_texts[i] for i in id_original_indices]
        
        y_id = dataset.y[id_mask].clone()
        dataset_ind.y = torch.tensor([id_class_map[label.item()] for label in y_id.squeeze()])
        
        edge_index, _ = subgraph(id_original_indices, dataset.edge_index, relabel_nodes=True)
        dataset_ind.edge_index = edge_index
        
    else:  # Transductive
        dataset_ind = deepcopy(dataset)
        dataset_ind.class_mapping = id_class_map
        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(dataset_ind, mask_key):
                dataset_ind[mask_key] = dataset_ind[mask_key] & id_mask

    # --- Create OOD Test Dataset ---
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_te.node_idx = ood_original_indices
    dataset_ood_te.train_mask = dataset_ood_te.val_mask = dataset_ood_te.test_mask = None

    # --- Create OOD Train (OE) Dataset ---
    dataset_ood_tr = None
    if use_OE:
        dataset_ood_tr = deepcopy(dataset)
        first_ood_class_mask = (y_squeezed == ood_classes[0])
        dataset_ood_tr.node_idx = torch.where(first_ood_class_mask)[0]
        dataset_ood_tr.train_mask = dataset_ood_tr.val_mask = dataset_ood_tr.test_mask = None
        
    return dataset_ind, dataset_ood_tr, dataset_ood_te


@cache_datasets("{dataset_name}_text_shift_{augmentation_type}")
def create_text_shift_datasets(
    dataset: Data,
    config: Dict[str, Any],
    text_encoder: TextEncoder,
    augmentation_type: str,
    noise_level: float,
    char_edit_prob: float,
    verbose: bool,
    random_seed: int,
    use_OE: bool,
) -> Tuple[Data, Optional[Data], Data]:
    """Generates text shift by augmenting raw text features."""
    if not hasattr(dataset, "raw_texts") or not dataset.raw_texts:
        raise ValueError("Dataset must have 'raw_texts' for this shift.")
    if not NLTK_AVAILABLE:
        raise RuntimeError("NLTK is required for 'text' shift.")

    dataset_ind = deepcopy(dataset)
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_tr = deepcopy(dataset) if use_OE else None

    noise_level = max(0.0, min(1.0, noise_level))
    if verbose:
        print(f"  Text Shift: Type={augmentation_type}, Noise={noise_level:.2f}")

    if noise_level == 0.0:
        return dataset_ind, dataset_ood_tr, dataset_ood_te

    build_replacement_cache(dataset.raw_texts, augmentation_type, verbose)

    # Generate shifted texts for OOD Test set
    shifted_texts_te = shift_texts_internal(
        dataset.raw_texts, augmentation_type, noise_level, char_edit_prob, random_seed, verbose
    )
    dataset_ood_te.raw_texts = shifted_texts_te
    dataset_ood_te.x = text_encoder.encode_texts(texts=shifted_texts_te)

    # Generate for OOD Train set if needed
    if use_OE and dataset_ood_tr:
        shifted_texts_tr = shift_texts_internal(
            dataset.raw_texts, augmentation_type, noise_level, char_edit_prob, random_seed + 1, verbose
        )
        dataset_ood_tr.raw_texts = shifted_texts_tr
        dataset_ood_tr.x = text_encoder.encode_texts(texts=shifted_texts_tr)
        
    return dataset_ind, dataset_ood_tr, dataset_ood_te


@cache_datasets("{dataset_name}_text_swap_{swap_scope}")
def create_text_swap_datasets(
    dataset: Data,
    config: Dict[str, Any],
    swap_scope: str,
    swap_ratio: float,
    verbose: bool,
    random_seed: int,
    use_OE: bool,
) -> Tuple[Data, Optional[Data], Data]:
    """Swaps text and features between nodes."""
    if not hasattr(dataset, "raw_texts"):
        raise ValueError("Dataset must have 'raw_texts' for text_swap shift.")
    
    dataset_ind = deepcopy(dataset)
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_tr = deepcopy(dataset) if use_OE else None
    
    swap_ratio = max(0.0, min(1.0, swap_ratio))
    if verbose:
        print(f"  Text Swap: Scope={swap_scope}, Ratio={swap_ratio:.2f}")

    if swap_ratio == 0.0:
        return dataset_ind, dataset_ood_tr, dataset_ood_te

    def _swap_and_get_map(seed):
        rng = np.random.RandomState(seed)
        n = dataset.num_nodes
        
        num_to_swap = int(n * swap_ratio)
        indices_to_swap = rng.choice(n, num_to_swap, replace=False)
        
        # Create pairs for swapping
        rng.shuffle(indices_to_swap)
        pairs = list(zip(indices_to_swap[::2], indices_to_swap[1::2]))
        
        # Create the mapping from old to new index
        reindex_map = np.arange(n)
        for i, j in pairs:
            # Optionally filter pairs based on scope (intra/inter-class)
            if swap_scope == 'intra' and dataset.y[i] != dataset.y[j]:
                continue
            if swap_scope == 'inter' and dataset.y[i] == dataset.y[j]:
                continue
            reindex_map[i], reindex_map[j] = reindex_map[j], reindex_map[i]
            
        return reindex_map

    # Perform swap for OOD Test
    te_map = _swap_and_get_map(random_seed)
    dataset_ood_te.raw_texts = [dataset.raw_texts[i] for i in te_map]
    if dataset.x is not None:
        dataset_ood_te.x = dataset.x[te_map]

    # Perform swap for OOD Train
    if use_OE and dataset_ood_tr:
        tr_map = _swap_and_get_map(random_seed + 1)
        dataset_ood_tr.raw_texts = [dataset.raw_texts[i] for i in tr_map]
        if dataset.x is not None:
            dataset_ood_tr.x = dataset.x[tr_map]
            
    return dataset_ind, dataset_ood_tr, dataset_ood_te

@cache_datasets("{dataset_name}_semantic_{selection_mode}_{threshold_percentile}")
def create_semantic_connection_datasets(
    dataset: Data,
    config: Dict[str, Any],
    text_encoder: Optional[TextEncoder],
    target_density_factor: float = 1.0,
    similarity_metric: str = 'cosine',
    selection_mode: str = 'threshold',
    threshold_percentile: float = 0.5,
    use_approximate: bool = False,
    batch_size: int = 10000,
    sampling_ratio: float = 0.01,
    verbose: bool = False,
    random_seed: int = 42,
    use_OE: bool = False,
) -> Tuple[Data, Optional[Data], Data]:
    """Rewires graph edges based on node feature similarity."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("Scikit-learn is required for semantic connection shift.")
        
    dataset_ind = deepcopy(dataset)
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_tr = deepcopy(dataset) if use_OE else None

    target_num_edges = _get_target_edge_count(dataset, target_density_factor, verbose)

    if target_num_edges == 0:
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        dataset_ood_te.edge_index = empty_edges
        if use_OE and dataset_ood_tr:
            dataset_ood_tr.edge_index = empty_edges
        return dataset_ind, dataset_ood_tr, dataset_ood_te
        
    node_features = dataset.x.cpu().numpy()
    n, d = node_features.shape
    is_large_graph = n > 50000

    new_edge_index = None
    if use_approximate and is_large_graph:
        if FAISS_AVAILABLE and selection_mode == 'top':
             new_edge_index = _select_edges_by_faiss(node_features, target_num_edges, verbose)
        else:
             new_edge_index = _select_edges_by_sampling(node_features, target_num_edges, selection_mode, threshold_percentile, sampling_ratio, verbose)
    else:
        new_edge_index = _select_edges_from_full_matrix(node_features, target_num_edges, selection_mode, threshold_percentile, verbose)
        
    dataset_ood_te.edge_index = new_edge_index
    if use_OE and dataset_ood_tr:
        # For simplicity, using the same graph for OE. Could generate another.
        dataset_ood_tr.edge_index = new_edge_index

    if verbose and dataset.y is not None:
        h_orig = homophily(dataset_ind.edge_index, dataset.y, method='edge')
        h_new = homophily(new_edge_index, dataset.y, method='edge')
        print(f"  Homophily (Edge) - Original: {h_orig:.4f}, Semantic ({selection_mode}): {h_new:.4f}")

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def create_arxiv_time_split_datasets(
    dataset: Data, time_bound_train: List[int], time_bound_test: List[int],
    inductive: bool, verbose: bool, random_seed: int, use_OE: bool, config: Dict[str, Any]
) -> Tuple[Data, Optional[Data], List[Data]]:
    ogb_dataset = dataset
    edge_index = ogb_dataset.edge_index
    node_feat = ogb_dataset.x
    label = ogb_dataset.y.squeeze()
    year = torch.as_tensor(dataset.node_year).squeeze()

    year_min, year_max = time_bound_train[0], time_bound_train[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    center_node_mask = (year <= year_min).squeeze(-1)
    inductive = True
    if inductive:
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (year <= year_max).squeeze(-1) * (year > year_min).squeeze(-1)
    if inductive:
        all_node_mask = (year <= year_max).squeeze(-1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in range(len(test_year_bound)-1):
        center_node_mask = (year <= test_year_bound[i+1]).squeeze(-1) * (year > test_year_bound[i]).squeeze(-1)
        if inductive:
            all_node_mask = (year <= test_year_bound[i+1]).squeeze(-1)
            ood_te_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_te_edge_index = edge_index

        dataset = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    id_node_idx = np.load("../embeddings/arxiv_ind_node_idx.npy", allow_pickle=True)
    train_idx = np.load("../embeddings/arxiv_ind_node_tr_idx.npy", allow_pickle=True)
    valid_idx = np.load("../embeddings/arxiv_ind_node_val_idx.npy", allow_pickle=True)
    test_idx = np.load("../embeddings/arxiv_ind_node_te_idx.npy", allow_pickle=True)

    print("tr_idx", train_idx, "val_idx", valid_idx, "te_idx", test_idx)

    dataset_ind.node_idx = id_node_idx
    train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    masks = {}
    masks['train_mask'] = train_mask
    masks['val_mask'] = val_mask
    masks['test_mask'] = test_mask

    for i in range(len(dataset_ood_te)):
        idx =  np.load(f"../embeddings/arxiv_ood_te_{i}_idx.npy")
        idx = torch.tensor(idx)
        dataset_ood_te[i].node_idx = idx

    for mask_key in ['train_mask', 'val_mask', 'test_mask']:
        dataset_ind[mask_key] = masks[mask_key]

    return dataset_ind, dataset_ood_tr, dataset_ood_te


# --- Utility Functions ---

def sample_unique_edges(edge_index: torch.Tensor, num_to_sample: int) -> torch.Tensor:
    """Samples unique directed edges from an edge_index tensor."""
    num_total_edges = edge_index.size(1)
    if num_to_sample >= num_total_edges:
        return edge_index
    if num_to_sample <= 0:
        return torch.empty((2, 0), dtype=edge_index.dtype)

    perm = torch.randperm(num_total_edges, device=edge_index.device)
    return edge_index[:, perm[:num_to_sample]]


def _get_target_edge_count(dataset: Data, factor: float, verbose: bool) -> int:
    """Calculate the target number of edges based on original density."""
    n = dataset.num_nodes
    if n <= 1: return 0
    
    num_orig_edges = to_undirected(dataset.edge_index, n).size(1) // 2
    max_possible_edges = n * (n - 1) // 2
    target_num_edges = min(int(num_orig_edges * factor), max_possible_edges)

    if verbose:
        print(f"  Original unique edges: {num_orig_edges}")
        print(f"  Target unique edges: {target_num_edges} (Factor: {factor:.2f})")
    return target_num_edges


def _select_edges_by_faiss(features: np.ndarray, k: int, verbose: bool) -> torch.Tensor:
    """Use FAISS for approximate nearest neighbor search to find top-k similar pairs."""
    if verbose: print("  Using FAISS for approximate nearest neighbors...")
    n, d = features.shape
    
    features_normalized = features.astype(np.float32)
    faiss.normalize_L2(features_normalized)
    
    index = faiss.IndexFlatIP(d)
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(features_normalized)
    
    # Estimate neighbors needed per node
    neighbors_per_node = min(max(2, int(2.5 * k / n)), 1000)
    
    _, indices = index.search(features_normalized, neighbors_per_node)
    
    source_nodes, target_nodes = [], []
    for i in range(n):
        for j in indices[i, 1:]: # Skip self
            if i < j: # Avoid duplicates
                source_nodes.append(i)
                target_nodes.append(j)

    edges = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # Sub-sample if we found too many edges
    if edges.size(1) > k:
        perm = torch.randperm(edges.size(1))
        edges = edges[:, perm[:k]]
        
    return to_undirected(edges, n)


def _select_edges_by_sampling(features: np.ndarray, k: int, mode: str, percentile: float, ratio: float, verbose: bool) -> torch.Tensor:
    """Use random sampling to find top-k, bottom-k, or threshold-based pairs."""
    if verbose: print(f"  Using random sampling (ratio={ratio:.3f}) for '{mode}' selection...")
    n = features.shape[0]
    num_samples = int(n * (n - 1) / 2 * ratio)
    num_samples = min(num_samples, 20_000_000) # Cap samples

    # Sample pairs without replacement
    rows = np.random.randint(0, n - 1, size=num_samples)
    cols = np.random.randint(rows + 1, n, size=num_samples)
    
    sims = np.sum(features[rows] * features[cols], axis=1) # Cosine on L2-normalized features
    
    if mode == 'top':
        indices = np.argpartition(sims, -k)[-k:]
    elif mode == 'bottom':
        indices = np.argpartition(sims, k)[:k]
    else: # threshold
        threshold_val = np.percentile(sims, percentile * 100)
        # Select k pairs centered around the threshold
        distances = np.abs(sims - threshold_val)
        indices = np.argpartition(distances, k)[:k]
        
    edges = torch.tensor([rows[indices], cols[indices]], dtype=torch.long)
    return to_undirected(edges, n)


def _select_edges_from_full_matrix(features: np.ndarray, k: int, mode: str, percentile: float, verbose: bool) -> torch.Tensor:
    """Calculate the full similarity matrix to select edges (for smaller graphs)."""
    if verbose: print("  Calculating full similarity matrix...")
    n = features.shape[0]
    
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, -np.inf if mode == 'top' else np.inf)
    
    row_idx, col_idx = np.triu_indices(n, k=1)
    sims = sim_matrix[row_idx, col_idx]
    
    if mode == 'top':
        indices = np.argpartition(sims, -k)[-k:]
    elif mode == 'bottom':
        indices = np.argpartition(sims, k)[:k]
    else: # threshold
        threshold_idx = int(len(sims) * percentile)
        sorted_indices = np.argsort(sims)
        start = max(0, threshold_idx - k // 2)
        end = start + k
        indices = sorted_indices[start:end]
        
    edges = torch.tensor([row_idx[indices], col_idx[indices]], dtype=torch.long)
    return to_undirected(edges, n)