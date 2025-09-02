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
            embedding_path = os.path.join(embedding_path, path_template_args['dataset_name'])

            os.makedirs(embedding_path, exist_ok=True)
            
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


# # --- Shift-Specific Dataset Creation Functions ---
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
    """
    Generate structure-shifted datasets via SBM–original edge interpolation.
    Includes a memory-efficient SBM generator for very large graphs.
    """
    # ---- Configurable thresholds for "large" workloads ----
    LARGE_GRAPH_NODE_THRESHOLD = 40000            # use efficient SBM above this
    WITHIN_BETWEEN_CHUNK = 10000000              # candidates per chunk
    EDGE_SAMPLE_RESERVOIR_CHUNK = 1000000        # edges per sampling chunk

    # ---- Copies for IND / OOD TE / OOD TR ----
    dataset_ind = deepcopy(dataset)
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_tr = deepcopy(dataset) if use_OE else None

    # ---- Node index (all nodes participate) ----
    n = dataset.num_nodes
    all_nodes = torch.arange(n)
    dataset_ind.node_idx = all_nodes
    dataset_ood_te.node_idx = all_nodes
    if use_OE:
        dataset_ood_tr.node_idx = all_nodes

    if n <= 1:
        return dataset_ind, dataset_ood_tr, dataset_ood_te

    # ---- Original undirected edges & density ----
    if dataset.edge_index is not None and dataset.edge_index.numel() > 0:
        original_edge_index = to_undirected(dataset.edge_index, num_nodes=n)
        dataset_ind.edge_index = original_edge_index
        num_orig_edges_undirected = original_edge_index.size(1) // 2
        possible_edges = float(n) * (n - 1) / 2.0
        density = (num_orig_edges_undirected / possible_edges) if possible_edges > 0 else 0.0
    else:
        original_edge_index = torch.empty((2, 0), dtype=torch.long)
        num_orig_edges_undirected, density = 0, 0.0

    # ---- Clamp noise level ----
    noise_level = max(0.0, min(1.0, noise_level))

    if verbose:
        print(
            f"  Structure Shift: orig_edges={dataset.edge_index.size(1) if dataset.edge_index is not None else 0}, "
            f"undirected={original_edge_index.size(1)}, half={num_orig_edges_undirected}, "
            f"density={density:.6f}, alpha={noise_level:.2f}"
        )

    # ---- Early exit: no shift ----
    if noise_level == 0.0:
        dataset_ood_te.edge_index = original_edge_index
        if use_OE:
            dataset_ood_tr.edge_index = original_edge_index
        return dataset_ind, dataset_ood_tr, dataset_ood_te

    # ---- Determine block sizes from labels ----
    if dataset.y is None:
        block_sizes = [n]
    else:
        y_squeezed = dataset.y.squeeze() if hasattr(dataset.y, "squeeze") else dataset.y
        if n > 1000000:
            # Chunked counting to reduce peak memory
            if verbose:
                print("  Counting labels in chunks for very large graph...")
            counts = {}
            CHUNK = 1000000
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                chunk = y_squeezed[start:end]
                uniq, cnt = torch.unique(chunk, return_counts=True)
                for u, c in zip(uniq.tolist(), cnt.tolist()):
                    counts[u] = counts.get(u, 0) + int(c)
                del chunk, uniq, cnt
                gc.collect()
            block_sizes = [counts[k] for k in sorted(counts)]
        else:
            uniq, cnt = torch.unique(y_squeezed, return_counts=True)
            order = torch.argsort(uniq)
            block_sizes = cnt[order].tolist()

        # sanity: ensure sum equals n
        if sum(block_sizes) != n:
            block_sizes = [bs for bs in block_sizes if bs > 0]
            if sum(block_sizes) != n:
                block_sizes.append(n - sum(block_sizes))

    if not block_sizes:
        block_sizes = [n]

    num_classes = len(block_sizes)

    # ---- SBM probabilities from density scaling ----
    p_ii = min(max(p_ii_factor * density, 0.0), 1.0)
    p_ij = min(max(p_ij_factor * density, 0.0), 1.0)
    edge_probs = torch.full((num_classes, num_classes), p_ij, dtype=torch.float32)
    edge_probs.fill_diagonal_(p_ii)

    if verbose:
        print(f"  BlockSizes={block_sizes}, p_ii={p_ii:.6f}, p_ij={p_ij:.6f}")

    # ---------- Helpers ----------

    def linear_to_triu_indices(idx: int, size: int) -> Tuple[int, int]:
        """Map linear index → (row, col) in upper triangle (excluding diagonal)."""
        i = int((2 * size - 1 - np.sqrt((2 * size - 1) ** 2 - 8 * idx)) / 2)
        j = idx - (i * (2 * size - i - 1)) // 2 + i + 1
        return i, j

    def sample_unique_edges_efficient(edge_index: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample edges without replacement; chunked reservoir for huge E."""
        if edge_index.numel() == 0 or num_samples <= 0:
            return torch.empty((2, 0), dtype=torch.long)

        E = edge_index.size(1)
        if num_samples >= E:
            return edge_index

        # Small enough: simple randperm
        if E <= EDGE_SAMPLE_RESERVOIR_CHUNK:
            idx = torch.randperm(E)[:num_samples]
            return edge_index[:, idx]

        # Reservoir sampling (indices only)
        if verbose:
            print(f"  Reservoir sampling {num_samples} from {E} edges...")
        import random as _rnd

        reservoir = list(range(num_samples))  # initial fill
        seen = num_samples
        while seen < E:
            # process in chunks
            chunk_end = min(seen + EDGE_SAMPLE_RESERVOIR_CHUNK, E)
            for i in range(seen, chunk_end):
                j = _rnd.randint(0, i)
                if j < num_samples:
                    reservoir[j] = i
            seen = chunk_end

        idx_tensor = torch.tensor(reservoir, dtype=torch.long)
        return edge_index[:, idx_tensor]

    # ---------- Memory-efficient SBM generation ----------

    def generate_sbm_efficient(valid_sizes: list, probs: torch.Tensor, seed_offset: int = 0) -> torch.Tensor:
        """
        Edge-level sampling with chunking:
          - within-block: sample upper-tri entries in chunks
          - between-block: sample full rectangle in chunks
        """
        torch.manual_seed(random_seed + seed_offset)
        total_nodes = sum(valid_sizes)
        if total_nodes != n or len(valid_sizes) == 0:
            if verbose:
                print("  [warn] generate_sbm_efficient: invalid block sizes, returning empty graph")
            return torch.empty((2, 0), dtype=torch.long)

        edges: list[Tuple[int, int]] = []
        offset_i = 0

        for i, sz_i in enumerate(valid_sizes):
            block_i = list(range(offset_i, offset_i + sz_i))

            # within-block (upper triangle only)
            p_within = float(probs[i, i].item())
            if p_within > 0 and sz_i >= 2:
                cand = sz_i * (sz_i - 1) // 2
                if verbose and cand > WITHIN_BETWEEN_CHUNK:
                    print(f"  Within block {i} candidates={cand:,} (chunking)")

                for start in range(0, cand, WITHIN_BETWEEN_CHUNK):
                    end = min(start + WITHIN_BETWEEN_CHUNK, cand)
                    r = torch.rand(end - start, dtype=torch.float32)
                    take = torch.nonzero(r < p_within, as_tuple=False).flatten()
                    if take.numel() > 0:
                        for idx in take.tolist():
                            lin = start + idx
                            a, b = linear_to_triu_indices(lin, sz_i)
                            u, v = block_i[a], block_i[b]
                            edges.append((u, v))
                    del r, take
                    gc.collect()

            # between-block
            offset_j = offset_i + sz_i
            for j in range(i + 1, len(valid_sizes)):
                sz_j = valid_sizes[j]
                p_between = float(probs[i, j].item())
                if p_between <= 0:
                    offset_j += sz_j
                    continue

                block_j = list(range(offset_j, offset_j + sz_j))
                cand = sz_i * sz_j
                if verbose and cand > WITHIN_BETWEEN_CHUNK:
                    print(f"  Between blocks {i}-{j} candidates={cand:,} (chunking)")

                for start in range(0, cand, WITHIN_BETWEEN_CHUNK):
                    end = min(start + WITHIN_BETWEEN_CHUNK, cand)
                    r = torch.rand(end - start, dtype=torch.float32)
                    take = torch.nonzero(r < p_between, as_tuple=False).flatten()
                    if take.numel() > 0:
                        for lin in (start + take).tolist():
                            a = lin // sz_j
                            b = lin % sz_j
                            u, v = block_i[a], block_j[b]
                            edges.append((u, v))
                    del r, take
                    gc.collect()

                offset_j += sz_j

            offset_i += sz_i

        if not edges:
            return torch.empty((2, 0), dtype=torch.long)

        edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        del edges
        gc.collect()
        # undirect here to avoid duplicating later
        return to_undirected(edge_tensor, num_nodes=n)

    def generate_sbm(seed_offset: int = 0) -> torch.Tensor:
        valid_sizes = [bs for bs in block_sizes if bs > 0]
        if sum(valid_sizes) != n or not valid_sizes:
            return torch.empty((2, 0), dtype=torch.long)

        # Use built-in for smaller graphs; efficient custom for large
        if n <= LARGE_GRAPH_NODE_THRESHOLD:
            if verbose:
                print(f"  Generating SBM (stochastic_blockmodel_graph) seed={seed_offset}")
            torch.manual_seed(random_seed + seed_offset)
            out = stochastic_blockmodel_graph(valid_sizes, edge_probs, directed=False)
            gc.collect()
            return out
        else:
            if verbose:
                print(f"  Generating SBM (efficient) seed={seed_offset}")
            out = generate_sbm_efficient(valid_sizes, edge_probs, seed_offset=seed_offset)
            gc.collect()
            return out

    # ---------- Interpolation of edges ----------
    def interpolate_edges(sbm_edges: torch.Tensor) -> torch.Tensor:
        if noise_level == 1.0:
            return sbm_edges

        # sample based on original undirected edge count
        num_from_orig = int(round((1.0 - noise_level) * num_orig_edges_undirected))
        num_from_sbm = int(round(noise_level * num_orig_edges_undirected))

        sampled_orig = sample_unique_edges_efficient(original_edge_index, num_from_orig)
        sampled_sbm = sample_unique_edges_efficient(sbm_edges, num_from_sbm)

        combined = torch.cat([sampled_orig, sampled_sbm], dim=1)
        combined = torch.unique(combined, dim=1)
        combined = combined[:, combined[0] != combined[1]]  # remove self-loops
        return combined

    # ---- Build OOD TE ----
    sbm_edges_base = generate_sbm(seed_offset=0)
    dataset_ood_te.edge_index = interpolate_edges(sbm_edges_base)

    # Make undirected (some datasets are already handled)
    if dataset.name not in {"bookchild", "bookhis", "elecomp", "elephoto", "arxiv", "reddit", "sportsfit"}:
        dataset_ood_te.edge_index = to_undirected(dataset_ood_te.edge_index, num_nodes=n)

    if verbose:
        print(f"  OOD_TE edges: {dataset_ood_te.edge_index.size(1)}")

    gc.collect()

    # ---- Build OOD TR (if OE) ----
    if use_OE and dataset_ood_tr is not None:
        sbm_edges_tr = generate_sbm(seed_offset=(1 if noise_level < 1.0 else 0))
        dataset_ood_tr.edge_index = interpolate_edges(sbm_edges_tr)
        gc.collect()

    # ---- Optional: homophily logging (if available) ----
    if verbose and dataset.y is not None:
        try:
            h_orig = homophily(dataset_ind.edge_index, dataset.y, method="edge") if dataset_ind.edge_index.numel() else float("nan")
            h_te = homophily(dataset_ood_te.edge_index, dataset.y, method="edge") if dataset_ood_te.edge_index.numel() else float("nan")
            if use_OE and dataset_ood_tr is not None and dataset_ood_tr.edge_index.numel():
                h_tr = homophily(dataset_ood_tr.edge_index, dataset.y, method="edge")
                print(f"  Homophily - Orig: {h_orig:.4f}, OOD_TE: {h_te:.4f}, OOD_TR: {h_tr:.4f}")
            else:
                print(f"  Homophily - Orig: {h_orig:.4f}, OOD_TE: {h_te:.4f}")
        except Exception as e:
            if verbose:
                print(f"  Homophily calc failed: {e}")

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
    dataset: Data, ood_class_to_leave_out: Union[int, List[int]], setting: str,
    verbose: bool, random_seed: int, use_OE: bool, config: Dict[str, Any],
) -> Tuple[Data, Optional[Data], Data]:
    """Generates label shift datasets by holding out classes."""
    if not hasattr(dataset, 'y') or dataset.y is None:
        raise ValueError("Label shift requires dataset labels ('y').")
    if setting not in ['inductive', 'transductive']:
        raise ValueError("Label shift setting must be 'inductive' or 'transductive'.")

    y_squeezed = dataset.y.squeeze()
    unique_classes, counts = torch.unique(y_squeezed, return_counts=True)
    actual_classes = sorted(unique_classes.tolist())
    num_classes = len(actual_classes)

    if num_classes < 2:
        raise ValueError(f"Label shift requires at least 2 classes, found {num_classes}.")

    # Determine OOD classes
    if ood_class_to_leave_out is None:
         ood_classes = [actual_classes[-1]] # Default: leave out last class
    elif isinstance(ood_class_to_leave_out, int):
        ood_classes = [ood_class_to_leave_out]
    else: # List
        ood_classes = sorted(list(set(ood_class_to_leave_out)))

    # Validate OOD classes
    invalid_classes = [c for c in ood_classes if c not in actual_classes]
    if invalid_classes:
        raise ValueError(f"Invalid OOD classes {invalid_classes}. Available: {actual_classes}.")
    if len(ood_classes) >= num_classes:
        raise ValueError("Cannot leave out all classes.")

    id_classes = [c for c in actual_classes if c not in ood_classes]
    if verbose:
        print(f"  Label Shift ({setting}): ID Classes={id_classes}, OOD Classes={ood_classes}")

    # Create masks
    ind_mask = torch.isin(y_squeezed, torch.tensor(id_classes))
    ood_mask = torch.isin(y_squeezed, torch.tensor(ood_classes))

    if ind_mask.sum() == 0: print("Warning: No ID nodes found after filtering.")
    if ood_mask.sum() == 0: print("Warning: No OOD nodes found for the specified classes.")

    all_original_indices = torch.arange(dataset.num_nodes)
    id_original_indices = all_original_indices[ind_mask]
    ood_original_indices = all_original_indices[ood_mask]

    # --- Create In-Distribution Dataset ---
    dataset_ind = Data()
    dataset_ind.node_idx = id_original_indices
    dataset_ind.num_nodes = dataset.num_nodes

    # Create mapping for relabeling
    id_class_map = {orig_label: new_label for new_label, orig_label in enumerate(id_classes)}

    if setting == 'inductive':
        # Filter nodes, features, labels, edges for ID subgraph
        dataset_ind.num_nodes = id_original_indices.numel()
        if hasattr(dataset, 'x') and dataset.x is not None:
            dataset_ind.x = dataset.x[ind_mask].clone()
        if hasattr(dataset, 'raw_text') and dataset.raw_text is not None:
                ind_mask_indices = torch.where(ind_mask)[0]
                dataset_ind.raw_text = [dataset.raw_text[i] for i in ind_mask_indices.tolist()]
        if hasattr(dataset, 'raw_texts') and dataset.raw_texts is not None:
            ind_mask_indices = torch.where(ind_mask)[0]
            dataset_ind.raw_text = [dataset.raw_texts[i] for i in ind_mask_indices.tolist()]

        if dataset.y is not None:
             # Relabel Y to be contiguous 0..N_id_classes-1
             y_original_id = dataset.y[ind_mask].clone()
             dataset_ind.y = torch.tensor([id_class_map[label.item()] for label in y_original_id.squeeze()],
                                          dtype=y_original_id.dtype).view_as(y_original_id) # Keep original shape
        # Create ID subgraph, relabeling nodes in edges
        if dataset.edge_index is not None and id_original_indices.numel() > 0:
            edge_attr = getattr(dataset, 'edge_attr', None)
            ind_edge_index, ind_edge_attr = subgraph(id_original_indices, dataset.edge_index, edge_attr,
                                                     relabel_nodes=True, num_nodes=dataset.num_nodes)
            dataset_ind.edge_index = ind_edge_index
            if ind_edge_attr is not None: dataset_ind.edge_attr = ind_edge_attr
        else:
            dataset_ind.edge_index = torch.empty((2, 0), dtype=torch.long)

        # Filter masks and remap indices
        id_subgraph_map = {orig_idx.item(): new_idx for new_idx, orig_idx in enumerate(id_original_indices)}
        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
            new_mask = torch.zeros(dataset_ind.num_nodes, dtype=torch.bool)
            if hasattr(dataset, mask_key) and dataset[mask_key] is not None:
                original_mask_nodes = all_original_indices[dataset[mask_key] & ind_mask] # Nodes in mask AND ID
                if original_mask_nodes.numel() > 0:
                     new_indices = torch.tensor([id_subgraph_map[idx.item()] for idx in original_mask_nodes], dtype=torch.long)
                     new_mask[new_indices] = True
            dataset_ind[mask_key] = new_mask

        
    else: # Transductive
        dataset_ind = deepcopy(dataset)
        dataset_ind.node_idx = id_original_indices
        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
             if hasattr(dataset_ind, mask_key) and dataset_ind[mask_key] is not None:
                  dataset_ind[mask_key] = dataset_ind[mask_key] & ind_mask
        # Store class mapping, but don't relabel Y in the data object itself
        dataset_ind.class_mapping = id_class_map

    dataset_ood_te = deepcopy(dataset)
    dataset_ood_te.node_idx = ood_original_indices # Mark ALL OOD nodes for testing
    dataset_ood_te.train_mask = dataset_ood_te.val_mask = dataset_ood_te.test_mask = None

    dataset_ood_tr = None
    if use_OE:
        dataset_ood_tr = deepcopy(dataset)
        first_ood_class = ood_classes[0]
        first_ood_mask = (y_squeezed == first_ood_class)
        dataset_ood_tr.node_idx = all_original_indices[first_ood_mask]
        dataset_ood_tr.train_mask = dataset_ood_tr.val_mask = dataset_ood_tr.test_mask = None

    common_keys = ['x', 'edge_index', 'y', 'node_idx', 'num_nodes', 'train_mask', 'val_mask', 'test_mask', 'edge_attr', 'raw_text']
    other_keys = [k for k in dataset.keys() if k not in common_keys]
    for data_obj in [dataset_ind, dataset_ood_tr, dataset_ood_te]:
        if data_obj is None: continue
        for key in other_keys:
            data_obj[key] = deepcopy(dataset[key]) # Deepcopy other attributes
            if key == "raw_texts":
                if hasattr(data_obj, "raw_text"):
                    continue
                else:
                    data_obj["raw_text"] = deepcopy(dataset[key]) 


    print("dataset_ind, dataset_ood_tr, dataset_ood_te", dataset_ind, dataset_ood_tr, dataset_ood_te)
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
    dataset_name = config.get('dataset_name', 'unknown')
    dataset_ood_te.raw_texts = shifted_texts_te
    dataset_ood_te.x = text_encoder.encode_texts(texts=shifted_texts_te, 
                                                 dataset_name = f"{getattr(dataset,'name','d')}_text_{augmentation_type}_{noise_level:.2f}_seed{random_seed}_te")

    # Generate for OOD Train set if needed
    if use_OE and dataset_ood_tr:
        shifted_texts_tr = shift_texts_internal(
            dataset.raw_texts, augmentation_type, noise_level, char_edit_prob, random_seed + 1, verbose
        )
        dataset_ood_tr.raw_texts = shifted_texts_tr
        dataset_ood_tr.x = text_encoder.encode_texts(texts=shifted_texts_tr, 
                                                     dataset_name = f"{getattr(dataset,'name','d')}_text_{augmentation_type}_{noise_level:.2f}_seed{random_seed}_tr")
        
    return dataset_ind, dataset_ood_tr, dataset_ood_te


@cache_datasets("{dataset_name}_text_swap_{swap_scope}")
def create_text_swap_datasets(
    dataset: Data,
    config: Dict[str, Any],
    text_encoder,
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
    """
    Creates OOD datasets by reconnecting edges based on node feature similarity.
    Maintains similar edge density to the original graph (scaled by `target_density_factor`).
    Selects edges by 'top', 'bottom', or percentile 'threshold' similarity.
    Features, labels, and texts remain unchanged.

    NOTE:
      - For 'threshold' mode, edges are selected around a percentile (e.g., 0.85).
      - For very large graphs, approximate FAISS flow or chunked/sampling strategies are used.
      - OOD Test uses the same semantic graph as OOD Train in this implementation.
    """
    if verbose:
        print(
            f"Creating semantic_connection shift datasets "
            f"(Mode: {selection_mode.upper()}, Target Density Factor: {target_density_factor:.2f})..."
        )

    dataset_ind = deepcopy(dataset)
    dataset_ood_te = deepcopy(dataset)
    dataset_ood_tr = deepcopy(dataset) if use_OE else None

    all_node_indices = torch.arange(dataset.num_nodes)
    dataset_ind.node_idx = all_node_indices
    dataset_ood_te.node_idx = all_node_indices
    if use_OE:
        dataset_ood_tr.node_idx = all_node_indices

    n = dataset.num_nodes
    is_very_large_graph = n > 50_000
    if is_very_large_graph and verbose:
        print(f"  Detected very large graph ({n:,} nodes). Using highly optimized strategies.")
        if use_approximate:
            print(f"  Using approximate methods (sampling_ratio={sampling_ratio:.4f}).")
        else:
            print("  Warning: Not using approximate methods. This may be very slow and memory intensive.")

    np.random.seed(random_seed)
    random.seed(random_seed)

    node_features = dataset.x.cpu().numpy()
    original_edge_index = dataset.edge_index
    num_orig_edges_undirected = 0
    if original_edge_index is not None and original_edge_index.numel() > 0:
        undirected_orig_edges = to_undirected(original_edge_index, num_nodes=n)
        num_orig_edges_undirected = undirected_orig_edges.size(1) // 2
        if dataset_ind.edge_index.size(1) != undirected_orig_edges.size(1):
            dataset_ind.edge_index = undirected_orig_edges
    else:
        print("Warning: Original dataset has no edges. Semantic graph will be built from scratch based on target density factor.")

    if is_very_large_graph:
        max_possible_edges = (n * (n - 1)) // 2
        if max_possible_edges < 0:
            max_possible_edges = float(n) * (n - 1) / 2
    else:
        max_possible_edges = (n * (n - 1)) // 2

    target_num_edges = int(round(num_orig_edges_undirected * target_density_factor))
    target_num_edges = min(target_num_edges, max_possible_edges)
    target_num_edges = max(0, target_num_edges)

    if verbose:
        print(f"  Original unique undirected edges: {num_orig_edges_undirected}")
        print(f"  Target unique undirected edges:  {target_num_edges} (Factor: {target_density_factor:.2f})")

    if target_num_edges == 0:
        if verbose:
            print("  Target number of edges is 0. Resulting OOD graphs will be empty.")
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        if use_OE:
            dataset_ood_tr.edge_index = empty_edge_index
        dataset_ood_te.edge_index = empty_edge_index

        # Copy through any extra attributes (retain exact original logic)
        targets = [dataset_ind, dataset_ood_te] if not use_OE else [dataset_ind, dataset_ood_tr, dataset_ood_te]
        for data_obj in targets:
            for key, value in dataset.items():
                if key not in ['x', 'edge_index', 'y', 'node_idx', 'num_nodes'] and key not in data_obj:
                    data_obj[key] = value

        return dataset_ind, dataset_ood_tr, dataset_ood_te

    # Decide computation mode
    if verbose:
        print(f"  Calculating pairwise {similarity_metric} similarity for {n} nodes...")

    if is_very_large_graph:
        use_approximate = True
        use_chunked_computation = False
        if verbose:
            print("  Using approximate methods for similarity computation...")
    elif n > 30_000:
        use_chunked_computation = True
        if verbose:
            print("  Using chunked computation for similarity calculation...")
    else:
        use_chunked_computation = False

    if verbose:
        print(f"  Using chunked computation: {use_chunked_computation}")

    # ---- Approximate path (FAISS / sampling) ----
    new_edge_index = None
    if use_approximate:
        try:
            import faiss
            FAISS_AVAILABLE = True
        except ImportError:
            FAISS_AVAILABLE = False
            if verbose:
                print("  Warning: FAISS library not available. Falling back to standard methods.")

        # GPU accel if possible
        use_gpu = False
        if FAISS_AVAILABLE:
            try:
                import faiss.contrib.torch_utils  # noqa: F401
                if torch.cuda.is_available():
                    use_gpu = True
                    if verbose:
                        print("  Using GPU acceleration for similarity computation.")
            except Exception:
                if verbose:
                    print("  FAISS GPU support not available. Using CPU.")

        if selection_mode == 'top' and FAISS_AVAILABLE:
            if verbose:
                print("  Using FAISS for approximate nearest neighbors search...")

            # Normalize for cosine sim
            features = node_features.astype(np.float32)
            faiss.normalize_L2(features)
            d = features.shape[1]

            # Build index
            if use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, d)
            else:
                index = faiss.IndexFlatIP(d)
            index.add(features)

            # Neighbors per node (approximate; capped as in original)
            k_per_node = min(1001, n)
            if verbose:
                print(f"  Searching for {k_per_node - 1} nearest neighbors per node...")

            neighbors_needed = min(k_per_node, int(2 * target_num_edges / n) + 5)
            neighbors_needed = max(2, neighbors_needed)

            all_edges = set()
            for start_idx in range(0, n, batch_size):
                end_idx = min(start_idx + batch_size, n)
                _, indices = index.search(features[start_idx:end_idx], neighbors_needed)

                # collect edges (skip self)
                for i in range(end_idx - start_idx):
                    node_idx = start_idx + i
                    for j in range(1, neighbors_needed):
                        if j < indices.shape[1]:
                            neighbor_idx = int(indices[i, j])
                            if neighbor_idx != node_idx and neighbor_idx < n:
                                a, b = (node_idx, neighbor_idx)
                                if a > b:
                                    a, b = b, a
                                all_edges.add((a, b))

            if verbose:
                print(f"  Found {len(all_edges):,} unique edges using approximate nearest neighbors.")

            if len(all_edges) > 0:
                edges_array = np.array(list(all_edges))
                src = torch.from_numpy(edges_array[:, 0]).long()
                dst = torch.from_numpy(edges_array[:, 1]).long()

                if len(src) > target_num_edges:
                    idx = torch.randperm(len(src))[:target_num_edges]
                    src = src[idx]
                    dst = dst[idx]

                pairs = torch.stack([src, dst], dim=0)
                new_edge_index = to_undirected(pairs, num_nodes=n)
            else:
                new_edge_index = torch.empty((2, 0), dtype=torch.long)

        elif selection_mode == 'bottom' and use_approximate:
            if verbose:
                print("  Using sampling approach to find dissimilar nodes...")

            num_samples = int(max_possible_edges * sampling_ratio)
            num_samples = min(num_samples, 100_000_000)
            num_samples = max(num_samples, target_num_edges * 10)
            if verbose:
                print(f"  Sampling {num_samples:,} random node pairs...")

            sampled_dissimilarities = []
            sampled_edges = []
            batch_samples = min(batch_size * 100, num_samples)

            for batch_start in range(0, num_samples, batch_samples):
                batch_end = min(batch_start + batch_samples, num_samples)
                count = batch_end - batch_start

                first_nodes = np.random.randint(0, n - 1, size=count)
                second_nodes = np.array([np.random.randint(i + 1, n) for i in first_nodes])

                # compute cosine similarities (vectorized loop kept as in original)
                for i in range(count):
                    u, v = first_nodes[i], second_nodes[i]
                    sim = cosine_similarity([node_features[u]], [node_features[v]])[0][0]
                    sampled_dissimilarities.append(-sim)  # negated for selecting most dissimilar
                    sampled_edges.append((u, v))

            sampled_dissimilarities = np.array(sampled_dissimilarities)
            k_to_select = min(target_num_edges, len(sampled_edges))
            if k_to_select < len(sampled_edges):
                top_idx = np.argpartition(sampled_dissimilarities, -k_to_select)[-k_to_select:]
                selected_edges = [sampled_edges[i] for i in top_idx]
            else:
                selected_edges = sampled_edges

            if verbose:
                print(f"  Selected {len(selected_edges):,} most dissimilar pairs from samples.")

            if selected_edges:
                edges_array = np.array(selected_edges)
                src = torch.from_numpy(edges_array[:, 0]).long()
                dst = torch.from_numpy(edges_array[:, 1]).long()
                pairs = torch.stack([src, dst], dim=0)
                new_edge_index = to_undirected(pairs, num_nodes=n)
            else:
                new_edge_index = torch.empty((2, 0), dtype=torch.long)

        elif selection_mode == 'threshold' and use_approximate:
            if verbose:
                print(f"  Using efficient sampling to find pairs around {threshold_percentile:.2%} threshold...")

            num_samples_for_threshold = min(1_000_000, int(max_possible_edges * 0.0001))
            if verbose:
                print(f"  Estimating threshold with {num_samples_for_threshold:,} samples...")

            # sample random upper triangle pairs
            first_indices = np.random.randint(0, n - 2, size=num_samples_for_threshold)
            offsets = np.random.randint(1, n - first_indices, size=num_samples_for_threshold)
            second_indices = first_indices + offsets

            f = node_features[first_indices]
            s = node_features[second_indices]

            # normalize for stable cosine
            f_norm = np.sqrt(np.sum(f * f, axis=1, keepdims=True))
            s_norm = np.sqrt(np.sum(s * s, axis=1, keepdims=True))
            f_n = f / (f_norm + 1e-8)
            s_n = s / (s_norm + 1e-8)
            sampled_sims = np.sum(f_n * s_n, axis=1)

            threshold_value = np.percentile(sampled_sims, threshold_percentile * 100)
            if verbose:
                print(f"  Estimated threshold at {threshold_percentile:.2%}: {threshold_value:.4f}")

            # progressive sampling around window
            window_size = 0.05
            lower_bound = max(-1.0, threshold_value - window_size)
            upper_bound = min(1.0, threshold_value + window_size)

            # precompute normalized features once
            if verbose:
                print("  Pre-computing normalized features (one-time operation)...")
            norms = np.sqrt(np.sum(node_features ** 2, axis=1, keepdims=True))
            norms[norms < 1e-8] = 1e-8
            normalized_features = node_features / norms
            del norms
            gc.collect()

            selected_edges: list[tuple[int, int]] = []
            max_samples = int(max_possible_edges * sampling_ratio)
            sample_count = 0
            micro_batch_size = 1000
            edges_found_counter = 0

            while len(selected_edges) < target_num_edges and sample_count < max_samples:
                first_nodes = np.random.randint(0, n, size=micro_batch_size)
                second_nodes = np.random.randint(0, n, size=micro_batch_size)

                valid = first_nodes != second_nodes
                first_nodes = first_nodes[valid]
                second_nodes = second_nodes[valid]

                swap = first_nodes > second_nodes
                tmp = first_nodes[swap].copy()
                first_nodes[swap] = second_nodes[swap]
                second_nodes[swap] = tmp
                del tmp, swap

                if len(first_nodes) == 0:
                    sample_count += micro_batch_size
                    continue

                similarities = np.zeros(len(first_nodes))
                mini_chunk = 100
                for j in range(0, len(first_nodes), mini_chunk):
                    j_end = min(j + mini_chunk, len(first_nodes))
                    fi = first_nodes[j:j_end]
                    si = second_nodes[j:j_end]
                    similarities[j:j_end] = np.sum(
                        normalized_features[fi] * normalized_features[si], axis=1
                    )

                mask = (lower_bound <= similarities) & (similarities <= upper_bound)
                if np.any(mask):
                    valid_f = first_nodes[mask]
                    valid_s = second_nodes[mask]
                    remaining = target_num_edges - len(selected_edges)
                    to_add = min(len(valid_f), remaining)
                    if to_add > 0:
                        for k in range(to_add):
                            selected_edges.append((int(valid_f[k]), int(valid_s[k])))
                        edges_found_counter += to_add

                # cleanup and progress
                del similarities, mask
                if 'valid_f' in locals():
                    del valid_f
                if 'valid_s' in locals():
                    del valid_s
                if sample_count % 100000 == 0:
                    gc.collect()

                sample_count += micro_batch_size

                # optional adaptive logging + window adjustment (kept as in original)
                if verbose and sample_count % 2_000_000 == 0:
                    target_rate = target_num_edges / max_samples if max_samples > 0 else 0
                    discovery_rate = edges_found_counter / 2_000_000
                    print(
                        f"  Progress: {min(100, sample_count / max_samples * 100 if max_samples else 100):.1f}% - "
                        f"Found {len(selected_edges):,}/{target_num_edges:,} edges "
                        f"(rate: {discovery_rate:.5f}, window: {lower_bound:.4f}-{upper_bound:.4f})"
                    )
                    edges_found_counter = 0
                    if target_rate > 0 and discovery_rate < target_rate * 0.1:
                        window_size *= 1.5
                        lower_bound = max(-1.0, threshold_value - window_size)
                        upper_bound = min(1.0, threshold_value + window_size)
                        if verbose:
                            print(f"  Widening window to {window_size:.4f}")
                    elif target_rate > 0 and discovery_rate > target_rate * 5.0 and window_size > 0.01:
                        window_size *= 0.8
                        lower_bound = max(-1.0, threshold_value - window_size)
                        upper_bound = min(1.0, threshold_value + window_size)
                        if verbose:
                            print(f"  Narrowing window to {window_size:.4f}")

            if len(selected_edges) > target_num_edges:
                random.shuffle(selected_edges)
                selected_edges = selected_edges[:target_num_edges]

            if verbose:
                print(
                    f"  Selected {len(selected_edges):,} edges around {threshold_percentile:.2%} threshold.\n"
                    f"  Final window size: {window_size:.4f} ({lower_bound:.4f} to {upper_bound:.4f})"
                )

            if selected_edges:
                edges_array = np.array(selected_edges)
                src = torch.from_numpy(edges_array[:, 0]).long()
                dst = torch.from_numpy(edges_array[:, 1]).long()
                pairs = torch.stack([src, dst], dim=0)
                new_edge_index = to_undirected(pairs, num_nodes=n)
            else:
                new_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            if verbose:
                print("  Falling back to chunked computation method...")
            use_chunked_computation = True

    if new_edge_index is None and use_chunked_computation:
        if verbose:
            print(f"  Using memory-efficient chunked computation (batch_size={batch_size})...")

        # 'top'/'bottom' use heaps; 'threshold' uses sampling bands
        if selection_mode in ['top', 'bottom']:
            import heapq

            k_to_select = min(target_num_edges, (n * (n - 1)) // 2)
            if selection_mode == 'top':
                heap = []
                heap_push = lambda val, i, j: heapq.heappush(heap, (-val, i, j))  # store as negative for max behavior
                worst_val = lambda: heap[0][0]
                better_mask_fn = lambda vals, worst: vals > -worst
                to_report = "HIGHEST"
                val_of = lambda x: -x
            else:  # 'bottom'
                heap = []
                heap_push = lambda val, i, j: heapq.heappush(heap, (val, i, j))
                worst_val = lambda: heap[0][0]
                better_mask_fn = lambda vals, worst: vals < worst
                to_report = "LOWEST"
                val_of = lambda x: x

            total_pairs = (n * (n - 1)) // 2
            pairs_processed = 0
            last_report = 0

            for start_idx in range(0, n, batch_size):
                end_idx = min(start_idx + batch_size, n)
                batch_features = node_features[start_idx:end_idx]
                batch_sim = cosine_similarity(batch_features, node_features)

                for row_local, (row_global, row_sim) in enumerate(zip(range(start_idx, end_idx), batch_sim)):
                    cols = np.arange(row_global + 1, n)
                    vals = row_sim[row_global + 1:]

                    if len(heap) < k_to_select:
                        for sim_val, col_idx in zip(vals, cols):
                            heap_push(sim_val, row_global, col_idx)
                    else:
                        worst = worst_val()
                        mask = better_mask_fn(vals, worst)
                        b_vals = vals[mask]
                        b_cols = cols[mask]
                        for sim_val, col_idx in zip(b_vals, b_cols):
                            heapq.heappop(heap)
                            heap_push(sim_val, row_global, col_idx)

                    pairs_processed += len(vals)
                    if verbose and pairs_processed - last_report >= 1_000_000:
                        # print progress occasionally (kept muted as in original)
                        last_report = pairs_processed

            sorted_pairs = [heapq.heappop(heap) for _ in range(len(heap))]
            if selection_mode == 'top':
                sorted_pairs.reverse()

            src = torch.tensor([p[1] for p in sorted_pairs], dtype=torch.long)
            dst = torch.tensor([p[2] for p in sorted_pairs], dtype=torch.long)
            pairs = torch.stack([src, dst], dim=0)
            new_edge_index = to_undirected(pairs, num_nodes=n)

            if verbose and len(sorted_pairs) > 0:
                min_val = val_of(sorted_pairs[-1][0])
                max_val = val_of(sorted_pairs[0][0])
                print(f"  Selected {len(sorted_pairs)} pairs with {to_report} similarity (range: {min_val:.4f} to {max_val:.4f})")

        elif selection_mode == 'threshold':
            if verbose:
                print("  Using sampling approach for threshold mode on large graph...")

            sample_size = min(10_000_000, (n * (n - 1)) // 2)
            sampled_similarities = np.zeros(sample_size)
            collected = 0
            last_report = 0
            gen_batch = min(100_000, sample_size)

            while collected < sample_size:
                take = min(gen_batch, sample_size - collected)
                rows = np.random.randint(0, n - 1, size=take)
                max_cols = n - rows
                col_offsets = np.random.rand(take) * (max_cols - 1) + 1
                cols = (rows + col_offsets).astype(np.int32)

                rf = node_features[rows]
                cf = node_features[cols]
                rn = np.linalg.norm(rf, axis=1)
                cn = np.linalg.norm(cf, axis=1)
                sims = np.sum(rf * cf, axis=1) / (rn * cn + 1e-12)

                sampled_similarities[collected:collected + take] = sims
                collected += take

                if verbose and collected - last_report >= 100_000:
                    last_report = collected

            thr_val = np.percentile(sampled_similarities, threshold_percentile * 100)
            if verbose:
                print(f"  Estimated similarity at {threshold_percentile:.2%} percentile: {thr_val:.4f}")
                print(f"  Sample similarity range: {sampled_similarities.min():.4f} to {sampled_similarities.max():.4f}")

            k_to_select = min(target_num_edges, (n * (n - 1)) // 2)
            window_size = 0.01
            max_window = 0.5

            # dynamic arrays (kept behavior)
            cap = min(k_to_select * 2, 1_000_000)
            sources = np.zeros(cap, dtype=np.int32)
            targets = np.zeros(cap, dtype=np.int32)
            count = 0

            while count < k_to_select and window_size <= max_window:
                lb = thr_val - window_size
                ub = thr_val + window_size
                count = 0

                for start_idx in range(0, n, batch_size):
                    end_idx = min(start_idx + batch_size, n)
                    batch_features = node_features[start_idx:end_idx]
                    batch_sim = cosine_similarity(batch_features, node_features)

                    for row_global, row_sim in zip(range(start_idx, end_idx), batch_sim):
                        cols = np.arange(row_global + 1, n)
                        vals = row_sim[row_global + 1:]
                        mask = (vals >= lb) & (vals <= ub)
                        match_cols = cols[mask]
                        num_m = len(match_cols)

                        if count + num_m > k_to_select:
                            remain = k_to_select - count
                            if num_m > 0:
                                sel = random.sample(range(num_m), remain)
                                match_cols = match_cols[sel]
                                num_m = remain

                        if count + num_m > len(sources):
                            new_cap = max(2 * len(sources), count + num_m)
                            sources = np.resize(sources, new_cap)
                            targets = np.resize(targets, new_cap)

                        sources[count:count + num_m] = row_global
                        targets[count:count + num_m] = match_cols
                        count += num_m

                        if count >= k_to_select:
                            break
                    if count >= k_to_select:
                        break

                if count < k_to_select:
                    window_size *= 2
                    if verbose:
                        print(f"  Expanding window to {window_size:.4f} (found {count:,}/{k_to_select:,} pairs)")

            src = torch.from_numpy(sources[:count]).long()
            dst = torch.from_numpy(targets[:count]).long()
            pairs = torch.stack([src, dst], dim=0)
            new_edge_index = to_undirected(pairs, num_nodes=n)
            if verbose:
                print(f"  Selected {count} pairs around the {threshold_percentile:.2%} threshold")
                print(f"  Final window size: {window_size:.4f}")

    if new_edge_index is None and not use_chunked_computation and not use_approximate:
        sim_mat = cosine_similarity(node_features)
        diag_fill_value = -np.inf if selection_mode == 'top' else np.inf
        np.fill_diagonal(sim_mat, diag_fill_value)
        if verbose:
            print("  Similarity calculation complete.")

        r, c = np.triu_indices(n, k=1)
        sims = sim_mat[r, c]
        num_pairs = len(sims)
        k_to_select = min(target_num_edges, num_pairs)

        if k_to_select > 0:
            if selection_mode == 'top':
                idx = np.argpartition(sims, -k_to_select)[-k_to_select:] if k_to_select < num_pairs else np.arange(num_pairs)
                if verbose:
                    print(f"  Selecting {k_to_select} pairs with HIGHEST similarity.")
            elif selection_mode == 'bottom':
                idx = np.argpartition(sims, k_to_select)[:k_to_select] if k_to_select < num_pairs else np.arange(num_pairs)
                if verbose:
                    print(f"  Selecting {k_to_select} pairs with LOWEST similarity.")
            elif selection_mode == 'threshold':
                if k_to_select < num_pairs:
                    sorted_idx = np.argsort(sims)
                    thr_idx = int(num_pairs * threshold_percentile)
                    half = k_to_select // 2
                    extra = k_to_select % 2
                    start = max(0, thr_idx - half)
                    end = min(num_pairs, thr_idx + half + extra)
                    if start == 0:
                        end = k_to_select
                    elif end == num_pairs:
                        start = num_pairs - k_to_select
                    idx = sorted_idx[start:end]
                    if verbose:
                        thr_val = sims[sorted_idx[thr_idx]]
                        print(f"  Selecting {k_to_select} pairs around the {threshold_percentile:.2%} similarity threshold.")
                        print(f"  Threshold similarity value: {thr_val:.4f}")
                else:
                    idx = np.arange(num_pairs)
                    if verbose:
                        print(f"  Selecting all {k_to_select} available pairs (threshold not applicable).")

            src = torch.from_numpy(r[idx]).long()
            dst = torch.from_numpy(c[idx]).long()
            pairs = torch.stack([src, dst], dim=0)
            new_edge_index = to_undirected(pairs, num_nodes=n)
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long)

    actual_new_edges = new_edge_index.size(1) // 2
    if verbose:
        print(f"  Constructed new undirected edge index with {actual_new_edges} unique edges.")

    if use_OE and dataset_ood_tr is not None:
        dataset_ood_tr.edge_index = new_edge_index
    dataset_ood_te.edge_index = new_edge_index

    if dataset_ind.y is not None:
        try:
            h_orig = homophily(
                dataset_ind.edge_index, dataset_ind.y, method='edge'
            ) if num_orig_edges_undirected > 0 else float('nan')
            h_new = homophily(
                new_edge_index, dataset_ind.y, method='edge'
            ) if actual_new_edges > 0 else float('nan')
            if verbose:
                print(f"  Homophily (Edge) - Original: {h_orig:.4f}, Semantic Rewired ({selection_mode}): {h_new:.4f}")
        except Exception as e:
            print(f"  Could not calculate homophily: {e}")
    else:
        if verbose:
            print("  Skipping homophily calculation as original labels are missing.")

    if verbose:
        print("Finished creating semantic_connection shift datasets.")

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

    id_node_idx = np.load("../datasets/arxiv/arxiv_ind_node_idx.npy", allow_pickle=True)
    train_idx = np.load("../datasets/arxiv/arxiv_ind_node_tr_idx.npy", allow_pickle=True)
    valid_idx = np.load("../datasets/arxiv/arxiv_ind_node_val_idx.npy", allow_pickle=True)
    test_idx = np.load("../datasets/arxiv/arxiv_ind_node_te_idx.npy", allow_pickle=True)

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
        idx =  np.load(f"../datasets/arxiv/arxiv_ood_te_{i}_idx.npy")
        idx = torch.tensor(idx)
        dataset_ood_te[i].node_idx = idx

    for mask_key in ['train_mask', 'val_mask', 'test_mask']:
        dataset_ind[mask_key] = masks[mask_key]

    return dataset_ind, dataset_ood_tr, dataset_ood_te