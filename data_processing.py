import torch
import time
import warnings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from torch_geometric.data import Data

warnings.filterwarnings("ignore", category=UserWarning, module='sentence_transformers')
warnings.filterwarnings("ignore", category=FutureWarning)

class TextEncoder:
    """Handles encoding of text data into embeddings using Sentence Transformers."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None, verbose: bool = True, random_seed: int = 42):
        self.model_name = model_name
        self.safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        self._verbose = verbose
        self.random_seed = random_seed

        self._set_random_seed(random_seed)

        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self._verbose:
            print(f"Initializing TextEncoder: Model='{model_name}', Device='{self.device}', Seed={random_seed}")

        try:
            self.encoder = SentenceTransformer(model_name, device=self.device)
            self._embedding_dim = self.encoder.get_sentence_embedding_dimension()
            if self._verbose:
                print(f"SentenceTransformer model loaded. Embedding dimension: {self._embedding_dim}")
        except Exception as e:
            raise RuntimeError(f"Error loading Sentence Transformer model '{model_name}': {e}") from e

    def _set_random_seed(self, seed: int):
        """Sets random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_embedding_filepath(self, dataset_name: str, embedding_base_path: Path) -> Path:
        """Constructs the filepath for cached embeddings."""
        filename = f"{dataset_name}_{self.safe_model_name}_seed{self.random_seed}.pt"

        embedding_base_path = Path(embedding_base_path)
        return embedding_base_path / filename

    def encode_texts(self,
                     texts: List[str],
                     dataset_name: str,
                     embedding_base_path: Path,
                     use_cache: bool = True
                     ) -> torch.Tensor:
        """
        Converts text list to embeddings. Loads from cache if available and use_cache=True,
        otherwise computes and saves to cache. Returns embeddings on CPU.
        """
        if not texts:
            return torch.empty((0, self._embedding_dim), dtype=torch.float, device='cpu')

        embedding_file = self._get_embedding_filepath(dataset_name, embedding_base_path)

        # Load from Cache
        if use_cache and embedding_file.exists():
            if self._verbose: print(f"Loading cached embeddings: {embedding_file}")
            try:
                embeddings = torch.load(embedding_file, map_location='cpu')
                if isinstance(embeddings, torch.Tensor) and \
                   embeddings.shape[0] == len(texts) and \
                   embeddings.shape[1] == self._embedding_dim:
                    if self._verbose: print(f"Cache loaded successfully. Shape: {embeddings.shape}")
                    return embeddings.float()
                else:
                    print("Warning: Cached embedding validation failed. Recomputing...")
            except Exception as e:
                print(f"Warning: Failed to load cached embeddings ({e}). Recomputing...")

        if self._verbose: print(f"Encoding {len(texts)} texts using '{self.model_name}' on {self.device}...")
        start_time = time.time()

        self.encoder.to(self.device)
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=self._verbose,
                device=self.device
            )

        embeddings = embeddings.cpu().float()
        duration = time.time() - start_time
        if self._verbose: print(f"Encoding finished in {duration:.2f}s. Shape: {embeddings.shape}")

        # Save to Cache
        if use_cache:
            try:
                embedding_base_path = Path(embedding_base_path)
                embedding_base_path.mkdir(parents=True, exist_ok=True)
                torch.save(embeddings, embedding_file)
                if self._verbose: print(f"Embeddings saved to cache: {embedding_file}")
            except Exception as e:
                print(f"Warning: Failed to save embeddings to cache ({e}).")

        return embeddings


class DatasetProcessor:
    """Handles validation and preparation of graph datasets."""

    @staticmethod
    def prepare_dataset(dataset: Data, verbose: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Prepares the dataset: ensures boolean masks, calculates num_classes, gathers info.
        """
        if verbose: print("Preparing dataset...")
        if not hasattr(dataset, 'y') or dataset.y is None:
            raise ValueError("Dataset missing labels 'y'.")
        if not hasattr(dataset, 'edge_index') or dataset.edge_index is None:
             print("Warning: Dataset missing 'edge_index'.")
        if not hasattr(dataset, 'num_nodes') or dataset.num_nodes == 0:
             print("Warning: Dataset reports 0 nodes or missing 'num_nodes'. Relying on label count.")
             dataset.num_nodes = len(dataset.y) if dataset.y is not None else 0

        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(dataset, mask_key) and dataset[mask_key] is not None:
                if dataset[mask_key].dtype != torch.bool:
                    if verbose: print(f"Converting '{mask_key}' to boolean type.")
                    dataset[mask_key] = dataset[mask_key].to(torch.bool)

        labels = dataset.y.squeeze()
        num_classes = 0
        class_distribution = []
        if labels.numel() > 0:
            if labels.is_floating_point():
                print("Warning: Labels are floating point. Attempting conversion to long.")
                try:
                    labels = labels.long()
                except Exception as e:
                     raise ValueError(f"Could not convert float labels to long: {e}")

            unique_labels = torch.unique(labels)
            num_classes = len(unique_labels)
            try:
                class_distribution = [int((labels == c).sum().item()) for c in sorted(unique_labels.tolist())]
            except Exception:
                 print("Warning: Could not compute class distribution.")
        else:
            print("Warning: Dataset labels are empty.")

        train_nodes = int(dataset.train_mask.sum().item()) if hasattr(dataset, 'train_mask') and dataset.train_mask is not None else 0
        val_nodes = int(dataset.val_mask.sum().item()) if hasattr(dataset, 'val_mask') and dataset.val_mask is not None else 0
        test_nodes = int(dataset.test_mask.sum().item()) if hasattr(dataset, 'test_mask') and dataset.test_mask is not None else 0

        dataset_info = {
            "num_nodes": dataset.num_nodes,
            "num_edges": dataset.num_edges if hasattr(dataset, 'num_edges') else 0,
            "num_classes": num_classes,
            "train_nodes": train_nodes,
            "val_nodes": val_nodes,
            "test_nodes": test_nodes,
            "class_distribution": class_distribution,
            "is_undirected": dataset.is_undirected() if hasattr(dataset, 'is_undirected') else "N/A"
        }

        if verbose:
            print("\n--- Dataset Info ---")
            for key, val in dataset_info.items():
                 print(f"  {key.replace('_', ' ').title()}: {val}")
            print("--------------------")

        return num_classes, dataset_info


def load_dataset(dataset_name: str, dataset_base_path: Path, verbose: bool = True) -> Data:
    """Loads a preprocessed PyTorch Geometric dataset object from disk."""

    dataset_base_path = Path(dataset_base_path)
    
    dataset_file = dataset_base_path / dataset_name / "geometric_data_processed.pt"
    if verbose: print(f"Attempting to load dataset from: {dataset_file}")

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    try:
        data_obj = torch.load(dataset_file, map_location='cpu')

        if isinstance(data_obj, (list, tuple)):
            if not data_obj: raise ValueError(f"Loaded object from {dataset_file} is an empty list/tuple.")
            dataset = data_obj[0]
        elif isinstance(data_obj, dict) and 'data' in data_obj:
            dataset = data_obj['data']
        elif isinstance(data_obj, Data):
            dataset = data_obj
        else:
            raise TypeError(f"Loaded object from {dataset_file} is not a recognized PyG Data format (type: {type(data_obj)}).")

        if not isinstance(dataset, Data):
             raise TypeError(f"Loaded object is not a PyG Data object (type: {type(dataset)}).")

        if verbose: print(f"Dataset '{dataset_name}' loaded successfully.")
        return dataset

    except Exception as e:
        print(f"Error loading dataset from {dataset_file}: {e}")
        import traceback
        traceback.print_exc()
        raise