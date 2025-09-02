import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from metrics import get_ood_metrics
from models import GCN

class OODBaseline:
    """Base class for OOD detection baseline methods."""

    def __init__(self, model: GCN, config: Dict[str, Any]):
        self.model = model.eval()  # Ensure model is in eval mode
        self.config = config
        self.name = "Base"
        self.device = next(model.parameters()).device

    def compute_scores(self, data: Data) -> torch.Tensor:
        """Compute OOD detection scores. Higher score = more OOD."""
        raise NotImplementedError

    def evaluate(self, ind_data: Data, ood_datasets: List[Data]) -> Dict[str, Any]:
        """Evaluate OOD detection performance."""
        self.model.eval()
        ind_data = ind_data.to(self.device)
        ind_scores = self.compute_scores(ind_data).to(self.device)

        if not hasattr(ind_data, "test_mask") or ind_data.test_mask is None or ind_data.test_mask.sum() == 0:
            print(f"Warning ({self.name}): IND data missing test_mask. Using all nodes for ID scores.")
            ind_scores_test = ind_scores
        else:
            ind_scores_test = ind_scores[ind_data.test_mask]

        if ind_scores_test.numel() == 0:
            print(f"Error ({self.name}): No ID scores could be computed based on test mask.")
            return {"average": {"auroc": 0.0, "aupr": 0.0, "fpr95": 1.0}, "datasets": []}

        all_results: List[Dict[str, Any]] = []
        all_ood_scores: List[np.ndarray] = []

        for i, ood_data in enumerate(ood_datasets):
            ood_data = ood_data.to(self.device)
            ood_scores_all = self.compute_scores(ood_data).to(self.device)

            if not hasattr(ood_data, "node_idx") or ood_data.node_idx is None:
                print(f"Warning ({self.name}): OOD dataset {i} missing node_idx. Using all nodes.")
                ood_scores_target = ood_scores_all
            else:
                ood_target_indices = ood_data.node_idx.to(self.device, dtype=torch.long)
                valid_mask = ood_target_indices < ood_scores_all.shape[0]
                valid_indices = ood_target_indices[valid_mask]
                if valid_indices.numel() > 0:
                    ood_scores_target = ood_scores_all[valid_indices]
                else:
                    print(f"Warning ({self.name}): OOD dataset {i} node_idx contained only invalid indices.")
                    ood_scores_target = torch.tensor([], device=self.device)

            all_ood_scores.append(ood_scores_target.detach().cpu().numpy())

            if ood_scores_target.numel() == 0:
                print(f"Warning ({self.name}): No OOD scores computed for dataset {i}. Skipping.")
                continue

            auroc, aupr, fpr95 = get_ood_metrics(
                ind_scores_test.detach().cpu().numpy(),
                ood_scores_target.detach().cpu().numpy(),
            )

            all_results.append(
                {
                    "dataset_index": i,
                    "auroc": auroc,
                    "aupr": aupr,
                    "fpr95": fpr95,
                    "id_score_mean": ind_scores_test.mean().item(),
                    "ood_score_mean": ood_scores_target.mean().item(),
                }
            )
            print(f"{self.name} - Dataset {i}:")
            print(f"    Auroc: {auroc * 100:.2f}")
            print(f"    AUPR: {aupr * 100:.2f}")
            print(f"    FPR95: {fpr95 * 100:.2f}")

        # Average metrics (preserve original default behavior)
        avg_metrics = {"auroc": 0.0, "aupr": 0.0, "fpr95": 1.0}
        if all_results:
            avg_metrics = {k: float(np.mean([r[k] for r in all_results])) for k in ["auroc", "aupr", "fpr95"]}

        return {"average": avg_metrics, "datasets": all_results}


class MSPBaseline(OODBaseline):
    """Maximum Softmax Probability (MSP) baseline."""

    def __init__(self, model: GCN, config: Dict[str, Any]):
        super().__init__(model, config)
        self.name = "MSP"

    @torch.no_grad()
    def compute_scores(self, data: Data) -> torch.Tensor:
        logits, _ = self.model(data)
        softmax_probs = F.softmax(logits, dim=1)
        max_probs, _ = softmax_probs.max(dim=1)
        # Original MSP uses max prob (lower = OOD). Negate for consistency (higher = OOD).
        return -max_probs


class EnergyBaseline(OODBaseline):
    """Energy-based baseline."""

    def __init__(self, model: GCN, config: Dict[str, Any], name: str = "Energy", use_propagation: bool = False):
        super().__init__(model, config)
        self.name = name
        self.temperature = config.get("T", 1.0)
        self.use_propagation = use_propagation
        if use_propagation:
            self.prop_layers = config.get("K", 3)
            self.alpha = config.get("alpha", 0.5)

    @torch.no_grad()
    def compute_scores(self, data: Data) -> torch.Tensor:
        self.model.eval()
        energy = self.model.calculate_energy(data, self.temperature)
        if self.use_propagation:
            energy = GCN.propagation(energy, data.edge_index, self.prop_layers, self.alpha)
        # Higher energy score indicates more OOD-like
        return energy


class NecoBaseline(OODBaseline):
    """
    OOD detection baseline using PCA energy retention ratio.
    Fits PCA on ID training features and scores nodes based on how much
    energy is retained when projecting onto top principal components.
    Assumes LOWER ratio indicates OOD. Score returned is NEGATIVE ratio (scaled by max logit).
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.name = "Neco"
        # Config (preserve defaults and behavior)
        self.use_scaler: bool = config.get("pca_use_scaler", True)
        self.n_components: int = config.get("pca_n_components", 50)  # kept (not used in original fit)
        self.feature_layer: int = config.get("pca_feature_layer", -2)

        # Fitted components
        self.scaler: Optional[StandardScaler] = None if not self.use_scaler else StandardScaler()
        self.pca: Optional[PCA] = None
        self.fitted: bool = False
        self.feature_dim_: int = -1

    def _get_features(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract (logits, features_for_pca) from the specified layer. Features returned on CPU."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(data.to(self.device))

        features: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None

        if isinstance(output, tuple) and len(output) == 2:
            logits, hidden = output
            if self.feature_layer == -1:
                features = logits
            elif isinstance(hidden, torch.Tensor):
                if self.feature_layer == -2:
                    features = hidden
                else:
                    print(f"Warning ({self.name}): Invalid feature_layer {self.feature_layer} for GCN hidden state.")
            elif isinstance(hidden, dict):
                num_feature_outputs = len(hidden.get("features", []))
                if self.feature_layer < 0 and abs(self.feature_layer) <= num_feature_outputs:
                    features = hidden["features"][self.feature_layer]
                else:
                    print(
                        f"Warning ({self.name}): Invalid feature_layer {self.feature_layer} for HyperCollapse. "
                        f"Num layers: {num_feature_outputs}"
                    )
            else:
                print(f"Warning ({self.name}): Unexpected model output format.")
        elif isinstance(output, torch.Tensor):
            logits = output
            if self.feature_layer == -1:
                features = output
            else:
                print(
                    f"Warning ({self.name}): Model only returned logits, cannot use feature_layer {self.feature_layer}."
                )

        if features is None or logits is None:
            print(f"Error ({self.name}): Could not extract features for layer {self.feature_layer}.")
            raise RuntimeError("Feature extraction failed.")

        return logits, features.detach().cpu()

    def fit(self, train_data: Data) -> None:
        """Fit StandardScaler and PCA on training data features."""
        print(f"Fitting {self.name} statistics...")
        self.model.eval()
        train_mask = train_data.train_mask.to("cpu")

        _, all_features = self._get_features(train_data)
        train_features_np = all_features[train_mask].numpy()
        self.feature_dim_ = train_features_np.shape[1]

        # 1) Fit scaler (kept unconditional if self.use_scaler as in original)
        print(f"  ({self.name}) Fitting StandardScaler...")
        self.scaler.fit(train_features_np)
        scaled_features = self.scaler.transform(train_features_np)

        # 2) Fit PCA (preserve original: use max feasible components)
        max_components = min(len(train_features_np), train_features_np.shape[1])
        print(f"  ({self.name}) Fitting PCA... {max_components} components...")
        self.pca = PCA(max_components)
        _ = self.pca.fit_transform(scaled_features)
        print("Fitted PCA successfully.")
        self.fitted = True

    def compute_scores(self, data: Data) -> torch.Tensor:
        """Compute PCA energy retention ratio score (negated and scaled by max logit)."""
        if not self.fitted or self.pca is None:
            raise RuntimeError(f"{self.name} must be fit() before compute_scores()")

        self.model.eval()
        logits, features_cpu = self._get_features(data)
        features_np = features_cpu.numpy()

        # 1) Scale features (optional)
        if self.use_scaler and self.scaler:
            scaled_features = self.scaler.transform(features_np)
        else:
            scaled_features = features_np

        # 2) Project onto top PCA components (preserve “top 100” usage) to ensure consistency and speed
        reduced_all = self.pca.transform(scaled_features)  # [N, n_components]
        score_maxlogit = logits.max(dim=-1).values.detach().cpu().numpy()
        reduced = reduced_all[:, :100]

        l_score: List[float] = []
        for i in range(reduced.shape[0]):
            sc_complete = np.linalg.norm(scaled_features[i, :])
            sc = np.linalg.norm(reduced[i, :])
            sc_final = sc / sc_complete if sc_complete != 0 else 0.0
            l_score.append(sc_final)

        score = np.array(l_score, dtype=np.float32)
        score *= -score_maxlogit  # Negate ratio and scale by confidence (unchanged behavior)

        return torch.from_numpy(score).float()


class ODINBaseline(OODBaseline):
    """ODIN baseline using input perturbation."""

    def __init__(self, model: GCN, config: Dict[str, Any]):
        super().__init__(model, config)
        self.name = "ODIN"
        self.temperature = config.get("T", 1.0)
        self.noise_magnitude = config.get("baseline_noise", 0.0014)

    def compute_scores(self, data: Data) -> torch.Tensor:
        self.model.eval()

        features = data.x.clone().detach().to(self.device)
        features.requires_grad = True
        edge_index = data.edge_index.to(self.device)

        outputs, _ = self.model.forward_with_features(features, edge_index)
        outputs_scaled = outputs / self.temperature

        labels = outputs.detach().argmax(dim=1)
        loss = F.cross_entropy(outputs_scaled, labels)

        self.model.zero_grad()
        grad = autograd.grad(loss, features, retain_graph=False, create_graph=False)[0]

        perturbed_features = features - self.noise_magnitude * grad.sign()

        with torch.no_grad():
            outputs_perturbed, _ = self.model.forward_with_features(perturbed_features, edge_index)
            outputs_perturbed_scaled = outputs_perturbed / self.temperature
            softmax_probs_perturbed = F.softmax(outputs_perturbed_scaled, dim=1)
            max_probs_perturbed, _ = softmax_probs_perturbed.max(dim=1)

        # Lower prob = OOD; negate for consistency
        return -max_probs_perturbed


class MahalanobisBaseline(OODBaseline):
    """Low-memory Mahalanobis OOD detection baseline."""

    def __init__(self, model: GCN, config: Dict[str, Any]):
        super().__init__(model, config)
        self.name = "Mahalanobis"
        self.noise_magnitude = config.get("baseline_noise", 0.0)  # kept for parity; not used here
        self.sample_class_mean: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.num_classes: Optional[int] = None
        self.fitted = False

    def fit(self, train_data: Data) -> None:
        print(f"Fitting {self.name} statistics...")
        self.model.eval()
        train_data = train_data.to(self.device)

        if not hasattr(train_data, "train_mask") or train_data.train_mask is None:
            raise ValueError("train_mask is required.")

        with torch.no_grad():
            _, features = self.model(train_data)

        train_features = features[train_data.train_mask].cpu()
        train_labels = train_data.y[train_data.train_mask].cpu()
        dim = train_features.size(1)
        self.num_classes = int(train_labels.max().item()) + 1

        self.sample_class_mean = torch.zeros(self.num_classes, dim)
        for c in range(self.num_classes):
            mask = train_labels == c
            if mask.any():
                self.sample_class_mean[c] = train_features[mask].mean(0)

        cov = torch.zeros(dim, dim)
        total_samples = 0
        for c in range(self.num_classes):
            mask = train_labels == c
            if mask.any():
                centered = train_features[mask] - self.sample_class_mean[c]
                cov += centered.t().mm(centered)
                total_samples += int(mask.sum().item())

        cov /= (total_samples - 1)
        cov += torch.eye(dim) * 1e-5  # regularization

        try:
            precision = torch.linalg.inv(cov)
        except RuntimeError:
            print("Using pseudo-inverse")
            precision = torch.linalg.pinv(cov)

        self.sample_class_mean = self.sample_class_mean.to(self.device)
        self.precision = precision.to(self.device)
        self.fitted = True
        print(f"{self.name} fitting complete.")

    def compute_scores(self, data: Data) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Mahalanobis baseline must be fit() before compute_scores()")

        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            _, features = self.model(data)

        # CPU scoring to keep memory low (as before)
        features = features.detach().cpu()
        sample_class_mean = self.sample_class_mean.detach().cpu()  # type: ignore[union-attr]
        precision = self.precision.detach().cpu()  # type: ignore[union-attr]

        best_scores = torch.full((features.size(0),), float("inf"))
        for c in range(self.num_classes or 0):
            diff = features - sample_class_mean[c]
            maha = torch.sum((diff @ precision) * diff, dim=1)
            best_scores = torch.min(best_scores, maha)

        return -best_scores.to(self.device)


def create_baseline(baseline_name: str, model: GCN, config: Dict[str, Any]) -> OODBaseline:
    """Factory function to create OOD baselines."""
    name_lower = baseline_name.lower()
    if name_lower == "msp":
        return MSPBaseline(model, config)
    if name_lower == "energy":
        return EnergyBaseline(model, config, name="Energy", use_propagation=False)
    if name_lower == "gnnsafe":
        return EnergyBaseline(model, config, name="GNNSafe", use_propagation=True)
    if name_lower == "nodesafe":
        return EnergyBaseline(model, config, name="NodeSafe", use_propagation=True)
    if name_lower == "odin":
        return ODINBaseline(model, config)
    if name_lower == "mahalanobis":
        return MahalanobisBaseline(model, config)
    if name_lower == "neco":
        return NecoBaseline(model, config)
    raise ValueError(f"Unknown baseline method: {baseline_name}")


def run_baseline_evaluation(
    model: GCN,
    ind_data: Data,
    ood_datasets: List[Data],
    baseline_names: List[str],
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run evaluation for multiple OOD detection baselines."""
    all_baseline_results: Dict[str, Any] = {}
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)

    for baseline_name in baseline_names:
        try:
            start_time = time.time()
            baseline = create_baseline(baseline_name, model, config)
            print(f"\n--- Evaluating Baseline: {baseline.name} ---")

            if isinstance(baseline, MahalanobisBaseline) and not baseline.fitted:
                baseline.fit(ind_data.to(device))
            elif isinstance(baseline, NecoBaseline) and not baseline.fitted:
                baseline.fit(ind_data.to(device))

            results = baseline.evaluate(ind_data, ood_datasets)
            all_baseline_results[baseline.name] = results

            end_time = time.time()
            print(f"Baseline evaluation time: {end_time - start_time:.2f} seconds")

        except ImportError as e:
            print(f"Skipping baseline {baseline_name}: {e}")
        except Exception as e:
            print(f"Error evaluating baseline {baseline_name}: {e}")
            traceback.print_exc()
            all_baseline_results[baseline_name] = {
                "average": {"auroc": 1, "aupr": 1, "fpr95": 1.0},
                "datasets": [
                    {
                        "dataset_index": 0,
                        "auroc": 1,
                        "aupr": 1,
                        "fpr95": 1,
                        "id_score_mean": 0,
                        "ood_score_mean": 0,
                    }
                ],
            }

    return all_baseline_results