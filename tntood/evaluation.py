import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from models import GCN
from metrics import get_ood_metrics
from torch_geometric.data import Data

try:
    import wandb
except ImportError:
    wandb = None

class OODDetector:
    """Handles Out-of-Distribution (OOD) Detection using energy scores."""

    def __init__(self, model: GCN, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.use_wandb = config.get('use_wandb', False) and wandb is not None
        self.verbose = config.get('verbose', True)

    @torch.no_grad()
    def _get_energies(self, data: Data) -> torch.Tensor:
        """Calculates energy scores, optionally applying propagation."""
        self.model.eval()
        temperature = self.config.get('T', 1.0)
        data = data.to(self.device)
        energy = self.model.calculate_energy(data, temperature)
        if self.config.get('use_prop', False):
            prop_layers = self.config.get('K', 3)
            alpha = self.config.get('alpha', 0.5)
            energy = energy.to(self.device)
            data = data.to(self.device)
            
            energy = GCN.propagation(energy, data.edge_index, prop_layers, alpha)
        
        return energy

    @torch.no_grad()
    def _get_accuracy(self, data: Data, target_indices: Optional[torch.Tensor] = None) -> Tuple[float, int]:
        """Calculates classification accuracy for the specified nodes."""
        self.model.eval()
        logits, _ = self.model(data)
        pred = logits.argmax(dim=1)

        if target_indices is None:
            if hasattr(data, 'test_mask') and data.test_mask is not None:
                target_indices = data.test_mask.nonzero(as_tuple=True)[0]
            else:
                target_indices = torch.arange(data.num_nodes, device=self.device)

        if not hasattr(data, 'y') or data.y is None or target_indices.numel() == 0:
            return 0.0, 0

        y_true = data.y[target_indices].squeeze()
        y_pred = pred[target_indices]
        correct = (y_pred == y_true).sum().item()
        total = len(target_indices)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy, total

    def evaluate_ood(self, ind_data_eval: Data, ood_datasets: List[Data], shift_name: str = "Unknown") -> Optional[Dict[str, Any]]:
        """Evaluates OOD detection performance."""
        if self.verbose: print(f"\n--- OOD Detection Evaluation: {shift_name.upper()} SHIFT ---")
        self.model.eval()
        if not hasattr(ind_data_eval, 'test_mask') or ind_data_eval.test_mask is None or ind_data_eval.test_mask.sum() == 0:
            print("Error: In-distribution data lacks a valid 'test_mask'. Cannot evaluate OOD.")
            return None

        all_energy_id = self._get_energies(ind_data_eval)
        id_accuracy, id_accuracy_count = self._get_accuracy(ind_data_eval)
        if self.verbose:
            print(f"ID Accuracy: {id_accuracy:.2f}% ({id_accuracy_count} nodes)")

        energy_id_test = all_energy_id[ind_data_eval.test_mask].cpu().numpy()
        if energy_id_test.size == 0:
            print("Warning: No ID test nodes found. Cannot evaluate OOD.")
            return None
        if self.verbose: print(f"Calculated energy for {len(energy_id_test)} ID test nodes.")

        all_ood_results = []
        if not ood_datasets:
            print("Warning: No OOD datasets provided.")
            return {'average': {'auroc': 0.0, 'aupr': 0.0, 'fpr95': 1.0, 'accuracy': 0.0}, 'datasets': []}

        for i, ood_data in enumerate(ood_datasets):
            if self.verbose: print(f"Processing OOD dataset {i+1}/{len(ood_datasets)}...")
            try:
                ood_data.to(self.device)
                all_energy_ood = self._get_energies(ood_data)
                ood_target_indices = ood_data.node_idx.to(self.device, dtype=torch.long)
                
                energy_ood_target = np.array([])
                if ood_target_indices.numel() > 0:
                    valid_mask = ood_target_indices < all_energy_ood.shape[0]
                    valid_indices = ood_target_indices[valid_mask]
                    if valid_indices.numel() > 0:
                        energy_ood_target = all_energy_ood[valid_indices].cpu().numpy()
                    else:
                        print(f"Warning: OOD dataset {i+1} 'node_idx' contained only out-of-bounds indices.")

                ood_accuracy = 0.0
                ood_accuracy_count = 0
                if shift_name != "label":
                    ood_accuracy, ood_accuracy_count = self._get_accuracy(ood_data, ood_target_indices)

                if energy_ood_target.size == 0:
                    print(f"  No OOD target node energies for dataset {i+1}. Skipping metrics.")
                    continue

                if self.verbose:
                    if shift_name != "label":
                        print(f"  OOD Accuracy: {ood_accuracy:.2f}% ({ood_accuracy_count} nodes)")

                # Higher energy -> more OOD is the assumption from calculate_energy
                auroc, aupr, fpr95 = get_ood_metrics(energy_id_test, energy_ood_target)

                dataset_result = {
                    'dataset_index': i, 'auroc': auroc, 'aupr': aupr, 'fpr95': fpr95,
                    'accuracy': ood_accuracy
                }
                all_ood_results.append(dataset_result)

                print("\n In evaluation Dataset {i+1} OOD Performance:")
                print(f"  AUROC: {auroc * 100:.2f}%")
                print(f"  AUPR: {aupr * 100:.2f}%")
                print(f"  FPR95: {fpr95 * 100:.2f}%")
                print(f"  OOD Accuracy: {ood_accuracy:.2f}%")

                if self.use_wandb and wandb.run:
                    try:
                        wandb.log({
                            f'ood/{shift_name}/ds_{i}/auroc': auroc, f'ood/{shift_name}/ds_{i}/aupr': aupr,
                            f'ood/{shift_name}/ds_{i}/fpr95': fpr95, f'ood/{shift_name}/ds_{i}/accuracy': ood_accuracy,
                        }, commit=False)
                    except Exception as log_e:
                        print(f"Wandb Error logging OOD dataset {i} results: {log_e}")

            except Exception as e:
                print(f"Error processing OOD dataset {i+1}: {e}")
                import traceback
                traceback.print_exc()

        # --- Calculate Average Metrics & Final Logging ---
        final_results = {'average': {'auroc': 0.0, 'aupr': 0.0, 'fpr95': 1.0, 'accuracy': 0.0},
                         'datasets': all_ood_results, 'id_accuracy': id_accuracy}

        if len(all_ood_results) > 1:
            avg_metrics = {key: np.mean([r[key] for r in all_ood_results if key in r])
                           for key in ['auroc', 'aupr', 'fpr95', 'accuracy']}
            final_results['average'] = avg_metrics

            print("\nAverage OOD Performance:")
            print(f"  AUROC: {avg_metrics['auroc'] * 100:.2f}")
            print(f"  AUPR: {avg_metrics['aupr'] * 100:.2f}")
            print(f"  FPR95: {avg_metrics['fpr95'] * 100:.2f}")
            print(f"  OOD Accuracy: {avg_metrics['accuracy']:.2f}%")
            print(f"  ID Accuracy: {id_accuracy:.2f}%")

            if self.use_wandb and wandb.run:
                try:
                    summary_updates = {f'ood/{shift_name}/avg_{k}': v for k, v in avg_metrics.items()}
                    summary_updates[f'ood/{shift_name}/id_accuracy'] = id_accuracy
                    wandb.run.summary.update(summary_updates)

                    log_updates = {f'ood/{shift_name}/avg/{k}': v for k, v in avg_metrics.items()}
                    log_updates[f'ood/{shift_name}/id_accuracy'] = id_accuracy
                    wandb.log(log_updates, commit=True)

                except Exception as log_e:
                    print(f"Wandb Error logging/updating summary: {log_e}")
        else:
             print("\nNo successful OOD dataset evaluations to calculate averages.")

        return final_results