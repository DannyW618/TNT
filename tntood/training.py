import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any
from torch_geometric.data import Data
from baseline_loss import UB_loss

try:
    import wandb
except ImportError:
    wandb = None

class Trainer:
    """Handles the model training loop, evaluation, and results tracking."""

    def __init__(self, model: torch.nn.Module, data: Data,
                 optimizer: optim.Optimizer, criterion: torch.nn.Module,
                 device: torch.device, config: Dict[str, Any]):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.use_wandb = config.get('use_wandb', False) and wandb is not None
        self.results: Dict[str, list] = {
            'epochs': [], 'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [],
            'train_f1': [], 'val_f1': [], 'test_f1': [], 'time_per_epoch': []
        }

        self.use_f1_metric = config.get('use_f1_metric', False) # Set during setup
        self.metric_name = "F1 score" if self.use_f1_metric else "accuracy"
        self.best_val_metric = 0.0
        self.epochs_no_improve = 0
        self.best_epoch = 0
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0) 
        self.best_val_loss = float('inf') 

    def _train_step(self, epoch = None) -> float:
        """Performs a single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        if self.config.get('use_tntood', False):
            out_logits, intermediates = self.model(self.data)
        else:
            out_logits, _ = self.model(self.data)

        if self.config.get('use_tntood', False):
            loss, loss_components = self.criterion(out_logits, intermediates, self.data.y, self.data.train_mask)
        else:
            baselines = self.config.get('use_baseline', [])
            loss = self.criterion(out_logits[self.data.train_mask], self.data.y[self.data.train_mask].squeeze())
            if "nodesafe" in baselines and len(baselines) == 1:
                lambda1 = self.config.get('lambda1', 0.999)
                m_criterion = UB_loss(lambda1)
                mloss_in, _ = m_criterion(_features = out_logits[self.data.train_mask],  labels = self.data.y[self.data.train_mask].squeeze().to(self.device), epoch = epoch)
                loss = loss + mloss_in * 0.5

        loss.backward()

        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _evaluate(self, epoch = None) -> Dict[str, float]:
        """Evaluates the model on train, validation, and test sets."""
        self.model.eval()
        out_logits, intermediates = self.model(self.data)
        pred = out_logits.argmax(dim=1)
        metrics = {}

        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
            base_key = mask_key.replace('_mask', '')
            acc_key = f"{base_key}_acc"
            f1_key = f"{base_key}_f1"

            metrics[acc_key] = 0.0
            metrics[f1_key] = 0.0
            mask = self.data[mask_key]

            true_labels = self.data.y[mask].squeeze()
            predictions = pred[mask]

            correct_predictions = (predictions == true_labels).sum().item()
            total_nodes = mask.sum().item()
            acc = float(correct_predictions) / float(total_nodes)
            metrics[acc_key] = acc

            unique_classes = torch.unique(true_labels)
            if self.use_f1_metric or len(unique_classes) <= 2:
                y_true_np = true_labels.cpu().numpy()
                y_pred_np = predictions.cpu().numpy()

                tp = ((y_pred_np == 1) & (y_true_np == 1)).sum()
                fp = ((y_pred_np == 1) & (y_true_np != 1)).sum()
                fn = ((y_pred_np != 1) & (y_true_np == 1)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                metrics[f1_key] = f1
            
            if mask_key == 'val_mask':
                if self.config.get('use_tntood', False):
                    val_loss, loss_components = self.criterion(out_logits, intermediates, self.data.y, mask)
                else:
                    val_loss = self.criterion(out_logits[self.data.val_mask], self.data.y[self.data.val_mask].squeeze())
                    baselines = self.config.get('use_baseline', [])
                    if "nodesafe" in baselines and len(baselines) == 1:
                        lambda1 = self.config.get('lambda1', 0.999)
                        m_criterion = UB_loss(lambda1)
                        mloss_in, _ = m_criterion(_features = out_logits[self.data.val_mask],  labels = self.data.y[self.data.val_mask].squeeze().to(self.device), epoch = epoch)
                        val_loss = val_loss + mloss_in * 0.5

        
        return metrics, val_loss.item()

    def _log_wandb(self, epoch: int, train_loss: float, metrics: Dict[str, float], duration: float):
        """Logs metrics to WandB."""
        if not self.use_wandb or wandb.run is None:
            return
        try:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/acc': metrics.get('train_acc', 0.0),
                'val/acc': metrics.get('val_acc', 0.0),
                'test/acc': metrics.get('test_acc', 0.0),
                'perf/epoch_time_s': duration
            }
            # Log F1 scores if they were calculated
            if self.use_f1_metric or metrics.get('val_f1', 0.0) > 0:
                log_dict.update({
                    'train/f1': metrics.get('train_f1', 0.0),
                    'val/f1': metrics.get('val_f1', 0.0),
                    'test/f1': metrics.get('test_f1', 0.0),
                })
            wandb.log(log_dict, step=epoch)
        except Exception as e:
            print(f"Wandb Error logging epoch metrics: {e}")

    def train(self, experiment_dir: Optional[Path] = None) -> Tuple[float, int]:
        """Runs the main training loop with early stopping."""
        print("\n--- Starting Training ---")
        train_start_time = time.time()

        max_epochs = self.config.get('epochs', 300)
        patience = self.config.get('early_stopping_patience', 100)

        progress_bar = tqdm(range(1, max_epochs + 1), desc="Training", unit="epoch", disable=not self.config.get('verbose', True))

        for epoch in progress_bar:
            epoch_start_time = time.time()
            train_loss = self._train_step(epoch = epoch)
            eval_metrics, _ = self._evaluate(epoch = epoch)
            epoch_duration = time.time() - epoch_start_time

            self.results['epochs'].append(epoch)
            self.results['train_loss'].append(train_loss)
            self.results['time_per_epoch'].append(epoch_duration)
            for key, value in eval_metrics.items():
                if key in self.results:
                    self.results[key].append(value)

            postfix_dict = {'loss': f"{train_loss:.4f}"}
            val_metric_value = eval_metrics.get('val_f1', 0.0) if self.use_f1_metric else eval_metrics.get('val_acc', 0.0)
            postfix_dict[f'val_{self.metric_name.split()[0]}'] = f"{val_metric_value:.4f}"
            progress_bar.set_postfix(postfix_dict)
            current_val_metric = eval_metrics.get(f'val_{"f1" if self.use_f1_metric else "acc"}', 0.0)

            if current_val_metric > self.best_val_metric:
                self.best_val_metric = current_val_metric
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if experiment_dir:
                    model_path = experiment_dir / 'best_model.pth'
                    try:
                        torch.save(self.best_model_state, model_path)
                    except Exception as e:
                        print(f"Error saving best model checkpoint: {e}")
            else:
                self.epochs_no_improve += 1


            if self.epochs_no_improve >= patience:
                print(f'\nEarly stopping at epoch {epoch}. Best val {self.metric_name}: {self.best_val_metric:.4f} at epoch {self.best_epoch}.')
                break
        else:
            print(f"\nTraining finished after {max_epochs} epochs. Best val loss: {self.best_val_metric:.4f} at epoch {self.best_epoch}.")


        train_duration = time.time() - train_start_time
        print(f"Total Training Time: {train_duration:.2f} seconds")

        if experiment_dir:
            history_path = experiment_dir / 'training_history.json'
            try:
                with open(history_path, 'w') as f:
                    json.dump(self.results, f, indent=2)
                if self.config.get('verbose', True): print(f"Training history saved to {history_path}")
                if self.use_wandb and wandb.run: wandb.save(str(history_path), base_path=str(experiment_dir))
            except Exception as e:
                 print(f"Error saving training history: {e}")

        if self.use_wandb and wandb.run:
            try:
                wandb.run.summary[f"best_val_{'f1' if self.use_f1_metric else 'acc'}"] = self.best_val_metric
                wandb.run.summary["best_epoch"] = self.best_epoch
                wandb.run.summary["training_time_s"] = train_duration
            except Exception as e:
                print(f"Wandb Error updating summary: {e}")

        # Load the best model state for final evaluation
        if self.best_model_state:
            if self.config.get('verbose', True): print(f"Loading best model from epoch {self.best_epoch}.")
            try:
                # Move state dict back to the correct device before loading
                device_state_dict = {k: v.to(self.device) for k, v in self.best_model_state.items()}
                self.model.load_state_dict(device_state_dict)
            except RuntimeError as e:
                 print(f"Error loading best model state_dict: {e}. Using final model state.")
        else:
            print("No best model saved. Using final model state.")

        return self.best_val_metric, self.best_epoch

    def final_evaluate(self) -> Dict[str, float]:
        """Performs final evaluation using the loaded best (or final) model state."""
        print("\n--- Final Evaluation ---")
        final_metrics, _ = self._evaluate(epoch=self.best_epoch)

        print(f"  Train Accuracy: {final_metrics.get('train_acc', 0.0):.4f}")
        print(f"  Valid Accuracy: {final_metrics.get('val_acc', 0.0):.4f}")
        print(f"  Test Accuracy:  {final_metrics.get('test_acc', 0.0):.4f}")
        if self.use_f1_metric or final_metrics.get('test_f1', 0.0) > 0:
            print(f"  Train F1 Score: {final_metrics.get('train_f1', 0.0):.4f}")
            print(f"  Valid F1 Score: {final_metrics.get('val_f1', 0.0):.4f}")
            print(f"  Test F1 Score:  {final_metrics.get('test_f1', 0.0):.4f}")

        # Log final metrics to wandb summary
        if self.use_wandb and wandb.run:
            try:
                wandb.run.summary["final_train_acc"] = final_metrics.get('train_acc', 0.0)
                wandb.run.summary["final_val_acc"] = final_metrics.get('val_acc', 0.0)
                wandb.run.summary["final_test_acc"] = final_metrics.get('test_acc', 0.0)
                if self.use_f1_metric or final_metrics.get('test_f1', 0.0) > 0:
                    wandb.run.summary["final_train_f1"] = final_metrics.get('train_f1', 0.0)
                    wandb.run.summary["final_val_f1"] = final_metrics.get('val_f1', 0.0)
                    wandb.run.summary["final_test_f1"] = final_metrics.get('test_f1', 0.0)
            except Exception as e:
                print(f"Wandb Error updating summary with final metrics: {e}")

        return final_metrics