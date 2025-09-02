import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from typing import Tuple

def get_ood_metrics(scores_id: np.ndarray, scores_ood: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate OOD detection metrics (AUROC, AUPR, FPR@95%TPR).
    Assumes **higher scores** indicate **more OOD**.

    Args:
        scores_id (np.ndarray): Scores for in-distribution samples.
        scores_ood (np.ndarray): Scores for out-of-distribution samples.

    Returns:
        tuple: (AUROC, AUPR, FPR@95%TPR)
               Returns (0.0, 0.0, 1.0) if calculation fails.
    """
    if torch.is_tensor(scores_id):
        scores_id = scores_id.cpu().detach().numpy()
    if torch.is_tensor(scores_ood):
        scores_ood = scores_ood.cpu().detach().numpy()

    scores_id = scores_id.flatten()
    scores_ood = scores_ood.flatten()

    if scores_id.size == 0 or scores_ood.size == 0:
        print("Warning: Cannot calculate OOD metrics with empty ID or OOD score arrays.")
        return 0.0, 0.0, 1.0

    labels_id = np.zeros_like(scores_id)
    labels_ood = np.ones_like(scores_ood)

    y_true = np.concatenate([labels_id, labels_ood])
    y_scores = np.concatenate([scores_id, scores_ood])

    if len(np.unique(y_true)) < 2:
        print("Warning: Only one class type found. Cannot calculate OOD metrics.")
        return 0.0, 0.0, 1.0

    try:
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        # Calculate FPR at 95% TPR
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        target_tpr = 0.95
        fpr_at_95_tpr = 1.0
        if np.any(tpr >= target_tpr):
            idx_tpr_95 = np.argmax(tpr >= target_tpr)
            fpr_at_95_tpr = fpr[idx_tpr_95]

        auroc = float(np.nan_to_num(auroc, nan=0.0))
        aupr = float(np.nan_to_num(aupr, nan=0.0))
        fpr_at_95_tpr = float(np.nan_to_num(fpr_at_95_tpr, nan=1.0))

        return auroc, aupr, fpr_at_95_tpr

    except ValueError as e:
        print(f"Error calculating OOD metrics (sklearn): {e}. Returning defaults.")
        return 0.0, 0.0, 1.0
    except Exception as e:
        print(f"Unexpected error calculating OOD metrics: {e}. Returning defaults.")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 1.0