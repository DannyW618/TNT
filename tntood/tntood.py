import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_softmax, scatter_add

from models import GCN


class CrossFusion(nn.Module):
    """
    Projects text and graph features into a shared 'dim' space, computes
    attention weights per target node over its incoming neighbors, and
    aggregates value vectors (from text) with a residual connection.
    """
    def __init__(self, t_dim: int, g_dim: int, dim: int) -> None:
        super().__init__()
        self.text_info = nn.Linear(t_dim, dim)
        self.graph_info = nn.Linear(g_dim, dim)
        self.value = nn.Linear(t_dim, dim)

    def forward(
        self,
        text_feat: torch.Tensor,
        graph_feat: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        t_info = self.text_info(text_feat)
        g_info = self.graph_info(graph_feat)
        v = self.value(text_feat)

        row, col = edge_index

        t_row = t_info[row]
        g_col = g_info[col]
        scores = (t_row * g_col).sum(dim=-1) / (t_info.size(-1) ** 0.5)
        attn = scatter_softmax(scores, row)

        v_col = v[col]
        out = scatter_add(attn.unsqueeze(-1) * v_col, row, dim=0, dim_size=text_feat.size(0))

        # Residual in input (text) space
        out = out + text_feat
        return out


class HyperProjectionHead(nn.Module):
    """
    Generates per-sample projection matrices (full-rank or low-rank) and
    applies them to inputs. Supports chunked/micro-batched evaluation to
    reduce peak memory on large graphs.

    Args:
        input_dim: Input dimensionality.
        proj_dim:  Output projection dimensionality.
        rank:      Rank for low-rank decomposition.
        use_low_rank_default: Default to low-rank generation when True.
    """
    def __init__(self, input_dim: int, proj_dim: int, rank: int = 16, use_low_rank_default: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.rank = rank
        self.use_low_rank_default = use_low_rank_default

        self.generator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, proj_dim * input_dim),
        )

        self.left_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, proj_dim * rank),
        )
        self.right_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, rank * input_dim),
        )

    def forward(
        self,
        fused_text: torch.Tensor,
        use_low_rank: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            If low-rank: (left_factor [B, proj_dim, rank], right_factor [B, rank, input_dim])
            Else:        full weights [B, proj_dim, input_dim]
        """
        if use_low_rank is None:
            use_low_rank = self.use_low_rank_default

        if use_low_rank:
            batch_size = fused_text.size(0)
            left_weights = self.left_generator(fused_text)
            right_weights = self.right_generator(fused_text)
            left_factor = left_weights.view(batch_size, self.proj_dim, self.rank)
            right_factor = right_weights.view(batch_size, self.rank, self.input_dim)
            return left_factor, right_factor

        weights = self.generator(fused_text)
        return weights.view(-1, self.proj_dim, self.input_dim)

    def apply_projection(
        self,
        x: torch.Tensor,
        weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        use_low_rank: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Applies per-sample projection to x.

        Returns:
            [B, proj_dim]
        """
        if use_low_rank is None:
            use_low_rank = isinstance(weights, tuple)

        if use_low_rank:
            left_factor, right_factor = weights
            temp = torch.bmm(right_factor, x.unsqueeze(-1)).squeeze(-1)
            result = torch.bmm(left_factor, temp.unsqueeze(-1)).squeeze(-1)
            return result

        return torch.bmm(weights, x.unsqueeze(-1)).squeeze(-1)

    def chunked_projection(
        self,
        x_fused: torch.Tensor,
        x: torch.Tensor,
        use_low_rank: Optional[bool] = None,
        chunk_size: int = 4096,
        micro_batch: int = 512,
    ) -> torch.Tensor:
        """
        Memory-efficient per-sample projection over large N.

        Returns:
            [N, proj_dim]
        """
        if use_low_rank is None:
            use_low_rank = self.use_low_rank_default

        results: list[torch.Tensor] = []
        for i in range(0, x.size(0), chunk_size):
            end_i = min(i + chunk_size, x.size(0))
            xf_chunk = x_fused[i:end_i]
            x_chunk = x[i:end_i]

            chunk_parts: list[torch.Tensor] = []
            for j in range(0, xf_chunk.size(0), micro_batch):
                end_j = min(j + micro_batch, xf_chunk.size(0))
                xf_micro = xf_chunk[j:end_j]
                x_micro = x_chunk[j:end_j]

                weights = self.forward(xf_micro, use_low_rank=use_low_rank)
                proj = self.apply_projection(x_micro, weights, use_low_rank=use_low_rank)
                chunk_parts.append(proj)

            results.append(torch.cat(chunk_parts, dim=0))
        return torch.cat(results, dim=0)


class TNTOODModel(nn.Module):
    """
    Text-and-Topology OOD model backbone.
    """
    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int,
        projection_dim: int = 128,
        dropout: float = 0.5,
        num_id_classes: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.graph_encoder = GCNConv(node_feature_dim, gnn_hidden_dim)
        self.dropout_p = dropout
        self.activation = nn.ReLU()

        # Fusion and projections
        self.fusion_gcn = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.text_encoder = nn.Linear(node_feature_dim, gnn_hidden_dim)
        self.cross_fusion = CrossFusion(384, projection_dim, node_feature_dim)

        self.hyper_proj = HyperProjectionHead(
            input_dim=node_feature_dim,
            proj_dim=projection_dim,
            rank=16,
            use_low_rank_default=False,
        )
        self.graph_proj = nn.Linear(projection_dim, projection_dim)
        self.text_proj = nn.Linear(node_feature_dim, projection_dim)
        self.id_classifier = GCNConv(gnn_hidden_dim, num_id_classes) if num_id_classes else None

    def forward(
        self,
        data: Data,
        proj_weights: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        g_emb = self.graph_encoder(x, edge_index)
        x_fused = self.cross_fusion(x, g_emb, edge_index)

        if x.size(0) > 40000: # only chunk for large graphs
            p_t = self.hyper_proj.chunked_projection(
                x_fused, x, use_low_rank=True, chunk_size=500
            )
            proj_weights = None
        else:
            proj_weights = self.hyper_proj(x_fused)
            p_t = torch.bmm(proj_weights, x.unsqueeze(-1)).squeeze(-1)

        g_emb = self.fusion_gcn(p_t, edge_index)
        g_emb = F.dropout(g_emb, p=self.dropout_p, training=self.training)
        p_g = self.graph_proj(g_emb)

        # Normalize for alignment
        p_t = F.normalize(p_t, dim=-1)
        p_g = F.normalize(p_g, dim=-1)

        alignment_scores = torch.sum(p_t * p_g, dim=1)
        logits = self.id_classifier(g_emb, edge_index) if self.id_classifier else None

        return logits, {
            "alignment_scores": alignment_scores,
            "p_t": p_t,
            "p_g": p_g,
            "features": g_emb,
            "proj_weights": proj_weights,
        }

    @torch.no_grad()
    def calculate_energy(
        self,
        data: Data,
        temperature: float = 1.0,
        proj_weights = None
    ) -> torch.Tensor:
        logits, _ = self(data)
        return -temperature * torch.logsumexp(logits / temperature, dim=-1)


class TNTOODLoss(nn.Module):
    """
    Combined contrastive (alignment) + ID classification loss.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        id_loss_weight: float = 0.0,
        use_batch: bool = False,
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.id_loss_weight = id_loss_weight
        self.ce = nn.CrossEntropyLoss()
        self.use_batch_contrastive = use_batch
        self.batch_size = batch_size

    def contrastive(self, p_t: torch.Tensor, p_g: torch.Tensor) -> torch.Tensor:
        p_t = F.normalize(p_t, dim=-1)
        p_g = F.normalize(p_g, dim=-1)
        sim = torch.matmul(p_t, p_g.t()) / self.temperature
        labels = torch.arange(sim.size(0), device=p_t.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

    def forward(
        self,
        logits: Optional[torch.Tensor],
        inter: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p_t, p_g = inter["p_t"], inter["p_g"]
        if mask is None:
            mask = torch.ones(p_t.size(0), dtype=torch.bool, device=p_t.device)

        total = torch.tensor(0.0, device=p_t.device)
        loss_c = torch.tensor(0.0, device=p_t.device)

        if self.contrastive_weight > 0:
            train_p_t = p_t[mask]
            train_p_g = p_g[mask]
            if self.use_batch_contrastive and train_p_t.shape[0] > self.batch_size:
                idx = torch.randperm(train_p_t.shape[0], device=train_p_t.device)[: self.batch_size]
                loss_c = self.contrastive(train_p_t[idx], train_p_g[idx])
            else:
                loss_c = self.contrastive(train_p_t, train_p_g)
            total = total + self.contrastive_weight * loss_c

        loss_id = torch.tensor(0.0, device=p_t.device)
        if logits is not None and labels is not None:
            loss_id = self.ce(logits[mask], labels[mask].long())
            total = total + self.id_loss_weight * loss_id

        return total, {
            "contrastive_loss": loss_c.item(),
            "id_task_loss": (self.id_loss_weight * loss_id).item(),
            "total_loss": total.item(),
        }


class TNTOODDetector:
    """
    OOD detector wrapper that combines energy and alignment and then
    propagates scores over the graph.
    """
    def __init__(self, model: TNTOODModel, config: Dict[str, Any]) -> None:
        self.model = model
        self.device = model.device
        self.alpha = config.get("alpha", 0.5)
        self.prop = config.get("K", 3)
        self.align_w = config.get("align_w", 1)
        self.projection_weights: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.fitted = False

    @torch.no_grad()
    def fit(self, data: Data) -> None:
        self.model.eval()
        _, intermediates = self.model(data)
        self.projection_weights = intermediates["proj_weights"]

    @torch.no_grad()
    def compute_scores(self, data: Data) -> torch.Tensor:
        self.model.eval()
        logits, inter = self.model(data, proj_weights=self.projection_weights)
        energy = self.model.calculate_energy(data, proj_weights=self.projection_weights)
        alignment = inter["alignment_scores"] * self.align_w
        print("alignment", alignment.mean(), alignment.std())
        score = energy - alignment
        score = GCN.propagation(score, data.edge_index, self.prop, self.alpha)
        return score

    @staticmethod
    def propagation(scores: torch.Tensor, edge_index: torch.Tensor, layers: int, alpha: float) -> torch.Tensor:
        """
        Simple label propagation utility (kept for backward-compatibility).
        """
        for _ in range(layers):
            row, col = edge_index
            agg = torch.zeros_like(scores).scatter_add_(0, row, scores[col])
            deg = torch.bincount(row, minlength=scores.size(0)).float().clamp(min=1)
            agg = agg / deg
            scores = alpha * scores + (1 - alpha) * agg
        return scores
