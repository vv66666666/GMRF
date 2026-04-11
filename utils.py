

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch



AMAZON_BEHAVIOR_FUSION_ALPHA: float = 0.5
AMAZON_POTTS_BETA: float = 0.3
AMAZON_LBP_MAX_ITERS: int = 50



TOPICS: Tuple[str, ...] = (
    "design",
    "quality",
    "price",
    "performance",
    "customer service",
)

EMOTIONS: Tuple[str, ...] = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "disgust",
    "surprise",
    "anticipation",
)


def preference_template(
    dominant: Tuple[str, str],
    rare: Tuple[str, str],
) -> str:
    t_max, e_max = dominant
    t_min, e_min = rare
    return (
        f"The user often expresses {e_max} about {t_max}, "
        f"while rarely expressing {e_min} about {t_min}."
    )


def dominant_and_rare_pairs(
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    if not pairs:
        raise ValueError("pairs must be non-empty")

    cnt = Counter(pairs)
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0])))
    dominant = items[0][0]

    items_rare = sorted(cnt.items(), key=lambda kv: (kv[1], str(kv[0])))
    rare = items_rare[0][0]

    return dominant, rare




def concat_temporal_features(n_i1_to_n_i5: Sequence[float]) -> np.ndarray:
    arr = np.asarray(n_i1_to_n_i5, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 5:
        raise ValueError("temporal block must have 5 scalars")
    return arr


def concat_rating_features(n_i6_to_n_i10: Sequence[float]) -> np.ndarray:
    arr = np.asarray(n_i6_to_n_i10, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 5:
        raise ValueError("rating block must have 5 scalars")
    return arr


def weighted_concat_behavior(
    x_time: Union[np.ndarray, torch.Tensor],
    x_rating: Union[np.ndarray, torch.Tensor],
    alpha: float = AMAZON_BEHAVIOR_FUSION_ALPHA,
) -> Union[np.ndarray, torch.Tensor]:

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    if isinstance(x_time, torch.Tensor):
        a = torch.as_tensor(alpha, dtype=x_time.dtype, device=x_time.device)
        left = a * x_time
        right = (1.0 - a) * x_rating
        return torch.cat([left, right], dim=-1)

    a = float(alpha)
    left = a * np.asarray(x_time, dtype=np.float64)
    right = (1.0 - a) * np.asarray(x_rating, dtype=np.float64)
    return np.concatenate([left, right], axis=-1)


def minmax_normalize_1d(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - xmin) / (np.maximum(xmax - xmin, eps))


def zscore_normalize_1d(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - mean) / (np.maximum(std, eps))


def numpy_to_torch_f32(x: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device)





def build_directed_edges_from_undirected(edge_index_undirected: torch.Tensor) -> torch.Tensor:
    src = edge_index_undirected[0]
    dst = edge_index_undirected[1]
    rev = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index_undirected, rev], dim=1)


def build_potts_edge_potentials(
    edge_strength: torch.Tensor,
    beta: float = AMAZON_POTTS_BETA,
) -> Dict[str, torch.Tensor]:
    p_same = torch.sigmoid(beta * edge_strength)
    same = torch.exp(beta * edge_strength)
    psi = torch.ones(edge_strength.size(0), 2, 2, device=edge_strength.device, dtype=edge_strength.dtype)
    psi[:, 0, 0] = same
    psi[:, 1, 1] = same
    return {"psi_undirected": psi, "p_same": p_same}


def edge_consistency_loss(
    p_same: torch.Tensor,
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
    edge_index_undirected: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    src = edge_index_undirected[0]
    dst = edge_index_undirected[1]
    edge_mask = labeled_mask[src] & labeled_mask[dst]

    if not edge_mask.any():
        return torch.tensor(0.0, device=y.device, dtype=p_same.dtype)

    t_ij = (y[src[edge_mask]] == y[dst[edge_mask]]).float()
    p = p_same[edge_mask].clamp(eps, 1.0 - eps)
    return -(t_ij * torch.log(p) + (1.0 - t_ij) * torch.log(1.0 - p)).sum()


def loopy_belief_propagation_binary(
    node_potential: torch.Tensor,
    edge_potential: torch.Tensor,
    edge_index_directed: torch.Tensor,
    num_iters: int = AMAZON_LBP_MAX_ITERS,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    src = edge_index_directed[0]
    dst = edge_index_directed[1]
    e_dir = src.numel()
    n = node_potential.size(0)

    messages = torch.ones(e_dir, 2, device=node_potential.device, dtype=node_potential.dtype)

    incoming = [[] for _ in range(n)]
    for e in range(e_dir):
        incoming[int(dst[e])].append(e)

    for _t in range(num_iters):
        new_messages = messages.clone()

        for e in range(e_dir):
            i = int(src[e])
            j = int(dst[e])

            prod = node_potential[i].clone()
            for e_in in incoming[i]:
                k = int(src[e_in])
                if k == j:
                    continue
                prod = prod * messages[e_in]

            m = torch.zeros(2, device=node_potential.device, dtype=node_potential.dtype)
            for y_j in range(2):
                val = 0.0
                for y_i in range(2):
                    val = val + edge_potential[e, y_i, y_j] * prod[y_i]
                m[y_j] = val

            m = m / (m.sum() + eps)
            new_messages[e] = m

        messages = new_messages

    belief = node_potential.clone()
    for i in range(n):
        for e_in in incoming[i]:
            belief[i] = belief[i] * messages[e_in]
        belief[i] = belief[i] / (belief[i].sum() + eps)

    return {"belief": belief, "messages": messages}
