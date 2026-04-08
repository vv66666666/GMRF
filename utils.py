
from __future__ import annotations

from typing import Dict

import torch


def build_directed_edges_from_undirected(edge_index_undirected: torch.Tensor) -> torch.Tensor:

  
    src = edge_index_undirected[0]
    dst = edge_index_undirected[1]
    rev = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index_undirected, rev], dim=1)


def build_potts_edge_potentials(edge_strength: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:

  
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
    num_iters: int,
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
