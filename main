from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class NodeCfg:
    proj_dim: int
    eps: float = 1e-8


class NodePotentialLearning(nn.Module):
    def __init__(self, cfg: NodeCfg):
        super().__init__()
        self.fc = nn.Linear(cfg.proj_dim, 1, bias=True)
        self.eps = cfg.eps

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(z).squeeze(-1)       
        p = torch.sigmoid(logits)             
        phi0 = 1.0 - p
        phi1 = p
        phi = torch.stack([phi0, phi1], dim=-1)  
        return {"p": p, "phi": phi, "logits": logits}

@dataclass
class RelationCfg:
    proj_dim: int
    gat_dim: int
    negative_slope: float = 0.2
    tau_mrf: float = 1.0


class DualChannelRelationWeights(nn.Module):
    def __init__(self, cfg: RelationCfg):
        super().__init__()
        D = cfg.gat_dim


        self.W_b = nn.Linear(cfg.proj_dim, D, bias=False)
        self.W_p = nn.Linear(cfg.proj_dim, D, bias=False)

     
        self.a_b = nn.Parameter(torch.empty(2 * D))
        self.a_p = nn.Parameter(torch.empty(2 * D))


        self.gamma_b = nn.Parameter(torch.tensor(0.0))
        self.gamma_p = nn.Parameter(torch.tensor(0.0))

        self.negative_slope = cfg.negative_slope
        self.tau_mrf = cfg.tau_mrf

        nn.init.xavier_uniform_(self.W_b.weight)
        nn.init.xavier_uniform_(self.W_p.weight)
        nn.init.xavier_uniform_(self.a_b.view(1, -1))
        nn.init.xavier_uniform_(self.a_p.view(1, -1))

    def _e_ij(self, h: torch.Tensor, edge_index_undirected: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        src = edge_index_undirected[0]  # i
        dst = edge_index_undirected[1]  # j

        hi = h[src]  # [E, D]
        hj = h[dst]  # [E, D]
        cat = torch.cat([hi, hj], dim=-1) 
        e = (cat * a).sum(dim=-1) 
        return F.leaky_relu(e, negative_slope=self.negative_slope)

    def forward(
        self,
        z_beh: torch.Tensor,  
        z_pref: torch.Tensor,  
        edge_index_undirected: torch.Tensor,  
    ) -> Dict[str, torch.Tensor]:

        h_b = self.W_b(z_beh)   
        h_p = self.W_p(z_pref)   


        e_b_ij = self._e_ij(h_b, edge_index_undirected, self.a_b) 
        e_b_ji = self._e_ij(h_b, edge_index_undirected.flip(0), self.a_b)  
        s_b = 0.5 * (e_b_ij + e_b_ji) 

        e_p_ij = self._e_ij(h_p, edge_index_undirected, self.a_p) 
        e_p_ji = self._e_ij(h_p, edge_index_undirected.flip(0), self.a_p) 
        s_p = 0.5 * (e_p_ij + e_p_ji) 
        
        gam = torch.stack([self.gamma_b, self.gamma_p], dim=0) 
        w = F.softmax(gam, dim=0)  
        w_b, w_p = w[0], w[1]


        s = w_b * s_b + w_p * s_p  # [E_u]


        alpha_ij = torch.sigmoid(s / self.tau_mrf)
        return {"alpha_ij": alpha_ij}




@dataclass
class EdgeCfg:
    beta: float  # Eq.(17) beta>0


class PottsEdgePotentialsAndLoss(nn.Module):
    def __init__(self, cfg: EdgeCfg):
        super().__init__()
        self.beta = cfg.beta

    def forward(self, alpha_ij: torch.Tensor) -> Dict[str, torch.Tensor]:
      
        P_same = torch.sigmoid(self.beta * alpha_ij) 

        same = torch.exp(self.beta * alpha_ij) 
        psi = torch.ones(alpha_ij.size(0), 2, 2, device=alpha_ij.device, dtype=alpha_ij.dtype)
        psi[:, 0, 0] = same
        psi[:, 1, 1] = same
        # psi[:,0,1] = 1, psi[:,1,0] = 1
        return {"psi_undirected": psi, "P_same": P_same}

    def edge_loss(
        self,
        P_same: torch.Tensor,          
        y: torch.Tensor,                 
        labeled_mask: torch.Tensor,    
        edge_index_undirected: torch.Tensor, 
        eps: float = 1e-8,
    ) -> torch.Tensor:

        src = edge_index_undirected[0]
        dst = edge_index_undirected[1]
        edge_mask = labeled_mask[src] & labeled_mask[dst] 

        if not edge_mask.any():
            return torch.tensor(0.0, device=y.device, dtype=y.dtype)


        t_ij = (y[src[edge_mask]] == y[dst[edge_mask]]).float() 
        P = P_same[edge_mask].clamp(eps, 1.0 - eps)  # 避免 log(0)

        L_edge = -(t_ij * torch.log(P) + (1.0 - t_ij) * torch.log(1.0 - P)).sum()
        return L_edge




class LoopyBeliefPropagation(nn.Module):

  
    def __init__(self, num_iters: int = 10, eps: float = 1e-8):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps

    def forward(
        self,
        phi: torch.Tensor,        
        psi_dir: torch.Tensor,   
        edge_index_dir: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        src = edge_index_dir[0]
        dst = edge_index_dir[1]
        E_dir = src.numel()
        N = phi.size(0)


        messages = torch.ones(E_dir, 2, device=phi.device, dtype=phi.dtype)


        incoming = [[] for _ in range(N)]
        for e in range(E_dir):
            incoming[int(dst[e])].append(e)


        for _t in range(self.num_iters):
            new_messages = messages.clone()

            for e in range(E_dir):
                i = int(src[e])
                j = int(dst[e])

                prod = phi[i].clone()  

                for e_in in incoming[i]: 
                    k = int(src[e_in])   
                    if k == j:
                        continue
                    prod = prod * messages[e_in]


                m = torch.zeros(2, device=phi.device, dtype=phi.dtype)
                for y_j in range(2):
                    val = 0.0
                    for y_i in range(2):
                        val = val + psi_dir[e, y_i, y_j] * prod[y_i]
                    m[y_j] = val


                m = m / (m.sum() + self.eps)
                new_messages[e] = m

            messages = new_messages


        beliefs = phi.clone()
        for i in range(N):
            for e_in in incoming[i]:
                beliefs[i] = beliefs[i] * messages[e_in]  
            beliefs[i] = beliefs[i] / (beliefs[i].sum() + self.eps)


        return {"beliefs": beliefs, "messages": messages}


def build_directed_from_undirected(edge_index_undirected: torch.Tensor) -> torch.Tensor:

    src = edge_index_undirected[0]
    dst = edge_index_undirected[1]
    return torch.cat([edge_index_undirected, torch.stack([dst, src], dim=0)], dim=1)

