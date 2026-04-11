
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



class TwoLayerMLP(nn.Module):
    """f(x) = W_2 ReLU(W_1 x + b_1) + b_2"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(F.relu(self.lin1(x)))


@dataclass
class DualBranchConfig:
    beh_dim: int
    pre_dim: int
    hidden_dim: int
    proj_dim: int
    temperature: float


class DualBranchUserEncoder(nn.Module):
    def __init__(self, cfg: DualBranchConfig):
        super().__init__()
        self.f_beh = TwoLayerMLP(cfg.beh_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.f_pre = TwoLayerMLP(cfg.pre_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.f_sh = TwoLayerMLP(cfg.hidden_dim, cfg.hidden_dim, cfg.proj_dim)
        self.cfg = cfg

    def forward(self, x_beh: torch.Tensor, x_pre: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_tilde_beh = self.f_beh(x_beh)
        x_tilde_pre = self.f_pre(x_pre)
        z_b = self.f_sh(x_tilde_beh)
        z_p = self.f_sh(x_tilde_pre)
        z = z_b + z_p
        return {"z_beh": z_b, "z_pref": z_p, "z": z}


def contrastive_loss_eq8(
    z_beh_batch: torch.Tensor,
    z_pref_bank: torch.Tensor,
    user_ids: torch.Tensor,
    temperature: float,
    num_negatives: int = 100,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("temperature tau must be > 0")

    n_users = z_pref_bank.size(0)
    if n_users <= 1:
        return torch.tensor(0.0, device=z_beh_batch.device, dtype=z_beh_batch.dtype)

    z_beh_batch = F.normalize(z_beh_batch, dim=-1)
    z_pref_bank = F.normalize(z_pref_bank, dim=-1)

    user_ids = user_ids.long().view(-1)
    if user_ids.shape[0] != z_beh_batch.shape[0]:
        raise ValueError("user_ids length must match batch dimension of z_beh_batch")
    if (user_ids < 0).any() or (user_ids >= n_users).any():
        raise ValueError("user_ids must be in [0, n_users)")

    bsz = z_beh_batch.size(0)
    device = z_beh_batch.device

    neg_idx = torch.randint(0, n_users - 1, (bsz, num_negatives), device=device)
    uid = user_ids.unsqueeze(1).expand(-1, num_negatives)
    neg_user = neg_idx + (neg_idx >= uid).long()

    z_pos = z_pref_bank[user_ids]
    pos = (z_beh_batch * z_pos).sum(dim=-1) / temperature

    z_neg = z_pref_bank[neg_user]
    sim_neg = (z_beh_batch.unsqueeze(1) * z_neg).sum(dim=-1) / temperature

    logits = torch.cat([pos.unsqueeze(1), sim_neg], dim=1)
    target = torch.zeros(bsz, dtype=torch.long, device=device)
    return F.cross_entropy(logits, target)


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
        phi = torch.stack([1.0 - p, p], dim=-1)
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
        src = edge_index_undirected[0]
        dst = edge_index_undirected[1]
        hi = h[src]
        hj = h[dst]
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

        s = w_b * s_b + w_p * s_p
        alpha_ij = torch.sigmoid(s / self.tau_mrf)
        return {"alpha_ij": alpha_ij}


@dataclass
class EdgeCfg:
    beta: float


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
        P = P_same[edge_mask].clamp(eps, 1.0 - eps)
        return -(t_ij * torch.log(P) + (1.0 - t_ij) * torch.log(1.0 - P)).sum()




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



@dataclass
class PaperGMRFConfig:
    dual: DualBranchConfig
    node: NodeCfg
    relation: RelationCfg
    edge: EdgeCfg
    lbp_iters: int = 10
    lbp_eps: float = 1e-8


class PaperGMRF(nn.Module):
    """
    Requires: DualBranchConfig.proj_dim == NodeCfg.proj_dim == RelationCfg.proj_dim.
    Node head uses fused z (Eq.7); relation uses z_beh, z_pref (Eq.9-14).
    """

    def __init__(self, cfg: PaperGMRFConfig):
        super().__init__()
        if not (cfg.dual.proj_dim == cfg.node.proj_dim == cfg.relation.proj_dim):
            raise ValueError("proj_dim must match across dual branch, node, and relation configs")

        self.cfg = cfg
        self.encoder = DualBranchUserEncoder(cfg.dual)
        self.node_head = NodePotentialLearning(cfg.node)
        self.relation = DualChannelRelationWeights(cfg.relation)
        self.potts = PottsEdgePotentialsAndLoss(cfg.edge)
        self.lbp = LoopyBeliefPropagation(num_iters=cfg.lbp_iters, eps=cfg.lbp_eps)

    def forward(
        self,
        x_beh: torch.Tensor,
        x_pre: torch.Tensor,
        edge_index_undirected: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x_beh, x_pre)
        z_beh = enc["z_beh"]
        z_pref = enc["z_pref"]
        z = enc["z"]

        node_out = self.node_head(z)
        phi = node_out["phi"]

        rel_out = self.relation(z_beh, z_pref, edge_index_undirected)
        alpha_ij = rel_out["alpha_ij"]

        pot_out = self.potts(alpha_ij)
        psi_u = pot_out["psi_undirected"]

        edge_dir = build_directed_from_undirected(edge_index_undirected)
        psi_dir = torch.cat([psi_u, psi_u], dim=0)

        lbp_out = self.lbp(phi, psi_dir, edge_dir)
        beliefs = lbp_out["beliefs"]
        refined_prob = beliefs[:, 1]

        return {
            **enc,
            **node_out,
            **rel_out,
            **pot_out,
            **lbp_out,
            "refined_prob": refined_prob,
            "edge_index_dir": edge_dir,
        }


