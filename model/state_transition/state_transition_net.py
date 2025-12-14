# state_transition_net.py
# 路径: MFSim/models/state_transition/state_transition_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class StateTransitionNet(nn.Module):
    """
    StateTransitionNet
    ------------------
    在上一时刻的平均场 (mu_{t-1}, C_{t-1}) 条件下，
    对“即插即用”的 agent 画像 u_i 进行状态转移建模，
    输出每个 agent 在 t 时刻为 正/中/负 的概率，
    并聚合得到 t 时刻的状态分布平均场 μ̂_t。

    数学对应：
      h_i^{in}(t) = [ u_i || μ_{t-1} || C_{t-1} ]
      z_i(t)      = f_θ(h_i^{in}(t))
      p_i(t)      = softmax(z_i(t))
      μ̂_t        = (1/N) Σ_i p_i(t)

    输入:
      - mu_prev:           (B, 3)
      - text_emb:          (B, d_c)
      - agent_features_list: 长度为 B 的 list，
                             每个元素为 (N_i, d_u) 的 tensor

    输出:
      - mu_pred: (B, 3)              # 预测的 t 时刻状态分布平均场 μ̂_t
      - p_list:  长度为 B 的 list，   # 每个元素为 (N_i, 3)，对应每个 agent 的状态概率
    """

    def __init__(
        self,
        agent_feat_dim: int,
        text_emb_dim: int,
        hidden_dim: int = 128,
        use_layernorm: bool = False,
    ):
        """
        整体是两层隐藏层的MLP，输出三分类结果logits
        """
        super().__init__()

        self.agent_feat_dim = agent_feat_dim
        self.text_emb_dim = text_emb_dim
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        # 输入维度: u_i (d_u) + mu_prev (3) + C_{t-1} (d_c)
        input_dim = agent_feat_dim + 3 + text_emb_dim

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        )
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, 3))  # 输出 logits: [pos, neu, neg]

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        mu_prev: torch.Tensor,
        text_emb: torch.Tensor,
        agent_features_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        mu_prev:            (B, 3)
        text_emb:           (B, d_c)
        agent_features_list: list of length B, each (N_i, d_u)

        返回:
          mu_pred: (B, 3)
          p_list:  list of length B, each (N_i, 3)
        """
        batch_size = mu_prev.size(0)
        assert len(agent_features_list) == batch_size, (
            "agent_features_list length must match batch size: "
            f"{len(agent_features_list)} vs {batch_size}"
        )

        mu_pred_list = []
        p_list: List[torch.Tensor] = []

        for b in range(batch_size):
            # 取出 batch 内第 b 个样本的输入
            mu_prev_b = mu_prev[b]         # (3,)
            text_emb_b = text_emb[b]       # (d_c,)
            agent_feats_b = agent_features_list[b]  # (N_b, d_u)

            if agent_feats_b.dim() != 2:
                raise ValueError(
                    f"agent_features_list[{b}] expected 2D (N_b, d_u), "
                    f"got shape {agent_feats_b.shape}"
                )

            N_b = agent_feats_b.size(0)
            if N_b == 0:
                # 边缘情况：没有 agent，直接复制上一时刻分布作为预测
                mu_pred_list.append(mu_prev_b)
                p_list.append(agent_feats_b.new_zeros((0, 3)))
                continue

            # 扩展 mu_prev_b 和 text_emb_b 以便与每个 agent 拼接
            mu_prev_exp = mu_prev_b.unsqueeze(0).expand(N_b, -1)      # (N_b, 3)
            text_emb_exp = text_emb_b.unsqueeze(0).expand(N_b, -1)    # (N_b, d_c)

            # h_i^{in}(t) = [ u_i || μ_{t-1} || C_{t-1} ]
            h_in = torch.cat(
                [agent_feats_b, mu_prev_exp, text_emb_exp],
                dim=-1,
            )  # (N_b, d_u + 3 + d_c)

            logits = self.mlp(h_in)              # (N_b, 3)
            probs = F.softmax(logits, dim=-1)    # (N_b, 3)

            # μ̂_t = (1/N_b) Σ_i p_i(t)
            mu_hat_b = probs.mean(dim=0)         # (3,)

            mu_pred_list.append(mu_hat_b)
            p_list.append(probs)

        mu_pred = torch.stack(mu_pred_list, dim=0)  # (B, 3)
        return mu_pred, p_list

    # 可选: 如果你以后希望支持 dense tensor 形式的 agent 特征，也可以用这个包装
    def forward_dense(
        self,
        mu_prev: torch.Tensor,
        text_emb: torch.Tensor,
        agent_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        一个可选接口：当所有样本的 agent 数量相同 N 时，可以用 dense 形式：
          mu_prev:   (B, 3)
          text_emb:  (B, d_c)
          agent_feats: (B, N, d_u)

        输出:
          mu_pred:   (B, 3)
          probs:     (B, N, 3)
        """
        B, N, d_u = agent_feats.shape
        assert d_u == self.agent_feat_dim, "agent_feat_dim mismatch."

        # 展开 batch 维 & agent 维
        mu_prev_exp = mu_prev.unsqueeze(1).expand(B, N, 3)       # (B, N, 3)
        text_emb_exp = text_emb.unsqueeze(1).expand(B, N, -1)    # (B, N, d_c)

        h_in = torch.cat(
            [agent_feats, mu_prev_exp, text_emb_exp],
            dim=-1,
        )  # (B, N, d_u+3+d_c)

        h_in_flat = h_in.reshape(B * N, -1)        # (B*N, input_dim)
        logits = self.mlp(h_in_flat)               # (B*N, 3)
        probs = F.softmax(logits, dim=-1)          # (B*N, 3)
        probs = probs.view(B, N, 3)                # (B, N, 3)

        # μ̂_t = (1/N) Σ_i p_i(t)
        mu_pred = probs.mean(dim=1)                # (B, 3)

        return mu_pred, probs
