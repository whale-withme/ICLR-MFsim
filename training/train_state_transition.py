# train_state_transition.py
# 路径: MFSim/training/train_state_transition.py

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 让 "MFSim" 根目录加入 sys.path，方便用绝对导入
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.state_transition.encoders import build_text_encoder
from model.state_transition.state_transition_net import StateTransitionNet


# ============================================================
# 配置结构（你可以以后改成读 YAML）
# ============================================================

@dataclass
class TrainConfig:
    # 模型
    encoder_type: str = "gru"        # "gru" 或 "bert"
    vocab_size: int = 50000          # 仅对 GRU encoder 有用
    text_emb_dim: int = 256
    agent_feat_dim: int = 32
    hidden_dim: int = 256
    use_layernorm: bool = False

    # 训练
    batch_size: int = 16
    num_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # 设备 & 日志 & 保存
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    save_dir: str = os.path.join(ROOT_DIR, "checkpoints")
    save_name: str = "state_transition.pt"


# ============================================================
# 这里假设你已经在 training/datasets.py 中实现了 Dataset
# 和 DataLoader 构造函数，batch 的结构是：
# {
#   "mu_prev": (B, 3),
#   "mu_t": (B, 3),
#   "text_ids": (B, L_max),
#   "attention_mask": (B, L_max),
#   "agent_features_list": [ (N_i, d_u) for i in batch ]
# }
# ============================================================

def build_dataloader(cfg: TrainConfig) -> DataLoader:
    """
    你需要在 training/datasets.py 中实现：
      - MeanFieldDataset
      - collate_fn
      - build_state_transition_dataset_or_loader(...)
    这里先放一个占位示例，你可以改成自己的实现。
    """
    from training.datasets import StateDataset, collate_fn, build_state_transition_dataset  # 自己实现

    dataset = build_state_transition_dataset()  # 你自己定义如何构建
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return loader


# ============================================================
# 构建模型
# ============================================================

def build_models(cfg: TrainConfig):
    # 文本编码器配置
    encoder_config: Dict[str, Any] = {
        "type": cfg.encoder_type,
    }

    if cfg.encoder_type.lower() == "gru":
        encoder_config.update(
            {
                "vocab_size": cfg.vocab_size,
                "emb_dim": cfg.text_emb_dim,
                "hidden_dim": cfg.text_emb_dim,
                "pad_token_id": 0,
            }
        )
    elif cfg.encoder_type.lower() == "bert":
        encoder_config.update(
            {
                "model_name": "bert-base-uncased",
                "output_dim": cfg.text_emb_dim,
                "freeze": True,
            }
        )
    else:
        raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")

    text_encoder = build_text_encoder(encoder_config)

    state_net = StateTransitionNet(
        agent_feat_dim=cfg.agent_feat_dim,
        text_emb_dim=cfg.text_emb_dim,
        hidden_dim=cfg.hidden_dim,
        use_layernorm=cfg.use_layernorm,
    )

    return text_encoder, state_net


# ============================================================
# 单个 epoch 的训练
# ============================================================

def train_one_epoch(
    epoch: int,
    cfg: TrainConfig,
    text_encoder: torch.nn.Module,
    state_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
):
    device = cfg.device
    text_encoder.train()
    state_net.train()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        mu_prev = batch["mu_prev"].to(device)             # (B, 3)
        mu_t = batch["mu_t"].to(device)                   # (B, 3)
        text_ids = batch["text_ids"].to(device)           # (B, L_max)
        attention_mask = batch["attention_mask"].to(device)

        agent_features_list = [
            feats.to(device) for feats in batch["agent_features_list"]
        ]  # list of (N_i, d_u)

        # 1. 文本平均场编码 C_{t-1}
        text_emb = text_encoder(text_ids, attention_mask)  # (B, d_c)

        # 2. 状态转移网络 → 预测 μ̂_t
        mu_pred, _ = state_net(mu_prev, text_emb, agent_features_list)  # (B, 3)

        # 3. Loss: MSE or KL
        loss = F.mse_loss(mu_pred, mu_t)

        optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(text_encoder.parameters()) + list(state_net.parameters()),
                max_norm=cfg.grad_clip,
            )

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            avg_loss = total_loss / total_batches
            print(
                f"[Epoch {epoch}] Step {batch_idx+1}/{len(train_loader)} "
                f"Loss: {loss.item():.6f} (avg: {avg_loss:.6f})"
            )

    avg_loss = total_loss / max(1, total_batches)
    print(f"[Epoch {epoch}] Training finished. Avg Loss: {avg_loss:.6f}")
    return avg_loss


# ============================================================
# 保存 & 主训练逻辑
# ============================================================

def save_checkpoint(
    cfg: TrainConfig,
    text_encoder: torch.nn.Module,
    state_net: torch.nn.Module,
    epoch: int,
    avg_loss: float,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    ckpt = {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "text_encoder_state": text_encoder.state_dict(),
        "state_net_state": state_net.state_dict(),
        "config": cfg.__dict__,
    }

    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    # 这里可以加命令行参数覆盖默认配置
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_type", type=str, default="gru")  # or "bert"
    args = parser.parse_args()

    cfg = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_type=args.encoder_type,
    )

    print("=== TrainConfig ===")
    print(cfg)

    device = cfg.device
    print(f"Using device: {device}")

    # 1. DataLoader
    train_loader = build_dataloader(cfg)

    # 2. 模型
    text_encoder, state_net = build_models(cfg)
    text_encoder.to(device)
    state_net.to(device)

    # 3. 优化器
    optimizer = torch.optim.Adam(
        list(text_encoder.parameters()) + list(state_net.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        avg_loss = train_one_epoch(
            epoch,
            cfg,
            text_encoder,
            state_net,
            optimizer,
            train_loader,
        )

        # 简单：按最优 loss 存一次
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(cfg, text_encoder, state_net, epoch, avg_loss)

    print("Training finished.")


if __name__ == "__main__":
    main()
