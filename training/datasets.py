import json
import pandas as pd
import torch
import random
import os
from torch.utils.data import Dataset
from typing import Dict, List

# 假设 encoders 在 models 目录下
from model.state_transition.encoders import build_text_encoder

class TrajectoryClusterDataset(Dataset):
    def __init__(
        self,
        trajectory_path: str,       # 10031994215_trajectory.csv (Ground Truth 分布)
        mf_path: str,               # 4264473811_mf.csv (舆论环境)
        test_data_path: str,        # 4264473811.json (真实用户流, 包含 uid)
        profile_path: str,          # cluster_core_user_profile.jsonl (核心用户池)
        encoder_config: dict,
        uid_mapping_path: str = None, # [关键] uid -> cluster 的映射文件 (json)
        batch_size: int = 16
    ):
        self.batch_size = batch_size
        
        # 1. 加载grouond truth状态分布
        self.traj_df = pd.read_csv(trajectory_path)
        
        # 2. 加载 MF Context
        mf_df = pd.read_csv(mf_path)
        self.mf_texts = mf_df['mean_field'].tolist()
        
        # 3. 加载测试集用户流 (Test Users)
        # 这里我们只需要 uid，按顺序排好
        with open(test_data_path, 'r', encoding='utf-8') as f:
            try:
                self.test_users = json.load(f) # 假设是 List[Dict]
            except json.JSONDecodeError:
                # 兼容 JSONL 格式
                f.seek(0)
                self.test_users = [json.loads(line) for line in f]
        
        print(f"✅ 加载测试用户: {len(self.test_users)} 人")

        # 4. 加载并向量化核心用户画像 (构建聚类池)
        # TODO: 已经做了聚类，这里应该是映射
        print("正在构建核心用户聚类池...")
        self.cluster_pools = self._build_cluster_pools(profile_path, encoder_config)
        self.available_clusters = list(self.cluster_pools.keys())
        print(f"✅ 聚类池构建完成: 共有 {len(self.available_clusters)} 个类 ({self.available_clusters})")
        
        # 5. 加载 UID -> Cluster 映射
        self.uid_map = self._load_or_mock_mapping(uid_mapping_path)

    # def _build_cluster_pools(self, jsonl_path, config) -> Dict[str, List[torch.Tensor]]:
    #     """
    #     读取画像 -> BERT编码 -> 按 'stance_label' 分组存储
    #     Returns: { '温和建制派': [Tensor1, Tensor2...], ... }
    #     """
    #     raw_profiles = []
    #     with open(jsonl_path, 'r', encoding='utf-8') as f:
    #         # 修复可能存在的 }{ 粘连
    #         content = f.read().replace('}{', '}\n{')
    #         for line in content.split('\n'):
    #             if not line.strip(): continue
    #             try:
    #                 item = json.loads(line)
    #                 # 提取聚类标签 (优先用 stance_label)
    #                 label = item.get('stance_label', 'unknown')
    #                 # 提取描述文本
    #                 desc = item.get('persona_description') or item.get('desc', "")
    #                 if desc:
    #                     raw_profiles.append({'label': label, 'text': desc})
    #             except:
    #                 continue

    #     # 批量编码
    #     pools = {}
    #     encoder = build_text_encoder(config)
    #     encoder.eval()
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     encoder.to(device)
        
    #     batch_size = 32
    #     texts = [p['text'] for p in raw_profiles]
    #     labels = [p['label'] for p in raw_profiles]
        
    #     all_vecs = []
    #     with torch.no_grad():
    #         for i in range(0, len(texts), batch_size):
    #             batch_txt = texts[i : i+batch_size]
    #             if config['type'] == 'bert':
    #                 inputs = encoder.tokenizer(batch_txt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    #                 emb = encoder(inputs['input_ids'], inputs['attention_mask'])
    #             else:
    #                 emb = torch.randn(len(batch_txt), config.get('emb_dim', 256))
    #             all_vecs.append(emb.cpu())
        
    #     # 分组存储
    #     flat_vecs = torch.cat(all_vecs, dim=0)
    #     for label, vec in zip(labels, flat_vecs):
    #         if label not in pools:
    #             pools[label] = []
    #         pools[label].append(vec)
            
    #     return pools

    def _load_or_mock_mapping(self, path):
        """加载 UID 映射，如果没有则生成随机映射（仅供测试！）"""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️ 警告: 未找到 UID 映射文件 ({path})。将为测试用户随机分配聚类标签以跑通代码。")
            mock_map = {}
            for u in self.test_users:
                # 这里的 key 必须和 test_users 里的 uid 类型一致 (建议统一转 str)
                uid = str(u.get('uid', ''))
                mock_map[uid] = random.choice(self.available_clusters)
            return mock_map

    def __len__(self):
        return len(self.traj_df)

    def __getitem__(self, idx):
        # 1. 获取当前 Batch 对应的测试用户片段
        # 逻辑：Trajectory 的每一行代表一个 Time Step，包含 batch_size 个新用户行为
        # 我们使用循环读取：假设 Test Set 是一个无限流
        start_idx = (idx * self.batch_size) % len(self.test_users)
        
        batch_profiles = []
        
        for i in range(self.batch_size):
            # 获取真实用户的 UID
            curr_idx = (start_idx + i) % len(self.test_users)
            user_item = self.test_users[curr_idx]
            uid = str(user_item.get('uid', ''))
            
            # [核心逻辑] UID -> Cluster -> Random Profile
            cluster_label = self.uid_map.get(uid)
            
            # 容错：如果映射表里没这个uid，或者该类没画像，随机选一个兜底
            if not cluster_label or cluster_label not in self.cluster_pools:
                cluster_label = random.choice(self.available_clusters)
            
            # 从该类中随机采样一个画像向量
            profile_vec = random.choice(self.cluster_pools[cluster_label])
            batch_profiles.append(profile_vec)
            
        # 堆叠画像向量 (16, Hidden_Dim)
        profile_vecs_tensor = torch.stack(batch_profiles)

        # 2. 获取其他信息 (Target Dist, MF Text, Prev Dist)
        row = self.traj_df.iloc[idx]
        target_dist = torch.tensor([
            row['batch_ratio_pos'], row['batch_ratio_neu'], row['batch_ratio_neg']
        ], dtype=torch.float32)

        # 上一时刻状态
        if idx == 0:
            mu_prev = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        else:
            prev_row = self.traj_df.iloc[idx - 1]
            mu_prev = torch.tensor([
                prev_row['cum_ratio_pos'], prev_row['cum_ratio_neu'], prev_row['cum_ratio_neg']
            ], dtype=torch.float32)
            
        # 舆论文本
        mf_text = str(self.mf_texts[idx % len(self.mf_texts)])

        return {
            "mu_prev": mu_prev,
            "mf_text": mf_text,
            "profile_vecs": profile_vecs_tensor, # (16, 768)
            "target_dist": target_dist
        }