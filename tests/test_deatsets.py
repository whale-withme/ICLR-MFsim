import torch
from torch.utils.data import DataLoader
import sys
import os

# 1. 路径 Hack: 确保能导入 datasets 和 models
# 假设脚本在 MFSim 根目录下运行，如果不是，请调整这里的路径
sys.path.append(os.getcwd()) 

try:
    from training.datasets import StateDataset, trajectory_collate
except ImportError:
    print("导入失败，请检查你的目录结构。确保 datasets.py 在 training/ 文件夹下")
    sys.exit(1)

def test_datasets_pipeline():
    print("=== 开始测试 Dataset 数据流 ===")
    
    # 2. 模拟配置
    # 注意：为了测试快一点，如果不方便加载 BERT，可以把 type 改成 'gru' 或者 mock
    # 但为了验证真实维度，建议还是加载 BERT
    encoder_config = {
        "type": "bert", 
        "model_name": "bert-base-chinese", # 假设数据是中文
        "output_dim": 768,
        "freeze": True
    }
    
    # 3. 实例化 Dataset
    # 请替换为你真实的文件路径
    traj_path = "/root/ICML/data/test_state_distribution/10031994215_trajectory.csv"
    mf_path = "/root/ICML/data/test_mf/10031994215_mf.csv"
    profile_path = "/root/ICML/data/profile/cluster_core_user_profile.jsonl"
    
    if not os.path.exists(traj_path):
        print(f"❌ 找不到文件: {traj_path}，请修改路径")
        return

    print("正在初始化 Dataset (这可能需要几秒钟加载 BERT)...")
    dataset = StateDataset(
        trajectory_path=traj_path,
        mf_path=mf_path,
        profile_path=profile_path,
        encoder_config=encoder_config
    )
    
    print(f"✅ Dataset 初始化成功! 总样本数: {len(dataset)}")
    
    # 4. 测试单个样本 (__getitem__)
    print("\n--- 测试单个样本 (Index 0) ---")
    item = dataset[0]
    
    print(f"Keys: {item.keys()}")
    print(f"Profile Vec Shape: {item['profile_vec'].shape} (期望: [768])")
    print(f"MF Text Preview: {item['mf_text'][:20]}...")
    print(f"Label: {item['label']} (类型: {type(item['label'])})")
    
    # 检查向量是否全是 0 (意味着 ID 没匹配上)
    if torch.sum(item['profile_vec']) == 0:
        print("⚠️ 警告: Profile Vector 全为 0。这意味着 Trajectory 里的 user_id 在 JSONL 里没找到。")
    else:
        print("✅ Profile Vector 数据正常 (非零)。")

    # 5. 测试 DataLoader (__collate_fn__)
    print("\n--- 测试 Batch Loading (Batch Size = 4) ---")
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=trajectory_collate
    )
    
    try:
        batch = next(iter(loader))
        print("Batch 读取成功!")
        print(f"Batch Profile Vecs: {batch['profile_vecs'].shape} (期望: [4, 768])")
        print(f"Batch Labels: {batch['labels'].shape} (期望: [4])")
        print(f"Batch MF Texts 数量: {len(batch['mf_texts'])} (期望: 4)")
        print(f"Batch Mu Prev: {batch['mu_prev'].shape} (期望: [4, 3])")
        print("✅ Collate 函数工作正常")
    except Exception as e:
        print(f"❌ Batch 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_datasets_pipeline()