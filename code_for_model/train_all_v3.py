# requirements:
# pip install torch sentence-transformers tqdm scikit-learn matplotlib

import os
import json
import math
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 移除了 KeyBERT 导入，因为它不是必需的
from sentence_transformers import SentenceTransformer, models
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1) 数据集
# -----------------------
class TripletTextDataset(Dataset):
    """
    Expect json list of {"anchor": "...", "positive": "...", "negative": "...", 
                        "anchor_idx": int, "positive_idx": int, "negative_idx": int}
    """
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        assert isinstance(self.data, list), "json must be a list of triplets"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        return t["anchor"], t["positive"], t["negative"], t["anchor_idx"], t["positive_idx"], t["negative_idx"]


def collate_fn(batch, tokenizer_model: SentenceTransformer, device="cpu"):
    """Turn batch of triplet texts into tensors of embeddings (or tokens if you prefer)"""
    anchors, positives, negatives, anchor_idxs, positive_idxs, negative_idxs = zip(*batch)
    # use SBERT.encode (fast) to produce numpy arrays, then tensor
    # encode with batch_size large for speed; we rely on SBERT inside CPU/GPU depending on model
    a_emb = tokenizer_model.encode(list(anchors), convert_to_tensor=True, device=device).clone().detach()
    p_emb = tokenizer_model.encode(list(positives), convert_to_tensor=True, device=device).clone().detach()
    n_emb = tokenizer_model.encode(list(negatives), convert_to_tensor=True, device=device).clone().detach()
    return a_emb, p_emb, n_emb, anchor_idxs, positive_idxs, negative_idxs


# -----------------------
# 2) 模型：SBERT（可选冻结） + MLP -> 2d
# -----------------------
class Bert2DMapper(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=(256, 64), out_dim=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h))
            self.norms.append(nn.BatchNorm1d(h))
            self.activations.append(nn.GELU())
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = h
        self.final = nn.Linear(in_dim, out_dim)
        # 不在这里做 tanh；留给训练后或loss控制

    def forward(self, x):
        # x: (batch, embed_dim)
        out = x
        for layer, norm, act, drop in zip(self.layers, self.norms, self.activations, self.dropouts):
            residual = out
            out = layer(out)
            out = norm(out)
            out = act(out)
            out = drop(out)
            # 残差连接：仅当输入输出维度一致时
            if residual.shape[-1] == out.shape[-1]:
                out = out + residual
        out = self.final(out)
        out = out / out.max(dim=0, keepdim=True)[0]  # 归一化到 [0, 1]
        out = out * 10  # 缩放到 [0, 10]
        return out  # (batch, 2)


# -----------------------
# 3) Repulsion term（batch内pairwise）
# -----------------------
def repulsion_loss(points: torch.Tensor, eps=1e-3):
    """
    points: (batch, 2)
    L_rep = sum_{i<j} 1 / (||p_i - p_j||^2 + eps)
    normalize by n_pairs to keep scale stable
    """
    b = points.size(0)
    if b < 2:
        return torch.tensor(0.0, device=points.device)
    # pairwise squared distances
    diffs = points.unsqueeze(1) - points.unsqueeze(0)  # (b,b,2)
    dist2 = (diffs ** 2).sum(dim=-1) + torch.eye(b, device=points.device) * 1e12  # mask diagonal large
    inv = 1.0 / (dist2 + eps)
    # only upper triangle
    triu_indices = torch.triu_indices(b, b, offset=1)
    vals = inv[triu_indices[0], triu_indices[1]]
    return vals.mean()  # mean keeps scale stable


# -----------------------
# 4) 层级拉近损失函数
# -----------------------
def hierarchical_pull_loss(coords: torch.Tensor, metadata: Dict, anchor_idxs: List[int], 
                          same_paper_weight=2.0, same_para_weight=4.0, diff_weight=1.0):
    """
    基于para_id和paper_id的层级拉近损失
    coords: (batch, 2) 坐标
    metadata: 包含所有句子元数据的字典，索引为键
    anchor_idxs: 当前batch中anchor的索引列表
    """
    n = coords.size(0)
    loss = 0.0
    count = 0
    
    for i in range(n):
        idx_i = anchor_idxs[i]
        meta_i = metadata.get(str(idx_i), {})
        paper_id_i = meta_i.get("paper_id", -1)
        para_id_i = meta_i.get("para_id", -1)
        
        for j in range(i+1, n):
            idx_j = anchor_idxs[j]
            meta_j = metadata.get(str(idx_j), {})
            paper_id_j = meta_j.get("paper_id", -2)
            para_id_j = meta_j.get("para_id", -2)
            
            # 计算当前距离
            dist = torch.norm(coords[i] - coords[j])
            
            # 根据层级关系确定目标距离和权重
            if paper_id_i == paper_id_j:
                if para_id_i == para_id_j:
                    # 同一段落，应该非常接近
                    target_dist = 1
                    weight = same_para_weight
                else:
                    # 同一论文但不同段落，应该比较接近
                    target_dist = 10
                    weight = same_paper_weight
            else:
                # 不同论文，应该较远
                target_dist = 30.0
                weight = diff_weight
            
            # 计算距离差异的平方损失
            loss += weight * (dist - target_dist).pow(2)
            count += 1
    
    return loss / (count if count > 0 else 1)


# -----------------------
# 5) 辅助：pairwise/ triplet loss
# -----------------------
triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: torch.norm(x - y, p=2, dim=-1),
    margin=5, 
    reduction='mean'
)


# -----------------------
# 6) 单阶段训练函数
# -----------------------
def train_single_stage(
    json_path: str,
    metadata_path: str,
    sbert_model_name: str = "/home/lxy/bgemodel",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embed_dim: int = 384,
    hidden_dims: tuple = (256, 64),
    batch_size: int = 128,
    epochs: int = 8,
    lr: float = 1e-3,
    lambda_repulsion: float = 1.0,
    lambda_hierarchical: float = 5.0,
    freeze_sbert: bool = True,
    save_path: str = "./model_2d.pt"
):
    # 加载元数据
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    
    # 转换为以索引为键的字典
    metadata_dict = {str(item.get("idx", i)): item for i, item in enumerate(metadata_list)}
    
    # 加载 SentenceTransformer 模型
    try:
        # 尝试直接加载模型
        sbert = SentenceTransformer(sbert_model_name, device=device)
        print("模型直接加载成功")
    except Exception as e:
        print(f"直接加载失败: {e}")
        print("尝试手动构建模型...")
        
        # 手动构建模型
        word_embedding_model = models.Transformer(sbert_model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        print("手动构建模型成功")
    
    # 检测嵌入维度
    sample_emb = sbert.encode("test", convert_to_tensor=True, device=device)
    actual_embed_dim = sample_emb.shape[-1]
    print("Detected embedding dim:", actual_embed_dim)
    if actual_embed_dim != embed_dim:
        embed_dim = actual_embed_dim

    # 数据集和数据加载器
    ds = TripletTextDataset(json_path)
    dataloader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, sbert, device), 
        drop_last=False
    )

    # 创建映射模型
    mapper = Bert2DMapper(embed_dim=embed_dim, hidden_dims=hidden_dims, out_dim=2).to(device)

    # 冻结 SBERT 参数
    if freeze_sbert:
        trainable_params = list(mapper.parameters())
    else:
        # 如果不冻结 SBERT，需要将其参数也加入优化器
        trainable_params = list(mapper.parameters()) + list(sbert.parameters())
        print("警告: 解冻 SBERT 参数进行微调")

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ---------- 单阶段：Triplet + Repulsion + 层级损失 ----------
    print("Single Stage: training with Triplet + Repulsion + Hierarchical Loss")
    mapper.train()
    
    # 记录损失历史
    loss_history = {
        'total': [],
        'triplet': [],
        'repulsion': [],
        'hierarchical': []
    }
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_trip_loss = 0.0
        total_repel_loss = 0.0
        total_hier_loss = 0.0
        n_batches = 0
        
        for a_emb, p_emb, n_emb, anchor_idxs, positive_idxs, negative_idxs in tqdm(dataloader):
            optimizer.zero_grad()
            a2d = mapper(a_emb)  # (B,2)
            p2d = mapper(p_emb)
            n2d = mapper(n_emb)
            
            # losses
            loss_trip = triplet_loss_fn(a2d, p2d, n2d)
            loss_repel = repulsion_loss(a2d)  # only repel anchors in batch
            loss_hier = hierarchical_pull_loss(a2d, metadata_dict, anchor_idxs)
            
            loss = loss_trip + lambda_repulsion * loss_repel + lambda_hierarchical * loss_hier
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_trip_loss += loss_trip.item()
            total_repel_loss += loss_repel.item()
            total_hier_loss += loss_hier.item()
            n_batches += 1
            
        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        avg_trip = total_trip_loss / max(1, n_batches)
        avg_repel = total_repel_loss / max(1, n_batches)
        avg_hier = total_hier_loss / max(1, n_batches)
        
        # 记录损失历史
        loss_history['total'].append(avg_loss)
        loss_history['triplet'].append(avg_trip)
        loss_history['repulsion'].append(avg_repel)
        loss_history['hierarchical'].append(avg_hier)
        
        print(f"[SingleStage] Epoch {epoch+1}/{epochs} avg loss: {avg_loss:.4f} "
              f"(triplet: {avg_trip:.4f}, repel: {avg_repel:.4f}, hier: {avg_hier:.4f})")
        
        # 每2个epoch保存一次检查点
        if (epoch + 1) % 2 == 0:
            checkpoint_path = save_path.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save({
                "mapper_state": mapper.state_dict(),
                "sbert_name": sbert_model_name,
                "embed_dim": embed_dim,
                "hidden_dims": hidden_dims,
                "loss_history": loss_history,
                "epoch": epoch
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # final save
    torch.save({
        "mapper_state": mapper.state_dict(),
        "sbert_name": sbert_model_name,
        "embed_dim": embed_dim,
        "hidden_dims": hidden_dims,
        "loss_history": loss_history
    }, save_path)
    print("Saved final model to", save_path)
    
    # 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['total'], label='Total Loss')
        plt.plot(loss_history['triplet'], label='Triplet Loss')
        plt.plot(loss_history['repulsion'], label='Repulsion Loss')
        plt.plot(loss_history['hierarchical'], label='Hierarchical Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path.replace('.pt', '_loss.png'))
        plt.close()
        print("Saved loss plot")
    except ImportError:
        print("Matplotlib not available, skipping loss plot")


# -----------------------
# 7) 使用示例
# -----------------------
if __name__ == "__main__":
    train_single_stage(
        json_path="pollution_result/contrastive_triplets_with_context_all_database_v2.0.json",
        metadata_path="pollution_result/formdatabase_v2.0.json",
        sbert_model_name="/home/lxy/bgemodel",
        device="cuda" if torch.cuda.is_available() else "cpu",
        embed_dim=384,
        hidden_dims=(256, 64, 32),
        batch_size=128,
        epochs=20,
        lr=1e-3,
        lambda_repulsion=0.4,
        lambda_hierarchical=10.0,
        freeze_sbert=True,
        save_path="pollution_result/bert2d_mapper_all_v3.1.pt"
    )