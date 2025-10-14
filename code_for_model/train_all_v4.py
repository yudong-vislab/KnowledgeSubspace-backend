# =========================================================
# train_all_v4.py  —— 稳定版多损失融合训练, 加入 Neighborhood CE 的稳定版训练脚本
# =========================================================
# pip install torch sentence-transformers tqdm scikit-learn matplotlib

import os
import json
from typing import List, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models
import matplotlib.pyplot as plt

# =========================================================
# 1) 数据集
# =========================================================
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
    anchors, positives, negatives, anchor_idxs, positive_idxs, negative_idxs = zip(*batch)
    a_emb = tokenizer_model.encode(list(anchors), convert_to_tensor=True, device=device).detach()
    p_emb = tokenizer_model.encode(list(positives), convert_to_tensor=True, device=device).detach()
    n_emb = tokenizer_model.encode(list(negatives), convert_to_tensor=True, device=device).detach()
    return a_emb, p_emb, n_emb, anchor_idxs, positive_idxs, negative_idxs


# =========================================================
# 2) 映射模型
# =========================================================
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
            self.norms.append(nn.LayerNorm(h))  # 更稳
            self.activations.append(nn.GELU())
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = h
        self.final = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = x
        for layer, norm, act, drop in zip(self.layers, self.norms, self.activations, self.dropouts):
            out = drop(act(norm(layer(out))))
        out = self.final(out)
        # 每样本 L2 归一化，锁定尺度
        out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        return out  # (batch, 2)


# =========================================================
# 3) 损失函数
# =========================================================
def repulsion_loss(points: torch.Tensor, min_dist=0.3):
    """只惩罚过近的点对，平滑稳定。"""
    b = points.size(0)
    if b < 2:
        return torch.tensor(0.0, device=points.device)
    diffs = points.unsqueeze(1) - points.unsqueeze(0)
    dist = torch.norm(diffs, dim=-1)
    mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), 1)
    loss = torch.relu(min_dist - dist[mask]) ** 2
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=points.device)


def hierarchical_pull_loss(coords: torch.Tensor, metadata: Dict, anchor_idxs: List[int],
                          r_para=0.3, r_same=0.8, r_diff=1.5,
                          w_para=2.0, w_same=1.0, w_diff=2.0):
    """
    带区间的层级约束：
    - 同段落: d ≤ r_para
    - 同论文不同段落: d ≤ r_same
    - 不同论文: d ≥ r_diff
    """
    n = coords.size(0)
    loss = 0.0
    count = 0
    for i in range(n):
        idx_i = anchor_idxs[i]
        meta_i = metadata.get(str(idx_i), {})
        paper_i = meta_i.get("paper_id", -1)
        para_i = meta_i.get("para_id", -1)
        for j in range(i + 1, n):
            idx_j = anchor_idxs[j]
            meta_j = metadata.get(str(idx_j), {})
            paper_j = meta_j.get("paper_id", -2)
            para_j = meta_j.get("para_id", -2)
            dist = torch.norm(coords[i] - coords[j])
            if paper_i == paper_j:
                if para_i == para_j:
                    loss += w_para * torch.relu(dist - r_para) ** 2
                else:
                    loss += w_same * torch.relu(dist - r_same) ** 2
            else:
                loss += w_diff * torch.relu(r_diff - dist) ** 2
            count += 1
    return loss / max(1, count)


def variance_reg(z, gamma=0.5):
    std = z.std(dim=0) + 1e-4
    return torch.relu(gamma - std).mean()


def radius_reg(z, target_r=1.0):
    mean_r = z.norm(dim=1).mean()
    return (mean_r - target_r).pow(2)


triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: torch.norm(x - y, p=2, dim=-1),
    margin=1.0,
    reduction='mean'
)

# ---------- 新增：邻域保持项（高维→低维的本地一致性） ----------
@torch.no_grad()
def _topk_neighbors(dist_mat: torch.Tensor, k: int):
    """返回每个 i 的 top-k 近邻索引（排除自身）"""
    k = min(k, dist_mat.size(1) - 1)
    idx = dist_mat.topk(k + 1, largest=False).indices[:, 1:]
    return idx

def neighborhood_ce(x_hd: torch.Tensor, z_ld: torch.Tensor, t_hd=2.0, t_ld=0.5, k=15):
    """
    x_hd: (N, D)  高维（SBERT）嵌入
    z_ld: (N, 2)  低维映射
    t_hd/t_ld: 温度
    k: 仅用 top-k 邻居形成软目标，稳定高效
    返回：平均交叉熵
    """
    N = x_hd.size(0)
    # 高维距离（不回传梯度）
    with torch.no_grad():
        d_hd = torch.cdist(x_hd, x_hd)  # (N,N)
        idx = _topk_neighbors(d_hd, k)  # (N,k)
        # 软目标分布 P(j|i)
        P = torch.zeros(N, k, device=x_hd.device, dtype=x_hd.dtype)
        for i in range(N):
            nei = idx[i]
            logits = -d_hd[i, nei] / t_hd
            P[i] = torch.softmax(logits, dim=0)

    # 低维分布 Q(j|i)
    d_ld = torch.cdist(z_ld, z_ld)  # (N,N)
    loss = 0.0
    for i in range(N):
        nei = idx[i]
        q_log = torch.log_softmax(-d_ld[i, nei] / t_ld, dim=0)
        loss += -(P[i] * q_log).sum()
    return loss / N


# =========================================================
# 4) 训练
# =========================================================
def train_single_stage(
    json_path: str,
    metadata_path: str,
    sbert_model_name: str = "/home/lxy/bgemodel",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embed_dim: int = 384,
    hidden_dims: tuple = (256, 64),
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    lambda_repulsion: float = 0.4,
    lambda_hierarchical: float = 10.0,
    lambda_neigh: float = 1.0,          # <—— 新增：邻域保持项权重
    freeze_sbert: bool = True,
    save_path: str = "./model_2d_v5.pt"
):
    # ---------- 元数据 ----------
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    metadata_dict = {str(item.get("idx", i)): item for i, item in enumerate(metadata_list)}

    # ---------- SBERT ----------
    try:
        sbert = SentenceTransformer(sbert_model_name, device=device)
        print("模型直接加载成功")
    except Exception as e:
        print(f"直接加载失败: {e}")
        print("尝试手动构建模型...")
        word_embedding_model = models.Transformer(sbert_model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        print("手动构建模型成功")

    sample_emb = sbert.encode("test", convert_to_tensor=True, device=device)
    actual_embed_dim = sample_emb.shape[-1]
    if actual_embed_dim != embed_dim:
        embed_dim = actual_embed_dim
        print(f"调整 embed_dim 为 {embed_dim}")

    # ---------- 数据 ----------
    ds = TripletTextDataset(json_path)
    dataloader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, sbert, device),
        drop_last=True
    )

    # ---------- 模型/优化 ----------
    mapper = Bert2DMapper(embed_dim=embed_dim, hidden_dims=hidden_dims, out_dim=2).to(device)
    params = list(mapper.parameters()) if freeze_sbert else list(mapper.parameters()) + list(sbert.parameters())
    if not freeze_sbert:
        print("警告: 解冻 SBERT 参数")
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ---------- 训练 ----------
    print("Training with Triplet + Repulsion + Hierarchical + NeighborhoodCE + Variance + Radius")
    mapper.train()

    loss_history = {k: [] for k in ["total", "triplet", "repulsion", "hierarchical", "neigh_ce", "variance", "radius"]}

    for epoch in range(epochs):
        total = trip = repel = hier = neigh = varr = radi = 0.0
        n_batches = 0

        for a_emb, p_emb, n_emb, anchor_idxs, positive_idxs, negative_idxs in tqdm(dataloader):
            optimizer.zero_grad()

            # 低维
            a2d, p2d, n2d = mapper(a_emb), mapper(p_emb), mapper(n_emb)
            all2d = torch.cat([a2d, p2d, n2d], dim=0)

            # 高维（作为邻域目标）
            all_hd = torch.cat([a_emb, p_emb, n_emb], dim=0)

            all_idxs = list(anchor_idxs) + list(positive_idxs) + list(negative_idxs)

            # 各项损失
            loss_trip  = triplet_loss_fn(a2d, p2d, n2d)
            loss_repel = repulsion_loss(all2d, min_dist=0.3)
            loss_hier  = hierarchical_pull_loss(all2d, metadata_dict, all_idxs,
                                                r_para=0.3, r_same=0.8, r_diff=1.5)
            loss_neigh = neighborhood_ce(all_hd, all2d, t_hd=2.0, t_ld=0.5, k=15)
            loss_var   = variance_reg(all2d, gamma=0.5)
            loss_rad   = radius_reg(all2d, target_r=1.0)

            loss = (
                1.0 * loss_trip +
                lambda_repulsion * loss_repel +
                lambda_hierarchical * loss_hier +
                lambda_neigh * loss_neigh +
                0.1 * loss_var +
                0.05 * loss_rad
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            total += loss.item(); trip += loss_trip.item()
            repel += loss_repel.item(); hier += loss_hier.item()
            neigh += loss_neigh.item(); varr += loss_var.item(); radi += loss_rad.item()
            n_batches += 1

        scheduler.step()
        hist_vals = [total, trip, repel, hier, neigh, varr, radi]
        for k, v in zip(loss_history.keys(), hist_vals):
            loss_history[k].append(v / n_batches)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"total={loss_history['total'][-1]:.4f}, trip={loss_history['triplet'][-1]:.4f}, "
              f"repel={loss_history['repulsion'][-1]:.4f}, hier={loss_history['hierarchical'][-1]:.4f}, "
              f"neigh={loss_history['neigh_ce'][-1]:.4f}, var={loss_history['variance'][-1]:.4f}, "
              f"rad={loss_history['radius'][-1]:.4f}")

        if (epoch + 1) % 2 == 0:
            ckpt = save_path.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save({
                "mapper_state": mapper.state_dict(),
                "sbert_name": sbert_model_name,
                "embed_dim": embed_dim,
                "hidden_dims": hidden_dims,
                "epoch": epoch,
                "loss_history": loss_history
            }, ckpt)
            print("Checkpoint saved:", ckpt)

    # ---------- 保存模型 ----------
    torch.save({
        "mapper_state": mapper.state_dict(),
        "sbert_name": sbert_model_name,
        "embed_dim": embed_dim,
        "hidden_dims": hidden_dims,
        "loss_history": loss_history
    }, save_path)
    print("Final model saved to", save_path)

    # ---------- 绘图 ----------
    plt.figure(figsize=(11,6))
    for k, v in loss_history.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path.replace(".pt", "_loss.png"))
    plt.close()
    print("Saved loss plot")


# =========================================================
# 5) 运行示例
# =========================================================
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
        lambda_neigh=1.0,                      # 可在 0.5–2.0 间微调
        freeze_sbert=True,
        save_path="pollution_result/bert2d_mapper_all_v5.pt"
    )
