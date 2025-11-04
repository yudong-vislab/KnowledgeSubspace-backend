# =========================================================
# train_all_v4_fixed.py —— 修复“圆环/椭圆”问题的稳定版
# 关键点：区间层级 + 半难负采样(diff) + 局部排斥 + 论文中心
# =========================================================
# pip install torch sentence-transformers tqdm matplotlib

import json
from typing import List, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models
import matplotlib.pyplot as plt
import torch.nn.functional as F


# =========================
# 1) 数据集
# =========================
class TripletTextDataset(Dataset):
    """json list of {anchor, positive, negative, anchor_idx, positive_idx, negative_idx}"""
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        assert isinstance(self.data, list)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        return t["anchor"], t["positive"], t["negative"], t["anchor_idx"], t["positive_idx"], t["negative_idx"]


def collate_fn(batch, tokenizer_model: SentenceTransformer, device="cpu"):
    anchors, positives, negatives, anchor_idxs, positive_idxs, negative_idxs = zip(*batch)
    a_emb = tokenizer_model.encode(list(anchors),  convert_to_tensor=True, device=device).detach()
    p_emb = tokenizer_model.encode(list(positives), convert_to_tensor=True, device=device).detach()
    n_emb = tokenizer_model.encode(list(negatives), convert_to_tensor=True, device=device).detach()
    return a_emb, p_emb, n_emb, list(anchor_idxs), list(positive_idxs), list(negative_idxs)


# =========================
# 2) 映射模型（无 per-sample L2）
# =========================
class Bert2DMapper(nn.Module):
    def __init__(self, embed_dim=768, hidden=(256, 64, 32), out_dim=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden:
            self.layers.append(nn.Linear(in_dim, h))
            self.norms.append(nn.LayerNorm(h))
            in_dim = h
        self.final = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        z = x
        for lin, ln in zip(self.layers, self.norms):
            z = self.drop(self.act(ln(lin(z))))
        z = self.final(z)       # 不做 L2/单位圆归一
        return z


# =========================
# 3) 损失函数
# =========================
def triplet_margin_loss(a, p, n, margin=1.0):
    d_pos = torch.norm(a - p, p=2, dim=-1)
    d_neg = torch.norm(a - n, p=2, dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()

def local_repulsion(points: torch.Tensor, k=15, min_dist=0.22):
    """仅对 top-k 近邻施加最小间距，不做全局均匀化"""
    n = points.size(0)
    if n < 2: return torch.tensor(0.0, device=points.device)
    D = torch.cdist(points, points)
    k = min(k, n-1)
    idx = D.topk(k+1, largest=False).indices[:,1:]
    rows = torch.arange(n, device=points.device).unsqueeze(1).expand_as(idx)
    near = D[rows, idx]
    return F.relu(min_dist - near).pow(2).mean()

def hierarchical_band_loss(coords: torch.Tensor, metadata: Dict, idxs: List[int],
                           para_low=0.18, para_high=0.45,
                           same_low=0.55, same_high=1.20,
                           w_para=2.0, w_same=1.0):
    """同段/同论文双边带宽（只惩罚越界），不对异论文做下限——交给 diff_sampled"""
    n = coords.size(0); loss=0.0; c=0
    for i in range(n):
        mi = metadata.get(str(idxs[i]), {}); pi, si = mi.get("paper_id",-1), mi.get("para_id",-1)
        for j in range(i+1, n):
            mj = metadata.get(str(idxs[j]), {}); pj, sj = mj.get("paper_id",-2), mj.get("para_id",-2)
            if pi != pj: continue
            d = torch.norm(coords[i]-coords[j])
            if si == sj:
                loss += w_para*(F.relu(d-para_high)**2 + F.relu(para_low-d)**2)
            else:
                loss += w_same*(F.relu(d-same_high)**2 + F.relu(same_low-d)**2)
            c += 1
    return loss / max(1, c)

def diff_sampled_margin(coords_2d: torch.Tensor, embeds_hd: torch.Tensor,
                        metadata: Dict, idxs: List[int], K=10, diff_low=1.35, w_diff=2.0):
    """
    只对每个点的 K 个“半难负样本”(异论文、高维相近)施加 d>=diff_low 的下限。
    这样不会用“所有 diff pair”把半径统统往外推。
    """
    N = coords_2d.size(0)
    if N < 2: return torch.tensor(0.0, device=coords_2d.device)
    # 高维近邻（找候选）
    D_hd = torch.cdist(embeds_hd, embeds_hd)                    # (N,N)
    _, hd_idx = D_hd.topk(min(K+1, N), largest=False)
    hd_idx = hd_idx[:,1:]                                       # 去掉自身
    # 低维距离
    D_ld = torch.cdist(coords_2d, coords_2d)                    # (N,N)
    loss = 0.0; cnt = 0
    for i in range(N):
        mi = metadata.get(str(idxs[i]), {}); pi = mi.get("paper_id",-1)
        for j in hd_idx[i]:
            j = j.item()
            mj = metadata.get(str(idxs[j]), {}); pj = mj.get("paper_id",-2)
            if pi == pj: continue
            d = D_ld[i, j]
            loss += w_diff * F.relu(diff_low - d)**2
            cnt += 1
    return loss / max(1, cnt)

def paper_center_loss(z2d: torch.Tensor, paper_ids: List[int], pull_w=0.5, sep_w=0.5, margin=1.0):
    """
    批内“论文中心”拉近 + 中心分离：让簇在平面里散开而不是靠同半径区分
    """
    device = z2d.device
    ids = torch.tensor(paper_ids, device=device)
    uniq = ids.unique()
    centers = []
    pull = 0.0
    for pid in uniq:
        mask = (ids == pid)
        c = z2d[mask].mean(0)
        centers.append(c)
        pull += ((z2d[mask] - c)**2).sum(dim=1).mean()
    pull = pull / max(1, len(uniq))
    if len(uniq) < 2:
        sep = torch.tensor(0.0, device=device)
    else:
        C = torch.stack(centers)                # (G,2)
        Dc = torch.cdist(C, C)
        mask = torch.triu(torch.ones_like(Dc, dtype=torch.bool), 1)
        sep = F.relu(margin - Dc[mask]).pow(2).mean()
    return pull_w * pull + sep_w * sep

def neighborhood_ce(x_hd: torch.Tensor, z_ld: torch.Tensor, t_hd=2.0, t_ld=0.75, k=10):
    """温和邻域保持（可关）。给低维更多自由度，避免固化成环。"""
    N = x_hd.size(0)
    if N < 3: return torch.tensor(0.0, device=x_hd.device)
    with torch.no_grad():
        d_hd = torch.cdist(x_hd, x_hd)
        k = min(k, N-1)
        idx = d_hd.topk(k+1, largest=False).indices[:,1:]
        P = torch.zeros(N, k, device=x_hd.device, dtype=x_hd.dtype)
        for i in range(N):
            logits = -d_hd[i, idx[i]]/t_hd
            P[i] = torch.softmax(logits, dim=0)
    d_ld = torch.cdist(z_ld, z_ld)
    loss = 0.0
    for i in range(N):
        q_log = torch.log_softmax(-d_ld[i, idx[i]]/t_ld, dim=0)
        loss += -(P[i]*q_log).sum()
    return loss / N

def variance_reg(z, gamma=0.30):
    std = z.std(dim=0) + 1e-4
    return F.relu(gamma - std).mean()


# =========================
# 4) 训练
# =========================
def train_single_stage(
    json_path: str,
    metadata_path: str,
    sbert_model_name: str = "/home/lxy/bgemodel",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embed_dim: int = 384,
    hidden_dims: tuple = (256, 64, 32),
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    # 损失权重（先让簇在平面里成形，再净化边界）
    w_triplet: float = 1.0,
    w_repulsion: float = 0.30,
    w_hier: float = 6.0,
    w_diff: float = 2.0,
    w_center: float = 0.50,
    w_neigh: float = 0.50,     # 如仍有粘连可加到 0.8
    w_var: float = 0.03,
    freeze_sbert: bool = True,
    save_path: str = "./model_2d_v4_fixed.pt"
):
    # ---- 元数据 ----
    with open(metadata_path, "r", encoding="utf-8") as f:
        metas = json.load(f)
    meta = {str(it.get("idx", i)): it for i, it in enumerate(metas)}

    # ---- SBERT ----
    try:
        sbert = SentenceTransformer(sbert_model_name, device=device)
    except Exception:
        word_embedding_model = models.Transformer(sbert_model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True)
        sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    sample_emb = sbert.encode("test", convert_to_tensor=True, device=device)
    if sample_emb.shape[-1] != embed_dim:
        embed_dim = sample_emb.shape[-1]

    # ---- DataLoader ----
    ds = TripletTextDataset(json_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, sbert, device), drop_last=True)

    # ---- 模型/优化器 ----
    mapper = Bert2DMapper(embed_dim=embed_dim, hidden=hidden_dims, out_dim=2).to(device)
    params = list(mapper.parameters()) if freeze_sbert else list(mapper.parameters())+list(sbert.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("Train: Triplet + 局部排斥 + 区间层级(同论文) + 采样异论文下限 + 论文中心 + 温和邻域 + 方差下限")
    loss_hist = {k: [] for k in ["total","triplet","repel","hier","diff","center","neigh","var"]}

    for ep in range(epochs):
        tot = {k:0.0 for k in loss_hist}
        n_batches = 0

        for a_emb, p_emb, n_emb, a_idx, p_idx, n_idx in tqdm(dl):
            optimizer.zero_grad()
            a2d, p2d, n2d = mapper(a_emb), mapper(p_emb), mapper(n_emb)
            all2d = torch.cat([a2d, p2d, n2d], dim=0)
            allhd = torch.cat([a_emb, p_emb, n_emb], dim=0)
            allidx = a_idx + p_idx + n_idx
            paper_ids = [meta[str(i)].get("paper_id",-1) for i in allidx]

            L_trip  = triplet_margin_loss(a2d, p2d, n2d, margin=1.0)
            L_rep   = local_repulsion(all2d, k=15, min_dist=0.22)
            L_hier  = hierarchical_band_loss(all2d, meta, allidx,
                                             para_low=0.18, para_high=0.45,
                                             same_low=0.55, same_high=1.20)
            L_diff  = diff_sampled_margin(all2d, allhd, meta, allidx, K=10, diff_low=1.35, w_diff=2.0)
            L_center= paper_center_loss(all2d, paper_ids, pull_w=0.5, sep_w=0.5, margin=1.0)
            L_neigh = neighborhood_ce(allhd, all2d, t_hd=2.0, t_ld=0.75, k=10)
            L_var   = variance_reg(all2d, gamma=0.30)

            loss = (w_triplet*L_trip + w_repulsion*L_rep + w_hier*L_hier +
                    w_diff*L_diff + w_center*L_center + w_neigh*L_neigh + w_var*L_var)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            tot["total"] += loss.item()
            tot["triplet"]+= L_trip.item(); tot["repel"]+=L_rep.item(); tot["hier"]+=L_hier.item()
            tot["diff"]+=L_diff.item(); tot["center"]+=L_center.item(); tot["neigh"]+=L_neigh.item(); tot["var"]+=L_var.item()
            n_batches += 1

        scheduler.step()
        for k in loss_hist: loss_hist[k].append(tot[k]/n_batches)

        print(f"[Ep {ep+1}/{epochs}] total={loss_hist['total'][-1]:.4f} "
              f"trip={loss_hist['triplet'][-1]:.3f} repel={loss_hist['repel'][-1]:.3f} "
              f"hier={loss_hist['hier'][-1]:.3f} diff={loss_hist['diff'][-1]:.3f} "
              f"center={loss_hist['center'][-1]:.3f} neigh={loss_hist['neigh'][-1]:.3f} var={loss_hist['var'][-1]:.3f}")

    # ---- 保存 ----
    torch.save({
        "mapper_state": mapper.state_dict(),
        "sbert_name": sbert_model_name,
        "embed_dim": embed_dim,
        "hidden_dims": hidden_dims,
        "loss_history": loss_hist,
    }, save_path)
    print("Saved to:", save_path)

    # ---- 绘图 ----
    plt.figure(figsize=(10,6))
    for k,v in loss_hist.items():
        plt.plot(v, label=k)
    plt.legend(); plt.grid(True)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss History (v4_fixed)")
    plt.savefig(save_path.replace(".pt","_loss.png")); plt.close()
    print("Saved loss curve")


# =========================
# 运行示例（与你当前路径一致）
# =========================
if __name__ == "__main__":
    train_single_stage(
        json_path="pollution_result/contrastive_triplets_with_context_all_database_v2.0.json",
        metadata_path="pollution_result/formdatabase_v2.0.json",
        sbert_model_name="/home/lxy/bgemodel",
        embed_dim=384,
        hidden_dims=(256, 64, 32),
        batch_size=128,
        epochs=20,
        lr=1e-3,
        save_path="pollution_result/bert2d_mapper_all_v4_fixed.pt"
    )
