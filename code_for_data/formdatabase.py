import json
import torch
from sentence_transformers import SentenceTransformer, models
import os
paragraphs = []
papers = []
# 1. 读取 alldata.json
# with open('case_engine/alldata_processed.json', 'r', encoding='utf-8') as f:
#     alldata = json.load(f)

# # 2. 读取 paragraphs.json 和 papers.json
# with open('case_engine/paragraphs.json', 'r', encoding='utf-8') as f:
#     para_data = json.load(f)
#     if isinstance(para_data, list):
#         paragraphs = {str(i): p for i, p in enumerate(para_data)}
#     elif isinstance(para_data, dict):
#         paragraphs = para_data
#     else:
#         raise ValueError('paragraphs.json 格式不支持')
# with open('case_engine/papers.json', 'r', encoding='utf-8') as f:
#     paper_data = json.load(f)
#     if isinstance(paper_data, list):
#         papers = {str(i): p for i, p in enumerate(paper_data)}
#     elif isinstance(paper_data, dict):
#         papers = paper_data
#     else:
#         raise ValueError('papers.json 格式不支持')
with open('case_engine/formdatabase.json', 'r', encoding='utf-8') as f:
    alldata = json.load(f)

# 3. 加载SBERT和映射模型（参考 inference_interactive.py）
sbert_path = '/home/lxy/bgemodel'  # 修改为你的实际路径
model_path = 'model_train/pollution_result/bert2d_mapper_all_v3.0.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SBERT加载
try:
    sbert = SentenceTransformer(sbert_path, device=device)
except Exception:
    word_embedding_model = models.Transformer(sbert_path)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# Bert2DMapper定义（与训练一致）
import torch.nn as nn
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
    def forward(self, x):
        out = x
        for layer, norm, act, drop in zip(self.layers, self.norms, self.activations, self.dropouts):
            residual = out
            out = layer(out)
            out = norm(out)
            out = act(out)
            out = drop(out)
            if residual.shape[-1] == out.shape[-1]:
                out = out + residual
        out = self.final(out)
        return out

ckpt = torch.load(model_path, map_location=device)
mapper = Bert2DMapper(embed_dim=ckpt['embed_dim'], hidden_dims=tuple(ckpt['hidden_dims']), out_dim=2)
mapper.load_state_dict(ckpt['mapper_state'])
mapper.to(device)
mapper.eval()

# 4. 构建新数据
newdata = []
with torch.no_grad():
    for idx, item in enumerate(alldata):
        sentence = item.get('sentence', '')
        emb = sbert.encode(sentence, convert_to_tensor=True, device=device).unsqueeze(0)
        coords = mapper(emb).cpu().numpy().tolist()[0]
        item['2d_coord'] = coords
        # item['para_info'] = paragraphs.get(str(item.get('para_id', '-1')), None)
        # item['paper_info'] = papers.get(str(item.get('paper_id', '-1')), None)
        newdata.append(item)

# 5. 保存到 formdatabase.json
with open('case_engine/formdatabase_v2.0.json', 'w', encoding='utf-8') as f:
    json.dump(newdata, f, ensure_ascii=False, indent=2)

