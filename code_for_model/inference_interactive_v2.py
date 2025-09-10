# inference_visualize_interactive.py
import json
import torch
import plotly.graph_objects as go
import pandas as pd
from sentence_transformers import SentenceTransformer,models
import torch.nn as nn

# Bert2DMapper 与 train_all_v3.py 保持一致（残差结构）
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
        return out  # (batch, 2)
import os
import time

def load_mapper(model_path, device):
    print(f"加载映射模型从: {model_path}")
    start_time = time.time()
    ckpt = torch.load(model_path, map_location=device)
    mapper = Bert2DMapper(embed_dim=ckpt["embed_dim"], hidden_dims=tuple(ckpt["hidden_dims"]), out_dim=2)
    mapper.load_state_dict(ckpt["mapper_state"])
    mapper.to(device)
    mapper.eval()
    end_time = time.time()
    print(f"映射模型加载完成，耗时: {end_time - start_time:.2f}秒")
    return mapper, ckpt["sbert_name"]

def load_sbert_from_path(model_path, device):
    """从本地路径加载SentenceTransformer模型"""
    print(f"加载SBERT模型从: {model_path}")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
    
    # 检查模型文件
    model_files = os.listdir(model_path)
    print(f"模型目录中的文件: {model_files}")
    
    # 加载模型
    start_time = time.time()
    try:
        # 尝试直接加载模型
        sbert = SentenceTransformer(model_path, device=device)
        print("模型直接加载成功")
    except Exception as e:
        print(f"直接加载失败: {e}")
        print("尝试手动构建模型...")
        
        # 手动构建模型
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        print("手动构建模型成功")
    end_time = time.time()
    print(f"SBERT模型加载完成，耗时: {end_time - start_time:.2f}秒")
    return sbert

def visualize_json(json_path, model_path, sbert_path, device="cpu", save_html="scatter_interactive.html"):
    print("=" * 50)
    print("开始可视化过程")
    print("=" * 50)
    
    # 1) 加载数据
    print(f"加载数据从: {json_path}")
    start_time = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    end_time = time.time()
    print(f"数据加载完成，共 {len(data)} 条记录，耗时: {end_time - start_time:.2f}秒")
    
    texts = []
    labels = []
    full_texts = []  # 存储完整文本用于显示
    metadata = []
    paper_ids = []
    para_ids = []
    with open("case_engine/alldata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # 只保留 rank=5 的 item
    # data = [item for item in metadata if item.get("rank", None) == 5]

    print("处理文本数据...")
    for i, item in enumerate(data):
        # 处理anchor
        # anchor_text = item["anchor"]
        if "sentence" not in item:
            print(f"跳过第 {i} 条记录，缺少 'sentence' 字段")
            continue
        anchor_text = item["sentence"]

        texts.append(anchor_text)  # 只取标题部分
        full_texts.append(anchor_text)  # 保存完整文本
        labels.append("anchor")
        anchor_id = i
        paper_id = metadata[anchor_id]["paper_id"] if anchor_id < len(metadata) else -1
        paper_ids.append(paper_id)
        para_id = metadata[anchor_id]["para_id"] if anchor_id < len(metadata) else -1
        para_ids.append(para_id)
        
        # # 处理positive（如果有）
        # if "positive" in item:
        #     positive_text = item["positive"]
        #     texts.append(positive_text.split("&&", 2)[2])
        #     full_texts.append(positive_text)
        #     labels.append("positive")
        
        # # 处理negative（如果有）
        # if "negative" in item:
        #     negative_text = item["negative"]
        #     texts.append(negative_text.split("&&", 2)[2])
        #     full_texts.append(negative_text)
        #     labels.append("negative")
            
        # 每处理100条记录输出一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 条记录...")
    
    print(f"总共处理了 {len(texts)} 个文本片段")

    # 2) 加载 mapper + sbert
    mapper, sbert_name = load_mapper(model_path, device)
    
    # 使用本地路径加载SBERT模型
    sbert = load_sbert_from_path(sbert_path, device)
    print(f"使用的SBERT模型: {sbert_name}")

    # 3) 编码文本
    print("开始编码文本...")
    start_time = time.time()
    with torch.no_grad():
        embs = sbert.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=True)
        print(f"文本编码完成，嵌入向量形状: {embs.shape}")
        
        # 使用映射模型将嵌入向量映射到2D空间
        print("将嵌入向量映射到2D空间...")
        coords = mapper(embs).cpu().numpy()
        print(f"2D坐标形状: {coords.shape}")
    
    end_time = time.time()
    print(f"编码和映射完成，总耗时: {end_time - start_time:.2f}秒")

    # 4) 创建DataFrame以便于处理
    print("准备可视化数据...")
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'label': labels,
        'text': texts,
        'full_text': full_texts,
        'paper_id': paper_ids,
        'para_id': para_ids,
        'sentence_id': list(range(len(texts)))
    })
    
    # 计算一些统计信息
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    print(f"X坐标范围: [{df['x'].min():.4f}, {df['x'].max():.4f}] (范围: {x_range:.4f})")
    print(f"Y坐标范围: [{df['y'].min():.4f}, {df['y'].max():.4f}] (范围: {y_range:.4f})")
    
    # 5) 创建交互式图表
    print("创建交互式图表...")
    fig = go.Figure()
    
    # 为每种标签添加不同的颜色
    # colors = {"anchor": "red", "positive": "green", "negative": "blue"}
    colors = {0:"red",1:"green"}
    # for label in df['label'].unique():
    #     label_df = df[df['label'] == label]
    #     fig.add_trace(go.Scatter(
    #         x=label_df['x'],
    #         y=label_df['y'],
    #         mode='markers',
    #         name=label,
    #         marker=dict(
    #             color=colors.get(label, "gray"),
    #             size=8,
    #             opacity=0.7
    #         ),
    #         text=label_df['full_text'],  # 悬停时显示的文本
    #         hoverinfo='text',
    #         customdata=label_df.index,  # 保存索引用于点击事件
    #         hovertemplate='<b>%{text}</b><extra></extra>'  # 自定义悬停模板
    #     ))
    #     print(f"添加了 {len(label_df)} 个{label}点")
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        # 为每个点分配颜色，避免 Series 作为 key
        marker_colors = [colors.get(pid, "gray") for pid in label_df['paper_id']]
        fig.add_trace(go.Scatter(
            x=label_df['x'],
            y=label_df['y'],
            mode='markers',
            name=str(label),
            marker=dict(
                color=marker_colors,
                size=8,
                opacity=0.7
            ),
            text=label_df['full_text'],  # 悬停时显示的文本
            hoverinfo='text',
            customdata=label_df.index,  # 保存索引用于点击事件
            hovertemplate='<b>%{text}</b><extra></extra>'  # 自定义悬停模板
        ))
        print(f"添加了 {len(label_df)} 个{label}点")
    # 连线：para_id相同的点
    # for para_id, group in df.groupby('para_id'):
    #     if len(group) > 1:
    #         fig.add_trace(go.Scatter(
    #             x=group['x'],
    #             y=group['y'],
    #             mode='lines',
    #             line=dict(color='rgba(100,100,100,0.3)', width=2),
    #             name=f'para_{para_id}_line',
    #             showlegend=False
    #         ))
    # 连线：sentence_id前后相连（仅当id相差1时才连线）
    for i in range(len(df)-1):
        if abs(df.loc[i, 'sentence_id'] - df.loc[i+1, 'sentence_id']) == 1:
            fig.add_trace(go.Scatter(
                x=[df.loc[i, 'x'], df.loc[i+1, 'x']],
                y=[df.loc[i, 'y'], df.loc[i+1, 'y']],
                mode='lines',
                line=dict(color='rgba(0,0,200,0.2)', width=1),
                name='sentence_order_line',
                showlegend=False
            ))
    
    
    # 更新布局
    fig.update_layout(
        title="2D Mapping Visualization (Hover over points to see text)",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        hovermode='closest',
        width=1000,
        height=700,
        showlegend=True
    )
    
    # 保存为HTML文件
    print(f"保存交互式图表到: {save_html}")
    fig.write_html(save_html)
    
    print("=" * 50)
    print("可视化完成!")
    print(f"请打开 {save_html} 在网页浏览器中查看交互式图表")
    print("=" * 50)
    
    # 显示一些样本点的信息
    print("\n样本点预览:")
    for i in range(min(5, len(df))):
        print(f"点 {i}: ({df.iloc[i]['x']:.4f}, {df.iloc[i]['y']:.4f}) - {df.iloc[i]['label']} - {df.iloc[i]['text'][:50]}...")
    
    return fig

if __name__ == "__main__":
    # 假设你的模型下载到了本地路径
    sbert_local_path = "/home/lxy/bgemodel"  # 替换为你的实际路径
    
    try:
        fig = visualize_json(
            json_path="case_engine/formdatabase.json",        # 你的测试json
            model_path="model_train/pollution_result/bert2d_mapper_all_v3.0.pt",  # 最终训练好的模型
            sbert_path=sbert_local_path,       # 本地SBERT模型路径
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_html="case_engine/scatter_interactive_v2.5.html"
        )
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()