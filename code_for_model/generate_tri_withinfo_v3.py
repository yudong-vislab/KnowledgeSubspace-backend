import json
import random
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util,models
from keybert import KeyBERT
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import os
import torch

# 下载 punkt 分词模型（运行一次）
print("working...")
# nltk.download("punkt")
model_path = "/home/lxy/bgemodel"
print(model_path)
# 自定义缩写，避免在 Fig.、e.g. 等处误断
# ABBREVS = ["fig", "figs", "e.g", "i.e", "al", "etc"]
# punkt_params = PunktParameters()
# punkt_params.abbrev_types = set(ABBREVS)
# SENT_TOKENIZER = PunktSentenceTokenizer(punkt_params)

# 初始化模型 - 使用更强的BGE模型
# embed_model = SentenceTransformer("bge-base-en-v2")  # 或 "BAAI/bge-large-en"
device="cuda" if torch.cuda.is_available() else "cpu",

try:
    # 尝试直接加载模型
    sbert = SentenceTransformer(model_path)
    print("模型直接加载成功")
except Exception as e:
    print(f"直接加载失败: {e}")
    print("尝试手动构建模型...")
    # 手动构建模型
    word_embedding_model = models.Transformer(model_path)
    word_embedding_dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(
        1024,
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("手动构建模型成功")
kw_model = KeyBERT(model=embed_model)

def extract_sentences_from_text(text):
    """从文本中提取句子"""
    return SENT_TOKENIZER.tokenize(text)

def load_and_process_data(data_path):
    """加载并处理数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 只提取所有句子
    all_sentences = []
    sentence_to_id = {}

    for idx, item in enumerate(data):
        sentence = item.get("sentence", "").strip()
        if sentence:
            all_sentences.append(sentence)
            sentence_to_id[sentence] = idx  # id即为alldata中的索引

    return all_sentences, sentence_to_id

def build_sentence_windows(sentences, window_size=1):
    """为每个句子构建上下文窗口"""
    sentence_windows = {}
    n = len(sentences)
    
    for i, sentence in enumerate(sentences):
        # 获取上下文句子
        start_idx = max(0, i - window_size)
        end_idx = min(n, i + window_size + 1)
        
        context_sentences = sentences[start_idx:end_idx]
        window_text = " . ".join(context_sentences) + " ."
        
        sentence_windows[sentence] = window_text
    
    return sentence_windows

def embedding_similarity(texts):
    """计算嵌入相似度（不使用上下文窗口）"""
    # 直接对原始文本进行嵌入
    embeddings = embed_model.encode(texts, normalize_embeddings=True)
    return cosine_similarity(embeddings)
def window_embed_similarity(texts, windows_dict):
    """计算嵌入相似度（使用上下文窗口）"""
    # 获取窗口文本
    window_texts = [windows_dict[text] for text in texts]
    
    # 计算嵌入
    embeddings = embed_model.encode(window_texts, normalize_embeddings=True)
    
    return cosine_similarity(embeddings)
def tfidf_similarity(texts):
    """计算TF-IDF相似度"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)

def keybert_similarity(texts, top_n=5):
    """计算KeyBERT关键词相似度"""
    all_keywords = []
    for text in texts:
        keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words="english")
        all_keywords.append(" ".join([kw for kw, _ in keywords]))
    vectorizer = TfidfVectorizer()
    kw_matrix = vectorizer.fit_transform(all_keywords)
    return cosine_similarity(kw_matrix)

def combined_similarity(texts, weights={"embedding":0.5, "tfidf":0.3, "keybert":0.2}):
    """计算融合相似度"""
    sim_embed = embedding_similarity(texts)
    sim_tfidf = tfidf_similarity(texts)
    sim_kw = keybert_similarity(texts)

    combined = (weights["embedding"] * sim_embed +
                weights["tfidf"] * sim_tfidf +
                weights["keybert"] * sim_kw)
    
    return {
        "embedding": sim_embed,
        "tfidf": sim_tfidf,
        "keybert": sim_kw,
        "combined": combined
    }

def plot_similarity_heatmaps(sim_dict, labels, output_dir="output"):
    """可视化相似度矩阵热力图"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for title, matrix in sim_dict.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, fmt=".2f", cmap="YlGnBu",
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{title}_heatmap.png")
        plt.close()

def generate_triplets_topk(texts, sim_matrix, pos_threshold=0.7, neg_threshold=0.3, top_k=5):
    """生成三元组（Top-K策略）"""
    n = len(texts)
    triplets = []
    
    for anchor_idx in range(n):
        positives = [(i, sim_matrix[anchor_idx][i]) 
                     for i in range(n) if i != anchor_idx and sim_matrix[anchor_idx][i] >= pos_threshold]
        negatives = [(i, sim_matrix[anchor_idx][i]) 
                     for i in range(n) if i != anchor_idx and sim_matrix[anchor_idx][i] <= neg_threshold]
        
        # 如果没有正样本或负样本，就跳过
        if not positives or not negatives:
            continue
        
        # 生成候选三元组，并计算优先级（相似度差）
        candidate_triplets = []
        for p_idx, p_sim in positives:
            for n_idx, n_sim in negatives:
                score = p_sim - n_sim  # 越大越好
                candidate_triplets.append((score, texts[anchor_idx], texts[p_idx], texts[n_idx]))
        
        # 排序并取 top_k
        candidate_triplets.sort(key=lambda x: x[0], reverse=True)
        top_triplets = candidate_triplets[:top_k]
        
        # 添加到结果
        for _, anchor, pos, neg in top_triplets:
            triplets.append((anchor, pos, neg))
    
    return triplets

def generate_category_based_triplets(sentences, sentence_to_category, num_triplets_per_anchor=3):
    """基于类别信息生成三元组"""
    # 按类别分组句子
    category_to_sentences = defaultdict(list)
    for sent in sentences:
        if sent in sentence_to_category:
            category_to_sentences[sentence_to_category[sent]].append(sent)
    
    # 如果没有类别信息，返回空列表
    if not category_to_sentences:
        return []
    
    categories = list(category_to_sentences.keys())
    triplets = []
    
    for sent in sentences:
        if sent not in sentence_to_category:
            continue
            
        anchor_category = sentence_to_category[sent]
        
        # 正样本：同类别
        positive_candidates = [s for s in category_to_sentences[anchor_category] if s != sent]
        if not positive_candidates:
            continue
            
        # 负样本：不同类别
        negative_candidates = []
        for cat in categories:
            if cat != anchor_category:
                negative_candidates.extend(category_to_sentences[cat])
        
        if not negative_candidates:
            continue
            
        # 生成多个三元组
        for _ in range(num_triplets_per_anchor):
            positive = random.choice(positive_candidates)
            negative = random.choice(negative_candidates)
            triplets.append((sent, positive, negative))
    
    return triplets

def main():
    # 加载数据
    # data_path = "pollution_result/alldata.json"    
    data_path = "pollution_result/formdatabase_v2.0.json"
    plot_output = "pollution_result/similarity_plots_v2.0"
    output_path = "pollution_result/contrastive_triplets_with_context_all_database_v2.0.json"

    # trainingdata_sentences = []
    # trainingdata_sentences_to_category = {}
    # trainingdata_sentences_to_paragraph = {}
    # trainingdata_sentences_to_paper = {}
    # for filename in os.listdir(input_dir):
    #     data_path = os.path.join(input_dir, filename, f"{filename}-rewrite.json")
    #     if not os.path.isfile(data_path):
    #         print(f"跳过不存在的文件: {data_path}")
    #         continue
    print(f"Loading data from {data_path}...")
    sentences, sentence_to_idx = load_and_process_data(data_path)
    print(f"Loaded {len(sentences)} unique sentences")
    # 构建句子窗口
    window_size = 1  # 前后各取1句
    # sentence_windows = build_sentence_windows(sentences, window_size)



    # 计算相似度
    weights = {"embedding": 0.4, "tfidf": 0.3, "keybert": 0.3}
    sim_matrices = combined_similarity(sentences, weights)
    # 可视化相似度矩阵
    labels = [f"Sent{i+1}" for i in range(len(sentences))]
    plot_similarity_heatmaps(sim_matrices, labels, plot_output)
    
    # 生成基于相似度的三元组
    all_triplets = generate_triplets_topk(
        sentences, 
        sim_matrices["combined"], 
        pos_threshold=0.55, 
        neg_threshold=0.3, 
        top_k=10
    )
    
    # 生成基于类别的三元组
    # triplets_category = generate_category_based_triplets(sentences, sentence_to_category)
    
    # 合并三元组
        # all_triplets = triplets_sim + triplets_category
    # all_triplets = triplets_sim
    # 去重
    unique_triplets = list(set(all_triplets))
    
    # print(f"Generated {len(unique_triplets)} triplets (similarity-based: {len(triplets_sim)}, category-based: {len(triplets_category)})")
    print(f"Generated {len(unique_triplets)} triplets (similarity-based: {len(all_triplets)}")
    # 保存三元组

    triplets_data = []
    for anchor, pos, neg in unique_triplets:
        # 获取上下文窗口
        # anchor_window = sentence_windows[anchor]
        # pos_window = sentence_windows[pos]
        # neg_window = sentence_windows[neg]
        
        # 获取类别信息（如果存在）
        # anchor_category = sentence_to_category.get(anchor, "Unknown")
        # pos_category = sentence_to_category.get(pos, "Unknown")
        # neg_category = sentence_to_category.get(neg, "Unknown")
        # anchor_paper = sentence_to_paper.get(anchor, "Unknown")
        # pos_paper = sentence_to_paper.get(pos, "Unknown")   
        # neg_paper = sentence_to_paper.get(neg, "Unknown")
        
        triplets_data.append({
            "anchor": anchor,
            "positive": pos,
            "negative": neg,
            "anchor_idx": sentence_to_idx.get(anchor, -1),
            "positive_idx": sentence_to_idx.get(pos, -1),   
            "negative_idx": sentence_to_idx.get(neg, -1),
            # "anchor_window": anchor_window,
            # "positive_window": pos_window,
            # "negative_window": neg_window,
            # "anchor_category": anchor_category,
            # "positive_category": pos_category,
            # "negative_category": neg_category,
            # "anchor_paper": anchor_paper,
            # "positive_paper": pos_paper,
            # "negative_paper": neg_paper
        })
    
    # 保存成 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triplets_data, f, ensure_ascii=False, indent=4)
    
    print(f"Triplets saved to {output_path}")
    
    # 打印几个示例
    print("\nSample triplets:")
    for i, triplet in enumerate(triplets_data[:3]):
        print(f"\nTriplet {i+1}:")
        print(f"  Anchor: {triplet['anchor'][:100]}...")
        print(f"  Positive: {triplet['positive'][:100]}...")
        print(f"  Negative: {triplet['negative'][:100]}...")

        # 新增：只保存索引三元组格式
        # triplet_idx_list = []
        # for triplet in unique_triplets:
        #     anchor_idx = sentence_to_idx.get(triplet[0])
        #     pos_idx = sentence_to_idx.get(triplet[1])
        #     neg_idx = sentence_to_idx.get(triplet[2])
        #     triplet_idx_list.append({"anchor": anchor_idx, "positive": pos_idx, "negative": neg_idx})

        # triplet_idx_dict = {
        #     "sentences": list(sentence_to_idx.keys()),
        #     "triplets": triplet_idx_list
        # }
        # with open("triplets_idx_only.json", "w", encoding="utf-8") as f:
        #     json.dump(triplet_idx_dict, f, ensure_ascii=False, indent=2)
        # print("Triplet index-only format saved to triplets_idx_only.json")

if __name__ == "__main__":
    main()