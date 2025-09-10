# enhanced_link_generator.py - 基于语义数据的六边形聚合链接生成器
import json
import numpy as np
from typing import Dict, List, Tuple, Set
import math
from collections import defaultdict

class SemanticLinkGenerator:
    def __init__(self, sentence_file: str, semantic_map_file: str = "../KnowledgeSubspace-backend/data/semantic_map_data.json"):
        self.sentence_data = self.load_json(sentence_file)
        self.semantic_map_data = self.load_json(semantic_map_file)
        
        # 从semantic数据中提取所有MSU_ids
        self.relevant_msu_ids = self.extract_relevant_msu_ids()
        
        # 过滤句子数据，只保留相关的句子
        self.filtered_sentences = self.filter_sentences_by_msu_ids()
        
        # 建立映射关系
        self.hex_to_sentences = self.map_hex_to_sentences()
        self.hex_to_panel_idx = self.map_hex_to_panel_idx()
        self.hex_to_country = self.assign_country_to_hex()
        
    def load_json(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_relevant_msu_ids(self) -> Set[int]:
        """从semantic_map_data.json中提取所有存在的MSU_ids"""
        msu_ids = set()
        
        for subspace in self.semantic_map_data.get('subspaces', []):
            hex_list = subspace.get('hexList', [])
            for hex_item in hex_list:
                msu_ids_in_hex = hex_item.get('msu_ids', [])
                msu_ids.update(msu_ids_in_hex)
        
        return msu_ids
    
    def filter_sentences_by_msu_ids(self) -> List[Dict]:
        """只保留在relevant_msu_ids中的句子"""
        return [sentence for sentence in self.sentence_data 
                if sentence.get('MSU_id') in self.relevant_msu_ids]
    
    def map_hex_to_sentences(self) -> Dict[Tuple[int, int], List[Dict]]:
        """将六边形坐标映射到对应的句子"""
        hex_map = defaultdict(list)
        
        # 创建MSU_id到句子的映射
        sentence_by_msu = {s['MSU_id']: s for s in self.filtered_sentences}
        
        # 遍历所有subspace中的六边形
        for subspace in self.semantic_map_data.get('subspaces', []):
            panel_idx = subspace.get('panelIdx')
            hex_list = subspace.get('hexList', [])
            
            for hex_item in hex_list:
                q = hex_item.get('q')
                r = hex_item.get('r')
                msu_ids = hex_item.get('msu_ids', [])
                
                if q is not None and r is not None:
                    coord = (q, r)
                    
                    # 获取该六边形对应的所有句子
                    for msu_id in msu_ids:
                        if msu_id in sentence_by_msu:
                            sentence = sentence_by_msu[msu_id]
                            hex_map[coord].append({
                                'msu_id': msu_id,
                                'sentence': sentence['sentence'],
                                'category': sentence['category'],
                                'embedding': sentence['2d_coord'],
                                'paper_id': sentence['paper_id'],
                                'para_id': sentence['para_id'],
                                'panel_idx': panel_idx
                            })
        
        return dict(hex_map)
    
    def map_hex_to_panel_idx(self) -> Dict[Tuple[int, int], int]:
        """将六边形坐标映射到对应的panelIdx"""
        panel_map = {}
        
        for subspace in self.semantic_map_data.get('subspaces', []):
            panel_idx = subspace.get('panelIdx')
            hex_list = subspace.get('hexList', [])
            
            for hex_item in hex_list:
                q = hex_item.get('q')
                r = hex_item.get('r')
                if q is not None and r is not None:
                    coord = (q, r)
                    panel_map[coord] = panel_idx
        
        return panel_map
    
    def assign_country_to_hex(self) -> Dict[Tuple[int, int], int]:
        """为六边形分配country_id（基于panelIdx的模运算）"""
        country_map = {}
        
        for coord, panel_idx in self.hex_to_panel_idx.items():
            # 使用panelIdx来分配country，可以根据需要调整
            country_map[coord] = (panel_idx % 10) + 1
        
        return country_map
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def generate_hexagon_pairs(self, similarity_threshold: float = 0.75) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[Dict]]:
        """生成六边形对之间的连接统计"""
        hexagon_connections = defaultdict(list)
        
        # 获取所有六边形
        hex_coords = list(self.hex_to_sentences.keys())
        
        # 遍历所有六边形对
        for i, hex1 in enumerate(hex_coords):
            for j, hex2 in enumerate(hex_coords[i+1:], i+1):
                connections = []
                
                # 检查两个六边形之间的所有句子对
                sentences1 = self.hex_to_sentences[hex1]
                sentences2 = self.hex_to_sentences[hex2]
                
                for sent1 in sentences1:
                    for sent2 in sentences2:
                        # 防止相同MSU_id的句子被链接
                        if sent1['msu_id'] == sent2['msu_id']:
                            continue
                            
                        similarity = self.cosine_similarity(
                            sent1['embedding'], 
                            sent2['embedding']
                        )
                        
                        if similarity >= similarity_threshold:
                            connections.append({
                                'similarity': similarity,
                                'from_msu_id': sent1['msu_id'],
                                'to_msu_id': sent2['msu_id'],
                                'from_category': sent1['category'],
                                'to_category': sent2['category'],
                                'from_paper_id': sent1['paper_id'],
                                'to_paper_id': sent2['paper_id'],
                                'from_sentence': sent1['sentence'][:100] + "...",
                                'to_sentence': sent2['sentence'][:100] + "..."
                            })
                
                if connections:
                    hexagon_connections[(hex1, hex2)] = connections
        
        return dict(hexagon_connections)
    
    def filter_by_connection_count(self, hexagon_connections: Dict, min_connections: int = 3) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[Dict]]:
        """过滤连接数不足的六边形对"""
        return {
            pair: connections 
            for pair, connections in hexagon_connections.items()
            if len(connections) >= min_connections
        }
    
    def determine_link_type(self, hex1: Tuple[int, int], hex2: Tuple[int, int]) -> str:
        """根据panelIdx确定link类型
        
        当两个六边形位于同一个panel时，返回'road'类型；
        当两个六边形位于不同panel时，返回'flight'类型。
        """
        panel_idx1 = self.hex_to_panel_idx.get(hex1, 0)
        panel_idx2 = self.hex_to_panel_idx.get(hex2, 1)
        
        # 根据panelIdx判断连接类型
        if panel_idx1 == panel_idx2:
            return 'road'
        else:
            return 'flight'
    
    def generate_final_links(self, filtered_connections: Dict, output_file: str) -> List[Dict]:
        """生成最终的links JSON"""
        frontend_links = []
        
        for (hex1, hex2), connections in filtered_connections.items():
            link_type = self.determine_link_type(hex1, hex2)
            
            # 获取country信息并转换为cX格式
            country1 = self.hex_to_country[hex1]
            country2 = self.hex_to_country[hex2]
            country_from_str = f"c{country1}"
            country_to_str = f"c{country2}"
            
            # 获取对应的panelIdx
            panel_idx_from = self.hex_to_panel_idx.get(hex1, 0)
            panel_idx_to = self.hex_to_panel_idx.get(hex2, 1)
            
            if link_type == 'road':
                # road格式 - 使用目标六边形的panelIdx
                frontend_link = {
                    'type': 'road',
                    'panelIdx': panel_idx_to,
                    'countryFrom': country_from_str,
                    'countryTo': country_to_str,
                    'path': [
                        {'q': hex1[0], 'r': hex1[1]},
                        {'q': hex2[0], 'r': hex2[1]}
                    ]
                }
            else:  # flight
                # flight格式 - 使用实际的panelIdx
                frontend_link = {
                    'type': 'flight',
                    'panelIdxFrom': panel_idx_from,
                    'panelIdxTo': panel_idx_to,
                    'from': {'q': hex1[0], 'r': hex1[1], 'panelIdx': panel_idx_from},
                    'to': {'q': hex2[0], 'r': hex2[1], 'panelIdx': panel_idx_to},
                    'path': [
                        {'q': hex1[0], 'r': hex1[1], 'panelIdx': panel_idx_from},
                        {'q': hex2[0], 'r': hex2[1], 'panelIdx': panel_idx_to}
                    ]
                }
            
            frontend_links.append(frontend_link)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(frontend_links, f, indent=2, ensure_ascii=False)
        
        return frontend_links
    
    def generate_connection_database(self, filtered_connections: Dict, db_file: str):
        """生成连接数据库JSON"""
        connection_db = {}
        
        for pair_id, ((hex1, hex2), connections) in enumerate(filtered_connections.items()):
            connection_db[f"pair_{pair_id}"] = {
                'from_hex': {'q': hex1[0], 'r': hex1[1]},
                'to_hex': {'q': hex2[0], 'r': hex2[1]},
                'from_country': self.hex_to_country[hex1],
                'to_country': self.hex_to_country[hex2],
                'from_panel_idx': self.hex_to_panel_idx.get(hex1, 0),
                'to_panel_idx': self.hex_to_panel_idx.get(hex2, 1),
                'connection_count': len(connections),
                'connections': connections
            }
        
        with open(db_file, 'w', encoding='utf-8') as f:
            json.dump(connection_db, f, indent=2, ensure_ascii=False)
    
    def run(self, similarity_threshold: float = 0.85, min_connections: int = 5):
        """运行完整的链接生成流程"""
        print("开始生成语义连接...")
        print(f"从semantic数据中提取了 {len(self.relevant_msu_ids)} 个相关MSU_ids")
        print(f"过滤后处理了 {len(self.filtered_sentences)} 个句子")
        
        # 1. 生成六边形对之间的连接
        hexagon_connections = self.generate_hexagon_pairs(similarity_threshold)
        print(f"找到 {len(hexagon_connections)} 个六边形对有连接")
        
        # 2. 过滤连接数
        filtered_connections = self.filter_by_connection_count(hexagon_connections, min_connections)
        print(f"过滤后剩余 {len(filtered_connections)} 个六边形对")
        
        # 3. 生成最终links
        final_links = self.generate_final_links(filtered_connections, "final_links_semantic_based.json")
        print(f"生成了 {len(final_links)} 个最终连接")
        
        # 4. 生成连接数据库
        self.generate_connection_database(filtered_connections, "connection_database_semantic.json")
        print("连接数据库已生成")
        
        return {
            'total_relevant_msu_ids': len(self.relevant_msu_ids),
            'processed_sentences': len(self.filtered_sentences),
            'total_hexagon_pairs': len(hexagon_connections),
            'filtered_pairs': len(filtered_connections),
            'final_links': len(final_links),
            'avg_connections_per_pair': sum(len(conns) for conns in filtered_connections.values()) / len(filtered_connections) if filtered_connections else 0
        }

# 使用示例
if __name__ == "__main__":
    generator = SemanticLinkGenerator(
        sentence_file="../KnowledgeSubspace-backend/case2/formdatabase.json"
    )
    
    # 使用更严格的参数以减少连接数量
    result = generator.run(
        similarity_threshold=0.90,  # 提高相似度阈值
        min_connections=800           # 提高最小连接数要求
    )
    
    print("\n=== 运行结果统计 ===")
    for key, value in result.items():
        print(f"{key}: {value}")