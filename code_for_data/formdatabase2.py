import json
from collections import defaultdict

# 读取数据
with open('formdatabase_v2.0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化存储链接的列表
links = []
link_idx = 0

# 处理每个段落
current_para_id = None
path = []
para_id_items = []

# 遍历数据，从前往后读取
for idx, item in enumerate(data):
    para_id = item.get('para_id', -1)

    # 如果 para_id 变化了，开始处理上一个段落
    if para_id != current_para_id:
        if current_para_id is not None:
            # 在段落结束时生成一个连线对象
            link_type = 'road'  # 默认是 road 类型
            if any(i.get('type') == 'figure' for i in para_id_items):
                link_type = 'river'  # 如果段落内有 figure 元素，则是 river 类型
            
            # 按照 MSU_id 排序，确保按句子的顺序连接
            para_id_items_sorted = sorted(para_id_items, key=lambda x: x['MSU_id'])
            
            # 创建 path，连接当前段落内所有句子
            path = [{'q': para_id_items_sorted[i]['MSU_id'], 'r': para_id_items_sorted[i + 1]['MSU_id']}
                    for i in range(len(para_id_items_sorted) - 1)]

            # 生成连线
            if path:
                link = {
                    'type': link_type,
                    'panelIdx': link_idx,
                    'countryFrom': current_para_id,
                    'countryTo': current_para_id,
                    'path': path
                }
                links.append(link)
                link_idx += 1

        # 重置当前段落，开始处理新段落
        current_para_id = para_id
        para_id_items = [item]  # 新段落，重新开始记录 items
    else:
        para_id_items.append(item)  # 在同一段落内，继续添加句子

# 最后一个段落处理（防止遗漏最后一个段落）
if para_id_items:
    link_type = 'road'  # 默认是 road 类型
    if any(i.get('type') == 'figure' for i in para_id_items):
        link_type = 'river'  # 如果段落内有 figure 元素，则是 river 类型
    
    para_id_items_sorted = sorted(para_id_items, key=lambda x: x['MSU_id'])
    
    # 创建 path，连接当前段落内所有句子
    path = [{'q': para_id_items_sorted[i]['MSU_id'], 'r': para_id_items_sorted[i + 1]['MSU_id']}
            for i in range(len(para_id_items_sorted) - 1)]

    # 生成连线
    if path:
        link = {
            'type': link_type,
            'panelIdx': link_idx,
            'countryFrom': current_para_id,
            'countryTo': current_para_id,
            'path': path
        }
        links.append(link)

# 保存到line.json
with open('line.json', 'w', encoding='utf-8') as f:
    json.dump({'links': links}, f, ensure_ascii=False, indent=2)

print('已生成 line.json，连线数量:', len(links))
