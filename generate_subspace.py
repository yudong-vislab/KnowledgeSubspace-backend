import json
from collections import defaultdict

# 假设数据已经存储在一个变量中
with open('/home/lxy/model_train/pollution_result/formdatabase_v2.0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将数据根据 category 分组
grouped_data = defaultdict(list)
for entry in data:
    grouped_data[entry['category']].append(entry)

# 为每个类别创建一个单独的 JSON 文件
for category, items in grouped_data.items():
    filename = f"{category}.json"
    with open(filename, 'w') as f:
        json.dump(items, f, indent=4)

print("JSON 文件已根据类别分组保存。")
