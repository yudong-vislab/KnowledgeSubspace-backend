import json
import re

# 读取数据
with open("/home/lxy/case_engine/alldata.json", "r", encoding="utf-8") as f:
    alldata = json.load(f)
with open("/home/lxy/case_engine/paragraphs.json", "r", encoding="utf-8") as f:
    paragraphs = json.load(f)

new_data = []
for i, item in enumerate(alldata):
    if item.get("type") == "text":
        new_data.append(item)
        continue
    elif item.get("type") == "figure":
        # 添加 paragraphs.json 中 paraid 对应内容
        paraid = item.get("para_id")
        para_text = paragraphs[paraid] if paraid is not None and paraid < len(paragraphs) else ""
        # print(para_text)
        new_item = dict(item)
        new_item["paragraph_text"] = para_text

        # 检查下一个元素是否有 fig/figure+数字（考虑大小写和多种写法）
        if i + 1 < len(alldata):
            nextparaid = alldata[i + 1].get("para_id", "")
            para_text = paragraphs[nextparaid] if nextparaid is not None and nextparaid < len(paragraphs) else ""
            print(para_text)
            # 支持 'Fig. 7'、'FIG. 7'、'figure 7' 等格式
            new_item["number"] = para_text
        new_data.append(new_item)

# 保存结果
with open("/home/lxy/case_engine/alldata_processed.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("已生成 alldata_processed.json")