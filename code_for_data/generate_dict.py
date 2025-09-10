import os
import json

ALLDATA = []
# SENTENCE; PARA_ID; PAPER_ID; CATEGORY; RANK;TYPE
PARA_LIST= []
PAPER_LIST = []
input_dir = "/home/lxy/case_engine"
paragraph_id = 0
paper_id = 0
sentence_id = 0
for filename in os.listdir(input_dir):
    data_path = os.path.join(input_dir, filename, f"{filename}_rewrite.json")
    if not os.path.isfile(data_path):
        print(f"跳过不存在的文件: {data_path}")
        continue
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    PAPER_LIST.append(filename)
    for section in data:
        para = section.get("paragraph", "unknown")
        PARA_LIST.append(para)
        type_ = section.get("type", "unknown")
        if type_ == "figure":
            ALLDATA.append({
            "type": type_,
            "para_id": paragraph_id,
            "paper_id": paper_id,
            })
        else:
            resultmsu = section.get("resultmsu", [])
            if not resultmsu:
                print(f"⚠️ Warning: No MSU results for paragraph ID {paragraph_id} in paper {paper_id}")
                print(f"Paragraph: {para[:50]}...")
                paragraph_id += 1   
                continue
            for msu in resultmsu:
                sentence = msu.get("sentence", "")
                category = msu.get("category", "missing")
                rank = msu.get("rank", -1)
                ALLDATA.append({
                    "sentence": sentence,
                    "category": category,
                    "rank": rank,
                    "type": type_,
                    "para_id": paragraph_id,
                    "paper_id": paper_id,

                })
                sentence_id += 1
        paragraph_id += 1   
    paper_id += 1
print(f"总共加载 {len(PARA_LIST)} 段落，共计{sentence_id}个句子，来自 {paper_id} 篇论文。")
with open("/home/lxy/case_engine/paragraphs.json", "w", encoding="utf-8") as f:
    json.dump(PARA_LIST, f, ensure_ascii=False, indent=2)
with open("/home/lxy/case_engine/papers.json", "w", encoding="utf-8") as f:
    json.dump(PAPER_LIST, f, ensure_ascii=False, indent=2)
with open("/home/lxy/case_engine/alldata.json", "w", encoding="utf-8") as f:
    json.dump(ALLDATA, f, ensure_ascii=False, indent=2)
print("已分别保存 paragraphs.json, papers.json, alldata.json")

