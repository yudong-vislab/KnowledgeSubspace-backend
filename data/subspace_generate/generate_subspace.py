<<<<<<< HEAD
import os
from openai import OpenAI
import json
import re
from pathlib import Path

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url="https://www.jcapikey.com/v1"
)

def clean_json_text(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()

def can_answer_question(paragraph: str, question: str) -> bool:
    """
    让大模型判断该段落能否回答问题，返回 True/False
    """
    prompt = f"""
你是科研论文智能助手。请判断下面的段落能否直接或间接回答指定问题。
请只返回“是”或“否”。

问题：{question}

段落：{paragraph}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return "是" in answer

def filter_paragraphs_by_question(item, question):
    """
    批量筛选能回答问题的段落
    """
    results = []
    para_id = 0
    text = item["sentence"]
    if can_answer_question(text, question):
        results.append(text)
 
    return results

def generate_subspace(jsonpath, question,method="filter_sentences"):
    paragraphs = []
    json_path = Path(jsonpath)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    extractmsu = []
    sentence_id = 0
    # methods = ["filter_sentences","generate_from_text"]
    if method == "filter_sentences":
        for item in data:
            text = item["sentence"]
            text = clean_json_text(text)
            filter_paragraphs_by_question([text], question)
    elif method == "generate_from_text":
        for item in data:
            text = clean_json_text(item.get("text", ""))
            paragraphs.append(text)
    else:
        raise ValueError("Unsupported method. Use 'filter_sentences' or 'generate_from_text'.")


    filtered = filter_paragraphs_by_question(paragraphs, question)
    print("能回答问题的段落：")
    for para in filtered:
        print(para)
        print("-----")
=======
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
>>>>>>> c340c0b2f9f973d753ba8e4c70d28b9ae83a855e
