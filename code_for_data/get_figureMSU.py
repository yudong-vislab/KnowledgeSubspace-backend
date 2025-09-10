import json
import os
import re
from pathlib import Path
from openai import OpenAI  
import base64 
# 1. 读取 formdatabase.json

client = OpenAI(
    api_key="sk-R5R6baAMrW9C8fJl25Bb213d723846C7B9E9E89a45B32685",
    base_url="https://www.jcapikey.com/v1"
)

def ask_llm_with_image_and_text(image_path, text):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64_image}}
            ]}
        ],
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return answer
def clean_json_text(text: str) -> str:
    # 去掉 markdown 代码块 ```json 或 ```
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()

with open('case_engine/formdatabase.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 按次序编号，添加 MSU_id 属性
for idx, item in enumerate(data):
    item['MSU_id'] = idx

# 3. 处理 figure 的 sentence 属性，并调用大模型
for idx, item in enumerate(data):
    if item.get('type') == 'figure' and idx + 1 < len(data):
        item['sentence'] = data[idx + 1].get('para_info', {})
        item['2d_coord'] = data[idx + 1].get('2d_coord', [0, 0])
        item['para_id'] = data[idx + 1].get('para_id', -1)
        # 调用大模型，假设图片路径在item['image_path']，文本在item['sentence'].get('sentence', '')
        image_path = f"case_engine/images/{item.get('paper_info', '')}/{item.get('para_info', '')}"        
        print(f"Processing figure MSU_id {item['MSU_id']} with image {image_path}")
        text = item['sentence']
        prompt = f"""
        You are a scientific paper assistant. I will give you an image from a scientific paper and a piece of text, which is the text directly below the image in the paper. This text may be a figure caption or may not be. Please describe the content of this image in several concise sentences.        about the sentence:
        Your task is to generate a concise description of the image based on the provided text named 'sentence'. And the category of this figure. And the rank you give to this picture.
        Your output should satisfy:
        1. Be a single, self-contained statement.
        2. Express only one scientific fact or idea.
        3. Do not emit important information like reasons and aims behind "to" clauses.
        about the category:
        4. Be classified into one of the categories: [Method, Experiment, Result, Conclusion,Background,others].
        5. If there are unimportant or vague sentences or author introduction, classify them as "others".
        about the rank:
        6. Rank the importance of each MSU within the paragraph on a scale from 1 to 5, where 5 is most important.
        7. Higher rank should be given to explanations which are specific to this paper.
        
        Please output the results in the following JSON format:
        [
        {{"sentence": "...", "category": "...", "rank": ...}},
        段落：{text}
        """
        text_output = ask_llm_with_image_and_text(image_path, prompt)
        text_output = clean_json_text(text_output)
        try:
            msus = json.loads(text_output)
            print(f"解析成功: {msus}")
            item['sentence'] = msus[0].get('sentence', '')
            item['category'] = msus[0].get('category', 'others')
            item['rank'] = msus[0].get('rank', -1)
        except json.JSONDecodeError:
            print("⚠️ JSON 解析失败，返回原始输出：")
            print(text_output)


with open('case_engine/formdatabase.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# # 4. 生成连线
# links = []
# link_idx = 0
# # 按 para_id 分组
# from collections import defaultdict
# groups = defaultdict(list)
# for item in data:
#     groups[item.get('para_id', -1)].append(item)
# # 4.1. 对每组 para_id 内部按 MSU_id 排序后逐个连线
# for para_id, items in groups.items():
#     items_sorted = sorted(items, key=lambda x: x['MSU_id'])
#     for i in range(len(items_sorted) - 1):
#         link = {
#             'type': 'road',
#             'panelIdx': link_idx,
#             'countryFrom': para_id,
#             'countryTo': para_id,
#             'path': [
#                 {'q': items_sorted[i]['MSU_id'], 'r': 0},
#                 {'q': items_sorted[i+1]['MSU_id'], 'r': 0}
#             ]
#         }
#         links.append(link)
#         link_idx += 1
# # 4.2. figure与下一个sentence连线
# for idx, item in enumerate(data):
#     if item.get('type') == 'figure' and idx + 1 < len(data):
#         link = {
#             'type': 'road',
#             'panelIdx': link_idx,
#             'countryFrom': item.get('para_id', -1),
#             'countryTo': data[idx+1].get('para_id', -1),
#             'path': [
#                 {'q': item['MSU_id'], 'r': 0},
#                 {'q': data[idx+1]['MSU_id'], 'r': 0}
#             ]
#         }
#         links.append(link)
#         link_idx += 1

# # 5. 保存所有连线到 line.json
# with open('line.json', 'w', encoding='utf-8') as f:
#     json.dump({'links': links}, f, ensure_ascii=False, indent=2)

# print('已生成 line.json，连线数量:', len(links))
