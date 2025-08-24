from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import json, os
from openai import OpenAI

# ========== 配置 OpenAI Key ==========
API_KEY ="sk-UJYl2J6urdV4kgORmsFP8xfoAtOl6AbVc75PPBCoz1qPdGhv"
client = OpenAI(api_key=API_KEY, base_url="https://api.chatanywhere.tech/v1")


app = Flask(__name__)
CORS(app)  # 开发期允许跨域
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "semantic_map_data.json")

def load_data():
    if not os.path.exists(DATA_PATH):
        return {"subspaces": [], "links": []}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.get("/api/semantic-map")
def get_semantic_map():
    data = load_data()
    # 这里可以做“数据解析/预处理”，返回给前端的是“干净的可视化数据”
    return jsonify(data)

@app.post("/api/subspaces")
def create_subspace():
    body = request.get_json(force=True)
    name = (body or {}).get("subspaceName") or None
    data = load_data()
    new_idx = len(data.get("subspaces", []))
    subspace = {
        "subspaceName": name or f"Subspace {new_idx + 1}",
        "hexList": body.get("hexList", []),
        "countries": body.get("countries", [])
    }
    data["subspaces"].append(subspace)
    save_data(data)
    return jsonify({"index": new_idx, "subspace": subspace}), 201

@app.patch("/api/subspaces/<int:idx>")
def rename_subspace(idx):
    body = request.get_json(force=True)
    new_name = (body or {}).get("subspaceName")
    if not new_name:
        abort(400, "subspaceName required")
    data = load_data()
    subs = data.get("subspaces", [])
    if idx < 0 or idx >= len(subs):
        abort(404, "subspace not found")
    subs[idx]["subspaceName"] = new_name
    save_data(data)
    return jsonify({"index": idx, "subspace": subs[idx]})


# # ========== 配置代理 (如 Clash 本地代理 7890) ==========
# # 如果你没有代理，这部分可以注释掉
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# ========== 系统提示 ==========
system_prompt = """
你是一个科研助手，专注于帮助用户检索学术文献。
当用户输入查询时，你要：
1. 理解领域和关键词（如气象、空气质量）。
2. 提供最近五年相关文献列表（可附作者、年份、论文标题）。
3. 尽量返回结构化信息，方便前端显示。
"""

# ========== Flask API ==========
@app.route("/api/query", methods=["POST"])
def query_gpt():
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            max_tokens=500,
            timeout=30.0
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        print("GPT调用出错：", e)
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
