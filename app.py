from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import json, os
from openai import OpenAI
from dotenv import load_dotenv

# ========== 加载环境变量 ==========
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1").strip()
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

app = Flask(__name__)
CORS(app)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "semantic_map_data.json")

# ---------- 系统提示（模块常量；可被环境变量覆盖） ----------
DEFAULT_SYSTEM_PROMPT = """
你是一个科研助手，专注于帮助用户检索学术文献。
当用户输入查询时，你要：
1. 理解领域和关键词（如气象、空气质量）。
2. 提供最近五年相关文献列表（可附作者、年份、论文标题）。
3. 尽量返回结构化信息，方便前端显示。
"""
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

def load_data():
    if not os.path.exists(DATA_PATH):
        return {"title": "Semantic Map", "subspaces": [], "links": []}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- 语义地图数据 ----------
@app.get("/api/semantic-map")
def get_semantic_map():
    return jsonify(load_data())

@app.post("/api/subspaces")
def create_subspace():
    body = request.get_json(force=True) or {}
    name = body.get("subspaceName") or None
    data = load_data()
    new_idx = len(data.get("subspaces", []))
    subspace = {
        "subspaceName": name or f"Subspace {new_idx + 1}",
        "hexList": body.get("hexList", []),
        "countries": body.get("countries", [])
    }
    data.setdefault("subspaces", []).append(subspace)
    save_data(data)
    return jsonify({"index": new_idx, "subspace": subspace}), 201

@app.patch("/api/subspaces/<int:idx>")
def rename_subspace(idx):
    body = request.get_json(force=True) or {}
    new_name = body.get("subspaceName")
    if not new_name:
        abort(400, "subspaceName required")
    data = load_data()
    subs = data.get("subspaces", [])
    if idx < 0 or idx >= len(subs):
        abort(404, "subspace not found")
    subs[idx]["subspaceName"] = new_name
    save_data(data)
    return jsonify({"index": idx, "subspace": subs[idx]})

# ✅ 补齐：改标题路由（配合你的前端 renameMapTitle）
@app.post("/api/semantic-map/title")
def update_map_title():
    body = request.get_json(force=True) or {}
    new_title = (body.get("title") or "").strip()
    if not new_title:
        abort(400, "title required")
    data = load_data()
    data["title"] = new_title
    save_data(data)
    return jsonify({"title": new_title})


        
# /api/query 路由：支持 history + model  // ★★ 修改点
@app.post("/api/query")
def query_gpt():
    body = request.get_json(force=True) or {}
    user_query = (body.get("query") or "").strip()
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        resp = client.chat.completions.create(
            model=body.get("model", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            max_tokens=800,
            timeout=30.0
        )
        answer = resp.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        print("GPT 调用出错：", e)
        return jsonify({"error": f"{e}"}), 500

    except Exception as e:
        print("GPT 调用出错：", e)
        return jsonify({"ok": False, "error": str(e)}), 500



if __name__ == "__main__":
    # 用环境变量管理 host/port 更稳妥
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
