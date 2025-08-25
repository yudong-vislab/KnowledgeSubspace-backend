# app.py
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import os, json, sys, pathlib
from dotenv import load_dotenv
from openai import OpenAI

# 让同目录的 prompts.py 一定可见（无论从哪里启动）
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

# 从 prompts 模块导入固定提示
from prompts import SYSTEM_PROMPT as PROMPT_BASE_SYSTEM
from prompts import PROMPT_LITERATURE_SEARCH, PROMPT_SUBSPACE_ANALYSIS

# ========== 环境变量 ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1").strip()
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# 系统提示允许被环境变量覆盖（如果 .env 里提供了 SYSTEM_PROMPT）
ENV_SYSTEM_PROMPT = (os.getenv("SYSTEM_PROMPT") or "").strip()
SYSTEM_PROMPT_ACTIVE = ENV_SYSTEM_PROMPT if ENV_SYSTEM_PROMPT else PROMPT_BASE_SYSTEM

app = Flask(__name__)
CORS(app)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "semantic_map_data.json")

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
    name = (body.get("subspaceName") or "").strip()
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
def rename_subspace(idx: int):
    body = request.get_json(force=True) or {}
    new_name = (body.get("subspaceName") or "").strip()
    if not new_name:
        abort(400, "subspaceName required")
    data = load_data()
    subs = data.get("subspaces", [])
    if idx < 0 or idx >= len(subs):
        abort(404, "subspace not found")
    subs[idx]["subspaceName"] = new_name
    save_data(data)
    return jsonify({"index": idx, "subspace": subs[idx]})

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

# ---------- LLM 路由 ----------
TASK_PROMPTS = {
    "literature": PROMPT_LITERATURE_SEARCH,
    "subspace":   PROMPT_SUBSPACE_ANALYSIS,
}

@app.post("/api/query")
def query_gpt():
    body = request.get_json(force=True) or {}
    user_query = (body.get("query") or "").strip()
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    task_type = (body.get("task") or "literature").lower()
    task_prompt = TASK_PROMPTS.get(task_type, "")

    try:
        resp = client.chat.completions.create(
            model=body.get("model", OPENAI_DEFAULT_MODEL),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ACTIVE},
                {"role": "user", "content": f"{task_prompt}\n\nUser query:\n{user_query}"}
            ],
            temperature=0.2,
            max_tokens=900,
            timeout=30.0
        )
        answer = resp.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        print("GPT 调用出错：", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
