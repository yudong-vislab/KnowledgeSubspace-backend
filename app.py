# app.py
from flask import Flask, jsonify, request, abort, make_response
from flask_cors import CORS
import os, json, sys, pathlib, hashlib, time
from dotenv import load_dotenv

# ---- Optional gzip (auto-enable if installed) ----
try:
    from flask_compress import Compress
    HAS_COMPRESS = True
except Exception:
    HAS_COMPRESS = False

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

# OpenAI 客户端（若你在本地调试不需要 GPT，可注释掉下面两行与 /api/query 路由）
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# 系统提示允许被环境变量覆盖（如果 .env 里提供了 SYSTEM_PROMPT）
ENV_SYSTEM_PROMPT = (os.getenv("SYSTEM_PROMPT") or "").strip()
SYSTEM_PROMPT_ACTIVE = ENV_SYSTEM_PROMPT if ENV_SYSTEM_PROMPT else PROMPT_BASE_SYSTEM

app = Flask(__name__)


CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}},
    supports_credentials=False
)

if HAS_COMPRESS:
    Compress(app)  # 自动 gzip 压缩较大的 JSON 响应

# ---- 数据路径 ----
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "semantic_map_data.json"

# ========== 工具函数 ==========
def _file_etag(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    stat = path.stat()
    base = f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _json_load(path: pathlib.Path):
    if not path.exists():
        return {"title": "Semantic Map", "subspaces": [], "links": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _json_save(path: pathlib.Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _ensure_full_package(data: dict) -> dict:
    """
    确保语义地图包具备前端一次请求所需的全部关键字段。
    若缺失 msu_index / indices，则补空壳，避免前端崩溃。
    """
    data = data or {}
    data.setdefault("title", "Semantic Map")
    data.setdefault("subspaces", [])
    data.setdefault("links", [])
    data.setdefault("msu_index", {})   # 全量原始 MSU 数据（由构建脚本写入）
    data.setdefault("indices", {})     # 反向索引（由构建脚本写入）
    data.setdefault("stats", {})       # 可选统计
    data.setdefault("version", "1.0")
    data.setdefault("build_info", {"ts": int(time.time()), "tool": "app.py@runtime"})
    return data

def load_data():
    return _ensure_full_package(_json_load(DATA_PATH))

def save_data(data):
    _json_save(DATA_PATH, data)

# ========== 语义地图（全量数据） ==========
@app.get("/api/semantic-map")
def get_semantic_map():
    """
    一次请求返回整包数据：subspaces + links + msu_index + indices。
    配置了 ETag/缓存头；前端可使用 If-None-Match 做条件 GET。
    """
    data = load_data()
    etag = _file_etag(DATA_PATH)
    inm = request.headers.get("If-None-Match")
    if inm and etag and inm == etag:
        resp = make_response("", 304)
        resp.headers["ETag"] = etag
        resp.headers["Cache-Control"] = "public, max-age=60"
        return resp

    resp = jsonify(data)
    if etag:
        resp.headers["ETag"] = etag
        resp.headers["Cache-Control"] = "public, max-age=60"
    return resp

@app.get("/api/semantic-map/indices")
def get_indices_only():
    """仅返回 indices，便于某些轻量场景快速拉取。"""
    data = load_data()
    return jsonify(data.get("indices", {}))

@app.get("/api/semantic-map/msu/<int:mid>")
def get_msu_detail(mid: int):
    """按 MSU_id 返回原始完整对象（来自 msu_index）。"""
    data = load_data()
    msu = data.get("msu_index", {}).get(str(mid)) or data.get("msu_index", {}).get(mid)
    if msu is None:
        abort(404, f"MSU {mid} not found")
    return jsonify(msu)

# ========== 基础写接口（保留你的原逻辑） ==========
@app.post("/api/subspaces")
def create_subspace():
    body = request.get_json(force=True) or {}
    name = (body.get("subspaceName") or "").strip()
    data = load_data()
    new_idx = len(data.get("subspaces", []))
    # 兼容：如果你要保持 panelIdx，请这里按 new_idx 写入
    subspace = {
        "panelIdx": new_idx,
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

# ========== LLM 路由 ==========
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

# ========== 入口 ==========
if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
