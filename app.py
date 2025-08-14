from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import json, os

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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
