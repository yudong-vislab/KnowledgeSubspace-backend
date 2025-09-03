#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_semantic_map.py (keep-all-raw + indices)

功能：
- 读取 data/hexagon_info.json（含 hex_coord/country/MSU_ids）
- 读取 data/formdatabase.json（含 MSU_id 与完整原始字段）
- 生成 semantic_map_data.json：
  {
    "version": "1.0",
    "build_info": {...},
    "subspaces": [ { panelIdx, subspaceName, hexList, countries } ],
    "links": [],
    "msu_index": { "<MSU_id>": <原始完整对象> },
    "indices": {
      "msu_to_hex": { "<MSU_id>": { panelIdx, q, r, country_id } },
      "panel_country_to_hex": { "<panelIdx>": { "<country_id>": [[q, r], ...] } },
      "category_to_cells": { "<category>": [ { panelIdx, q, r }, ... ] }
    },
    "stats": { ... }
  }

特性：
- ✅ 完整保留每条 MSU 的原始数据（不丢字段）：存入 msu_index
- ✅ hexList 中包含每格的 msu_ids（可选再内嵌 msu_details=完整对象）
- ✅ 不改变现有前端读取方式，同时新增 indices 反向索引，减少二次拼接/多次请求
- ✅ 可选：--embed-msu-details（为每个 hex 内嵌完整 MSU 对象数组，体积更大但零查表）
- ✅ 可选：--strict-format（仅保留 q,r,modality,country_id，不在 hexList 放 msu_ids/msu_details）
- ✅ 可选：--include-unknown-msu（若 hex 引用了 form 中不存在的 MSU_id，生成占位对象以“保留”）

用法（在项目根目录）：
  python build_semantic_map.py \
    --hex-info data/hexagon_info.json \
    --form data/formdatabase.json \
    --out data/semantic_map_data.json

可选参数：
  --embed-msu-details
  --strict-format
  --include-unknown-msu
  --panel-idx 0
  --panel-name Auto
  --country-prefix c
  --country-offset 0
"""

import json
import argparse
import copy
import time
from collections import Counter, defaultdict
from typing import Dict, Any, List, Iterable


# -----------------------
# I/O
# -----------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------
# 基础工具
# -----------------------

def to_int_list(values: Iterable[Any]) -> List[int]:
    out = []
    for v in values or []:
        s = str(v).strip()
        if s == "":
            continue
        try:
            out.append(int(float(s)))
        except Exception:
            # 跳过无法转 int 的
            continue
    return out

def majority_modality(msu_ids: Iterable[int], msu_map: Dict[int, Dict[str, Any]]) -> str:
    """基于该 hex 的 MSU type 多数表决得到 text/image（不影响原始数据保留）。"""
    votes = []
    for mid in msu_ids:
        rec = msu_map.get(mid)
        if rec is None:
            continue
        t = (rec.get("type") or "").lower()
        if t in ("text", "image"):
            votes.append(t)
    if not votes:
        return "text"
    c = Counter(votes)
    return "image" if c["image"] > c["text"] else "text"

def build_countries(hex_items, country_prefix: str, country_offset: int):
    """从 hex_items 聚合国家 -> hex 列表（与之前前端结构一致）"""
    agg = defaultdict(list)
    for h in hex_items:
        cid = f"{country_prefix}{h['__country_raw'] + country_offset}"
        agg[cid].append({"q": h["q"], "r": h["r"]})
    countries = [{"country_id": cid, "hexes": hexes} for cid, hexes in agg.items()]
    countries.sort(key=lambda x: x["country_id"])
    return countries

def build_msu_index_full(form: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """原样保留每个 MSU 的完整对象（深拷贝避免副作用）。键使用 int（写 JSON 时会转字符串）。"""
    msu_map: Dict[int, Dict[str, Any]] = {}
    for rec in form:
        if "MSU_id" not in rec:
            continue
        try:
            mid = int(rec["MSU_id"])
        except Exception:
            continue
        msu_map[mid] = copy.deepcopy(rec)
    return msu_map

def make_unknown_stub(mid: int) -> Dict[str, Any]:
    """为缺失的 MSU 生成占位对象（仅在 --include-unknown-msu 时使用），以“保留引用”"""
    return {
        "MSU_id": mid,
        "_missing": True,
        "sentence": None,
        "category": None,
        "type": None,
        "para_id": None,
        "paper_id": None,
        "paper_info": None,
        "paragraph_info": None,
        "2d_coord": None
    }


# -----------------------
# indices 构建（反向索引）
# -----------------------

def build_indices(subspaces: List[Dict[str, Any]], msu_index: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成三类轻量反向索引，降低前端二次拼接成本：
      - msu_to_hex: MSU_id -> { panelIdx, q, r, country_id }
      - panel_country_to_hex: panelIdx -> country_id -> [[q, r], ...]
      - category_to_cells: category -> [{ panelIdx, q, r }, ...]
    注意：为了 JSON 兼容，索引字典的 key 统一转为字符串。
    """
    msu_to_hex: Dict[str, Dict[str, Any]] = {}
    panel_country_to_hex: Dict[str, Dict[str, List[List[int]]]] = defaultdict(lambda: defaultdict(list))
    category_to_cells: Dict[str, List[Dict[str, int]]] = defaultdict(list)

    # 为了快速按 MSU_id 查类别，构建一个 int->category 的映射
    msu_id_to_cat: Dict[int, Any] = {}
    for mid, obj in msu_index.items():
        # mid 是 int；obj 里存有原始字段
        cat = obj.get("category")
        msu_id_to_cat[mid] = cat

    for sp in subspaces:
        pidx = sp["panelIdx"]
        for cell in sp["hexList"]:
            q, r = cell["q"], cell["r"]
            cid = cell["country_id"]
            panel_country_to_hex[str(pidx)][cid].append([q, r])

            for mid in cell.get("msu_ids", []):
                # 反向索引：MSU_id -> 坐标
                msu_to_hex[str(mid)] = {"panelIdx": pidx, "q": q, "r": r, "country_id": cid}

                cat = msu_id_to_cat.get(mid)
                if cat:
                    category_to_cells[str(cat)].append({"panelIdx": pidx, "q": q, "r": r})

    return {
        "msu_to_hex": msu_to_hex,
        "panel_country_to_hex": panel_country_to_hex,
        "category_to_cells": category_to_cells
    }


# -----------------------
# 组装单一子空间（默认）
# -----------------------

def build_single_subspace(
    hex_info, msu_map,
    panel_idx, panel_name,
    country_prefix, country_offset,
    strict_format, embed_msu_details,
    include_unknown_msu
):
    hex_items = []
    unknown_msu_ids = 0
    total_msu_refs = 0

    for cell in hex_info:
        coord = cell.get("hex_coord")
        if not (isinstance(coord, list) and len(coord) == 2):
            continue
        q, r = int(coord[0]), int(coord[1])
        country_raw = int(cell.get("country", 0))
        msu_ids = to_int_list(cell.get("MSU_ids"))

        # 统计未知 / 需要时生成占位
        for mid in msu_ids:
            total_msu_refs += 1
            if mid not in msu_map:
                unknown_msu_ids += 1
                if include_unknown_msu:
                    msu_map[mid] = make_unknown_stub(mid)

        modality = majority_modality(msu_ids, msu_map)
        country_id = f"{country_prefix}{country_raw + country_offset}"

        item = {
            "q": q,
            "r": r,
            "modality": modality,
            "country_id": country_id,
            "__country_raw": country_raw,
        }
        if not strict_format:
            item["msu_ids"] = msu_ids
            if embed_msu_details:
                # 原样内嵌完整对象（体积更大，但前端无需再查 msu_index）
                item["msu_details"] = [msu_map[mid] for mid in msu_ids if mid in msu_map]

        hex_items.append(item)

    countries = build_countries(hex_items, country_prefix, country_offset)
    for h in hex_items:
        h.pop("__country_raw", None)

    subspace = {
        "panelIdx": panel_idx,
        "subspaceName": panel_name,
        "hexList": hex_items,
        "countries": countries
    }
    stats = {
        "total_cells": len(hex_items),
        "total_countries": len(countries),
        "total_msu_refs": total_msu_refs,
        "unknown_msu_ids": unknown_msu_ids
    }
    return subspace, stats


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Merge hexagon_info.json and formdatabase.json -> semantic_map_data.json (keep all raw MSU data + indices)")
    ap.add_argument("--hex-info", default="data/hexagon_info.json", help="path to hexagon_info.json")
    ap.add_argument("--form", default="data/formdatabase.json", help="path to formdatabase.json")
    ap.add_argument("--out", default="data/semantic_map_data.json", help="output path")
    ap.add_argument("--panel-idx", type=int, default=0, help="panelIdx for the generated subspace")
    ap.add_argument("--panel-name", default="Auto", help="subspaceName for the generated subspace")
    ap.add_argument("--country-prefix", default="c", help='prefix for country_id (default "c")')
    ap.add_argument("--country-offset", type=int, default=0, help="offset added to numeric country when forming ID")
    ap.add_argument("--strict-format", action="store_true",
                    help="do NOT include msu_ids/msu_details in hexList entries")
    ap.add_argument("--embed-msu-details", action="store_true",
                    help="embed full raw MSU objects into each hex cell (bigger file but zero lookup)")
    ap.add_argument("--include-unknown-msu", action="store_true",
                    help="if hex references an unknown MSU_id, create a stub object so that everything is preserved")
    args = ap.parse_args()

    hex_info = load_json(args.hex_info)
    form = load_json(args.form)

    # 1) 构建完整 msu_index（原样保留所有字段；键为 int，JSON 会转字符串）
    msu_map = build_msu_index_full(form)

    # 2) 构建一个子空间（若需按 category 拆分，可在此基础上扩展）
    subspace, stats = build_single_subspace(
        hex_info, msu_map,
        panel_idx=args.panel_idx, panel_name=args.panel_name,
        country_prefix=args.country_prefix, country_offset=args.country_offset,
        strict_format=args.strict_format,
        embed_msu_details=args.embed_msu_details,
        include_unknown_msu=args.include_unknown_msu
    )
    subspaces = [subspace]

    # 3) 构建反向索引（不改变原有访问方式，仅补充）
    indices = build_indices(subspaces, msu_map)

    # 4) 汇总统计信息
    totals_msu = len(msu_map)
    stats_out = {
        "subspaces": [{"panelIdx": sp["panelIdx"], "cells": len(sp["hexList"]), "countries": len(sp["countries"])} for sp in subspaces],
        "totals": {"cells": sum(s["cells"] for s in [{"cells": len(sp["hexList"])} for sp in subspaces]),
                   "countries": sum(s["countries"] for s in [{"countries": len(sp["countries"])} for sp in subspaces]),
                   "msu_count": totals_msu},
        "msu_refs_total": stats["total_msu_refs"],
        "msu_refs_missing": stats["unknown_msu_ids"]
    }

    # 5) 输出
    out_obj = {
        "version": "1.0",
        "build_info": {
            "ts": int(time.time()),
            "tool": "build_semantic_map.py",
            "options": {
                "strict_format": args.strict_format,
                "embed_msu_details": args.embed_msu_details,
                "include_unknown_msu": args.include_unknown_msu,
                "country_prefix": args.country_prefix,
                "country_offset": args.country_offset
            }
        },
        "subspaces": subspaces,
        "links": [],
        "msu_index": msu_map,   # ✅ 原样保留：前端可通过 MSU_id 读取任何原始字段
        "indices": indices,     # ✅ 反向索引：降低前端二次拼接成本
        "stats": stats_out
    }

    write_json(args.out, out_obj)

    print(f"[done] wrote {args.out}")
    print(f"  subspaces        : {len(subspaces)}")
    print(f"  total cells      : {stats['total_cells']}")
    print(f"  total countries  : {stats['total_countries']}")
    print(f"  msu refs (total) : {stats['total_msu_refs']}")
    print(f"  msu refs (miss)  : {stats['unknown_msu_ids']}")
    print(f"  strict-format    : {args.strict_format}")
    print(f"  embed-msu-details: {args.embed_msu_details}")
    print(f"  include-unknown  : {args.include_unknown_msu}")
    print(f"  output path      : {args.out}")


if __name__ == "__main__":
    main()
