#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_semantic_map.py
把 hexagon_info.json 与 formdatabase.json 关联，输出 semantic_map_data.json（subspaces/hexList/countries 结构）。
- hexagon_info.json: [{"hex_coord":[q,r], "country": int, "MSU_ids":[...]}]
- formdatabase.json : [{"MSU_id": int, "type": "text"|"image", "category": "...", ...}, ...]

默认：
- 生成单一子空间 (panelIdx=0, subspaceName="Auto")
- country 映射为 "c{country + country_offset}"（默认偏移 0 => c0, c1, ...）
- modality = MSU_ids 中记录的多数类型（若都无或冲突则回退为 "text"）
- 在 hexList 中额外保留 "msu_ids" 字段，便于后续溯源（如需严格兼容旧前端，可用 --strict-format 禁用）
"""

import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def majority_modality(msu_ids, msu_map) -> str:
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
    if c["image"] > c["text"]:
        return "image"
    return "text"

def build_countries(hex_items, country_prefix: str, country_offset: int):
    """从 hex_items 聚合国家 -> hex 列表"""
    agg = defaultdict(list)
    for h in hex_items:
        cid = f"{country_prefix}{h['__country_raw'] + country_offset}"
        agg[cid].append({"q": h["q"], "r": h["r"]})
    countries = [{"country_id": cid, "hexes": hexes} for cid, hexes in agg.items()]
    countries.sort(key=lambda x: x["country_id"])
    return countries

def main():
    ap = argparse.ArgumentParser(description="Merge hexagon_info.json and formdatabase.json -> semantic_map_data.json")
    ap.add_argument("--hex-info", required=True, help="path to hexagon_info.json")
    ap.add_argument("--form", required=True, help="path to formdatabase.json")
    ap.add_argument("--out", default="semantic_map_data.json", help="output path")
    ap.add_argument("--panel-idx", type=int, default=0, help="panelIdx of the generated subspace")
    ap.add_argument("--panel-name", default="Auto", help="subspaceName of the generated subspace")
    ap.add_argument("--country-prefix", default="c", help='prefix for country_id (default "c")')
    ap.add_argument("--country-offset", type=int, default=0, help="offset added to numeric country when forming ID")
    ap.add_argument("--strict-format", action="store_true",
                    help="do NOT include msu_ids in hexList entries (strict backward-compat mode)")
    args = ap.parse_args()

    hex_info = load_json(args.hex_info)
    form = load_json(args.form)

    # 建 MSU_id -> 记录 的索引
    msu_map: Dict[int, Dict[str, Any]] = {}
    for rec in form:
        mid = rec.get("MSU_id")
        if mid is None:
            continue
        msu_map[int(mid)] = rec

    hex_items = []
    unknown_msu_ids = 0
    total_msu_refs = 0

    for cell in hex_info:
        coord = cell.get("hex_coord")
        if not (isinstance(coord, list) and len(coord) == 2):
            continue
        q, r = int(coord[0]), int(coord[1])
        country_raw = int(cell.get("country", 0))
        msu_ids_list = cell.get("MSU_ids") or []
        msu_ids = [int(x) for x in msu_ids_list if str(x).strip() != ""]
        # 统计未知
        for mid in msu_ids:
            total_msu_refs += 1
            if mid not in msu_map:
                unknown_msu_ids += 1

        modality = majority_modality(msu_ids, msu_map)
        country_id = f"{args.country_prefix}{country_raw + args.country_offset}"

        item = {
            "q": q,
            "r": r,
            "modality": modality,
            "country_id": country_id,
            "__country_raw": country_raw,  # 内部临时字段，用于 countries 聚合
        }
        if not args.strict_format:
            item["msu_ids"] = msu_ids

        hex_items.append(item)

    # countries 聚合
    countries = build_countries(hex_items, args.country_prefix, args.country_offset)

    # 去除内部字段
    for h in hex_items:
        h.pop("__country_raw", None)

    subspace = {
        "panelIdx": args.panel_idx,
        "subspaceName": args.panel_name,
        "hexList": hex_items,
        "countries": countries
    }

    out_obj = {
        "subspaces": [subspace],
        "links": []  # 暂无连线信息
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # 日志
    total_cells = len(hex_items)
    total_countries = len(countries)
    print(f"[done] wrote {args.out}")
    print(f"  subspaces        : 1 (panelIdx={args.panel_idx}, name={args.panel_name})")
    print(f"  hex cells        : {total_cells}")
    print(f"  countries        : {total_countries}")
    print(f"  msu refs (total) : {total_msu_refs}")
    print(f"  msu refs (miss)  : {unknown_msu_ids}")
    print(f"  strict-format    : {args.strict_format}")

if __name__ == "__main__":
    main()
