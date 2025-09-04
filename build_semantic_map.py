#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_semantic_map.py (multi-subspaces, keep-all-raw + indices)

功能：
- 读取 data/hexagon_info.json（含 hex_coord/country/MSU_ids）
- 读取 1~N 份 form JSON（与原 formdatabase.json 相同结构；可按 category 拆分 Background/Conclusion/...）
- 生成 semantic_map_data.json：
  {
    "version": "1.1",
    "build_info": {...},
    "subspaces": [ { panelIdx, subspaceName, hexList, countries }... ],
    "links": [],
    "msu_index": { "<MSU_id>": <原始完整对象> },      # 全量 MSU 原始对象（来自所有 form）
    "indices": {
      "msu_to_hex": { "<MSU_id>": { panelIdx, q, r, country_id } },
      "panel_country_to_hex": { "<panelIdx>": { "<country_id>": [[q, r], ...] } },
      "category_to_cells": { "<category>": [ { panelIdx, q, r }, ... ] }
    },
    "stats": { ... }
  }

特性：
- ✅ 多子空间：每个 form JSON => 一个子空间（panel）
- ✅ 子空间 hexList 仅包含该 form 内的 MSU_id（与 hexagon_info 的交集）
- ✅ 完整保留每条 MSU 的原始数据（msu_index）
- ✅ 不改变现有前端读取方式，同时新增 indices 反向索引
- ✅ 可选：--embed-msu-details（将完整 MSU 对象内嵌到 hex）
- ✅ 可选：--strict-format（仅保留 q,r,modality,country_id，不在 hex 内放 msu_ids/msu_details）
- ✅ 可选：--include-unknown-msu（hex 引用 form 中不存在的 MSU_id 时，生成占位对象以“保留”）
- ✅ 子空间名默认取文件名（不带扩展名），也可通过 --subspace-names 显式指定

用法示例：
  # 单子空间（向后兼容）
  python build_semantic_map.py \
    --hex-info data/hexagon_info.json \
    --form data/formdatabase.json \
    --out data/semantic_map_data.json

  # 多子空间（推荐）：把多个 form JSON 分别当作一个子空间
  python build_semantic_map.py \
    --hex-info data/hexagon_info.json \
    --form-files data/Background.json data/Method.json data/Conclusion.json \
    --subspace-names Background Methods Conclusion \
    --out data/semantic_map_data.json

可选参数：
  --embed-msu-details
  --strict-format
  --include-unknown-msu
  --country-prefix c
  --country-offset 0
"""

import json
import argparse
import copy
import time
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List, Iterable, Tuple


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
            continue
    return out

def majority_modality(msu_ids: Iterable[int], msu_map: Dict[int, Dict[str, Any]]) -> str:
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
    agg = defaultdict(list)
    for h in hex_items:
        cid = f"{country_prefix}{h['__country_raw'] + country_offset}"
        agg[cid].append({"q": h["q"], "r": h["r"]})
    countries = [{"country_id": cid, "hexes": hexes} for cid, hexes in agg.items()]
    countries.sort(key=lambda x: x["country_id"])
    return countries

def build_msu_index_full(form_list: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    接受一个“记录列表”的列表（可来自多个 form 文件），
    构建全量 msu_map（key 为 int MSU_id；value 为原始完整对象 deep copy）。
    重复 MSU_id 后者覆盖前者（通常不会重复）。
    """
    msu_map: Dict[int, Dict[str, Any]] = {}
    for form in form_list:
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

def infer_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name or base or "Subspace"


# -----------------------
# indices 构建（反向索引）
# -----------------------

def build_indices(subspaces: List[Dict[str, Any]], msu_index: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    msu_to_hex: Dict[str, Dict[str, Any]] = {}
    panel_country_to_hex: Dict[str, Dict[str, List[List[int]]]] = defaultdict(lambda: defaultdict(list))
    category_to_cells: Dict[str, List[Dict[str, int]]] = defaultdict(list)

    msu_id_to_cat: Dict[int, Any] = {}
    for mid, obj in msu_index.items():
        msu_id_to_cat[mid] = obj.get("category")

    for sp in subspaces:
        pidx = sp["panelIdx"]
        for cell in sp["hexList"]:
            q, r = cell["q"], cell["r"]
            cid = cell["country_id"]
            panel_country_to_hex[str(pidx)][cid].append([q, r])
            for mid in cell.get("msu_ids", []):
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
# 组装子空间
# -----------------------

def build_subspace_from_formslice(
    hex_info: List[Dict[str, Any]],
    msu_map_slice: Dict[int, Dict[str, Any]],
    msu_map_global: Dict[int, Dict[str, Any]],
    panel_idx: int,
    panel_name: str,
    country_prefix: str,
    country_offset: int,
    strict_format: bool,
    embed_msu_details: bool,
    include_unknown_msu: bool
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    - 针对“某个子空间的 MSU 子集”（msu_map_slice）构建一个 subspace。
    - hex_info 仍是统一的 hexagon_info（通过 MSU 交集筛选到该子空间）。
    - modality/countries/hexList 都基于“属于该子空间的 MSU_ids”计算。
    """
    # 该子空间允许的 MSU 集
    allowed_mids = set(msu_map_slice.keys())

    hex_items = []
    unknown_msu_ids = 0
    total_msu_refs = 0

    for cell in hex_info:
        coord = cell.get("hex_coord")
        if not (isinstance(coord, list) and len(coord) == 2):
            continue
        q, r = int(coord[0]), int(coord[1])
        country_raw = int(cell.get("country", 0))

        # 只保留该子空间相关的 MSU_ids（交集）
        msu_ids_all = to_int_list(cell.get("MSU_ids"))
        msu_ids = [mid for mid in msu_ids_all if mid in allowed_mids]

        if not msu_ids:
            # 这个 hex 在该子空间没有内容，跳过
            continue

        # 统计未知（理论上不会，因为 allowed_mids 来自 form 切片）
        for mid in msu_ids:
            total_msu_refs += 1
            if mid not in msu_map_global:
                unknown_msu_ids += 1
                if include_unknown_msu:
                    msu_map_global[mid] = make_unknown_stub(mid)

        modality = majority_modality(msu_ids, msu_map_global)
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
                item["msu_details"] = [msu_map_global[mid] for mid in msu_ids if mid in msu_map_global]

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
    ap = argparse.ArgumentParser(description="Merge hexagon_info.json + 1..N form json(s) -> semantic_map_data.json (multi subspaces, keep all raw MSU data + indices)")
    ap.add_argument("--hex-info", default="data/hexagon_info.json", help="path to hexagon_info.json")
    # 单文件（向后兼容）
    ap.add_argument("--form", default=None, help="single formdatabase.json (legacy single subspace)")
    # 多文件（新）
    ap.add_argument("--form-files", nargs="*", default=None, help="multiple form json files, each becomes a subspace")
    ap.add_argument("--subspace-names", nargs="*", default=None, help="names for subspaces in the same order as --form-files")
    ap.add_argument("--out", default="data/semantic_map_data.json", help="output path")
    ap.add_argument("--country-prefix", default="c", help='prefix for country_id (default "c")')
    ap.add_argument("--country-offset", type=int, default=0, help="offset added to numeric country when forming ID")
    ap.add_argument("--strict-format", action="store_true", help="do NOT include msu_ids/msu_details in hexList entries")
    ap.add_argument("--embed-msu-details", action="store_true", help="embed full raw MSU objects into each hex cell (bigger file but zero lookup)")
    ap.add_argument("--include-unknown-msu", action="store_true", help="if hex references an unknown MSU_id, create a stub object so that everything is preserved")
    args = ap.parse_args()

    # 读取 hexagon_info（全局）
    hex_info = load_json(args.hex_info)

    # 读取 form 数据（单文件 or 多文件）
    form_files: List[str] = []
    subspace_names: List[str] = []

    if args.form_files:
        form_files = list(args.form_files)
        if args.subspace_names and len(args.subspace_names) == len(form_files):
            subspace_names = list(args.subspace_names)
        else:
            # 未提供名字，则按文件名推断
            subspace_names = [infer_name_from_path(p) for p in form_files]
    elif args.form:
        form_files = [args.form]
        subspace_names = [infer_name_from_path(args.form)]
    else:
        raise RuntimeError("必须提供 --form（单文件）或 --form-files（多文件）。")

    # 读入所有 form
    forms_raw: List[List[Dict[str, Any]]] = [load_json(p) for p in form_files]

    # 1) 全量 msu_index（合并所有 form）
    msu_map_global = build_msu_index_full(forms_raw)

    subspaces: List[Dict[str, Any]] = []
    stats_by_panel: List[Dict[str, int]] = []

    # 2) 逐个 form 切片 => 构建子空间
    for i, (form_records, name) in enumerate(zip(forms_raw, subspace_names)):
        # 该子空间的 MSU 子集
        msu_slice = build_msu_index_full([form_records])
        subspace, stats = build_subspace_from_formslice(
            hex_info=hex_info,
            msu_map_slice=msu_slice,
            msu_map_global=msu_map_global,
            panel_idx=i,
            panel_name=name,
            country_prefix=args.country_prefix,
            country_offset=args.country_offset,
            strict_format=args.strict_format,
            embed_msu_details=args.embed_msu_details,
            include_unknown_msu=args.include_unknown_msu
        )
        subspaces.append(subspace)
        stats_by_panel.append(stats)

    # 3) 反向索引（覆盖所有子空间）
    indices = build_indices(subspaces, msu_map_global)

    # 4) 汇总统计
    totals_msu = len(msu_map_global)
    stats_out = {
        "subspaces": [
            {"panelIdx": sp["panelIdx"], "name": sp["subspaceName"], "cells": len(sp["hexList"]), "countries": len(sp["countries"])}
            for sp in subspaces
        ],
        "totals": {
            "cells": sum(len(sp["hexList"]) for sp in subspaces),
            "countries": sum(len(sp["countries"]) for sp in subspaces),
            "msu_count": totals_msu
        },
        "msu_refs_total": sum(s["total_msu_refs"] for s in stats_by_panel),
        "msu_refs_missing": sum(s["unknown_msu_ids"] for s in stats_by_panel)
    }

    # 5) 输出
    out_obj = {
        "version": "1.1",
        "build_info": {
            "ts": int(time.time()),
            "tool": "build_semantic_map.py",
            "options": {
                "strict_format": args.strict_format,
                "embed_msu_details": args.embed_msu_details,
                "include_unknown_msu": args.include_unknown_msu,
                "country_prefix": args.country_prefix,
                "country_offset": args.country_offset,
                "forms": form_files
            }
        },
        "subspaces": subspaces,
        "links": [],            # 如需，也可在这里补你的静态 links
        "msu_index": msu_map_global,
        "indices": indices,
        "stats": stats_out,
        "title": "Semantic Map View"
    }

    write_json(args.out, out_obj)

    # 日志
    print(f"[done] wrote {args.out}")
    print(f"  subspaces        : {len(subspaces)} ({', '.join(subspace_names)})")
    for i, s in enumerate(stats_by_panel):
        print(f"    - panel {i} '{subspace_names[i]}': cells={s['total_cells']}, countries={s['total_countries']}, msu_refs={s['total_msu_refs']} (miss={s['unknown_msu_ids']})")
    print(f"  totals           : cells={stats_out['totals']['cells']}, countries={stats_out['totals']['countries']}, msu_count={totals_msu}")
    print(f"  strict-format    : {args.strict_format}")
    print(f"  embed-msu-details: {args.embed_msu_details}")
    print(f"  include-unknown  : {args.include_unknown_msu}")
    print(f"  output path      : {args.out}")


if __name__ == "__main__":
    main()
