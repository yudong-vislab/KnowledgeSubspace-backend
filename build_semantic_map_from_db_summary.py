#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_semantic_map_from_db_summary.py

适配“database-xxxx.json” + “summary-xxxx.json”成对输入，生成 semantic_map_data.json。
- 子空间名 = 文件名中 '-' 之后的后缀（去除扩展名），如：
  database-background.json / summary-background.json  => 子空间名 'background'

输入假设：
- database-*.json：包含本子空间的 MSU 全量记录列表（每项含 MSU_id、type、category...）
- summary-*.json ：包含本子空间下每个 hex 的信息（含坐标、国家、MSU_ids、summary）
  为了兼容不同写法，summary文件里的每个 cell 支持以下键名的任一组合：
    - 坐标：["hex_coord"] = [q, r]  或  "q","r"
    - 国家： "country"(数字)  或  "country_id"("c12" 之类，会自动抽出数字12)
    - MSU：  "MSU_ids" 或 "msu_ids"
    - 摘要： "summary" 或 "hex_summary"

输出结构（与原先一致）：
{
  "version": "1.1",
  "build_info": {...},
  "subspaces": [ { panelIdx, subspaceName, hexList, countries }... ],
  "links": [],
  "msu_index": { "<MSU_id>": <原始完整对象> },
  "indices": { msu_to_hex, panel_country_to_hex, category_to_cells },
  "stats": {...},
  "title": "Semantic Map View"
}

可选参数：
  --strict-format           # 不把 msu_ids/msu_details 放进 hexList
  --embed-msu-details       # 在每个 hex 内嵌完整 MSU 对象（文件会变大）
  --include-unknown-msu     # 若 summary 中引用了 database 不存在的 MSU_id，生成占位对象保留痕迹
  --country-prefix c        # 生成 country_id 的前缀
  --country-offset 0        # 给数字国家码加偏移后再拼接为 country_id

用法示例（推荐自动发现配对文件）：
  python build_semantic_map_from_db_summary.py \
    --case-dir case1 \
    --out data/case1/semantic_map_data.json

  python build_semantic_map_from_db_summary.py \
    --case-dir case2 \
    --out data/case2/semantic_map_data.json

  python build_semantic_map_from_db_summary.py \
    --case-dir case3 \
    --out data/case3/semantic_map_data.json

也支持显式传入（同长度、同顺序）：
  python build_semantic_map_from_db_summary.py \
    --database-files data/database-background.json data/database-methods.json \
    --summary-files  data/summary-background.json  data/summary-methods.json \
    --out data/semantic_map_data.json
"""

import os
import re
import json
import time
import argparse
import copy
from collections import defaultdict, Counter
from typing import Any, Dict, List, Iterable, Tuple

# -----------------------
# I/O
# -----------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------
# 小工具
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

def infer_suffix_name(filename: str) -> str:
    """
    从文件名中提取 '-' 之后到扩展名之前的后缀，作为子空间名。
    e.g. database-background.json -> background
         summary-Conclusion.json  -> Conclusion
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    if "-" in name:
        return name.split("-", 1)[1]
    return name  # 若没有 '-', 退化为整个文件名

def numeric_from_country_id(country_or_id: Any) -> int:
    """
    传入可能是数字(12)或字符串("c12")的国家字段，提取其数字部分；无法解析则返 0。
    """
    if isinstance(country_or_id, (int, float)):
        return int(country_or_id)
    if isinstance(country_or_id, str):
        m = re.search(r"(\d+)", country_or_id)
        if m:
            return int(m.group(1))
    return 0

def normalize_summary_cell(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 summary 文件中的单个 hex 单元规范化为：
      { "q": int, "r": int, "country_raw": int, "msu_ids": [int...], "summary": str }
    支持多种键名。
    """
    q, r = None, None
    if isinstance(raw.get("hex_coord"), list) and len(raw["hex_coord"]) == 2:
        q, r = int(raw["hex_coord"][0]), int(raw["hex_coord"][1])
    else:
        # 兼容 q/r 分散写法
        q = int(raw.get("q", 0))
        r = int(raw.get("r", 0))

    # 国家：数字或带前缀的 country_id
    country_raw = 0
    if "country" in raw:
        country_raw = numeric_from_country_id(raw.get("country"))
    elif "country_id" in raw:
        country_raw = numeric_from_country_id(raw.get("country_id"))

    # MSU 列表：MSU_ids / msu_ids
    msu_ids = raw.get("MSU_ids", raw.get("msu_ids", []))
    msu_ids = to_int_list(msu_ids)

    # 摘要：summary / hex_summary
    summary_text = raw.get("summary", raw.get("hex_summary", ""))
    if summary_text is None:
        summary_text = ""

    return {
        "q": q,
        "r": r,
        "country_raw": country_raw,
        "msu_ids": msu_ids,
        "summary": summary_text
    }

def build_countries(hex_items: List[Dict[str, Any]], country_prefix: str, country_offset: int):
    agg = defaultdict(list)
    for h in hex_items:
        cid = f"{country_prefix}{h['__country_raw'] + country_offset}"
        agg[cid].append({"q": h["q"], "r": h["r"]})
    countries = [{"country_id": cid, "hexes": hexes} for cid, hexes in agg.items()]
    countries.sort(key=lambda x: x["country_id"])
    return countries

def build_msu_index_full(forms: List[List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
    msu_map: Dict[int, Dict[str, Any]] = {}
    for rec_list in forms:
        for rec in rec_list:
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

# -----------------------
# indices 构建
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
# 从一对子空间文件组装 subspace
# -----------------------

def build_subspace_from_pair(
    db_records: List[Dict[str, Any]],
    summary_cells_raw: List[Dict[str, Any]],
    panel_idx: int,
    panel_name: str,
    msu_map_global: Dict[int, Dict[str, Any]],
    country_prefix: str,
    country_offset: int,
    strict_format: bool,
    embed_msu_details: bool,
    include_unknown_msu: bool
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    使用当前子空间的 database 记录 + summary cells 构建 subspace。
    - 仅保留 summary 中列出的 msu_ids。
    - 只有在 db_records 中出现的 MSU_id 才是该子空间的“允许集合”（允许引用 unknown 由选项控制）。
    """
    # 当前子空间允许的 MSU 集
    allowed_mids = {int(rec["MSU_id"]) for rec in db_records if "MSU_id" in rec}

    hex_items = []
    unknown_msu_ids = 0
    total_msu_refs = 0

    for raw in summary_cells_raw:
        cell = normalize_summary_cell(raw)

        # 与本子空间交集
        msu_ids = [mid for mid in cell["msu_ids"] if mid in allowed_mids]
        if not msu_ids:
            # 该 hex 在此子空间没有内容，跳过
            continue

        # 统计并处理未知
        for mid in msu_ids:
            total_msu_refs += 1
            if mid not in msu_map_global:
                unknown_msu_ids += 1
                if include_unknown_msu:
                    msu_map_global[mid] = make_unknown_stub(mid)

        modality = majority_modality(msu_ids, msu_map_global)
        country_id = f"{country_prefix}{cell['country_raw'] + country_offset}"

        item = {
            "q": cell["q"],
            "r": cell["r"],
            "modality": modality,
            "country_id": country_id,
            "__country_raw": cell["country_raw"],
            "summary": cell["summary"],  # 无论 strict 与否，都保留 summary
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
# 自动发现成对文件
# -----------------------

def discover_pairs(case_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    在 case_dir 下寻找 database-*.json 与 summary-*.json，按 * 后缀配对。
    返回 (db_files, summary_files, subspace_names) —— 三者顺序一一对应。
    """
    if not os.path.isdir(case_dir):
        raise RuntimeError(f"--case-dir not found: {case_dir}")

    entries = [f for f in os.listdir(case_dir) if f.lower().endswith(".json")]
    db_map = {}
    sum_map = {}
    for f in entries:
        lower = f.lower()
        if lower.startswith("database-"):
            suf = lower.split("database-", 1)[1]
            suf = os.path.splitext(suf)[0]
            db_map[suf] = os.path.join(case_dir, f)
        elif lower.startswith("summary-"):
            suf = lower.split("summary-", 1)[1]
            suf = os.path.splitext(suf)[0]
            sum_map[suf] = os.path.join(case_dir, f)

    # 求交集（只配对双方都存在的）
    common = sorted(set(db_map.keys()) & set(sum_map.keys()))
    if not common:
        raise RuntimeError(
            f"No pairs found in {case_dir}. Expect files like database-*.json and summary-*.json."
        )

    db_files = [db_map[s] for s in common]
    summary_files = [sum_map[s] for s in common]
    subspace_names = common  # 子空间名 = 后缀
    return db_files, summary_files, subspace_names

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Merge database-*.json + summary-*.json pairs -> semantic_map_data.json (multi subspaces, keep all raw MSU data + indices)")
    ap.add_argument("--case-dir", default=None, help="directory that contains database-*.json and summary-*.json")
    ap.add_argument("--database-files", nargs="*", default=None, help="explicit database-*.json files (order matches --summary-files)")
    ap.add_argument("--summary-files", nargs="*", default=None, help="explicit summary-*.json files (order matches --database-files)")
    ap.add_argument("--out", default="data/semantic_map_data.json", help="output json path")
    ap.add_argument("--country-prefix", default="c", help='prefix for country_id (default "c")')
    ap.add_argument("--country-offset", type=int, default=0, help="offset added to numeric country when forming ID")
    ap.add_argument("--strict-format", action="store_true", help="do NOT include msu_ids/msu_details in hexList entries")
    ap.add_argument("--embed-msu-details", action="store_true", help="embed full raw MSU objects into each hex cell (bigger file but zero lookup)")
    ap.add_argument("--include-unknown-msu", action="store_true", help="if summary references an unknown MSU_id, create a stub object so that everything is preserved")
    args = ap.parse_args()

    # 1) 发现/整理成对的文件 + 子空间名
    if args.case_dir:
        db_files, sum_files, subspace_names = discover_pairs(args.case_dir)
    else:
        if not (args.database_files and args.summary_files):
            raise RuntimeError("必须提供 --case-dir，或同时提供 --database-files 与 --summary-files。")
        if len(args.database_files) != len(args.summary_files):
            raise RuntimeError("--database-files 与 --summary-files 数量不一致。")
        db_files = list(args.database_files)
        sum_files = list(args.summary_files)
        # 子空间名来源：优先取文件名 '-' 之后的后缀；退化为去扩展名的文件名
        subspace_names = []
        for df, sf in zip(db_files, sum_files):
            # 取更可靠的：若两者后缀一致则用它；否则各自取后缀，优先数据库文件
            dname = infer_suffix_name(os.path.basename(df))
            sname = infer_suffix_name(os.path.basename(sf))
            subspace_names.append(dname if dname else sname)

    # 2) 读入所有 database（按子空间）
    all_db_records: List[List[Dict[str, Any]]] = [load_json(p) for p in db_files]
    # 全量 msu 索引（跨子空间）
    msu_map_global = build_msu_index_full(all_db_records)

    # 3) 逐对子空间构建 subspace
    subspaces: List[Dict[str, Any]] = []
    stats_by_panel: List[Dict[str, int]] = []

    for i, (db_path, sum_path, sp_name) in enumerate(zip(db_files, sum_files, subspace_names)):
        db_records = load_json(db_path)
        summary_cells_raw = load_json(sum_path)

        # 容错：若 summary 顶层是 dict 且有 "hexList"/"cells" 字段
        if isinstance(summary_cells_raw, dict):
            if "hexList" in summary_cells_raw:
                summary_cells_raw = summary_cells_raw["hexList"]
            elif "cells" in summary_cells_raw:
                summary_cells_raw = summary_cells_raw["cells"]

        if not isinstance(summary_cells_raw, list):
            raise RuntimeError(f"Summary file not a list: {sum_path}")

        subspace, stats = build_subspace_from_pair(
            db_records=db_records,
            summary_cells_raw=summary_cells_raw,
            panel_idx=i,
            panel_name=sp_name,
            msu_map_global=msu_map_global,
            country_prefix=args.country_prefix,
            country_offset=args.country_offset,
            strict_format=args.strict_format,
            embed_msu_details=args.embed_msu_details,
            include_unknown_msu=args.include_unknown_msu
        )
        subspaces.append(subspace)
        stats_by_panel.append(stats)

    # 4) 反向索引与统计
    indices = build_indices(subspaces, msu_map_global)
    stats_out = {
        "subspaces": [
            {"panelIdx": sp["panelIdx"], "name": sp["subspaceName"], "cells": len(sp["hexList"]), "countries": len(sp["countries"])}
            for sp in subspaces
        ],
        "totals": {
            "cells": sum(len(sp["hexList"]) for sp in subspaces),
            "countries": sum(len(sp["countries"]) for sp in subspaces),
            "msu_count": len(msu_map_global)
        },
        "msu_refs_total": sum(s["total_msu_refs"] for s in stats_by_panel),
        "msu_refs_missing": sum(s["unknown_msu_ids"] for s in stats_by_panel)
    }

    # 5) 输出
    out_obj = {
        "version": "1.1",
        "build_info": {
            "ts": int(time.time()),
            "tool": "build_semantic_map_from_db_summary.py",
            "options": {
                "strict_format": args.strict_format,
                "embed_msu_details": args.embed_msu_details,
                "include_unknown_msu": args.include_unknown_msu,
                "country_prefix": args.country_prefix,
                "country_offset": args.country_offset,
                "database_files": db_files,
                "summary_files": sum_files
            }
        },
        "subspaces": subspaces,
        "links": [],
        "msu_index": msu_map_global,
        "indices": indices,
        "stats": stats_out,
        "title": "Semantic Map View"
    }

    write_json(args.out, out_obj)

    # 日志
    print(f"[done] wrote {args.out}")
    print(f"  subspaces : {len(subspaces)} ({', '.join(subspace_names)})")
    for i, s in enumerate(stats_by_panel):
        print(f"    - panel {i} '{subspace_names[i]}': cells={s['total_cells']}, countries={s['total_countries']}, "
              f"msu_refs={s['total_msu_refs']} (miss={s['unknown_msu_ids']})")
    print(f"  totals    : cells={stats_out['totals']['cells']}, countries={stats_out['totals']['countries']}, "
          f"msu_count={stats_out['totals']['msu_count']}")
    print(f"  strict-format    : {args.strict_format}")
    print(f"  embed-msu-details: {args.embed_msu_details}")
    print(f"  include-unknown  : {args.include_unknown_msu}")
    print(f"  output path      : {args.out}")

if __name__ == "__main__":
    main()
