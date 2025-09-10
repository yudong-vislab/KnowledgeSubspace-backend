import re
import json
import os
from pathlib import Path
# input_path = Path("datamd/airvis/airvis.md")
input_path = Path("/home/lxy/case_engine/largeeddy/largeeddy.md")
# output_path = Path("airvis.md")
output_json = Path("/home/lxy/case_engine/largeeddy/largeeddy.json")

def strip_bold(text: str) -> str:
    """
    去除外层的 **…** 或 __…__，只处理包裹整个字符串的情况。
    """
    if (text.startswith("**") and text.endswith("**")) or (text.startswith("__") and text.endswith("__")):
        return text[2:-2].strip()
    return text

def correct_markdown_header_levels(md_text: str) -> str:
    """
    根据标题编号（如 3, 3.1, 3.1.1）自动重构 Markdown 标题层级：
      - 忽略原有的 `#` 数量，仅根据编号深度决定新的 `#` 数量
      - 无论原行是否带 `#`，只要以编号开头（如 "3 ", "3.1 ", "3.1.1 "），均识别为标题
      - `3`      -> `# 3 …`
      - `3.1`    -> `## 3-1 …`
      - `3.1.1`  -> `### 3-1-1 …`
    对既不带编号也不以 `#` 开头的行保持原样。
    """
    # 匹配可选的 #，后跟编号（1、1.2、1.2.3…），再跟空格或连字符，然后是标题文本
    header_re = re.compile(r'^\s*#+\s*(\d+(?:\.\d+)*)(?:\s+)(.*)$')
    out_lines = []

    for line in md_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('#'):
            # 去除前导 '#' 及空格，获取标题主体
            body = stripped.lstrip('#').strip()
            # 去除外层加粗
            body = strip_bold(body)
            m = header_re.match(f"# {body}")  # 人为加一个 '#' 让正则能统一处理去掉的 #
            if m:
                num = m.group(1)                # "3"、"3.1"、"3.1.1"
                print(num)
                title_text = m.group(2).strip() # 去除后剩下的标题内容
                level = num.count('.') + 1      # 点数量+1 决定 # 数量
                hashes = '#' * level
                out_lines.append(f"{hashes} {num} {title_text}")
                continue
        # 不满足上述情况，保持原样
        out_lines.append(line)

    return '\n'.join(out_lines)
def parse_markdown_to_json(md_text: str,abstract_summary: str) -> list:
    """
    将已修正标题层级的 Markdown 文本按标题分组，并将每个标题与其所有上级标题合并：
    title 格式："<顶级标题>-<次级标题>-...-<本级标题>"
    返回 JSON 结构：
    [
      {
        "title": "4 PATTERN MINING-4.1 Modelling Pollutant Transportation",
        "paragraphs": [ ... ]
      },
      ...
    ]
    """
    sections = []
    current_section = None
    parent_titles = {}  # 存储各层级最新标题，如 {1: "4 PATTERN MINING", 2: "4.1 Modelling Pollutant Transportation"}

    for line in md_text.splitlines():
        # 标题行：以 # 开头
        if line.lstrip().startswith('#'):
            m = re.match(r'^(#+)\s+(.*)$', line)
            if not m:
                continue
            level = len(m.group(1))
            title_text = m.group(2).strip()
            # 更新当前层级标题，并清除更深层级
            parent_titles[level] = title_text
            for deeper in [l for l in parent_titles if l > level]:
                del parent_titles[deeper]
            # 合并所有层级标题
            sorted_levels = sorted(parent_titles)
            full_title = "-".join(parent_titles[l] for l in sorted_levels)
            # 新建节
            current_section = {"title": full_title, "paragraphs": []}
            sections.append(current_section)
            continue

        # 非标题行：属于当前节的正文或图片
        if current_section is None:
            continue
        if not line.strip():
            continue
        # 跳过列表、公式或代码块
        if re.match(r'\s*([-*]\s+|\d+\.\s+|\$.*\$|```)', line):
            continue
        # 检测图片
        fig = re.search(r'!\[.*?\]\((.*?)\)', line)
        if fig:
            para = {
                "type": "figure",
                "origin_text": fig.group(1),
                "abstract_summary": abstract_summary
            }
        else:
            para = {
                "type": "text",
                "origin_text": line.strip(),
                "abstract_summary": abstract_summary
            }
        current_section["paragraphs"].append(para)

    return sections
# def parse_markdown_to_json(md_text: str, paper_title: str, abstract_summary: str) -> list:
    """
    将已修正标题层级的 Markdown 文本按标题分组，生成 JSON 结构：
    [
      {
        "title": "<标题编号及文本>",
        "paragraphs": [
          {
            "type": "text"|"figure",
            "origin_text": "...",
            "paper_title": "...",
            "abstract_summary": "..."
          },
          ...
        ]
      },
      ...
    ]
    """
    sections = []
    current_section = None
    for line in md_text.splitlines():
        # 遇到标题行（以 # 开头）
        if line.startswith('#'):
            title = line.lstrip('#').strip()
            current_section = {
                "title": title,
                "paragraphs": []
            }
            sections.append(current_section)
            continue

        if current_section is None:
            continue
        # 跳过空行
        if not line.strip():
            continue
        # 跳过列表、公式或代码块
        if re.match(r'\s*([-*]\s+|\d+\.\s+|\$.*\$|```)', line):
            continue

        # 检测图片
        fig_match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if fig_match:
            para = {
                "type": "figure",
                "origin_text": fig_match.group(1),
                "paper_title": paper_title,
                "abstract_summary": abstract_summary
            }
        else:
            para = {
                "type": "text",
                "origin_text": line.strip(),
                "paper_title": paper_title,
                "abstract_summary": abstract_summary
            }
        current_section["paragraphs"].append(para)

    return sections

def batch_process(root_dir: str, abstract_summary: str):
    root = Path(root_dir)
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        md_path = sub/f"{name}.md"
        json_path = sub/f"{name}.json"
        output_mdpath = sub/f"{name}_reorgan.md"

        if not md_path.exists():
            print(f"跳过 {sub}: 未找到 {md_path.name}")
            continue

        md_text = md_path.read_text(encoding="utf-8")
        md_fixed = correct_markdown_header_levels(md_text)
        output_mdpath.write_text(md_fixed, encoding="utf-8")
        sections = parse_markdown_to_json(
            md_fixed,
            abstract_summary=abstract_summary
        )
        json_path.write_text(
            json.dumps(sections, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✔ 生成 {json_path.relative_to(root_dir)}")
def main():
    # DATA_ROOT = "usefor_air"
    GLOBAL_ABSTRACT_SUMMARY = "在此填写论文摘要的精简版"
    
    md_text = input_path.read_text(encoding="utf-8")
    md_fixed = correct_markdown_header_levels(md_text)
    sections = parse_markdown_to_json(
        md_fixed,
        abstract_summary=GLOBAL_ABSTRACT_SUMMARY
    )
    output_json.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✔ 生成 {output_json}")
    
    # batch_process(DATA_ROOT, GLOBAL_ABSTRACT_SUMMARY)


if __name__ == "__main__":
    main()


