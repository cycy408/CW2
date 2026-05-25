"""将当前目录所有 .py 转为 notebooks/*.ipynb，保留源文件"""
import json, os, re

NOTEBOOK_DIR = "notebooks"
KERNEL_SPEC = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

TITLE_MAP = {
    "main": "01-Main Pipeline",
    "data_prep": "02-Data Preparation",
    "feature_selection": "03-Feature Selection",
    "models": "04-Models",
    "evaluation": "05-Evaluation",
    "model_fusion": "06-Model Fusion",
    "advanced": "07-Advanced Layer",
    "analysis": "08-Outlier Analysis",
    "visualization": "09-Visualization",
}

def is_section_header(line):
    """匹配 # =====...===== 格式的 section 注释（允许缩进）"""
    return bool(re.match(r'^\s*#\s*=+\s*[^=]', line))

def is_top_level_def(line):
    """顶层函数或类定义"""
    return bool(re.match(r'^(def |class |if __name__)', line))

def split_into_sections(code):
    """按 section header 将代码切成 (header_text, code_lines) 段"""
    lines = code.split('\n')
    sections = []  # [(header_or_None, code_lines)]
    current_code = []

    for line in lines:
        if is_section_header(line):
            if current_code:
                sections.append((None, current_code))
                current_code = []
            header_text = line.strip().lstrip('# ').strip()
            sections.append((header_text, []))
        else:
            current_code.append(line)

    if current_code:
        sections.append((None, current_code))

    return sections

def code_lines_to_cells(code_lines):
    """将一段代码行列表拆分为 cell 列表 [(type, text)]"""
    if not code_lines:
        return []

    text = '\n'.join(code_lines).strip()
    if not text:
        return []

    # 尝试拆分为多个 cell：按顶层函数/类定义拆分
    cells = []
    buf = []

    for line in code_lines:
        if is_top_level_def(line) and buf and buf != ['']:
            t = '\n'.join(buf).strip()
            if t:
                cells.append(("code", t))
            buf = [line]
        else:
            buf.append(line)

    if buf:
        t = '\n'.join(buf).strip()
        if t:
            cells.append(("code", t))

    return cells

def strip_main_wrapper(code):
    """移除 main.py 的 def main(): 包装和 if __name__ 守卫，返回平铺代码行"""
    lines = code.split('\n')
    result = []
    in_main = False
    main_indent = None
    skip_rest = False

    for line in lines:
        if skip_rest:
            continue
        if line.startswith('def main():'):
            in_main = True
            continue
        if in_main:
            if main_indent is None and line.strip():
                main_indent = len(line) - len(line.lstrip())
            if line.strip() == '':
                result.append('')
                continue
            # 检测是否退出了 main 函数（unindented non-empty line after main body）
            current_indent = len(line) - len(line.lstrip())
            if main_indent is not None and current_indent < main_indent and line.strip():
                in_main = False
                # 处理 main() 调用后的行
                if line.startswith('if __name__'):
                    skip_rest = True
                    continue
                result.append(line)
                continue
            # 去掉 main 函数体的一级缩进
            if main_indent is not None and line.strip():
                result.append(line[main_indent:])
            else:
                result.append('')
        else:
            result.append(line)

    # 去掉末尾的 if __name__ ...（如果还在 result 里）
    text = '\n'.join(result)
    text = re.sub(r'\n+if __name__\s*==\s*["\']__main__["\']\s*:\s*\n\s+main\(\)\s*\n*$', '', text)
    return text

def convert_file(py_file):
    """转换单个 .py 文件，返回 notebook dict"""
    with open(py_file, 'r', encoding='utf-8') as f:
        code = f.read()

    base = py_file.replace('.py', '')
    is_main = (base == 'main')

    if is_main:
        code = strip_main_wrapper(code)

    sections = split_into_sections(code)

    nb_cells = []
    # 标题 cell
    nb_cells.append({
        "cell_type": "markdown", "metadata": {},
        "source": [f"# {TITLE_MAP.get(base, base)}\n"]
    })

    for header, code_lines in sections:
        if header:
            nb_cells.append({
                "cell_type": "markdown", "metadata": {},
                "source": [f"## {header}\n"]
            })
        for ctype, text in code_lines_to_cells(code_lines):
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [text + '\n']
            })

    return {
        "cells": nb_cells,
        "metadata": KERNEL_SPEC,
        "nbformat": 4,
        "nbformat_minor": 5
    }

def main():
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    py_files = sorted([f for f in os.listdir('.') if f.endswith('.py') and not f.startswith('_')])

    for py_file in py_files:
        nb = convert_file(py_file)
        notebook_name = py_file.replace('.py', '.ipynb')
        out_path = os.path.join(NOTEBOOK_DIR, notebook_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
        md_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
        print(f"  -> {notebook_name}  ({code_cells} code + {md_cells} markdown cells)")

    print(f"\nDone. {len(py_files)} notebooks in '{NOTEBOOK_DIR}/'")

if __name__ == '__main__':
    main()
