#!/usr/bin/env python3
"""
生成 folders.json 文件，列出所有可用的数据集文件夹
"""

import os
import json

# 获取当前目录下所有 output_* 文件夹
folders = ["default"]
for item in os.listdir("."):
    if item.startswith("output_") and os.path.isdir(item):
        folders.append(item)

# 按名称排序（default 在前，其他按时间戳排序）
folders_sorted = ["default"] + sorted(
    [f for f in folders if f != "default"], reverse=True
)

# 写入 JSON 文件
with open("folders.json", "w", encoding="utf-8") as f:
    json.dump(folders_sorted, f, indent=2, ensure_ascii=False)

print(f"Generated folders.json with {len(folders_sorted)} folders:")
for folder in folders_sorted:
    print(f"  - {folder}")
