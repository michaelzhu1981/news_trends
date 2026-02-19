"""配置与路径管理模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# 项目根目录（news_trends 的上一级上一级）
ROOT = Path(__file__).resolve().parents[2]
# 配置文件目录
CONFIG_DIR = ROOT / "config"
# 数据目录
DATA_DIR = ROOT / "data"
# 原始数据目录
RAW_DIR = DATA_DIR / "raw"
# 处理后数据目录（含评分结果 JSON）
PROCESSED_DIR = DATA_DIR / "processed"
# 报告输出目录（Markdown 报告）
REPORT_DIR = ROOT / "reports"


def load_yaml(path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件，返回解析后的字典。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feeds() -> list[dict[str, str]]:
    """加载 RSS 源配置，返回 feeds 列表。"""
    data = load_yaml(CONFIG_DIR / "feeds.yaml")
    return data.get("feeds", [])


def load_scoring_config() -> dict[str, Any]:
    """加载打分配置（主题权重、关键词、规则等）。"""
    return load_yaml(CONFIG_DIR / "scoring.yaml")
