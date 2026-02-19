"""命令行入口：运行指定日期的新闻趋势日报流程。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 项目根目录与源码路径
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends.pipeline import run_daily_pipeline  # noqa: E402


def main() -> None:
    """解析 --date 参数，执行日报流程并打印 JSON 结果。"""
    parser = argparse.ArgumentParser(description="Run daily news trend scoring pipeline")
    parser.add_argument("--date", type=str, default=None, help="Target date: YYYY-MM-DD")
    args = parser.parse_args()

    result = run_daily_pipeline(target_date=args.date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
