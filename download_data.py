"""
数据下载脚本
============
从Kaggle下载 Store Sales 比赛数据。

使用方式:
    python download_data.py                     # 下载全部数据
    python download_data.py --list              # 列出数据文件

需要先配置Kaggle API:
    1. 在 https://www.kaggle.com/settings/api 创建API Token
    2. 将 kaggle.json 放到 ~/.kaggle/ 目录下
    3. 或设置环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY
"""

import os
import sys
from pathlib import Path

COMPETITION = "store-sales-time-series-forecasting"
DATA_DIR = Path(__file__).parent / "data"


def download_with_api():
    """使用 kagglehub 下载（推荐方式，无需API token）"""
    try:
        import kagglehub
        print(f"正在通过 kagglehub 下载 {COMPETITION} 数据...")
        path = kagglehub.competition_download(COMPETITION, path=str(DATA_DIR))
        print(f"数据已下载到: {path}")
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"kagglehub 下载失败: {e}")
        return False


def download_with_kaggle_cli():
    """使用 kaggle CLI 下载（需要API token）"""
    try:
        import subprocess
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        cmd = [
            "kaggle", "competitions", "download",
            "-c", COMPETITION,
            "-p", str(DATA_DIR)
        ]
        print(f"正在通过 kaggle CLI 下载...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("下载成功！")
            # 解压
            import zipfile
            for f in DATA_DIR.glob("*.zip"):
                with zipfile.ZipFile(f, "r") as zf:
                    zf.extractall(DATA_DIR)
                f.unlink()  # 删除zip
            return True
        else:
            print(f"下载失败: {result.stderr}")
            return False
    except FileNotFoundError:
        return False


def manual_instructions():
    """打印手动下载说明"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              手动下载数据步骤                                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 打开比赛页面:                                             ║
║     https://www.kaggle.com/competitions/                     ║
║     store-sales-time-series-forecasting/data                 ║
║                                                              ║
║  2. 点击 "Download All" 按钮                                  ║
║                                                              ║
║  3. 将下载的 ZIP 文件解压到:                                   ║
║     {}     ║
║                                                              ║
║  4. 解压后应包含以下文件:                                       ║
║     - train.csv                                              ║
║     - test.csv                                               ║
║     - stores.csv                                             ║
║     - oil.csv                                                ║
║     - holidays_events.csv                                    ║
║     - transactions.csv                                       ║
║     - sample_submission.csv                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """.format(DATA_DIR.absolute()))


def main():
    print("=" * 60)
    print("  Store Sales 数据下载工具")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # 检查数据是否已存在
    required_files = [
        "train.csv", "test.csv", "stores.csv",
        "oil.csv", "holidays_events.csv", "transactions.csv"
    ]
    existing = [f for f in required_files if (DATA_DIR / f).exists()]

    if len(existing) == len(required_files):
        print(f"\n✅ 所有数据文件已存在 ({DATA_DIR})")
        return

    if existing:
        print(f"\n部分数据已存在 ({len(existing)}/{len(required_files)})，继续下载缺失文件...")

    # 尝试自动下载
    print("\n尝试自动下载...")

    # 方法1: kagglehub
    print("\n[方法1] 尝试 kagglehub...")
    if not download_with_api():
        # 方法2: kaggle CLI
        print("\n[方法2] 尝试 kaggle CLI...")
        if not download_with_kaggle_cli():
            # 方法3: 手动
            print("\n❌ 自动下载失败")
            manual_instructions()


if __name__ == "__main__":
    main()
