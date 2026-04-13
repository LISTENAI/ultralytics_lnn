#!/usr/bin/env python3
"""
第三方依赖下载脚本
用于克隆 LISTENAI 工具链 (linger + thinker)

使用方法:
    python thirdparty/download_thirdparty.py

依赖:
    - Git
    - Python 3.8+
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, cwd=None):
    """执行命令并返回结果"""
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    return True


def clone_repo(url, target_dir):
    """克隆或更新 Git 仓库"""
    target_path = Path(target_dir)

    if target_path.exists() and any(target_path.iterdir()):
        print(f"✅ {target_path.name} 已存在，跳过克隆")
        return True

    print(f"📥 克隆 {url}...")
    if run_command(["git", "clone", "--depth", "1", url, str(target_path)]):
        print(f"✅ {target_path.name} 克隆完成")
        return True
    return False


def main():
    script_dir = Path(__file__).parent.resolve()
    thirdparty_dir = script_dir.parent / "thirdparty"

    print("=" * 60)
    print("📦 克隆 LISTENAI 第三方依赖")
    print("=" * 60)
    print()

    # 确保目录存在
    thirdparty_dir.mkdir(exist_ok=True)

    # 克隆 linger
    linger_url = "https://github.com/LISTENAI/linger.git"
    linger_dir = thirdparty_dir / "linger"
    clone_repo(linger_url, linger_dir)
    print()

    # 克隆 thinker
    thinker_url = "https://github.com/LISTENAI/thinker.git"
    thinker_dir = thirdparty_dir / "thinker"
    clone_repo(thinker_url, thinker_dir)
    print()

    # 打印安装说明
    print("=" * 60)
    print("📋 安装说明")
    print("=" * 60)
    print()
    print("如需安装 linger:")
    print(f"  cd {thirdparty_dir / 'linger'} && pip install -e .")
    print()
    print("如需安装 thinker:")
    print(f"  cd {thirdparty_dir / 'thinker'} && pip install -e .")
    print(f"  cd {thirdparty_dir / 'thinker' / 'tools'} && pip install -e .")
    print()
    print("=" * 60)
    print("✅ 第三方依赖准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()