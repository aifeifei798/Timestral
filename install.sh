#!/usr/bin/env bash
# Timestral API 环境安装脚本：创建 .venv 并安装依赖（torch>=2.9 的 PyPI 默认构建即 cu130）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "[install] 未找到 uv，改用 python3 -m venv + pip"
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
else
    [ -d .venv ] || uv venv --python 3.12 .venv
    # 全部命中 uv 缓存时离线可装；缓存缺失则自动联网
    uv pip install -r requirements.txt || {
        echo "[install] 离线解析失败，回退联网安装..."
        uv pip install -r requirements.txt --offline=false
    }
fi

echo "[install] 校验 torch + CUDA..."
.venv/bin/python - <<'EOF'
import torch
print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("警告: CUDA 不可用，服务将以 CPU 模式运行（需 NVIDIA 驱动 >= 580）")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

echo "[install] 完成。启动服务: ./server.sh"
