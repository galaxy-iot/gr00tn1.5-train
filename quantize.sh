#!/usr/bin/env bash
# Isolated uv environment for Qualcomm GR00T quantization and float ONNX export.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
quant_env="${GR00T_QUANT_ENV:-$project_root/.venv-quant}"
qai_package="qai-hub-models${QAI_HUB_MODELS_VERSION:+==$QAI_HUB_MODELS_VERSION}"
stage="${1:-help}"
if (($#)); then shift; fi

case "$stage" in
  help|-h|--help)
    cat <<'EOF'
Usage:
  bash quantize.sh setup             Create Python 3.12 uv env and install dependencies
  bash quantize.sh check             Check CUDA and installed dependencies
  bash quantize.sh quantize [args]   Run scripts/quantize_model.py
  bash quantize.sh onnx [args]       Run scripts/export_onnx.py (float ONNX)
  bash quantize.sh preprocess [args] Export offline tokenizer and checkpoint embeddings
  bash quantize.sh qnn [args]        Compile calibrated models using AI Hub (uploads model)

Install published qai-hub-models from the configured package index.
Set QAI_HUB_MODELS_VERSION to pin a release (otherwise latest stable).
Set GR00T_QUANT_ENV to override .venv-quant.
Run setup first; it installs the published model package and CLI, then the
official grootn15 dependency recipe. Quantization requires a CUDA GPU;
the official recipe recommends 40 GB VRAM for the 3B model.
EOF
    exit 0 ;;
  setup|check|quantize|onnx|qnn|preprocess) ;;
  *) echo "Unknown stage: $stage (use --help)" >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo 'uv is required: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }

if [[ "$stage" == setup ]]; then
  if [[ ! -x "$quant_env/bin/python" ]]; then
    uv venv --python 3.12 "$quant_env"
  fi
else
  [[ -x "$quant_env/bin/python" ]] || { echo 'Run bash quantize.sh setup first.' >&2; exit 1; }
fi

# The official installer invokes uv pip itself; target this env even inside conda.
export VIRTUAL_ENV="$quant_env"
export PATH="$quant_env/bin:$PATH"
export UV_PYTHON="$quant_env/bin/python"
export NO_ALBUMENTATIONS_UPDATE=1
unset PYTHONHOME PYTHONPATH

if [[ "$stage" == setup ]]; then
  uv pip install --python "$quant_env/bin/python" --upgrade "$qai_package"
  "$quant_env/bin/qai-hub-models" install grootn15 --yes
  uv pip freeze --python "$quant_env/bin/python" > "$quant_env/requirements-installed.txt"
  echo "Setup complete. Dependency snapshot: $quant_env/requirements-installed.txt"
  exit 0
fi

if [[ "$stage" == check || "$stage" == quantize ]]; then
  "$quant_env/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import onnxruntime as ort
for name in ('qai-hub-models', 'qai-hub-models-cli', 'torch', 'aimet-onnx', 'onnxruntime-gpu'):
    print(f'{name}: {metadata.version(name)}')
assert torch.cuda.is_available(), 'Quantization requires CUDA; CUDA is unavailable'
assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'ORT CUDA provider is missing'
free, total = torch.cuda.mem_get_info(0)
print(f'GPU: {torch.cuda.get_device_name(0)}; free/total: {free/2**30:.1f}/{total/2**30:.1f} GiB')
if total < 40 * 2**30:
    print('Notice: official 3B quantization recommends 40 GB VRAM; this GPU may run out of memory.')
PY
fi
case "$stage" in
  check) exit 0 ;;
  quantize) exec "$quant_env/bin/python" "$project_root/scripts/quantize_model.py" "$@" ;;
  onnx) exec "$quant_env/bin/python" "$project_root/scripts/export_onnx.py" "$@" ;;
  preprocess) exec "$quant_env/bin/python" "$project_root/scripts/export_preprocessing.py" "$@" ;;
  qnn) exec "$quant_env/bin/python" "$project_root/scripts/export_qnn.py" "$@" ;;
esac
