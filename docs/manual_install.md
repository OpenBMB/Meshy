# Manual Installation

This guide sets up Meshy without Docker. Use it if you cannot use the [prebuilt image or the Dockerfile](../README.md#quick-start).

## Prerequisites

- NVIDIA GPU with CUDA 12.9 support
- Ubuntu 24.04 (or a compatible Debian-based distribution)
- Python 3.12+

## Steps

1. **Install system dependencies** (Ubuntu/Debian):

   ```bash
   apt-get update && apt-get install -y --no-install-recommends \
       git vim python3 python3-dev python3-venv
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv /opt/meshy
   source /opt/meshy/bin/activate
   ```

3. **Upgrade pip and install `uv`** (a fast package installer):

   ```bash
   pip install --upgrade pip
   pip install uv
   ```

4. **Install SGLang** (pre-release build):

   ```bash
   uv pip install sglang==0.15.8
   ```

5. **Install PyTorch with CUDA 12.9**:

   ```bash
   uv pip install torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
       --index-url https://download.pytorch.org/whl/cu129 --force-reinstall
   ```

6. **Install SGLang kernels and DeepGEMM**:

   ```bash
   uv pip install sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/ --force-reinstall
   uv pip install sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --force-reinstall --no-deps
   ```

7. **Install TorchTitan** from a pre-built wheel:

   ```bash
   uv pip install third_party/torchtitan-0.1.0.dev20260501+cu126-py3-none-any.whl
   ```

8. **Install TransferQueue**:

   ```bash
   uv pip install TransferQueue==0.1.9 --no-deps
   ```

9. **Install Meshy with its dependencies**:

   ```bash
   uv pip install -e .
   ```

10. **Verify the installation**:

   ```bash
   python -c "import sglang, torch, torchaudio, torchvision, torchtitan; print(torch.__version__, torch.version.cuda)"
   ```

   You should see the PyTorch version and CUDA version printed without errors.

## Next step

Return to the README to [run a recipe](../README.md#run-a-recipe).
