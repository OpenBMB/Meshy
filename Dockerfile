FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
ARG SGLANG_INDEX_URL=https://docs.sglang.ai/whl/cu129/

ENV VIRTUAL_ENV=/opt/meshy \
    PATH=/opt/meshy/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        python3 \
        python3-dev \
        python3-venv && \
    python3 -m venv "${VIRTUAL_ENV}" && \
    python3 -m pip install --upgrade pip --index-url "${PYPI_INDEX_URL}" && \
    python3 -m pip install uv --index-url "${PYPI_INDEX_URL}" && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        --prerelease=allow sglang==0.5.18 \
        --index-url "${PYPI_INDEX_URL}" && \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
        --index-url "${PYTORCH_INDEX_URL}" \
        --force-reinstall && \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        sglang-kernel \
        --index-url "${SGLANG_INDEX_URL}" \
        --force-reinstall && \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        sgl-deep-gemm \
        --index-url "${SGLANG_INDEX_URL}" \
        --force-reinstall \
        --no-deps

COPY third_party/torchtitan-0.1.0.dev20260501+cu126-py3-none-any.whl /tmp/torchtitan-0.1.0.dev20260501+cu126-py3-none-any.whl

RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        pylatexenc \
        --index-url "${PYPI_INDEX_URL}" && \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        /tmp/torchtitan-0.1.0.dev20260501+cu126-py3-none-any.whl \
        --index-url "${PYPI_INDEX_URL}" && \
    rm /tmp/torchtitan-0.1.0.dev20260501+cu126-py3-none-any.whl && \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        TransferQueue==0.1.9 \
        --index-url "${PYPI_INDEX_URL}" \
        --no-deps && \
    python -c "import sglang, torch, torchaudio, torchvision, torchtitan; print(torch.__version__, torch.version.cuda)"
