# 最小可运行镜像：推理链路（无 LLM 判官、无训练）
#
#   docker build -t poisonguard:latest .
#   docker run --rm -p 8000:8000 \
#       -v $PWD/outputs:/app/outputs -v $PWD/data:/app/data \
#       poisonguard:latest serve --port 8000
#
# 说明：编码器权重（CoROM ~400MB）不烤进镜像——挂载 model_cache 或用
# `download-encoder` 拉。判官链路要 torch+transformers，用 --target judge 另建。
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POISONGUARD_LOG_LEVEL=INFO

WORKDIR /app

# 先装依赖，利用层缓存：代码改动不会重装 sklearn/numpy
COPY pyproject.toml README.md ./
COPY poisonguard ./poisonguard
RUN pip install --no-cache-dir -e ".[serve]" \
    && python -c "import poisonguard, sklearn, fastapi; print('deps ok')"

COPY configs ./configs
COPY docs ./docs
COPY scripts ./scripts

# 非 root 跑（容器逃逸面小一点；写入目录给 outputs）
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/outputs /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["python", "-m", "poisonguard.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]


# 判官/训练镜像：需要 torch（CPU 版）
FROM base AS judge
USER root
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers>=4.40,<5" safetensors
USER appuser
