# Reproducible evaluation image. CPU by default; for CUDA use the pytorch/pytorch base and the same pip steps.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEADEYE_CACHE=/cache HF_HOME=/cache/hf
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[hf,dev]" tabulate
VOLUME ["/cache", "/app/results"]
ENTRYPOINT ["deadeye"]
CMD ["--help"]
