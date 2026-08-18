# Checkpoint Phase 4 D.2B: real production chatbot-ui image (phase6, not
# the Phase 1 skeleton this Dockerfile used before). Self-contained build
# context (this submodule only) -- the frontend never imports root
# packages or another submodule's code, only talks to agent-system-a over
# HTTP (phase6/api_client.py).
#
# `phase6` is installed as a real package (pyproject.toml, `pip install
# --no-deps .` below) -- this is the declared mechanism `phase6/app.py`'s
# own `from phase6 import ...` imports rely on to resolve correctly no
# matter what the process's current working directory or the invoked
# script's own directory is. Never a PYTHONPATH environment variable and
# never a sys.path mutation anywhere in this image or in application
# source (a prior draft of this Dockerfile/app.py briefly did the latter
# -- a source-level sys.path insertion -- to work around
# `streamlit run phase6/app.py` only adding its own script directory to
# sys.path; that workaround was removed in favor of this real install,
# per this project's fixed "no production sys.path/PYTHONPATH hacks"
# rule, matching the same real-declared-dependency-path precedent root
# `pyproject.toml` already established for orchestration/system_a
# (Checkpoint D.2A/D.2B).
FROM python:3.12-slim

WORKDIR /app

RUN groupadd --system voyager && useradd --system --gid voyager --home /app voyager

COPY phase6/requirements.in ./phase6/requirements.in
RUN pip install --no-cache-dir -r phase6/requirements.in

COPY pyproject.toml ./pyproject.toml
COPY phase6 ./phase6
RUN pip install --no-cache-dir --no-deps .

ENV PORT=8501
# Explicit placeholder -- must be set by the environment (root
# docker-compose.yml sets it to the internal Compose service name,
# never localhost) or the app fails with a clear configuration error
# rather than guessing an internal hostname.
ENV AGENT_SYSTEM_A_BASE_URL=""

RUN chown -R voyager:voyager /app
USER voyager

EXPOSE 8501

HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=6 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health',timeout=2).status==200 else 1)"

CMD ["sh", "-c", "streamlit run phase6/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true"]
