FROM python:3.12-slim

WORKDIR /app

COPY phase1/requirements.in ./phase1/requirements.in
RUN pip install --no-cache-dir -r phase1/requirements.in

COPY phase1/app.py ./phase1/app.py

ENV PORT=8501

EXPOSE 8501

HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=6 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health',timeout=2).status==200 else 1)"

CMD ["sh", "-c", "streamlit run phase1/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true"]
