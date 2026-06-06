# Eval harness: builds the Streamlit viewer + the runner in one image.
FROM python:3.11-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, prompts, the stable corpus, the gold set, and persisted results.
COPY harness/ ./harness/
COPY prompts/ ./prompts/
COPY data/ ./data/
COPY results/ ./results/

# Streamlit serves here. Concurrency is a RUNTIME env var (CONCURRENCY), not baked
# in — override with `-e CONCURRENCY=16` without rebuilding.
ENV CONCURRENCY=8
EXPOSE 8501

# SI_API_KEY must be provided at run time:
#   docker run -e SI_API_KEY=doo_v1_... -p 8501:8501 doctl-eval
CMD ["streamlit", "run", "harness/ui.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
