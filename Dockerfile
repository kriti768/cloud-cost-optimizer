FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install uv (fast package installer)
RUN pip install uv

# Copy dependency files first (Docker layer cache)
COPY pyproject.toml .

# Install dependencies with uv
RUN uv pip install --system fastapi uvicorn[standard] pydantic openai \
    openenv-core httpx websockets gradio 2>/dev/null || \
    pip install fastapi uvicorn[standard] pydantic openai openenv-core httpx websockets gradio

# Copy all source files
COPY . .

# HF Spaces standard port
EXPOSE 7860

# Health check — checklist requires GET / to return 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# Start server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
