# Review console on CPU:  docker build -t chipqc . && docker run --rm -p 8501:8501 chipqc
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src ./src
COPY models ./models
COPY examples ./examples
COPY .streamlit ./.streamlit
COPY app.py inference.py ./

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
