# Сервис мониторинга природных пожаров (CPU). Сборка из корня репозитория:
#   docker build -f service/Dockerfile -t firemon-service .
#   docker run --rm -p 8000:8000 firemon-service
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install -r service/requirements.txt

# firemon нужен только для необязательного /api/v1/analyze/* (без torch эндпоинт отвечает 503)
COPY firemon/ firemon/
COPY service/ service/

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
