FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY frontend/ ./frontend/

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data/documents /app/data/graph

VOLUME ["/app/data"]

ENV CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR=/app/data/documents \
    CLINICAL_GRAPH_GRAPH_STORAGE_DIR=/app/data/graph

EXPOSE 8000

CMD ["uvicorn", "clinical_graph.main:app", "--host", "0.0.0.0", "--port", "8000"]
