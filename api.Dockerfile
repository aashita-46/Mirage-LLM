FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn
COPY api api
COPY data/datasets data/datasets
COPY config config
COPY scripts scripts
COPY backend.py .
EXPOSE 8000
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
