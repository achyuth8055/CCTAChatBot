FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models.py .
COPY fill_db.py .
COPY templates/ templates/

RUN mkdir -p data chroma_db uploads

EXPOSE 5001

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV ANONYMIZED_TELEMETRY=False

CMD ["python", "app.py"]
