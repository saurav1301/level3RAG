# docker/Dockerfile
FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /app/
RUN apt-get update && apt-get install -y build-essential git && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8080
CMD ["python", "app/rag_server.py"]
