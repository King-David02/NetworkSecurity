FROM python:3.12-slim
WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y awscli && \
    pip install -r requirements.txt

CMD ["python3", "app.py"]