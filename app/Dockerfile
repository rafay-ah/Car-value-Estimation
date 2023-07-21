
FROM python:3.7.7-slim-buster
ENV PYTHONUNBUFFERED 1
EXPOSE 8000

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -U pip

RUN pip3 install -r requirements.txt

COPY .. .

CMD gunicorn --bind 0.0.0.0:8000 wsgi
