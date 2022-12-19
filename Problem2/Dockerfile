FROM ubuntu:latest
WORKDIR /app
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY training.py ./app
COPY inference.py ./app
