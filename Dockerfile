ARG PYTHON_VERSION
FROM python:3.9-slim-buster

MAINTAINER Union AI Team 
LABEL org.opencontainers.image.source https://github.com/unionai-oss/stanford-alpaca

WORKDIR /root
ENV PYTHONPATH /root

ARG VERSION
ARG DOCKER_IMAGE

RUN apt-get update && apt-get install build-essential -y

COPY . /root

WORKDIR /root
# Pod tasks should be exposed in the default image
RUN pip install -r requirements.txt

ENV FLYTE_INTERNAL_IMAGE "$DOCKER_IMAGE"
