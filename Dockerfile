FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL org.opencontainers.image.source https://github.com/unionai-oss/stanford-alpaca

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

ARG VERSION
ARG DOCKER_IMAGE

RUN apt-get update && apt-get install build-essential -y

COPY . /root

WORKDIR /root
# Pod tasks should be exposed in the default image
RUN pip install -r requirements.txt

ENV FLYTE_INTERNAL_IMAGE "$DOCKER_IMAGE"
