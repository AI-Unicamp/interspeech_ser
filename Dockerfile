
ARG BASE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM ${BASE}

RUN apt-get update

RUN apt-get install -y nano git

WORKDIR /env

COPY benchmark/requirements.txt /env/requirements.txt

RUN pip install -r requirements.txt

RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /root
