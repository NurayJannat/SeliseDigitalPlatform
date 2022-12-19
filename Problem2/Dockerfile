FROM continuumio/miniconda3
RUN conda create -n env python=3.8
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip python3-opencv

RUN pip install --upgrade pip
# COPY requirements.txt .

RUN pip install protobuf==3.20.*
RUN pip install tensorflow==2.8
RUN pip install opencv-python

COPY training.py .
COPY inference.py .

