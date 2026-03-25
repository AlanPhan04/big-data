FROM spark:4.0.1-scala2.13-java21-ubuntu

USER root


RUN apt update && \
    apt install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt clean

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER spark