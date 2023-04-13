FROM nvidia/cuda:12.0.0-runtime-ubuntu18.04
FROM continuumio/miniconda3:4.10.3

COPY environment.yml .

RUN apt-get update -q \
  && apt-get install --no-install-recommends -qy g++ gcc

RUN conda env update -n root -f environment.yml && \
    conda clean -a