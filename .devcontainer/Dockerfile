FROM rust:1.69.0

RUN rustup component add rustfmt &&\
    rustup component add clippy

# for eval
RUN apt update &&\
    apt install -y \
    python3-pip \
    python3 \
    liblapack3 \
    liblapack-dev \
    libblas3 \
    libblas-dev \
    gfortran \
    ffmpeg

COPY TrackEval/requirements.txt /tmp

RUN pip3 install -r tmp/requirements.txt