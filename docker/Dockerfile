FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    zip \
    libopencv-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev curl \
    wget libffi-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG UID
RUN useradd docker -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker

ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

ENV PYTHON_VERSION 3.7.6
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

RUN pip install -U pip setuptools

COPY requirements/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# mount fedavg.pytorch to /work
WORKDIR /work

ENTRYPOINT ["/bin/bash"]