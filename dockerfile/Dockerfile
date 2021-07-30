FROM ixws/tf-gpu:pip3.6-tf1.15-ub18.04
MAINTAINER ixime

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE=C.UTF-8
ENV LC_CTYPE=C.UTF-8 
ENV LC_MESSAGES=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

ARG NEWUID=1010
ARG NEWGID=1090

RUN set -ex && \
    apt-get update -yq && \
    apt-get upgrade -yq && \
    apt-get install -y software-properties-common \
        apt-utils \
        curl \
        tmux \
        sudo \
        git \
        htop \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        jq \
        nano && \
        curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - && \
        apt-get install -y nodejs && \
        pip3 --no-cache-dir install --upgrade pip setuptools wheel

RUN groupadd -g ${NEWGID} common_user && \
    useradd -ms /bin/bash -d /home/common_user -g common_user -u ${NEWUID} -o -c "" -m common_user && \
    echo "common_user ALL=(ALL:ALL) NOPASSWD:ALL" | (EDITOR="tee -a" visudo) && \
    usermod -aG sudo common_user

RUN pip --no-cache-dir install numpy pandas && \
    pip --no-cache-dir install jupyter jupyterlab && \
    pip --no-cache-dir install matplotlib scipy scikit-learn scikit-image && \
    pip --no-cache-dir install opencv-python && \
    pip --no-cache-dir install boto3 && \
    pip --no-cache-dir install python-dotenv && \
    pip --no-cache-dir install shapely

RUN apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

RUN mkdir -p /home/common_user/workspace && \
    chown -R common_user: /home/common_user/workspace

EXPOSE 8888 6006

USER common_user
WORKDIR /home/common_user
