# Use Ubuntu 20.04 LTS
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    unzip \ 
                    curl \
                    g++-7 \
                    gcc-7 \
                    make \
                    git \
	            libboost-all-dev \
	            zlib1g \
                    zlib1g-dev \
                    software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10
Run update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10

# Get newer qt5
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    qt5-qmake \
    qt5-default \
    libqt5charts5-dev \
    libqt5opengl5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/tipl \
  && mkdir build && cd build \
  && qmake ../src && make \
  && cd /opt/dsi-studio \
  && curl -sSLO 'https://www.dropbox.com/s/xha3srev45at7vx/dsi_studio_64.zip' \
  && unzip dsi_studio_64.zip \
  && rm dsi_studio_64.zip \
  && cd dsi_studio_64 \
  && rm *.dll \
  && rm *.exe \
  && rm -rf iconengines \
  && rm -rf imageformats \
  && rm -rf platforms \
  && rm -rf styles \
  && mv ../build/dsi_studio . \
  && rm -rf /opt/dsi-studio/src /opt/dsi-studio/build
