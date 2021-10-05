FROM ubuntu:20.04

# Prepare environment
RUN apt update && apt full-upgrade -y && \
  apt install -y --no-install-recommends \
  unzip \
  curl \
  make \
  git \
  libboost-all-dev \
  zlib1g-dev \
  ca-certificates \
  qt5-qmake \
  qt5-default \
  libqt5charts5-dev \
  libqt5opengl5-dev \
  gcc \
  g++ && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD "https://api.github.com/repos/frankyeh/DSI-Studio/commits?per_page=1" latest_commit
ADD "https://api.github.com/repos/frankyeh/TIPL/commits?per_page=1" latest_commit

RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/tipl \
  && mkdir -p /opt/dsi-studio/build \
  && cd /opt/dsi-studio/build \
  && qmake ../src/dsi_studio.pro \
  && make

RUN cd /opt/dsi-studio \
  && mv build/dsi_studio . \
  && chmod 755 dsi_studio \
  && cp -R src/other/* . \
  && rm -rf src build \
  && git clone https://github.com/frankyeh/DSI-Studio-atlas.git \
  && rm -fr DSI-Studio-atlas/.git \
  && mv DSI-Studio-atlas atlas
ENV PATH="$PATH:/opt/dsi-studio" 


