FROM ubuntu:20.04 as builder-stage

ENV DEBIAN_FRONTEND noninteractive

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
RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/tipl \
  && mkdir -p /opt/dsi-studio/build \
  && cd /opt/dsi-studio/build \
  && qmake ../src/dsi_studio.pro \
  && make -k -j$(nproc) \
  && cd /opt/dsi-studio \
  && curl -sSLO 'https://www.dropbox.com/s/pib533irglhnwy7/dsi_studio_64.zip' \
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

#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/dsi-studio/dsi_studio_64 /
