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

#We clone here the specific 2021_06 version, and the closest TIPL commit
RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone --branch 2021_06 https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && (cd TIPL && git checkout 0635e9dc932437189c590a7c6b5e2eee16b1debf) \
  && mv TIPL src/tipl \
  && sed -i 's/std::vector<typename ImageType::value_type> pixels;/std::vector<float> pixels;/g' /opt/dsi-studio/src/tipl/vis/march_cube.hpp \
  && mkdir -p /opt/dsi-studio/build \
  && cd /opt/dsi-studio/build \
  && qmake ../src/dsi_studio.pro \
  && make -k -j$(nproc) \
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

#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/dsi-studio/dsi_studio_64 /
