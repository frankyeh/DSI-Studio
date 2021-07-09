FROM ubuntu:18.04 as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
# We use a ppa for the qt because 18.04 is tool old for DSI
RUN apt update && apt full-upgrade -y && \
  apt install --no-install-recommends -y software-properties-common && \
  add-apt-repository -y ppa:beineri/opt-qt-5.12.8-bionic && \
  apt install -y --no-install-recommends \
  unzip \
  curl \
  make \
  git \
  libboost-all-dev \
  zlib1g-dev \
  ca-certificates \
  qt512base \
  qt512charts-no-lgpl \
  mesa-common-dev \
  libglu1-mesa-dev \
  gcc \
  g++ && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#Need to use a different shell so the QT ENV script works
SHELL ["/bin/bash", "-c"]

#We clone here the specific 2021_06 version, and the closest TIPL commit
RUN source /opt/qt512/bin/qt512-env.sh \
  && mkdir /opt/dsi-studio \
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
