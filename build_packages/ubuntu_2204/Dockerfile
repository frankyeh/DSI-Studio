FROM ubuntu:22.04 as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
RUN apt update && apt upgrade -y && \
  apt install -y \
  unzip \
  curl \
  cmake \
  ninja-build \
  git \
  zlib1g-dev \
  libglu1-mesa-dev \
  qt6-base-dev \
  libqt6charts6-dev \
  libqt6opengl6-dev \
  qt6-image-formats-plugins \
#  nvidia-cuda-toolkit \
#  nvidia-cuda-dev \
#  nvidia-cuda-toolkit-gcc \
  gcc \
  g++ && \
  apt-get -y autoremove --purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD "https://api.github.com/repos/frankyeh/DSI-Studio/commits?per_page=1" latest_commit
ADD "https://api.github.com/repos/frankyeh/TIPL/commits?per_page=1" latest_commit

RUN cd /opt \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && cd /opt/src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mkdir -p /opt/src/build

RUN cd /opt/src \
  && cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE:STRING=Release -DTIPL_DIR=. \
  && cmake --build ./build --parallel --config Release

RUN mkdir -p /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && mv /opt/src/build/dsi_studio . \
  && chmod 755 dsi_studio \
  && cp -R /opt/src/other/* . \
  && git clone https://github.com/frankyeh/DSI-Studio-atlas.git \
  && rm -fr DSI-Studio-atlas/.git \
  && mv DSI-Studio-atlas atlas

ENV PATH="$PATH:/opt/dsi-studio" 

#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/dsi-studio /
