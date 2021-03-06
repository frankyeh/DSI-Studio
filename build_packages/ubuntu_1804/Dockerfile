FROM ubuntu:18.04 as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
# We use a ppa for the qt because 18.04 is tool old for DSI
RUN apt update && apt full-upgrade -y && \
  apt install --no-install-recommends -y software-properties-common && \
  add-apt-repository -y ppa:beineri/opt-qt-5.12.8-bionic && \
  add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
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
  gcc-9 \
  g++-9 && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 \
                    99 \
                    --slave   /usr/bin/cc cc /usr/bin/gcc-9 \
                    --slave   /usr/bin/c++ c++ /usr/bin/g++-9 \
                    --slave   /usr/bin/g++ g++ /usr/bin/g++-9 \
                    --slave   /usr/bin/gcov gcov /usr/bin/gcov-9 \
                    --slave   /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-9 \
                    --slave   /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-9 \
                    --slave   /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 \
                    --slave   /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-9 \
                    --slave   /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9

RUN update-alternatives --auto gcc

#Need to use a different shell so the QT ENV script works
SHELL ["/bin/bash", "-c"]

RUN source /opt/qt512/bin/qt512-env.sh \
  && mkdir /opt/dsi-studio \
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
