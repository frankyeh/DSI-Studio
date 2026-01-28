FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
ARG TAG_ANTS
RUN apt-get update && apt-get full-upgrade -y && \
  apt-get install --no-install-recommends -y \
  software-properties-common \
  unzip \
  curl \
  wget \
  make \
  git \
  libboost-all-dev \
  zlib1g-dev \
  ca-certificates \
  qt6-base-dev \
  qt6-base-dev-tools \
  qt6-base-private-dev \
  qt6-tools-dev \
  qt6-tools-dev-tools \
  qt6-l10n-tools \
  libqt6charts6-dev \
  libqt6opengl6-dev \
  libzip-dev \
  mesa-common-dev \
  libglu1-mesa-dev \
  build-essential \
  gnupg \
  bc \
  ninja-build \
  apt-transport-https && \
  wget -O /tmp/kitware-archive.key https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null && \
  gpg --dearmor /tmp/kitware-archive.key && \
  mv /tmp/kitware-archive.key.gpg /usr/share/keyrings/kitware-archive-keyring.gpg && \
  echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/kitware.list && \
  apt-get update && \
  apt-get -y install cmake cmake-data && \
  update-alternatives --install /usr/bin/qmake qmake /usr/lib/qt6/bin/qmake6 100 && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /opt/dsi-studio/src
RUN cd /opt/dsi-studio \
  && sed -i '/color_bar_dialog/d;/filebrowser/d;/mac_filesystem/d' src/dsi_studio.pro \
  && sed -i 's/DSI_STUDIO_LOGIN/""/g' src/mainwindow.cpp \
  && mkdir build && cd build \
  && qmake ../src/dsi_studio.pro && make -j 1

ARG ATLAS_SHA=59a1ca250845e569813ca6a2c5702d74f4d4cb4a
ARG UNET_SHA=3ec6aa9513c0c8bbb90399a7af16245b594cd09b

RUN cd /opt/dsi-studio \
  && mv build/dsi_studio . \
  && chmod 755 dsi_studio \
  && cp -R src/other/* . \
  && rm -rf src build \
  && curl -sSLO https://github.com/frankyeh/UNet-Studio-Data/archive/${UNET_SHA}.zip \
  && unzip ${UNET_SHA}.zip \
  && rm ${UNET_SHA}.zip \
  && mv UNet-Studio-Data-${UNET_SHA}/network/ . \
  && rm -rf UNet-Studio-Data-${UNET_SHA} \
  && curl -sSLO https://github.com/data-others/atlas/archive/${ATLAS_SHA}.zip \
  && unzip ${ATLAS_SHA}.zip \
  && rm -rf DSI-Studio-atlas-${ATLAS_SHA}/.git \
  && mv atlas-${ATLAS_SHA} atlas \
  && rm ${ATLAS_SHA}.zip
