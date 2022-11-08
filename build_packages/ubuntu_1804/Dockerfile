FROM ubuntu:18.04 as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
# We use a ppa for the qt because 18.04 is tool old for DSI
RUN apt update && apt full-upgrade -y && \
  apt install --no-install-recommends -y software-properties-common && \
  add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
  apt install -y --no-install-recommends \
  unzip \
  curl \
  make \
  git \
  zlib1g-dev \
  ca-certificates \
  qt5-qmake \
  qt5-default \
  libqt5charts5-dev \
  libqt5opengl5-dev \
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

ADD "https://api.github.com/repos/frankyeh/DSI-Studio/commits?per_page=1" latest_commit
ADD "https://api.github.com/repos/frankyeh/TIPL/commits?per_page=1" latest_commit

ENV PATH="$PATH:/opt/qt512/bin" 

RUN source /opt/qt512/bin/qt512-env.sh \
  && mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/TIPL \
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


RUN curl -sL https://github.com/probonopd/linuxdeployqt/releases/download/7/linuxdeployqt-7-x86_64.AppImage > linuxdeployqt \
  && chmod a+x linuxdeployqt \
  && ./linuxdeployqt --appimage-extract \
  && ./squashfs-root/AppRun /opt/dsi-studio/dsi_studio -unsupported-bundle-everything -no-translations -bundle-non-qt-libs \
  && rm -fr squashfs-root \
  && rm -fr linuxdeployqt

ENV PATH="$PATH:/opt/dsi-studio" 


#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/dsi-studio /
