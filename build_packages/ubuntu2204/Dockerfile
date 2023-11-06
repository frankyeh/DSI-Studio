FROM dsistudio/ubuntu2204_qt6_cuda:latest as builder-stage

ENV DEBIAN_FRONTEND noninteractive

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
  && git clone https://github.com/frankyeh/UNet-Studio-Data.git \
  && mv UNet-Studio-Data/network . \
  && rm -fr UNet-Studio-Data \
  && git clone https://github.com/frankyeh/DSI-Studio-atlas.git \
  && rm -fr DSI-Studio-atlas/.git \
  && mv DSI-Studio-atlas atlas

RUN curl -sLk https://github.com/probonopd/linuxdeployqt/releases/download/7/linuxdeployqt-7-x86_64.AppImage > linuxdeployqt \
  && chmod a+x linuxdeployqt \
  && ./linuxdeployqt --appimage-extract

RUN ./squashfs-root/AppRun /opt/dsi-studio/dsi_studio -unsupported-bundle-everything -no-translations -bundle-non-qt-libs \
  && rm -fr squashfs-root \
  && rm -fr linuxdeployqt

ENV PATH="$PATH:/opt/dsi-studio" 

#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/dsi-studio /
