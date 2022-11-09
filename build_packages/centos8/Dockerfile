FROM roboxes/centos8 as builder-stage

RUN yum -y install qt*

RUN yum -y install centos-release-scl 
RUN yum -y install devtoolset-9-gcc*

RUN yum -y install zlib-devel
RUN yum -y install freeglut-devel


SHELL [ "/usr/bin/scl", "enable", "devtoolset-9"]
RUN gcc --version

RUN mkdir /opt/dsi-studio \
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
