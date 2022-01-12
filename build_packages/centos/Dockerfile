FROM juangburgos/jenkins-centos7-qt5.13.2 as builder-stage

RUN yum -y install centos-release-scl \
  && yum -y install devtoolset-8-gcc devtoolset-8-gcc-c++ \
  && scl enable devtoolset-8 bash

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

RUN /linuxdeployqt /opt/dsi-studio/dsi_studio -unsupported-bundle-everything -no-translations -bundle-non-qt-libs

ENV PATH="$PATH:/opt/dsi-studio" 

#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/dsi-studio /
