FROM roboxes/centos8 as builder-stage

RUN yum -y install wget && \
  mkdir /tmp/downloads && \
  cd /tmp/downloads && \
  wget https://download.qt.io/official_releases/qt/5.13/5.13.2/qt-opensource-linux-x64-5.13.2.run && \
  wget https://download.qt.io/official_releases/qt-installer-framework/3.1.1/QtInstallerFramework-linux-x64.run

# missing dependencies
RUN yum -y update && \
yum -y group install "Development Tools" && \
yum -y install \
git \
kernel-headers kernel-devel \
mesa-libGL-devel \
glibc.i686 \
fuse \
fuse-devel \
ncurses \
ncurses-devel

# install qt and qt-tools
RUN cd /tmp/downloads && \
wget https://raw.githubusercontent.com/juangburgos/JenkinsCentOs7Docker/master/qt5.13.2/qt5.13.2-headless.qs && \
wget https://github.com/juangburgos/JenkinsCentOs7Docker/raw/master/qt5.13.2/qtifw-3.1.1-headless.qs && \
chmod +x qt-opensource-linux-x64-5.13.2.run && \
chmod +x QtInstallerFramework-linux-x64.run && \
./qt-opensource-linux-x64-5.13.2.run -platform minimal --script qt5.13.2-headless.qs --verbose && \
./QtInstallerFramework-linux-x64.run -platform minimal --script qtifw-3.1.1-headless.qs --verbose

RUN \
ln -s /opt/Qt5.13.2/5.13.2/gcc_64/bin/qmake /usr/bin/qmake && \
ln -s /opt/Qt5.13.2/5.13.2/gcc_64/bin/linuxdeployqt /usr/bin/linuxdeployqt && \
ln -s /opt/QtIFW-3.1.1/bin/binarycreator /usr/bin/binarycreator

# install jq to ba able to parse REST API responses from bash
RUN cd /tmp/downloads && \
wget https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64 && \
mv jq-linux64 /usr/bin/jq && \
chmod +x /usr/bin/jq

# cleanup
RUN rm -rf /tmp/downloads


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
