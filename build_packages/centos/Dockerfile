ARG QT_VERSION=5.12.9

FROM centos:centos8 as builder-stage

RUN yum -y update && yum clean all

RUN yum install -y epel-release centos-release-scl-rh && yum install -y \
	alsa-lib \
	cmake3 \
	cups-libs \
	devtoolset-10-gcc \
	devtoolset-10-gcc-c++ \
	file \
	git \
	fontconfig \
	libxkbcommon-x11 \
	libXcomposite-devel \
	libXcursor-devel \
	libXi-devel \
	libXrandr-devel \
	libXtst-devel \
	mesa-libGL-devel \
	make \
	ninja-build \
	perl \
	postgresql-devel \
	unixODBC \
	unzip \
	zlib && yum clean all -y && rm -rf /var/cache/yum

RUN ln -s /usr/bin/cmake3 /usr/bin/cmake
ENV BASH_ENV=/opt/rh/devtoolset-10/enable \
	ENV=/opt/rh/devtoolset-10/enable \
	PROMPT_COMMAND=". /opt/rh/devtoolset-10/enable"

ADD https://github.com/probonopd/linuxdeployqt/releases/download/7/linuxdeployqt-7-x86_64.AppImage /opt/linuxdeployqt.AppImage
RUN chmod +x /opt/linuxdeployqt.AppImage
RUN /opt/linuxdeployqt.AppImage --appimage-extract
RUN rm -rf /opt/linuxdeployqt.AppImage
RUN mv /squashfs-root/usr /linuxdeployqt && rm -rf /squashfs-root/
RUN chmod -R 755 /linuxdeployqt

### Install Qt using aqt installer
RUN yum install -y \
	python3 \
	python3-devel \
	python3-pip && yum clean all -y && rm -rf /var/cache/yum
RUN source /opt/rh/devtoolset-10/enable && pip3 install aqtinstall
RUN aqt install -O /qt $QT_VERSION linux desktop -m qtwebengine

ENV QTDIR=/qt/$QT_VERSION/gcc_64
ENV PATH=$QTDIR/bin:/linuxdeployqt/bin:$PATH
ENV QT_PLUGIN_PATH=$QTDIR/plugins


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
