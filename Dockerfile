# Use Ubuntu 16.04 LTS
FROM ubuntu:16.04

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    ca-certificates \
                    xvfb \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config \
                    libfontconfig1 \
                    libfreetype6 \
                    libgl1-mesa-dev \
                    libglu1-mesa-dev \
                    libgomp1 \
                    libice6 \
                    libxcursor1 \
                    libxft2 \
                    libxinerama1 \
                    libxrandr2 \
                    libxrender1 \
                    libxt6 \
                    wget \
                    libboost-all-dev \
                    zlib1g \
                    zlib1g-dev \
                    unzip \
                    libgl1-mesa-dev \
                    libglu1-mesa-dev \
                    freeglut3-dev \
                    mesa-utils \
                    g++ \
                    gcc \
                    make \
                    zlib1g-dev \
                    git \
                    software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Get newer qt5
RUN add-apt-repository ppa:beineri/opt-qt-5.12.2-xenial \
    && apt-get update \
    && apt install -y --no-install-recommends \
    freetds-common libclang1-5.0 libllvm5.0 libodbc1 libsdl2-2.0-0 libsndio6.1 \
    libsybdb5 libxcb-xinerama0 qt5123d qt512base qt512canvas3d \
    qt512connectivity qt512declarative qt512graphicaleffects \
    qt512imageformats qt512location qt512multimedia qt512scxml qt512svg \
    qt512wayland qt512x11extras qt512xmlpatterns qt512charts-no-lgpl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache notebook
ENV NB_USER dsistudiouser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}
RUN adduser --disabled-password \
            --gecos "Default user" \
            --uid ${NB_UID} \
            ${NB_USER}
WORKDIR ${HOME}


# Install DSI Studio
ENV QT_BASE_DIR="/opt/qt512"
ENV QTDIR="$QT_BASE_DIR" \
    PATH="$QT_BASE_DIR/bin:$PATH:/opt/dsi-studio/dsi_studio_64" \
    LD_LIBRARY_PATH="$QT_BASE_DIR/lib/x86_64-linux-gnu:$QT_BASE_DIR/lib:$LD_LIBRARY_PATH" \
    PKG_CONFIG_PATH="$QT_BASE_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/tipl \
  && mkdir build && cd build \
  && /opt/qt512/bin/qmake ../src && make \
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
