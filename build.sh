apt update
apt full-upgrade -y
apt install --no-install-recommends -y software-properties-common
add-apt-repository -y ppa:beineri/opt-qt-5.12.8-bionic
add-apt-repository -y ppa:ubuntu-toolchain-r/test
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
  g++-9
apt-get clean

rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

source /opt/qt512/bin/qt512-env.sh

mkdir /opt/dsi-studio
cd /opt/dsi-studio
git clone https://github.com/frankyeh/DSI-Studio.git
mv DSI-Studio src
git clone https://github.com/frankyeh/TIPL.git
mv TIPL src/tipl
mkdir -p /opt/dsi-studio/build
cd /opt/dsi-studio/build
qmake ../src/dsi_studio.pro
make -k -j1
cd /opt/dsi-studio
curl -sSLO 'https://www.dropbox.com/s/pib533irglhnwy7/dsi_studio_64.zip'
unzip dsi_studio_64.zip
rm dsi_studio_64.zip
cd dsi_studio_64
rm *.dll
rm *.exe
rm -rf iconengines
rm -rf imageformats
rm -rf platforms
rm -rf styles
mv ../build/dsi_studio .
rm -rf /opt/dsi-studio/src /opt/dsi-studio/build