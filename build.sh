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


echo "Update alternatives"

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 99 \
                    --slave   /usr/bin/g++ g++ /usr/bin/g++-9 \

update-alternatives --auto gcc

echo "CHECK GCC G++ versions"
gcc --version
g++ --version


filepath=$(pwd)

echo "COMPILE DSI STUDIO"

cd $SRC_DIR/src
/opt/qt512/bin/qmake dsi_studio.pro
make -k -j1


