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

cd $SRC_DIR
/opt/qt512/bin/qmake ./src/dsi_studio.pro
make -k -j1

echo "DOWNLOAD ATLAS PACKAGES"

-curl -sSLO 'https://www.dropbox.com/s/pib533irglhnwy7/dsi_studio_64.zip'
unzip dsi_studio_64.zip
rm dsi_studio_64.zip
cd dsi_studio_64
rm *.dll
rm *.exe
rm -rf iconengines
rm -rf imageformats
rm -rf platforms
rm -rf styles
cd ..
mv ./dsi_studio dsi_studio_64


