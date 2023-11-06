brew update
brew install qt5
brew install git
brew install mesa
brew install glew
brew install mas
mas install 497799835

cd ~ 
git clone https://github.com/frankyeh/DSI-Studio.git 
mv DSI-Studio dsi-studio 

cd ~/dsi-studio
git clone https://github.com/frankyeh/TIPL.git
git clone https://github.com/frankyeh/DSI-Studio-atlas.git
git clone https://github.com/frankyeh/UNet-Studio-Data.git
rm -fr DSI-Studio-atlas/.git

mkdir -p ~/dsi-studio/build
cd ~/dsi-studio/build
export PATH="/usr/local/opt/qt@5/bin:$PATH" && export LDFLAGS="-L/usr/local/opt/qt@5/lib" && export CPPFLAGS="-I/usr/local/opt/qt@5/include" && export PKG_CONFIG_PATH="/usr/local/opt/qt@5/lib/pkgconfig"
qmake ../dsi_studio.pro
make

cd ~/dsi-studio
mv build/dsi_studio.app . 
macdeployqt dsi_studio.app 
curl -sSLO "https://github.com/frankyeh/TinyFSL/releases/download/2022.08.03/tiny_fsl_macos-12.zip" 
unzip tiny_fsl_macos-12.zip -d dsi_studio.app/Contents/MacOS/plugin 
rm -fr dsi_studio.app/Contents/MacOS/plugin/tiny_fsl 
mv other/* dsi_studio.app/Contents/MacOS/ 
mv DSI-Studio-atlas dsi_studio.app/Contents/MacOS/atlas 
mv UNet-Studio-Data/network dsi_studio.app/Contents/MacOS/ 
mv dsi_studio.icns dsi_studio.app/Contents/Resources/
