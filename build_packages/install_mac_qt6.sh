brew update
brew install qt6
brew install cmake
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
cd ~/dsi-studio
cmake -S . -B build -DCMAKE_BUILD_TYPE:STRING=Release -DTIPL_DIR=.
cmake --build ./build --parallel --config Release
make

cd ~/dsi-studio
mv build/dsi_studio.app . 
macdeployqt dsi_studio.app 
curl -sSLO "https://github.com/frankyeh/TinyFSL/releases/download/2022.08.03/tiny_fsl_macos-13.zip" 
unzip tiny_fsl_macos-13.zip -d dsi_studio.app/Contents/MacOS/plugin 
rm -fr dsi_studio.app/Contents/MacOS/plugin/tiny_fsl 
mv other/* dsi_studio.app/Contents/MacOS/ 
mv DSI-Studio-atlas dsi_studio.app/Contents/MacOS/atlas 
mv UNet-Studio-Data/network dsi_studio.app/Contents/MacOS/ 
mv dsi_studio.icns dsi_studio.app/Contents/Resources/
