
echo "Updating brew and installing dependencies..."
brew update
brew install qt6
brew install cmake
brew install git
brew install mesa
brew install glew
brew install mas
mas install 497799835


echo "Cloning necessary repositories..."
cd ~ 
git clone https://github.com/frankyeh/DSI-Studio.git 
mv DSI-Studio dsi-studio 

cd ~/dsi-studio
git clone https://github.com/frankyeh/TIPL.git
git clone https://github.com/frankyeh/DSI-Studio-atlas.git
git clone https://github.com/frankyeh/UNet-Studio-Data.git
rm -fr DSI-Studio-atlas/.git

echo "Building the application..."
mkdir -p ~/dsi-studio/build
cd ~/dsi-studio
cmake -S . -B build -DCMAKE_BUILD_TYPE:STRING=Release -DTIPL_DIR=.
cmake --build ./build --parallel --config Release
make

echo "Constructing the application package..."
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


echo "Copying the application to the Applications folder..."
cp -R ~/dsi-studio/dsi_studio.app /Applications/

echo "Cleaning up the dsi-studio directory..."
rm -fr ~/dsi-studio

echo "Installation complete. dsi_studio.app has been copied to the Applications folder."
