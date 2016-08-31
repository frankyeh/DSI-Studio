#!/bin/bash
cd
rm -rf DSI-Studio
rm -rf TIPL
git clone http://github.com/frankyeh/DSI-Studio.git
git clone http://github.com/frankyeh/TIPL.git
cd DSI-Studio
rm Makefile*
rm *.Debug
rm *.Release
cd
mkdir DSI-Studio-master
cd DSI-Studio-master
mkdir image
cd
cp -R DSI-Studio/* DSI-Studio-master/
cp -R TIPL/* DSI-Studio-master/image/
rm -rf DSI-Studio
rm -rf TIPL
cd DSI-Studio-master
qmake -project
qmake -config release
make clean
make
macdeployqt dsi_studio.app -dmg
mv dsi_studio.dmg dsi_studio_64.dmg
cp dsi_studio_64.dmg /Users/frank/Dropbox/DSI\ Studio/


