FROM dsistudio/macos12_qt5:latest as builder-stage

RUN cd /opt \ 
    && git clone https://github.com/frankyeh/TIPL.git \
    && git clone https://github.com/frankyeh/DSI-Studio-atlas.git \
    && git clone https://github.com/frankyeh/UNet-Studio-Data.git \
    && rm -fr DSI-Studio-atlas/.git

RUN mkdir -p /opt/src/build \
    && cd /opt/src/build \
    && qmake ../dsi_studio.pro \
    && make

RUN cd /opt/src \
    && mv build/dsi_studio.app . \
    && macdeployqt dsi_studio.app \
    && curl -sSLO "https://github.com/frankyeh/TinyFSL/releases/download/2022.08.03/tiny_fsl_macos-12.zip" \
    && 7z e tiny_fsl_macos-12.zip -odsi_studio.app/Contents/MacOS/plugin \
    && rm -fr dsi_studio.app/Contents/MacOS/plugin/tiny_fsl \
    && mv other/* dsi_studio.app/Contents/MacOS/ \
    && mv DSI-Studio-atlas dsi_studio.app/Contents/MacOS/atlas \
    && mv UNet-Studio-Data/network dsi_studio.app/Contents/MacOS/ \
    && mv dsi_studio.icns dsi_studio.app/Contents/Resources/


#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/dsi_studio.app /