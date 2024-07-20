FROM dsistudio/ubuntu2004_qt5:latest

ENV DEBIAN_FRONTEND noninteractive

ADD "https://api.github.com/repos/frankyeh/DSI-Studio/commits?per_page=1" latest_commit
ADD "https://api.github.com/repos/frankyeh/TIPL/commits?per_page=1" latest_commit

# Fix issues: Singularity container cannot load libQt5Core.so.5 on CentOS 7
RUN strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5
RUN ldconfig
ENV PATH="$PATH:/opt/dsi-studio" 

RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone https://github.com/frankyeh/DSI-Studio.git \
  && mv DSI-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/TIPL \
  && mkdir -p /opt/dsi-studio/build \
  && cd /opt/dsi-studio/build \
  && qmake ../src/dsi_studio.pro \
  && make

RUN cd /opt/dsi-studio \
  && mv build/dsi_studio . \
  && chmod 755 dsi_studio \
  && cp -R src/other/* . \
  && rm -rf src build \
  && git clone https://github.com/frankyeh/UNet-Studio-Data.git \
  && mv UNet-Studio-Data/network . \
  && rm -fr UNet-Studio-Data \
  && git clone https://github.com/frankyeh/DSI-Studio-atlas.git \
  && rm -fr DSI-Studio-atlas/.git \
  && mv DSI-Studio-atlas atlas


# FSL
RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.5.2-centos7_64.tar.gz | tar zxv --no-same-owner -C /opt \
    --exclude='fsl/doc' \
    --exclude='fsl/refdoc' \
    --exclude='fsl/python/oxford_asl' \
    --exclude='fsl/data/possum' \
    --exclude='fsl/data/first' \
    --exclude='fsl/data/mist' \
    --exclude='fsl/data/atlases' \
    --exclude='fsl/data/xtract_data' \
    --exclude='fsl/extras/doc' \
    --exclude='fsl/extras/man' \
    --exclude='fsl/extras/src' \
    --exclude='fsl/src'

ENV OS="Linux" \
    FSLDIR="/opt/fsl" \
    FSL_DIR="$FSLDIR" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    LD_LIBRARY_PATH="$FSLDIR/lib:$LD_LIBRARY_PATH" \
    FSLTCLSH="/usr/bin/tclsh" \
    FSLWISH="/usr/bin/wish" \
    PATH="$FSLDIR/bin:$PATH"


