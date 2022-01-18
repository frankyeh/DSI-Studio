FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment 
RUN apt update && apt full-upgrade -y && \
  apt install -y --no-install-recommends \
  libopenblas-dev \
  file \
  libquadmath0 \
  tclsh \
  wish \ 
  unzip \
  curl \
  make \
  git \
  zlib1g-dev \
  libboost-all-dev \
  ca-certificates \
  qt5-qmake \
  qt5-default \
  libqt5charts5-dev \
  libqt5opengl5-dev \
  gcc \
  g++ && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
  && mv TIPL src/tipl \
  && mkdir -p /opt/dsi-studio/build \
  && cd /opt/dsi-studio/build \
  && qmake ../src/dsi_studio.pro \
  && make

RUN cd /opt/dsi-studio \
  && mv build/dsi_studio . \
  && chmod 755 dsi_studio \
  && cp -R src/other/* . \
  && rm -rf src build \
  && git clone https://github.com/frankyeh/DSI-Studio-atlas.git \
  && rm -fr DSI-Studio-atlas/.git \
  && mv DSI-Studio-atlas atlas


# FSL
RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.4-centos7_64.tar.gz | tar zxv --no-same-owner -C /opt \
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


# TINYFSL

RUN cd /opt \
  && git clone https://github.com/frankyeh/TIPL.git \
  && git clone https://github.com/frankyeh/TinyFSL.git \
  && mv TinyFSL tinyfsl \
  && mv TIPL tipl \
  && mkdir -p /opt/tinyfsl/include/ \
  && mv tipl /opt/tinyfsl/include/

RUN cd /opt/tinyfsl \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-armawrap-0.6.0-h2bc3f7f_0.tar.bz2 --output fsl-armawrap-0.6.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-avwutils-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-avwutils-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-bet2-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-bet2-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-basisfield-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-basisfield-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-cprob-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-cprob-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-eddy-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-eddy-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-meshclass-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-meshclass-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-miscmaths-2111.1-h2bc3f7f_0.tar.bz2 --output fsl-miscmaths-2111.1-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newnifti-4.0.0-h2bc3f7f_0.tar.bz2 --output fsl-newnifti-4.0.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-topup-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-topup-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-utils-2111.1-h2bc3f7f_0.tar.bz2 --output fsl-utils-2111.1-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-warpfns-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-warpfns-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-znzlib-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-znzlib-2111.0-h2bc3f7f_0.tar.bz2 \
  && bunzip2 *.bz2 \
  && for i in *.tar;do tar -xf "$i";done \
  && cd src \
  && for i in "fsl-"*;do mv -v -f "$i" "${i#"fsl-"}";done \
  && for i in *;do cp -v -r -f "$i" ../include/;done \
  && cd .. \
  && cd patches \
  && for i in *;do cp -v -r -f "$i" ../include/;done \
  && cd .. 


RUN mkdir -p /opt/tinyfsl/build \
  && cd /opt/tinyfsl/build \
  && qmake ../eddy.pro \
  && make \
  && qmake ../topup.pro \
  && make \
  && qmake ../applytopup.pro \
  && make \
  && qmake ../bet2.pro \
  && make \
  && chmod 755 eddy \
  && chmod 755 topup \
  && chmod 755 applytopup \
  && chmod 755 bet2 \
  && mkdir -p /opt/tinyfsl_bin \
  && mv eddy /opt/tinyfsl_bin \
  && mv topup /opt/tinyfsl_bin \
  && mv applytopup /opt/tinyfsl_bin \
  && mv bet2 /opt/tinyfsl_bin


RUN cd /opt \
  && rm -fr tinyfsl \
  && mkdir tinyfsl \
  && mv tinyfsl_bin tinyfsl \
  && cd /opt/tinyfsl \
  && mv tinyfsl_bin bin \
  && cp -fr /opt/tinyfsl/bin/topup /opt/fsl/bin


