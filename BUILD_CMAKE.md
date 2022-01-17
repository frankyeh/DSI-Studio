# Building DSI Studio using CMake

This note is brief review of how I built DSI Studio, using CMake.

I used a Mac running MacOS 11.6.1 BigSur, but things should be 
similar at least for Linux flavors.

__I have not tested this on any other system than my Mac so far. Your Mileage May Vary__

## Getting the dependent libraries.

* Get Qt Open Source Distribution version 5.12.2 from [The Qt Company](https://www.qt.io/download-open-source) and install
  using the GUI Installer

  On my Mac the installation left several versions on my computer in `~/Qt`. To get to the right version I needed to use 
  `~Qt/5.12.2/clang_64` as the root directory for Qt or (I will refer to it as `<QT_ROOT>` and ignore the other stuff there
   (there were also IOS and Android versions).

* Get TIPL from [The TIPL GitHub Repository](https://github.com/frankyeh/TIPL.git) -- Install this using CMake 
  into some directory (we will call the installation directory `<TIPL_ROOT>`) as follows
  ```bash$
  $ cd TIPL
  $ mkdir build
  $ cd build
  $ cmake ..
  $ cmake --install . --prefix <TIPL_ROOT>
  ```

## Configure the CMake Build for DSI Studio

 * Set up the CMake Prefix Path (I assume we are using bash)
 ```bash$
  export CMAKE_PREFIX_PATH=<TIPL_ROOT>:<QT_ROOT>:<BOOST_ROOT>
 ```
 * Make a build directory parallel to the cloned Source directory
  ```bash$
   mkdir DSI_Studio_build; cd DSI_Studio_build
  ```
 * Invoke CMake and build
  ```bash$
    cmake ../DSI_Studio
    cmake --build . -j 4 -v
  ```

  After a successful build the executable should be available in the top level of the build directory

## TODO
  - Installation and packaging 
  - Cross platform building (e.g. Windows, Linux, Currently hampered by my lack of access to Windows)
   
