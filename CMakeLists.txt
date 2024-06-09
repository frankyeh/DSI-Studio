cmake_minimum_required(VERSION 3.19)
project(DSI_Studio VERSION "2023.08.26" LANGUAGES CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 COMPONENTS Core Gui OpenGL Charts Network Widgets REQUIRED)
find_package(ZLIB REQUIRED)
set(ZLIB_USE_STATIC_LIBS "ON")

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit)


if(CUDAToolkit_FOUND)
    project(DSI_Studio VERSION "2023.07.19" LANGUAGES CUDA CXX;C)
endif()

set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/tracking
  ${CMAKE_SOURCE_DIR}/reconstruction
  ${CMAKE_SOURCE_DIR}/dicom
  ${CMAKE_SOURCE_DIR}/connectometry)

set(DSI_STUDIO_HEADERS mainwindow.h
    dicom/dicom_parser.h
    dicom/dwi_header.hpp
    libs/dsi/tessellated_icosahedron.hpp
    libs/dsi/odf_process.hpp
    libs/dsi/image_model.hpp
    libs/dsi/gqi_process.hpp
    libs/dsi/gqi_mni_reconstruction.hpp
    libs/dsi/dti_process.hpp
    libs/dsi/basic_voxel.hpp
    SliceModel.h
    tracking/tracking_window.h
    reconstruction/reconstruction_window.h
    tracking/slice_view_scene.h
    opengl/glwidget.h
    opengl/renderingtablewidget.h
    opengl/tract_render.hpp
    opengl/region_render.hpp
    libs/tracking/tracking_method.hpp
    libs/tracking/roi.hpp
    libs/tracking/fib_data.hpp
    libs/tracking/basic_process.hpp
    libs/tracking/tract_cluster.hpp
    tracking/region/regiontablewidget.h
    tracking/region/Regions.h
    libs/tracking/tract_model.hpp
    tracking/tract/tracttablewidget.h
    qcolorcombobox.h
    libs/tracking/tracking_thread.hpp
    libs/mapping/atlas.hpp
    view_image.h
    manual_alignment.h
    tracking/tract_report.hpp
    tracking/color_bar_dialog.hpp
    tracking/connectivity_matrix_dialog.h
    tracking/atlasdialog.h
    filebrowser.h
    qcompletelineedit.h
    libs/mapping/connectometry_db.hpp
    connectometry/createdbdialog.h
    connectometry/match_db.h
    connectometry/db_window.h
    connectometry/group_connectometry.hpp
    connectometry/group_connectometry_analysis.h
    regtoolbox.h
    auto_track.h
    tracking/device.h
    tracking/devicetablewidget.h
    libs/dsi/hist_process.hpp
    xnat_dialog.h
    console.h
    cmd/img.hpp
    reg.hpp)

set(DSI_STUDIO_FORMS mainwindow.ui
    tracking/tracking_window.ui
    reconstruction/reconstruction_window.ui
    dicom/dicom_parser.ui
    view_image.ui
    manual_alignment.ui
    tracking/tract_report.ui
    tracking/color_bar_dialog.ui
    tracking/connectivity_matrix_dialog.ui
    tracking/atlasdialog.ui
    filebrowser.ui
    connectometry/createdbdialog.ui
    connectometry/match_db.ui
    connectometry/db_window.ui
    connectometry/group_connectometry.ui
    regtoolbox.ui
    auto_track.ui
    console.ui
    xnat_dialog.ui)

set(DSI_STUDIO_RESOURCES icons.qrc)

set(DSI_STUDIO_SOURCES
    main.cpp
    mainwindow.cpp
    dicom/dicom_parser.cpp
    dicom/dwi_header.cpp
    libs/dsi/dsi_interface_imp.cpp
    libs/dsi/gqi_process.cpp
    libs/tracking/tract_cluster.cpp
    libs/tracking/roi.cpp
    SliceModel.cpp
    tracking/tracking_window.cpp
    tracking/tracking_window_action.cpp
    reconstruction/reconstruction_window.cpp
    tracking/slice_view_scene.cpp
    opengl/glwidget.cpp
    opengl/tract_render.cpp
    opengl/renderingtablewidget.cpp
    opengl/region_render.cpp
    tracking/region/regiontablewidget.cpp
    tracking/region/Regions.cpp
    libs/tracking/tract_model.cpp
    tracking/tract/tracttablewidget.cpp
    qcolorcombobox.cpp
    cmd/trk.cpp
    cmd/rec.cpp
    cmd/src.cpp
    cmd/img.cpp
    libs/mapping/atlas.cpp
    cmd/ana.cpp
    view_image.cpp
    manual_alignment.cpp
    tracking/tract_report.cpp
    tracking/color_bar_dialog.cpp
    cmd/exp.cpp
    tracking/connectivity_matrix_dialog.cpp
    libs/dsi/tessellated_icosahedron.cpp
    cmd/atl.cpp
    tracking/atlasdialog.cpp
    cmd/vis.cpp
    filebrowser.cpp
    qcompletelineedit.cpp
    libs/tracking/fib_data.cpp
    libs/tracking/tracking_thread.cpp
    cmd/ren.cpp
    libs/mapping/connectometry_db.cpp
    connectometry/createdbdialog.cpp
    connectometry/match_db.cpp
    connectometry/db_window.cpp
    connectometry/group_connectometry.cpp
    connectometry/group_connectometry_analysis.cpp
    regtoolbox.cpp
    cmd/cnn.cpp
    cmd/cnt.cpp
    cmd/qc.cpp
    libs/dsi/basic_voxel.cpp
    libs/dsi/image_model.cpp
    cmd/reg.cpp
    auto_track.cpp
    cmd/atk.cpp
    tracking/device.cpp
    tracking/devicetablewidget.cpp
    cmd/xnat.cpp
    xnat_dialog.cpp
    console.cpp)

set(DSI_STUDIO_OTHER_FILES options.txt
    dicom_tag.txt
    FreeSurferColorLUT.txt)

if(WIN32)
    set(DSI_STUDIO_EXEC
        WIN32
        dsi_studio.rc)
endif()
if(APPLE)
    set(DSI_STUDIO_EXEC
        MACOSX_BUNDLE)
endif()

if(CUDAToolkit_FOUND)
    set(DSI_STUDIO_CUDA
        cuda/reg.cu)
endif()


add_executable(dsi_studio ${DSI_STUDIO_EXEC} ${DSI_STUDIO_SOURCES} ${DSI_STUDIO_CUDA} ${DSI_STUDIO_FORMS} ${DSI_STUDIO_HEADERS} ${DSI_STUDIO_RESOURCES})
set_target_properties(dsi_studio PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)
target_compile_definitions(dsi_studio PUBLIC DSISTUDIO_RELEASE_NAME="Chen")
target_compile_definitions(dsi_studio PUBLIC DSISTUDIO_RELEASE_CODE=11770345)
target_compile_definitions(dsi_studio PUBLIC QT6_PATCH)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
    target_compile_definitions(dsi_studio PUBLIC TIPL_USE_CUDA)
    target_compile_definitions(dsi_studio PUBLIC TIPL_USE_QT)
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23.0")
        set_property(TARGET dsi_studio PROPERTY CUDA_ARCHITECTURES all)
    else()
        set_property(TARGET dsi_studio PROPERTY CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86")
    endif()
endif(CUDAToolkit_FOUND)

target_include_directories(dsi_studio PUBLIC
  ${CMAKE_SOURCE_DIR}/libs
  ${CMAKE_SOURCE_DIR}/libs/dsi
  ${CMAKE_SOURCE_DIR}/libs/tracking
  ${CMAKE_SOURCE_DIR}/libs/mapping
  ${CMAKE_SOURCE_DIR}/dicom
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
  ${TIPL_DIR})

target_link_libraries(dsi_studio Qt6::Core Qt6::Gui Qt6::OpenGL Qt6::Charts Qt6::Network Qt6::Widgets ZLIB::ZLIB OpenGL::GL OpenGL::GLU)
if(CUDAToolkit_FOUND)
    target_link_libraries(dsi_studio ${CUDA_LIBRARIES})
endif(CUDAToolkit_FOUND)
