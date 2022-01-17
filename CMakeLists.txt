cmake_minimum_required(VERSION 3.19)
project(DSI_Studio VERSION "202112.03" LANGUAGES CXX;C)


find_package(Qt5 COMPONENTS Core Gui OpenGL Charts Network Widgets REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenGL REQUIRED)
find_package(TIPL REQUIRED)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/tracking
  ${CMAKE_SOURCE_DIR}/reconstruction
  ${CMAKE_SOURCE_DIR}/dicom
  ${CMAKE_SOURCE_DIR}/connectometry)

set(CMAKE_AUTOMOC ON)

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
    libs/tracking/tracking_method.hpp 
    libs/tracking/roi.hpp 
    libs/tracking/interpolation_process.hpp 
    libs/tracking/fib_data.hpp 
    libs/tracking/basic_process.hpp 
    libs/tracking/tract_cluster.hpp 
    tracking/region/regiontablewidget.h 
    tracking/region/Regions.h 
    tracking/region/RegionModel.h 
    libs/tracking/tract_model.hpp 
    tracking/tract/tracttablewidget.h 
    opengl/renderingtablewidget.h 
    qcolorcombobox.h 
    libs/tracking/tracking_thread.hpp 
    libs/prog_interface_static_link.h 
    libs/mapping/atlas.hpp 
    view_image.h 
    libs/gzip_interface.hpp 
    manual_alignment.h 
    tracking/tract_report.hpp 
    tracking/color_bar_dialog.hpp 
    tracking/connectivity_matrix_dialog.h 
    tracking/atlasdialog.h 
    filebrowser.h 
    program_option.hpp 
    qcompletelineedit.h 
    libs/mapping/connectometry_db.hpp 
    connectometry/createdbdialog.h 
    connectometry/individual_connectometry.hpp 
    connectometry/match_db.h 
    connectometry/db_window.h 
    connectometry/group_connectometry.hpp 
    connectometry/group_connectometry_analysis.h 
    regtoolbox.h
    connectometry/nn_connectometry.h 
    connectometry/nn_connectometry_analysis.h 
    auto_track.h 
    tracking/device.h 
    tracking/devicetablewidget.h 
    mac_filesystem.hpp 
    libs/dsi/hist_process.hpp 
    xnat_dialog.h)

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
    connectometry/individual_connectometry.ui 
    connectometry/match_db.ui 
    connectometry/db_window.ui 
    connectometry/group_connectometry.ui 
    regtoolbox.ui 
    connectometry/nn_connectometry.ui 
    auto_track.ui 
    xnat_dialog.ui)

set(DSI_STUDIO_RESOURCES icons.qrc)

set(DSI_STUDIO_SOURCES main.cpp 
    mainwindow.cpp 
    dicom/dicom_parser.cpp 
    dicom/dwi_header.cpp 
    libs/utility/prog_interface.cpp 
    libs/dsi/dsi_interface_imp.cpp 
    libs/tracking/interpolation_process.cpp 
    libs/tracking/tract_cluster.cpp 
    SliceModel.cpp 
    tracking/tracking_window.cpp 
    reconstruction/reconstruction_window.cpp 
    tracking/slice_view_scene.cpp 
    opengl/glwidget.cpp 
    tracking/region/regiontablewidget.cpp 
    tracking/region/Regions.cpp 
    tracking/region/RegionModel.cpp 
    libs/tracking/tract_model.cpp 
    tracking/tract/tracttablewidget.cpp 
    opengl/renderingtablewidget.cpp 
    qcolorcombobox.cpp 
    cmd/trk.cpp 
    cmd/rec.cpp 
    cmd/src.cpp 
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
    connectometry/individual_connectometry.cpp 
    connectometry/match_db.cpp 
    connectometry/db_window.cpp 
    connectometry/group_connectometry.cpp 
    connectometry/group_connectometry_analysis.cpp 
    regtoolbox.cpp 
    cmd/cnn.cpp 
    cmd/qc.cpp 
    libs/dsi/basic_voxel.cpp 
    libs/dsi/image_model.cpp 
    connectometry/nn_connectometry.cpp 
    connectometry/nn_connectometry_analysis.cpp 
    cmd/reg.cpp 
    auto_track.cpp 
    cmd/atk.cpp 
    tracking/device.cpp 
    tracking/devicetablewidget.cpp 
    libs/gzip_interface.cpp 
    cmd/xnat.cpp 
    xnat_dialog.cpp)

set(DSI_STUDIO_OTHER_FILES options.txt 
    dicom_tag.txt 
    FreeSurferColorLUT.txt 
    shader_fragment.txt 
    shader_vertex.txt)

set(DIS_STUDIO_DISTFILES shader_fragment2.txt 
    shader_vertex2.txt 
    LICENSE)


add_executable(dsi_studio ${DSI_STUDIO_SOURCES})
set_target_properties(dsi_studio PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)
target_compile_definitions(dsi_studio PUBLIC DSISTUDIO_RELEASE_NAME="Chen")
target_compile_definitions(dsi_studio PUBLIC DSISTUDIO_RELEASE_CODE=11770345)
target_include_directories(dsi_studio PUBLIC 
  ${CMAKE_SOURCE_DIR}/libs 
  ${CMAKE_SOURCE_DIR}/libs/dsi 
  ${CMAKE_SOURCE_DIR}/libs/tracking 
  ${CMAKE_SOURCE_DIR}/libs/mapping 
  ${CMAKE_SOURCE_DIR}/dicom
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR})

target_link_libraries(dsi_studio TIPL::tipl Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::Charts Qt5::Network Qt5::Widgets ZLIB::ZLIB OpenGL::GL OpenGL::GLU)

