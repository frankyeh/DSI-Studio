# -------------------------------------------------
# Project created by QtCreator 2011-01-20T20:02:59
# -------------------------------------------------
QT += core \
    gui \
    opengl
#CONFIG += console
TARGET = dsi_studio
TEMPLATE = app
win32 {
INCLUDEPATH += C:\frank\myprog\include
}

win32-g++ {
LIBS += -L. -lboost_thread-mgw45-mt-1_45.dll \
     -L. -lboost_program_options-mgw45-mt-1_45.dll
}

linux {
QMAKE_CXXFLAGS += -fpermissive
LIBS += -lboost_thread \
        -lboost_program_options \
        -lGLU \
        -lz
}

mac{
LIBS += -L/opt/local/lib -lboost_system \
        -L/opt/local/lib -lboost_thread \
        -L/opt/local/lib -lboost_program_options
}


# you may need to change the include directory of boost library
INCLUDEPATH += libs \
    libs/dsi \
    libs/tracking \
    libs/mapping
HEADERS += mainwindow.h \
    dicom/dicom_parser.h \
    dicom/dwi_header.hpp \
    libs/mat_file.hpp \
    libs/dsi/stdafx.h \
    libs/dsi/tessellated_icosahedron.hpp \
    libs/dsi/space_mapping.hpp \
    libs/dsi/shaping_model.hpp \
    libs/dsi/sh_process.hpp \
    libs/dsi/sample_model.hpp \
    libs/dsi/racian_noise.hpp \
    libs/dsi/qbi_process.hpp \
    libs/dsi/odf_process.hpp \
    libs/dsi/odf_deconvolusion.hpp \
    libs/dsi/odf_decomposition.hpp \
    libs/dsi/mix_gaussian_model.hpp \
    libs/dsi/layout.hpp \
    libs/dsi/image_model.hpp \
    libs/dsi/gqi_process.hpp \
    libs/dsi/gqi_mni_reconstruction.hpp \
    libs/dsi/dti_process.hpp \
    libs/dsi/dsi_process.hpp \
    libs/dsi/common.hpp \
    libs/dsi/basic_voxel.hpp \
    libs/dsi_interface_static_link.h \
    SliceModel.h \
    libs/tracking_static_link.h \
    tracking/tracking_window.h \
    reconstruction/reconstruction_window.h \
    tracking/slice_view_scene.h \
    opengl/glwidget.h \
    libs/tracking/tracking_model.hpp \
    libs/tracking/tracking_method.hpp \
    libs/tracking/tracking_info.hpp \
    libs/tracking/roi.hpp \
    libs/tracking/interpolation_process.hpp \
    libs/tracking/fib_data.hpp \
    libs/tracking/basic_process.hpp \
    libs/tracking/tract_cluster.hpp \
    tracking/region/regiontablewidget.h \
    tracking/region/Regions.h \
    tracking/region/RegionModel.h \
    libs/tracking/tract_model.hpp \
    tracking/tract/tracttablewidget.h \
    opengl/renderingtablewidget.h \
    qcolorcombobox.h \
    libs/coreg/linear.hpp \
    libs/coreg/lddmm.hpp \
    libs/coreg/coreg_interface.h \
    libs/tracking/tracking_thread.hpp \
    libs/prog_interface_static_link.h \
    simulation.h \
    reconstruction/vbcdialog.h \
    libs/vbc/vbc.hpp \
    libs/mapping/atlas.hpp \
    libs/mapping/fa_template.hpp \
    libs/mapping/normalization.hpp \
    plot/qcustomplot.h \
    view_image.h \
    libs/vbc/vbc_database.h \
    libs/gzip_interface.hpp \
    libs/dsi/racian_noise.hpp \
    libs/dsi/mix_gaussian_model.hpp \
    libs/dsi/layout.hpp
FORMS += mainwindow.ui \
    tracking/tracking_window.ui \
    reconstruction/reconstruction_window.ui \
    dicom/dicom_parser.ui \
    simulation.ui \
    reconstruction/vbcdialog.ui \
    view_image.ui
RESOURCES += \
    icons.qrc
SOURCES += main.cpp \
    mainwindow.cpp \
    dicom/dicom_parser.cpp \
    dicom/dwi_header.cpp \
    libs/utility/prog_interface.cpp \
    libs/dsi/common.cpp \
    libs/dsi/sample_model.cpp \
    libs/dsi/dsi_interface_imp.cpp \
    libs/tracking/tracking_interface_imp.cpp \
    libs/tracking/interpolation_process.cpp \
    libs/tracking/tract_cluster.cpp \
    SliceModel.cpp \
    tracking/tracking_window.cpp \
    reconstruction/reconstruction_window.cpp \
    tracking/slice_view_scene.cpp \
    opengl/glwidget.cpp \
    tracking/region/regiontablewidget.cpp \
    tracking/region/Regions.cpp \
    tracking/region/RegionModel.cpp \
    libs/tracking/tract_model.cpp \
    tracking/tract/tracttablewidget.cpp \
    opengl/renderingtablewidget.cpp \
    qcolorcombobox.cpp \
    libs/coreg/lddmm_interface.cpp \
    libs/coreg/coreg_interface.cpp \
    cmd/trk.cpp \
    cmd/rec.cpp \
    simulation.cpp \
    gzlib/zutil.c \
    gzlib/uncompr.c \
    gzlib/trees.c \
    gzlib/inftrees.c \
    gzlib/inflate.c \
    gzlib/inffast.c \
    gzlib/infback.c \
    gzlib/gzwrite.c \
    gzlib/gzread.c \
    gzlib/gzlib.c \
    gzlib/gzclose.c \
    gzlib/deflate.c \
    gzlib/crc32.c \
    gzlib/compress.c \
    gzlib/adler32.c \
    reconstruction/vbcdialog.cpp \
    cmd/src.cpp \
    libs/mapping/atlas.cpp \
    libs/mapping/fa_template.cpp \
    plot/qcustomplot.cpp \
    libs/vbc/vbc.cpp \
    cmd/ana.cpp \
    view_image.cpp \
    libs/vbc/vbc_database.cpp
