#-------------------------------------------------
#
# Project created by QtCreator 2011-01-21T22:58:19
#
#-------------------------------------------------

QT       -= core gui

TARGET = mapping
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += c:/frank/myprog/include \
    ..

SOURCES += mapping.cpp \
    interface.cpp \
    mni_norm.cpp \
    talairach.cpp

HEADERS += mapping.h \
    mni_norm.hpp \
    talairach.hpp \

RESOURCES +=
