#ifndef STDAFX_DSI_H
#define STDAFX_DSI_H
#define _USE_MATH_DEFINES

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <boost/mpl/vector.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/thread.hpp>


#ifndef DEFINE_PARAM
#define DEFINE_PARAM
// TODO: 在此參考您的程式所需要的其他標頭
//---
const unsigned int qcode_count = 203;
const int display_range = 3;
const int min_display = -(1 << (display_range-1));
const int max_display = (1 << (display_range-1))-1;
const double odf_min_radius = 2.1;
const double odf_max_radius = 6.0;
const double odf_sampling_interval = 0.2;

const int dsi_range = 4;
const unsigned int space_length = 1 << dsi_range;             //16
const unsigned int space_half_length = 1 << (dsi_range-1);    //8
const int space_min_offset = -(1 << (dsi_range-1));                //-8
const int space_max_offset = space_half_length-1;                  //7
const unsigned int qspace_size = 1 << (dsi_range * 3);         //4096

#endif

#endif
