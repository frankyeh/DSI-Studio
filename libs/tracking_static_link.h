#ifndef TRACKING_INTERFACE_H
#define TRACKING_INTERFACE_H
class ODFModel;
extern "C"
{
     void tracking_get_slices_dir_color(ODFModel* handle,unsigned short order,unsigned int* pixels);
     const float* get_odf_direction(ODFModel* handle,unsigned int index);
     const unsigned short* get_odf_faces(ODFModel* handle,unsigned int index);
     const float* get_odf_data(ODFModel* odf_model,unsigned int index);
}


#endif
