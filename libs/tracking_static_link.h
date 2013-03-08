#ifndef TRACKING_INTERFACE_H
#define TRACKING_INTERFACE_H
class ODFModel;
extern "C"
{
     void tracking_get_slices_dir_color(ODFModel* handle,unsigned short order,unsigned int* pixels);
     bool tracking_get_voxel_dir(ODFModel* handle,unsigned int x,unsigned int y,unsigned int z,float* fa,float* dir);
     const float* get_odf_direction(ODFModel* handle,unsigned int index);
     const unsigned short* get_odf_faces(ODFModel* handle,unsigned int index);
     const float* get_odf_data(ODFModel* odf_model,unsigned int index);
     void* tract_cluster_create(unsigned int method,const float* param);
     void tract_cluster_add_tract(void* tract_cluster,const float* points,unsigned int count);
     unsigned int tract_cluster_get_cluster_count(void* tract_cluster);
     const unsigned int* tract_cluster_get_cluster(void* tract_cluster,unsigned int cluster_index,unsigned int& cluster_size);
     void tract_cluster_free(void* tract_cluster);

}


#endif
