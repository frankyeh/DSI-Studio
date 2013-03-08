#include <memory>
#include <algorithm>
#include <string>
#include <set>

#include "prog_interface_static_link.h"
#include "mat_file.hpp"
typedef class ReadMatFile MatReader;
typedef class WriteMatFile MatWriter;

#include "roi.hpp"
#include "tracking_model.hpp"
#include "tract_cluster.hpp"
#include "tracking_method.hpp"
#include "image/image.hpp"

extern "C"
    void tracking_get_slices_dir_color(ODFModel* odf_model,unsigned short order,unsigned int* pixels)
{
    odf_model->getSlicesDirColor(order,pixels);
}

extern "C"
    bool tracking_get_voxel_dir(ODFModel* odf_model,unsigned int x,unsigned int y,unsigned int z,
                                float* fa,float* dir)
{
    FibData& fib_data = odf_model->fib_data;
    image::pixel_index<3> pos(x,y,z,odf_model->fib_data.dim);
    if (pos.index() >= odf_model->fib_data.total_size || fib_data.fib.getFA(pos.index(),0) == 0.0)
        return false;
    unsigned int limit = std::min<unsigned int>(3,fib_data.fib.num_fiber);
    for (unsigned int index = 0;index < limit;++index)
    {
        fa[index] = fib_data.fib.getFA(pos.index(),index);
        const image::vector<3,float>& fdir = fib_data.fib.getDir(pos.index(),index);
        std::copy(fdir.begin(),fdir.end(),dir);
        dir += 3;
    }
    for (unsigned int index = limit;index < 3;++index)
    {
        fa[index] = 0.0;
        std::fill(dir,dir+3,0.0);
        dir += 3;
    }
    return true;
}


extern "C"
    const float* get_odf_direction(ODFModel* odf_model,unsigned int index)
{
    return &*odf_model->fib_data.fib.odf_table[index].begin();
}
extern "C"
    const unsigned short* get_odf_faces(ODFModel* odf_model,unsigned int index)
{
    return &*odf_model->fib_data.fib.odf_faces[index].begin();
}

extern "C"
    const float* get_odf_data(ODFModel* odf_model,unsigned int index)
{
    return odf_model->fib_data.fib.get_odf_data(index);
}

extern "C"
    void* tract_cluster_create(unsigned int method,const float* param)
{
    switch (method)
    {
    case 0:
        return new TractCluster(param);
    case 1:
        return new FeatureBasedClutering<k_means<double,unsigned char> >(param);
    case 2:
        return new FeatureBasedClutering<expectation_maximization<double,unsigned char> >(param);
    }
    return 0;
}

extern "C"
    void tract_cluster_add_tract(BasicCluster* tract_cluster,const float* points,unsigned int count)
{
    if (points)
        tract_cluster->add_tract(points,count);
    else
        tract_cluster->run_clustering();
}

extern "C"
    unsigned int tract_cluster_get_cluster_count(BasicCluster* tract_cluster)
{
    return tract_cluster->get_cluster_count();
}


extern "C"
    const unsigned int* tract_cluster_get_cluster(BasicCluster* tract_cluster,unsigned int cluster_index,unsigned int& cluster_size)
{
    return tract_cluster->get_cluster(cluster_index,cluster_size);
}


extern "C"
    void tract_cluster_free(BasicCluster* tract_cluster)
{
    delete tract_cluster;
}
