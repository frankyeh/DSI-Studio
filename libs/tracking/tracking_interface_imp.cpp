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
    unsigned char limit = std::min((unsigned char)3,fib_data.fib.cur_odf_record);
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
    void set_voxel_information(ODFModel* odf_model,unsigned int x,unsigned int y,unsigned int z,float* data)
{
//    odf_model->setVoxelInformation(x,y,z,data);
}



extern "C"
    const short* select_bundle(ODFModel* odf_model,short* position,
                               float angle,float fa_threshold,unsigned int& count)
{
    float cos_angle = std::cos(angle);
    image::basic_image<char,3> level_map(odf_model->fib_data.dim);
    std::deque<image::vector<3,float> > seed_dir;
    std::deque<image::pixel_index<3> > seed_pos;
    seed_pos.push_back(image::pixel_index<3>(position[0],position[1],position[2],odf_model->fib_data.dim));
    seed_pos.push_back(image::pixel_index<3>(position[0],position[1],position[2],odf_model->fib_data.dim));
    seed_dir.push_back(odf_model->fib_data.fib.getDir(seed_pos.back().index(),0));
    seed_dir.push_back(-odf_model->fib_data.fib.getDir(seed_pos.back().index(),0));
    level_map[seed_pos.back().index()] = 1;


    while (!seed_pos.empty())
    {
        std::vector<image::pixel_index<3> > neighbors;
        image::get_neighbors(seed_pos.front(),odf_model->fib_data.dim,neighbors);
        image::vector<3,float> center(seed_pos.front());
        for (unsigned int index = 0;index < neighbors.size();++index)
        {
            // select front
            if (level_map[neighbors[index].index()])
                continue;

            image::vector<3,float> displace(neighbors[index]);
            displace -= center;
            if (displace*seed_dir.front() < 0.2)
                continue;
            image::vector<3,float> new_dir;
            if (!odf_model->fib_data.get_nearest_dir(neighbors[index].index(),seed_dir.front(),new_dir,fa_threshold,cos_angle))
                continue;

            if (std::abs(new_dir*seed_dir.front()) < cos_angle)
                continue;

            if (new_dir*seed_dir.front() < 0)
                new_dir = -new_dir;
            seed_pos.push_back(neighbors[index]);
            seed_dir.push_back(new_dir);
            level_map[neighbors[index].index()] = 1;
        }
        seed_pos.pop_front();
        seed_dir.pop_front();
    }
    //image::morphology_smoothing(level_map);
    static std::vector<short> sel_regions;
    sel_regions.clear();
    for (image::pixel_index<3> index;index.valid(odf_model->fib_data.dim);index.next(odf_model->fib_data.dim))
        if (level_map[index.index()])
        {
            sel_regions.push_back(index.x());
            sel_regions.push_back(index.y());
            sel_regions.push_back(index.z());
        }

    count = sel_regions.size();
    return &*sel_regions.begin();
}


extern "C"
    const char* compare_fiber_directions(const char* file_name1,const char* file_name2,
                                         const short *points,unsigned int number)
{
    static std::string result;
    ODFModel odf_model1,odf_model2;
    if (!odf_model1.load_from_file(file_name1) || !odf_model2.load_from_file(file_name2))
        return 0;
    std::string report_name(file_name2);
    report_name += ".report.txt";
    // originally, this is a output file write to report_name
    std::ostringstream out;
    odf_model1.fib_data.compare_fiber_directions(odf_model2.fib_data,points,number,result,out);
    return &*result.c_str();
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
