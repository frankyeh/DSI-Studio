#include <set>
#include "tract_cluster.hpp"
#include "tipl/tipl.hpp"

struct compare_cluster
{

        bool operator()(const std::shared_ptr<Cluster>& lhs,const std::shared_ptr<Cluster>& rhs)
        {
            return lhs->tracts.size() > rhs->tracts.size();
        }

};

void BasicCluster::sort_cluster(void)
{
    std::sort(clusters.begin(),clusters.end(),compare_cluster());

    for (unsigned int index = 0;index < clusters.size();++index)
        clusters[index]->index = index;
}

TractCluster::TractCluster(const float* param):error_distance(param[3])
{
    tipl::vector<3,float> fdim(param);
    fdim /= error_distance;
    fdim += 1.0;
    fdim.floor();
    dim[0] = fdim[0];
    dim[1] = fdim[1];
    dim[2] = fdim[2];
    w = dim[0];
    wh = dim[0]*dim[1];
    voxel_connection.resize(dim.size());

}

void TractCluster::set_tract_labels(Cluster* from,Cluster* to)
{
    std::vector<unsigned int>::const_iterator iter = from->tracts.begin();
    std::vector<unsigned int>::const_iterator end = from->tracts.end();
    for (;iter != end;++iter)
        tract_labels[*iter] = to;

}

int TractCluster::get_index(short x,short y,short z)
{
    int index = z;
    index *= dim[1];
    index += y;
    index *= dim[0];
    index += x;
    return index;
}
/**
prerequite:
    tract_index1 > tract_index2
*/
void TractCluster::merge_tract(unsigned int tract_index1,unsigned int tract_index2)
{
    if (tract_index1 == tract_index2)
        return;
    std::lock_guard<std::mutex> lock(lock_merge);
    Cluster* cluster_index = tract_labels[tract_index1];
    if (cluster_index == 0) // no cluster
    {
        cluster_index = tract_labels[tract_index2];
        if (cluster_index == 0) // create a cluster for both
        {
            std::shared_ptr<Cluster> new_cluster_index(new Cluster);
            new_cluster_index->index = clusters.size();
            clusters.push_back(new_cluster_index);

            new_cluster_index->tracts.push_back(tract_index2);
            tract_labels[tract_index2] = new_cluster_index.get();
            new_cluster_index->tracts.push_back(tract_index1);
            tract_labels[tract_index1] = new_cluster_index.get();
            return;
        }
        else
        {
            cluster_index->tracts.push_back(tract_index1);
            tract_labels[tract_index1] = cluster_index;
            return;
        }
    }
    else
    {
        Cluster* another_cluster_index = tract_labels[tract_index2];
        if (another_cluster_index == 0) // add tract2 to the cluster
        {
            cluster_index->tracts.push_back(tract_index2);
            tract_labels[tract_index2] = cluster_index;
            return;
        }
        if (another_cluster_index == cluster_index) // already in the same group
            return;
        // merge two clusters
        if (another_cluster_index->tracts.size() > cluster_index->tracts.size())
            std::swap(cluster_index,another_cluster_index);

        cluster_index->tracts.resize(another_cluster_index->tracts.size()+cluster_index->tracts.size());
        std::copy(another_cluster_index->tracts.begin(),another_cluster_index->tracts.end(),
                  cluster_index->tracts.end()-another_cluster_index->tracts.size());

        set_tract_labels(another_cluster_index,cluster_index);
        clusters.back()->index = another_cluster_index->index;
        clusters[another_cluster_index->index] = clusters.back();
        clusters.pop_back();

    }
}

void TractCluster::add_tracts(const std::vector<std::vector<float> >& tracks)
{
    tract_labels.clear();
    tract_passed_voxels.clear();
    tract_ranged_voxels.clear();
    tract_labels.resize(tracks.size());
    tract_length.resize(tracks.size());
    tract_passed_voxels.resize(tracks.size());
    tract_ranged_voxels.resize(tracks.size());
    for(unsigned int tract_index = 0;tract_index < tracks.size();++tract_index)
        tract_length[tract_index] = tracks[tract_index].size();

    // build passing points and ranged points
    tipl::par_for(tracks.size(),[&](unsigned int tract_index)
    {
        if(tracks[tract_index].empty())
            return;
        unsigned int count = tracks[tract_index].size();
        const float* points = &tracks[tract_index][0];
        const float* points_end = points + count;
        std::vector<unsigned short>& passed_points = tract_passed_voxels[tract_index];
        std::vector<unsigned short>& ranged_points = tract_ranged_voxels[tract_index];

        for (;points_end != points;points += 3)
        {
            tipl::vector<3,float> cur_point(points);
            cur_point /= error_distance;
            cur_point.round();
            if(!dim.is_valid(cur_point))
                continue;
            tipl::pixel_index<3> center(cur_point[0],cur_point[1],cur_point[2],dim);
            passed_points.push_back(center.index() & 0xFFFF);
            std::vector<tipl::pixel_index<3> > iterations;
            tipl::get_neighbors(center,dim,iterations);
            for(unsigned int index = 0;index < iterations.size();++index)
                if (dim.is_valid(iterations[index]))
                    ranged_points.push_back(iterations[index].index() & 0xFFFF);
        }

        // delete repeated points
        {
            std::set<unsigned short> unique_passed_points(passed_points.begin(),passed_points.end());
            passed_points = std::vector<unsigned short>(unique_passed_points.begin(),unique_passed_points.end());

            std::set<unsigned short> unique_ranged_points(ranged_points.begin(),ranged_points.end());
            ranged_points = std::vector<unsigned short>(unique_ranged_points.begin(),unique_ranged_points.end());
        }
    });
    // book keeping passing points
    for(unsigned int tract_index = 0;tract_index < tracks.size();++tract_index)
    {
        voxel_connection[tract_passed_voxels[tract_index].front()].push_back(tract_index);
        voxel_connection[tract_passed_voxels[tract_index].back()].push_back(tract_index);
    }

    tipl::par_for(tracks.size(),[&](unsigned int tract_index)
    {
        if(tracks[tract_index].empty())
            return;
        unsigned int count = tracks[tract_index].size();
        std::vector<unsigned short>& passed_points = tract_passed_voxels[tract_index];
        std::vector<unsigned short>& ranged_points = tract_ranged_voxels[tract_index];
        if(passed_points.empty() || ranged_points.empty())
            return;

        // get the eligible fibers for merging, and also register the ending points
        std::vector<unsigned int> passing_tracts;
        {
            std::vector<unsigned int> p1(voxel_connection[tract_passed_voxels[tract_index].front()]);
            std::vector<unsigned int> p2(voxel_connection[tract_passed_voxels[tract_index].back()]);
            p1.insert(p1.end(),p2.begin(),p2.end());
            p1.erase(std::remove(p1.begin(),p1.end(),tract_index),p1.end());
            std::set<unsigned int> unique_tracts(p1.begin(),p1.end());
            passing_tracts = std::vector<unsigned int>(unique_tracts.begin(),unique_tracts.end());
        }

        // check each tract to see if anyone is included in the error range
        for (int i = 0;i < passing_tracts.size();++i)
        {
            unsigned int cur_index = passing_tracts[i];
            Cluster* label1 = tract_labels[tract_index];
            Cluster* label2 = tract_labels[cur_index];
            if (label1 != 0 && label1 == label2)
                continue;
            unsigned int cur_count = tract_length[cur_index];
            float dif = cur_count;
            dif -= (float) count;
            dif /= (float)std::max(cur_count,count);
            if (std::abs(dif) > 0.2)
                continue;
            if (std::includes(ranged_points.begin(),ranged_points.end(),
                              tract_passed_voxels[cur_index].begin(),tract_passed_voxels[cur_index].end()) &&
                std::includes(tract_ranged_voxels[cur_index].begin(),tract_ranged_voxels[cur_index].end(),
                                  passed_points.begin(),passed_points.end()))
                merge_tract(tract_index,cur_index);
        }
    });
}
