#include <set>
#include "tract_cluster.hpp"

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
    tract_mid_voxels.clear();
    tract_end1.clear();
    tract_end2.clear();
    tract_labels.resize(tracks.size());
    tract_length.resize(tracks.size());
    tract_mid_voxels.resize(tracks.size());
    tract_end1.resize(tracks.size());
    tract_end2.resize(tracks.size());
    tipl::par_for(tracks.size(),[&](unsigned int tract_index)
    {
        if(tracks[tract_index].size() >= 6)
            tract_length[tract_index] = float(tracks[tract_index].size())*
                    float((tipl::vector<3>(&tracks[tract_index][0])-tipl::vector<3>(&tracks[tract_index][3])).length());
    });

    // build passing points and ranged points
    tipl::par_for(tracks.size(),[&](unsigned int tract_index)
    {
        if(tracks[tract_index].empty())
            return;
        tipl::vector<3,float> p_end1(&tracks[tract_index][0]);
        tipl::vector<3,float> p_end2(&tracks[tract_index][tracks[tract_index].size()-3]);
        if(p_end1 > p_end2)
            std::swap(p_end1,p_end2);
        tract_end1[tract_index] = p_end1;
        tract_end2[tract_index] = p_end2;

        // get mid point in reduced space
        tipl::vector<3,float> p_mid(&tracks[tract_index][(tracks[tract_index].size()/6)*3]);
        p_mid /= error_distance;
        p_mid.round();
        if(!dim.is_valid(p_mid))
            return;
        tract_mid_voxels[tract_index] = tipl::pixel_index<3>(p_mid[0],p_mid[1],p_mid[2],dim).index();

    });
    // book keeping passing points
    for(unsigned int tract_index = 0;tract_index < tracks.size();++tract_index)
    {
        voxel_connection[tract_mid_voxels[tract_index]].push_back(tract_index);
        tipl::for_each_connected_neighbors(
                    tipl::pixel_index<3>(tract_mid_voxels[tract_index],dim),dim,
                    [&](const auto& pos)
            {
                voxel_connection[pos.index()].push_back(tract_index);
            });
    }
    tipl::par_for(voxel_connection.size(),[&](unsigned int pos)
    {
        std::set<unsigned int> tmp(voxel_connection[pos].begin(),voxel_connection[pos].end());
        voxel_connection[pos] = std::vector<unsigned int>(tmp.begin(),tmp.end());
    });
    tipl::par_for(tracks.size(),[&](unsigned int tract_index)
    {
        if(tracks[tract_index].empty())
            return;
        auto& passing_tracts = voxel_connection[tract_mid_voxels[tract_index]];
        // check each tract to see if anyone is included in the error range
        for (size_t i = 0;i < passing_tracts.size();++i)
        {
            unsigned int cur_index = passing_tracts[i];
            if(cur_index <= tract_index)
                continue;
            Cluster* label1 = tract_labels[tract_index];
            Cluster* label2 = tract_labels[cur_index];
            if (label1 != nullptr && label1 == label2)
                continue;

            if(std::fabs(tract_end1[tract_index][0]-tract_end1[cur_index][0]) > error_distance ||
               (tract_end1[tract_index]-tract_end1[cur_index]).length() > double(error_distance) ||
               (tract_end2[tract_index]-tract_end2[cur_index]).length() > double(error_distance))
                continue;
            if (std::fabs((tract_length[cur_index]-tract_length[tract_index])) > double(error_distance)*2.0)
                continue;
            merge_tract(tract_index,cur_index);
        }
    });
}
