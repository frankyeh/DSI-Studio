#ifndef TRACT_CLUSTER_HPP
#define TRACT_CLUSTER_HPP
#include <vector>
#include <map>
#include "zlib.h"
#include "TIPL/tipl.hpp"

struct Cluster
{

    std::vector<unsigned int> tracts;
    unsigned int index;

};
class BasicCluster
{
protected:
    std::vector<std::shared_ptr<Cluster> > clusters;
    void sort_cluster(void);
public:
    virtual ~BasicCluster(void){}
    virtual void add_tracts(const std::vector<std::vector<float> >& tracks) = 0;
    virtual void run_clustering(void) = 0;
public:
    unsigned int get_cluster_count(void) const
    {
        return clusters.size();
    }
    const unsigned int* get_cluster(unsigned int cluster_index,unsigned int& cluster_size) const
    {
        cluster_size = clusters[cluster_index]->tracts.size();
        return &*clusters[cluster_index]->tracts.begin();
    }
};

template<class method_type>
class FeatureBasedClutering : public BasicCluster
{
    std::vector<std::vector<double> > features;
    std::vector<unsigned char> classifications;
    mutable std::vector<unsigned int> result;
    method_type clustering_method;
    unsigned int cluster_number;
public:
    FeatureBasedClutering(const float* param):clustering_method(param[0]),cluster_number(param[0]){}
    virtual ~FeatureBasedClutering(void) {}

public:
    virtual void add_tracts(const std::vector<std::vector<float> >& tracks)
    {
        for(int i = 0;i < tracks.size();++i)
            if(!tracks[i].empty())
            {
                const float* points = &tracks[i][0];
                unsigned int count = tracks[i].size();
                std::vector<double> feature(10);
                std::copy_n(points,3,feature.begin());
                std::copy_n(points+count-3,3,feature.begin()+3);
                count >>= 1;
                count -= count%3;
                std::copy_n(points+count-3,3,feature.begin()+6);
                feature.back() = count;
                features.push_back(feature);
            }
    }
    virtual void run_clustering(void)
    {
        classifications.resize(features.size());
        clustering_method(features.begin(),features.end(),10,classifications.begin());
        std::map<unsigned char,std::vector<unsigned int> > cluster_map;
        for (unsigned int index = 0;index < classifications.size();++index)
            cluster_map[classifications[index]].push_back(index);
		clusters.resize(cluster_map.size());
                std::map<unsigned char,std::vector<unsigned int> >::iterator iter = cluster_map.begin();
                std::map<unsigned char,std::vector<unsigned int> >::iterator end = cluster_map.end();
                for(unsigned int index = 0;iter != end;++iter,++index)
		{
            clusters[index] = std::make_shared<Cluster>();
			clusters[index]->tracts.swap(iter->second);
			clusters[index]->index= index;
		}
		sort_cluster();
    }

};



class TractCluster : public BasicCluster
{
    tipl::shape<3> dim;
    unsigned int w,wh;
    float error_distance;
    std::mutex  lock_merge;
private:


    void set_tract_labels(Cluster* from,Cluster* to);
    void merge_tract(unsigned int tract_index1,unsigned int tract_index2);
    int get_index(short x,short y,short z);
private:
    std::vector<std::vector<unsigned int> > voxel_connection;
private:
    std::vector<Cluster*> tract_labels;// 0 is no cluster
    std::vector<unsigned int> tract_mid_voxels;
    std::vector<tipl::vector<3> > tract_end1;
    std::vector<tipl::vector<3> > tract_end2;
    std::vector<float> tract_length;



public:
    TractCluster(const float* param);
    void add_tracts(const std::vector<std::vector<float> >& tracks);
	void run_clustering(void){sort_cluster();}

};




#endif//TRACT_CLUSTER_HPP
