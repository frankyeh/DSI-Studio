#include "vbc.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include <boost/math/distributions/students_t.hpp>


float permutation_test(std::vector<float>& data,
                       unsigned int num1,unsigned num2,
                       unsigned int num_trial,float p_value_threshold,
                       float& dif,
                       bool& greater)
{
    const float* g1 = &data[0];
    const float* g2 = &data[num1];
    const float* g1_end = g1 + num1;
    const float* g2_end = g2 + num2;
    // single subject condition
    if(num1 == 1)
    {
        float g1_value = *g1;
        float mean2 = std::accumulate(g2,g2_end,0.0f)/(float)num2;
        dif = g1_value-mean2;
        if(dif < 0)
        {
            dif = -dif;
            greater = false;
            unsigned int count = 0;
            for(;g2 != g2_end;++g2)
                if(*g2 <= g1_value)
                    ++count;
            return (float)(count)/((float)num2);
        }
        else
        {
            greater = true;
            unsigned int count = 0;
            for(;g2 != g2_end;++g2)
                if(*g2 >= g1_value)
                    ++count;
            return (float)(count)/((float)num2);
        }
    }
    else
    {
        unsigned int critical_num = p_value_threshold*num_trial;
        float mean1 = std::accumulate(g1,g1_end,0.0f)/(float)num1;
        float mean2 = std::accumulate(g2,g2_end,0.0f)/(float)num2;
        dif = mean1-mean2;
        if(dif < 0)
        {
            dif = -dif;
            unsigned int sum = 0;
            for(int j = 0;j < num_trial;++j)
            {
                std::random_shuffle(data.begin(),data.end());
                if(std::accumulate(g2,g2_end,0.0f)/(float)num2
                   -std::accumulate(g1,g1_end,0.0f)/(float)num1 > dif)
                    ++sum;
                // surely failed to achieve the significance
                if(sum > critical_num)
                    return 0.5;
            }
            greater = false;
            return (float)(sum)/(float)num_trial;
        }
        else
        {
            unsigned int sum = 0;
            for(int j = 0;j < num_trial;++j)
            {
                std::random_shuffle(data.begin(),data.end());
                if(std::accumulate(g1,g1_end,0.0f)/(float)num1
                   -std::accumulate(g2,g2_end,0.0f)/(float)num2 > dif)
                    ++sum;
                // surely failed to achieve the significance
                if(sum > critical_num)
                    return 0.5;
            }
            greater = true;
            return (float)(sum)/(float)num_trial;
        }
    }
}

vbc::~vbc(void)
{
    terminated = true;
    threads->join_all();
}

bool vbc::load_fiber_template(const char* filename)
{
    fib_file.reset(new ODFModel);
    if(!fib_file->load_from_file(filename))
        return false;

    num_fiber = fib_file->fib_data.fib.fa.size();
    findex = fib_file->fib_data.fib.findex;
    fa = fib_file->fib_data.fib.fa;
    vertices = fib_file->fib_data.fib.odf_table;
    dim = image::geometry<3>(fib_file->fib_data.dim);

    vertices_cos.resize(vertices.size());
    for (unsigned int i = 0; i < vertices.size(); ++i)
    {
        vertices_cos[i].resize(vertices.size());
        for (unsigned int j = 0; j < vertices.size(); ++j)
            vertices_cos[i][j] = std::fabs(vertices[i]*vertices[j]);
    }
    return true;
}

const char* vbc::load_subject_data(const std::vector<std::string>& file_names,
                            unsigned int num_files1_,float qa_threshold)
{
    static std::string error_msg;
    // initialize all ODF sapces
    total_num_subjects = file_names.size();
    num_files1 = num_files1_;
    num_files2 = total_num_subjects-num_files1;
    unsigned int half_vertex_count = 0;
    {
        MatFile odf_reader;
        if(!odf_reader.load_from_file(file_names[0].c_str()))
        {
            error_msg = "failed to load file ";
            error_msg += file_names[0];
            return error_msg.c_str();
        }
        odf_bufs_size.clear();
        for (unsigned int odf_index = 0;1;++odf_index)
        {
            unsigned int row,col;
            std::ostringstream out;
            out << "odf" << odf_index;
            const float* odf_buf = 0;
            odf_reader.get_matrix(out.str().c_str(),half_vertex_count,col,odf_buf);
            if (!odf_buf)
                break;
            odf_bufs_size.push_back(half_vertex_count*col);
        }
        // initialize index mapping
        index_mapping.resize(odf_bufs_size.size());
        for(int index = 0;index < index_mapping.size();++index)
            index_mapping[index].resize(odf_bufs_size[index]/half_vertex_count);
        for(int index = 0,i1 = 0,i2 = 0;index < dim.size();++index)
            if(fa[0][index] != 0.0)
            {
                index_mapping[i1][i2] = index;
                ++i2;
                if(i2 >= index_mapping[i1].size())
                {
                    i2 = 0;
                    ++i1;
                }
            }
    }

    // initialize subject odfs
    subject_odfs.resize(odf_bufs_size.size());
    try{
        begin_prog("allocating memory");
        for(int odf_block_index = 0;check_prog(odf_block_index,subject_odfs.size());++odf_block_index)
        {
            subject_odfs[odf_block_index].
                resize(odf_bufs_size[odf_block_index]/half_vertex_count);
            for(int voxel_index = 0;
                voxel_index < subject_odfs[odf_block_index].size();++voxel_index)
            {
                subject_odfs[odf_block_index][voxel_index].resize(num_fiber);
                //unsigned int findex_location = index_mapping[odf_index+odf_block_index][voxel_index];
                unsigned int findex_location = index_mapping[odf_block_index][voxel_index];
                for(int dir_index = 0;dir_index < num_fiber;++dir_index)
                {
                    if(fa[dir_index][findex_location] <= qa_threshold)
                        break;
                    subject_odfs[odf_block_index][voxel_index][dir_index].resize(total_num_subjects);
                }
            }
        }
    }
    catch(...)
    {
        error_msg = "Insufficient Memory";
        return error_msg.c_str();
    }
    begin_prog("reading fib files");
    for(int subject_index = 0;check_prog(subject_index,total_num_subjects);++subject_index)
    {
        MatFile reader;
        if(!reader.load_from_file(file_names[subject_index].c_str()))
        {
            error_msg = "failed to load file ";
            error_msg += file_names[subject_index];
            return error_msg.c_str();
        }

        for(int odf_block_index = 0;odf_block_index < subject_odfs.size();++odf_block_index)
        {
            std::ostringstream out;
            out << "odf" << (odf_block_index);
            const float* odf_buf = 0;
            unsigned int row,col;
            reader.get_matrix(out.str().c_str(),row,col,odf_buf);
            if (!odf_buf)
                break;
            for(int voxel_index = 0;voxel_index < subject_odfs[odf_block_index].size();
                                                ++voxel_index,odf_buf += half_vertex_count)
            {
                float min_value = *std::min_element(odf_buf, odf_buf + half_vertex_count);
                //unsigned int findex_location = index_mapping[odf_index+odf_block_index][voxel_index];
                unsigned int findex_location = index_mapping[odf_block_index][voxel_index];
                for(int dir_index = 0;dir_index < num_fiber;++dir_index)
                {
                    if(subject_odfs[odf_block_index][voxel_index][dir_index].empty())
                        continue;
                    subject_odfs[odf_block_index][voxel_index][dir_index][subject_index] =
                            odf_buf[findex[dir_index][findex_location]]-min_value;
                }
            }
        }
    }
    if (prog_aborted())
        return "Aborted";
    return 0;
}


void vbc::calculate_statistics(float alpha,vbc_clustering& vbc,bool is_null) const
{
    vbc.dif.resize(num_fiber);
    vbc.t.resize(num_fiber);
    for(int index = 0;index < num_fiber;++index)
    {
        vbc.dif[index].resize(dim.size());
        std::fill(vbc.dif[index].begin(),vbc.dif[index].end(),0.0);
        vbc.t[index].resize(dim.size());
        std::fill(vbc.t[index].begin(),vbc.t[index].end(),0.0);
    }

    std::vector<float> permu(total_num_subjects);
    float n = total_num_subjects;
    float sqrt_var_S = std::sqrt(n*(n-1)*(2.0*n+5.0)/18.0);
    boost::math::normal gaussian;

    for(int odf_block_index = 0;odf_block_index < subject_odfs.size();++odf_block_index)
    for(int voxel_index = 0;voxel_index < subject_odfs[odf_block_index].size();++voxel_index)
    {
        unsigned int findex_location = index_mapping[odf_block_index][voxel_index];
        const std::vector<std::vector<float> >& subject_odf_voxel
                = subject_odfs[odf_block_index][voxel_index];
        for(unsigned char dir_index = 0;dir_index < subject_odf_voxel.size() && !terminated;++dir_index)
            if(subject_odf_voxel[dir_index].size() == total_num_subjects)
            {


                const std::vector<float>& subject_odf_voxel_dir = subject_odf_voxel[dir_index];
                permu = subject_odf_voxel_dir;
                if(is_null)
                    std::random_shuffle(permu.begin(),permu.end());

                //for(unsigned int p_index = 0;p_index < permu.size();++p_index)
                //    permu[p_index] = subject_odf_voxel_dir[mapping[p_index]];

                if(num_files1 == 1)// single subject test
                {

                    bool greater;
                    float cur_dif = 0;
                    float p_value = ::permutation_test(permu,
                        num_files1,num_files2,1000,alpha,cur_dif,greater);
                    if(p_value >= alpha)
                        continue;
                    vbc.dif[dir_index][findex_location] = greater ? cur_dif:-cur_dif;
                    vbc.t[dir_index][findex_location] = greater ?
                                -std::log(std::max(p_value,0.00000001f)):
                                std::log(std::max(p_value,0.00000001f));
                }
                else
                if(num_files2 == 0)// trend test
                {
                    int S = 0;
                    for(unsigned int i = 0;i < total_num_subjects;++i)
                        for(unsigned int j = i+1;j < total_num_subjects;++j)
                        if(permu[j] > permu[i])
                            ++S;
                        else
                            if(permu[j] < permu[i])
                                --S;
                    float Z = ((S > 0) ? (float)(S-1):(float)(S+1))/sqrt_var_S;
                    float p_value = Z < 0.0 ? boost::math::cdf(gaussian,Z):
                                        boost::math::cdf(boost::math::complement(gaussian, Z));
                    if(p_value >= alpha)
                        continue;
                    vbc.dif[dir_index][findex_location] = Z;
                    vbc.t[dir_index][findex_location] = Z;
                }
                else
                    //t-test
                {
                    double Sm1 = image::mean(permu.begin(),permu.begin()+num_files1);
                    double Sm2 = image::mean(permu.begin()+num_files1,permu.end());
                    double Sd1 = image::standard_deviation(permu.begin(),permu.begin()+num_files1,Sm1);
                    double Sd2 = image::standard_deviation(permu.begin()+num_files1,permu.end(),Sm2);
                    double Sn1 = num_files1;
                    double Sn2 = num_files2;
                    // Degrees of freedom:
                    double v = Sd1 * Sd1 / Sn1 + Sd2 * Sd2 / Sn2;
                    v *= v;
                    double t1 = Sd1 * Sd1 / Sn1;
                    t1 *= t1;
                    t1 /=  (Sn1 - 1);
                    double t2 = Sd2 * Sd2 / Sn2;
                    t2 *= t2;
                    t2 /= (Sn2 - 1);
                    v /= (t1 + t2);



                    //cout << setw(55) << left << "Degrees of Freedom" << "=  " << v << "\n";
                    // t-statistic:
                    double t_stat = (Sm1 - Sm2) / sqrt(Sd1 * Sd1 / Sn1 + Sd2 * Sd2 / Sn2);
                    //cout << setw(55) << left << "T Statistic" << "=  " << t_stat << "\n";
                    boost::math::students_t dist(v);
                    double p_value;
                    if(Sm1 > Sm2)
                    {
                        if((p_value = boost::math::cdf(boost::math::complement(dist, t_stat))) > alpha)
                            continue;
                    }
                    else
                    {
                        if((p_value = boost::math::cdf(dist, t_stat)) > alpha)
                            continue;
                    }
                    vbc.dif[dir_index][findex_location] = Sm1-Sm2;
                    vbc.t[dir_index][findex_location] = t_stat;
                }


            }
    }
}

void vbc::calculate_mapping(const char* file_name,float p_value_threshold)
{
    {
        thread_data.resize(1);

        //std::vector<unsigned short> mapping(total_num_subjects);
        //for(unsigned int index = 0;index < total_num_subjects;++index)
        //    mapping[index] = index;

        calculate_statistics(p_value_threshold,thread_data[0],false);
    }
    std::vector<std::vector<float> >& dif = thread_data[0].dif;
    std::vector<std::vector<float> >& pv = thread_data[0].t;

    std::vector<unsigned int> group_voxel_index_list;
    std::vector<unsigned int> group_id_map;
    calculate_cluster(thread_data[0],group_voxel_index_list,group_id_map);


    // regularize the cluster table
    {
        std::multiset<unsigned int> sorted_size;
        for(unsigned int index = 0;index < group_voxel_index_list.size();++index)
            if(group_voxel_index_list[index])
                sorted_size.insert(group_voxel_index_list[index]);
        std::vector<unsigned int> sorted_size_table(sorted_size.begin(),sorted_size.end());
        for(unsigned int index = 0;index < sorted_size_table.size();++index)
        {
            unsigned int cluster_id =
                std::find(group_voxel_index_list.begin(),group_voxel_index_list.end(),
                          sorted_size_table[index])-group_voxel_index_list.begin()+1;
            std::replace(group_id_map.begin(),group_id_map.end(),cluster_id,index+1);
            group_voxel_index_list[cluster_id-1] = 0;
        }
        sorted_size_table.swap(group_voxel_index_list);
    }

    MatFile& matfile = fib_file->fib_data.mat_reader;
    matfile.write_to_file(file_name);
    for(unsigned int fib = 0;fib < num_fiber;++fib)
    {
        std::string name1 = "cluster0",name2 = "t0",name3 = "dif0";
        name1[7] += fib;
        name2[1] += fib;
        name3[3] += fib;
        matfile.add_matrix(name1.c_str(),&*group_id_map.begin()+fib*dim.size(),1,dim.size());
        matfile.add_matrix(name2.c_str(),&*pv[fib].begin(),1,pv[fib].size());
        matfile.add_matrix(name3.c_str(),&*dif[fib].begin(),1,dif[fib].size());
    }

    // single threshold
    /*
    {
        std::sort(max_statistics.begin(),max_statistics.end(),std::greater<float>());
        float critical_statistics = max_statistics[p_value_threshold*max_statistics.size()];

        for(unsigned int fib = 0;fib < num_fiber;++fib)
        {
            std::string name2 = "t_st0",name3 = "dif_st0";
            name2[4] += fib;
            name3[6] += fib;
            matfile.add_matrix(name2.c_str(),&*pv_t[fib].begin(),1,pv_t[fib].size());
            matfile.add_matrix(name3.c_str(),&*dif_t[fib].begin(),1,dif_t[fib].size());
        }
        matfile.add_matrix("critical_statistics",&critical_statistics,1,1);
        matfile.add_matrix("max_statistics",&*max_statistics.begin(),1,max_statistics.size());

    }
    */
    // supratheshold clustering
    {
        std::sort(max_cluster_size.begin(),max_cluster_size.end(),std::greater<float>());
        //float critical_cluster_size = max_cluster_size[p_value_threshold*max_cluster_size.size()/group_voxel_index_list.size()];

        std::vector<float> cluster_pv(group_voxel_index_list.size());
        for(int index = 0;index < cluster_pv.size();++index)
        {
            int rank = 0;
            for(;rank < max_cluster_size.size();++rank)
                if(max_cluster_size[rank] < group_voxel_index_list[index])
                    break;
            cluster_pv[index] = (float)(rank)/(float)(max_cluster_size.size());
            cluster_pv[index] *= (float)group_voxel_index_list.size();
        }
        std::vector<float> sum_dif(group_voxel_index_list.size());
        {
            std::vector<float> sum_mean(group_voxel_index_list.size());
            for(unsigned int fib = 0;fib < num_fiber;++fib)
            {
                unsigned int shift = fib*dim.size();
                for(unsigned int index = 0;index < dim.size();++index)
                    if(group_id_map[index+shift] == 0)
                        continue;
                    else
                    {
                        sum_dif[group_id_map[index+shift]-1] += dif[fib][index];
                        sum_mean[group_id_map[index+shift]-1] += fa[fib][index];
                    }
            }
            for(int index = 0;index < sum_dif.size();++index)
                if(sum_mean[index] == 0)
                    sum_dif[index] = 0;
            else
                    sum_dif[index] *= 100.0/sum_mean[index];

        }


        if(cluster_pv.back() < p_value_threshold)
        {

            std::vector<std::vector<float> > greater(num_fiber),lesser(num_fiber);


            for(unsigned int fib = 0;fib < num_fiber;++fib)
            {
                greater[fib].resize(dif[fib].size());
                lesser[fib].resize(dif[fib].size());
                unsigned int shift = fib*dim.size();
                for(unsigned int index = 0;index < dim.size();++index)
                    if(group_id_map[index+shift] == 0)
                        continue;
                    else
                    {
                        int group_id = group_id_map[index+shift]-1;
                        if(cluster_pv[group_id] > p_value_threshold)
                        {
                            dif[fib][index] = 0;
                            pv[fib][index] = 0;
                        }
                        else
                        {
                            pv[fib][index] = 1.0/std::max<float>(cluster_pv[group_id],0.0001);
                            dif[fib][index] = sum_dif[group_id];
                            if(dif[fib][index] > 0)
                                greater[fib][index] = dif[fib][index];
                            else
                                lesser[fib][index] = -dif[fib][index];
                        }
                    }
            }


            for(unsigned int fib = 0;fib < num_fiber;++fib)
            {
                std::string name2 = "pv_sc0",name3 = "dif_sc0",name4 = "greater0",name5 = "lesser0";
                name2[5] += fib;
                name3[6] += fib;
                name4[7] += fib;
                name5[6] += fib;
                matfile.add_matrix(name2.c_str(),&*pv[fib].begin(),1,pv[fib].size());
                matfile.add_matrix(name3.c_str(),&*dif[fib].begin(),1,dif[fib].size());
                matfile.add_matrix(name4.c_str(),&*greater[fib].begin(),1,greater[fib].size());
                matfile.add_matrix(name5.c_str(),&*lesser[fib].begin(),1,lesser[fib].size());

            }
        }
        matfile.add_matrix("cluster_size",&*group_voxel_index_list.begin(),1,group_voxel_index_list.size());
        matfile.add_matrix("max_cluster_size",&*max_cluster_size.begin(),1,max_cluster_size.size());

    }
    matfile.close_file();
}

/*
void vbc::run_tracking(void)
{
    float param[8] = {1,60,60,0.03,0.0,10.0,500.0};
    param[0] = fib_file->fib_data.vs[0]/2.0; //step size
    param[2] = param[1] = 60.0; // turning angle
    param[1] *= 3.1415926/180.0;
    param[2] *= 3.1415926/180.0;
    param[3] = 0.00001; //vm["fa_threshold"].as<float>();
    param[4] = 0.0; // vm["smoothing"].as<float>();
    param[5] = 0.0; // vm["min_length"].as<float>();
    param[6] = 300.0;//vm["max_length"].as<float>();
    unsigned int termination_count = 10000;
    unsigned char methods[5];
    methods[0] = 0;//vm["method"].as<int>();
    methods[1] = 0;//vm["initial_dir"].as<int>();
    methods[2] = 0;//vm["interpolation"].as<int>();
    methods[3] = 1;//stop_by_track;
    methods[4] = 0;//vm["seed_plan"].as<int>();
    std::auto_ptr<ThreadData> thread_handle(
            ThreadData::new_thread(fib_file.get(),param,methods,termination_count));

    {
        std::vector<image::vector<3,short> > seed;
        for(image::pixel_index<3> index;index.valid(dim);index.next(dim))
            if(fa[0][index.index()] > 0.0)
                seed.push_back(image::vector<3,short>(index.x(),index.y(),index.z()));
        thread_handle->setRegions(seed,3);
    }
    thread_handle->run_until_terminate(1);// no multi-thread
    tract_model.reset(new TractModel(fib_file.get(),dim,fib_file->fib_data.vs));
    thread_handle->fetchTracks(tract_model.get());
}
*/

/**
  group_voxel_index_list records the voxel index of each group
  group_id_map the mapping of the group id
  */
void vbc::calculate_cluster(
        const vbc_clustering& data,
        std::vector<unsigned int>& group_voxel_index_list,
        std::vector<unsigned int>& group_id_map)
{
    std::vector<unsigned int> shift(num_fiber);
    for(unsigned int index = 1;index < num_fiber;++index)
        shift[index] = dim.size()*index;

    image::disjoint_set dset;
    dset.label.resize(num_fiber*dim.size());
    dset.rank.resize(num_fiber*dim.size());
    for(unsigned int index = 0;index < dim.size();++index)
    {
        for(unsigned char fib = 0;fib < num_fiber;++fib)
            if(data.dif[fib][index] != 0.0)
                dset.label[index+shift[fib]] = index+shift[fib];
    }

    std::vector<image::pixel_index<3> > neighbour;
    for(unsigned char fib = 0;fib < num_fiber;++fib)
    {
        for(image::pixel_index<3> index;dim.is_valid(index);index.next(dim))
        {
            float cur_dif = data.dif[fib][index.index()];
            if(cur_dif == 0.0)
                continue;
            image::get_neighbors(index,dim,neighbour);
            neighbour.push_back(index);
            unsigned int cur_set_pos = index.index()+shift[fib];
            short main_dir = findex[fib][index.index()];

            for(unsigned int i = 0;i < neighbour.size();++i)
            {
                for(unsigned char j = 0;j < num_fiber;++j)
                    if(((cur_dif > 0.0 && data.dif[j][neighbour[i].index()] > 0.0) ||
                        (cur_dif < 0.0 && data.dif[j][neighbour[i].index()] < 0.0)) &&
                        vertices_cos[main_dir][findex[j][neighbour[i].index()]] > angle_threshold_cos)
                            dset.join_set(dset.find_set(cur_set_pos),
                                          dset.find_set(neighbour[i].index()+shift[j]));
            }
        }
    }

    for(unsigned int index = 0;index < dset.label.size();++index)
    {
        if(!dset.label[index])
            continue;
        unsigned int set = dset.find_set(index);
        if(set > group_voxel_index_list.size())
            group_voxel_index_list.resize(set);
        ++group_voxel_index_list[set-1];
    }
    group_id_map.swap(dset.label);
}

void vbc::run_thread(unsigned int thread_count,unsigned int thread_id,unsigned int permutation_num,float alpha)
{
    //std::vector<unsigned short> mapping(total_num_subjects);
    //for(unsigned int index = 0;index < total_num_subjects;++index)
    //    mapping[index] = index;

    for(unsigned int per_index = thread_id;per_index < permutation_num;per_index += thread_count,++cur_prog)
    {
        // if single subject
        /*
        if(num_files1 == 1)
        {
            for(unsigned int index = 0;index < total_num_subjects;++index)
                mapping[index] = index;
            std::swap(mapping[0],mapping[per_index]);
        }
        else
            std::random_shuffle(mapping.begin(),mapping.end());
            */

        calculate_statistics(alpha,thread_data[thread_id],true);


        // supra-threshod
        std::vector<unsigned int> group_voxel_index_list;
        std::vector<unsigned int> group_id_map;

        calculate_cluster(thread_data[thread_id],group_voxel_index_list,group_id_map);
        //calculate the size of each cluster
        {
            boost::mutex::scoped_lock lock(lock_function);
            // single threshold
            max_statistics.push_back(
                    *std::max_element(thread_data[thread_id].t[0].begin(),thread_data[thread_id].t[0].end()));
            max_statistics.push_back(
                    -*std::min_element(thread_data[thread_id].t[0].begin(),thread_data[thread_id].t[0].end()));

            /*
            if(max_cluster_size.empty() ||
               *std::max_element(max_cluster_size.begin(),max_cluster_size.end()) <
               *std::max_element(group_voxel_index_list.begin(),group_voxel_index_list.end()))
                max_mapping = mapping;
                */

            if(!group_voxel_index_list.empty())
                max_cluster_size.push_back(*std::max_element(group_voxel_index_list.begin(),group_voxel_index_list.end()));
            else
                max_cluster_size.push_back(0);
        }
    }
}

void vbc::calculate_permutation(unsigned int thread_count,unsigned int permutation_num,float alpha)
{
    terminated = false;
    threads.reset(new boost::thread_group);
    thread_data.clear();
    thread_data.resize(thread_count);
    // if single subject
    if(num_files1 == 1)
        permutation_num = num_files2;
    total_prog = permutation_num;
    cur_prog = 0;
    max_cluster_size.clear();
    max_statistics.clear();
    for (unsigned int thread_id = 0;thread_id < thread_count;++thread_id)
        threads->add_thread(new boost::thread(&vbc::run_thread,this,thread_count,thread_id,permutation_num,alpha));
}
