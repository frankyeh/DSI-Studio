#include "vbc.hpp"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include <boost/math/distributions/students_t.hpp>


float permutation_test(std::vector<float>& data,
                       unsigned int num1,unsigned num2,
                       unsigned int num_trial,
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
        unsigned int critical_num = 0.5*num_trial;
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
    if(threads.get())
        threads->join_all();
}


bool vbc::load_fiber_template(const char* filename)
{
    fib_file.reset(new ODFModel);
    if(!fib_file->load_from_file(filename))
        return false;

    dim = image::geometry<3>(fib_file->fib_data.dim);
    num_fiber = fib_file->fib_data.fib.fa.size();
    findex.resize(num_fiber);
    fa.resize(num_fiber);
    for(unsigned int index = 0;index < num_fiber;++index)
    {
        findex[index].resize(dim.size());
        fa[index].resize(dim.size());
        std::copy(fib_file->fib_data.fib.findex[index],
                  fib_file->fib_data.fib.findex[index]+dim.size(),
                  findex[index].begin());
        std::copy(fib_file->fib_data.fib.fa[index],
                  fib_file->fib_data.fib.fa[index]+dim.size(),
                  fa[index].begin());
    }
    vertices = fib_file->fib_data.fib.odf_table;
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
                                   unsigned int num_files1_)
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
                    if(fa[dir_index][findex_location] == 0.0)
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
        float max_iso = 0.0;
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
                if(min_value > max_iso)
                    max_iso = min_value;
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

        // normalization
        if(max_iso + 1.0 != 1.0)
        for(int odf_block_index = 0;odf_block_index < subject_odfs.size();++odf_block_index)
        {
            for(int voxel_index = 0;voxel_index < subject_odfs[odf_block_index].size();++voxel_index)
            {
                for(int dir_index = 0;dir_index < subject_odfs[odf_block_index][voxel_index].size();++dir_index)
                    if(!subject_odfs[odf_block_index][voxel_index][dir_index].empty())
                        subject_odfs[odf_block_index][voxel_index][dir_index][subject_index] /= max_iso;
            }
        }
    }
    if (prog_aborted())
        return "Aborted";
    return 0;
}


void vbc::calculate_statistics(float qa_threshold,std::vector<std::vector<float> >& vbc,unsigned int is_null) const
{
    vbc.resize(num_fiber);
    for(int index = 0;index < num_fiber;++index)
    {
        vbc[index].resize(dim.size());
        std::fill(vbc[index].begin(),vbc[index].end(),0.0);
    }

    std::vector<float> permu(total_num_subjects);
    std::vector<unsigned short> mapping(total_num_subjects);
    float n = total_num_subjects;
    float sqrt_var_S = std::sqrt(n*(n-1)*(2.0*n+5.0)/18.0);
    //boost::math::normal gaussian;
    if(is_null)
    {
        for(unsigned int index = 0;index < n;++index)
            mapping[index] = index;
        if(num_files1 == 1)
            std::swap(mapping[0],mapping[is_null]);
        else
            std::random_shuffle(mapping.begin(),mapping.end());
    }

    for(int odf_block_index = 0;odf_block_index < subject_odfs.size();++odf_block_index)
    for(int voxel_index = 0;voxel_index < subject_odfs[odf_block_index].size();++voxel_index)
    {
        unsigned int findex_location = index_mapping[odf_block_index][voxel_index];
        const std::vector<std::vector<float> >& subject_odf_voxel
                = subject_odfs[odf_block_index][voxel_index];
        for(unsigned char dir_index = 0;dir_index < subject_odf_voxel.size() && !terminated;++dir_index)
            if(subject_odf_voxel[dir_index].size() == total_num_subjects)
            {
                if(fa[dir_index][findex_location] < qa_threshold)
                    break;
                const std::vector<float>& subject_odf_voxel_dir = subject_odf_voxel[dir_index];
                permu = subject_odf_voxel_dir;
                if(is_null)
                    for(unsigned int p_index = 0;p_index < permu.size();++p_index)
                        permu[p_index] = subject_odf_voxel_dir[mapping[p_index]];

                if(num_files1 == 1)// single subject test
                {
                    if(permu[0] == 0.0)
                        break;
                    unsigned int rank = 1;
                    for(unsigned int index = 1;index < permu.size();++index)
                        if(permu[index] > permu[0])
                            ++rank;
                    double percentile = (double)rank/(double)(permu.size()+1);
                    vbc[dir_index][findex_location] = 1.0/(percentile > 0.5 ? 1.0-percentile:-percentile);
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
                    //float p_value = Z < 0.0 ? boost::math::cdf(gaussian,Z):
                    //                    boost::math::cdf(boost::math::complement(gaussian, Z));
                    vbc[dir_index][findex_location] = Z;

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
                    /*
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
                    */
                    vbc[dir_index][findex_location] = t_stat;
                }
        }
    }
}
void vbc::set_fib(bool greater,const std::vector<std::vector<float> >& t)
{

    std::vector<short*> fib_index(num_fiber);
    std::vector<float*> fib_fa(num_fiber);


    for(unsigned int fib = 0;fib < num_fiber;++fib)
    {
        fib_fa[fib] = (float*)fib_file->fib_data.fib.fa[fib];
        fib_index[fib] = (short*)fib_file->fib_data.fib.findex[fib];
    }

    for(unsigned int index = 0;index < dim.size();++index)
    {
        std::map<float,short,std::greater<float> > fmap;
        for(unsigned int fib = 0;fib < num_fiber;++fib)
        {
            fib_fa[fib][index] = 0;
            fib_index[fib][index] = 0;
            if(greater && t[fib][index] > 0)
                fmap[t[fib][index]] = findex[fib][index];
            if(!greater && t[fib][index] < 0)
                fmap[-t[fib][index]] = findex[fib][index];
        }
        std::map<float,short,std::greater<float> >::const_iterator iter = fmap.begin();
        std::map<float,short,std::greater<float> >::const_iterator end = fmap.end();
        for(unsigned int fib = 0;iter != end;++iter,++fib)
        {
            fib_fa[fib][index] = iter->first;
            fib_index[fib][index] = iter->second;
        }

    }

}

void vbc::output_greater_lesser_mapping(const char* file_name,float qa_threshold)
{

    MatFile& matfile = fib_file->fib_data.mat_reader;
    std::vector<std::vector<float> > data;
    terminated = false;
    calculate_statistics(qa_threshold,data,0);

    // set greater mapping
    set_fib(true,data);
    matfile.write_to_file((std::string(file_name)+".greater.fib.gz").c_str());
    matfile.close_file();

    for(unsigned int fib = 0;fib < num_fiber;++fib)
    {
        std::string name = "t0";
        name[1] += fib;
        matfile.add_matrix(name.c_str(),&*data[fib].begin(),1,data[fib].size());
    }

    // set lesser mapping
    set_fib(false,data);
    matfile.write_to_file((std::string(file_name)+".lesser.fib.gz").c_str());
    matfile.close_file();
    for(unsigned int fib = 0;fib < num_fiber;++fib)
    {
        std::string name = "t0";
        name[1] += fib;
        matfile.add_matrix(name.c_str(),&*data[fib].begin(),1,data[fib].size());
    }
}


void vbc::run_tracking(float t_threshold,std::vector<std::vector<float> > &tracts)
{
    float param[8] = {1,60,60,0.03,0.0,10.0,500.0};
    param[0] = fib_file->fib_data.vs[0]/2.0; //step size
    param[2] = param[1] = 60.0; // turning angle
    param[1] *= 3.1415926/180.0;
    param[2] *= 3.1415926/180.0;
    param[3] = t_threshold; //vm["fa_threshold"].as<float>();
    param[4] = 0.0; // vm["smoothing"].as<float>();
    param[5] = 0.0; // vm["min_length"].as<float>();
    param[6] = 300.0;//vm["max_length"].as<float>();
    unsigned int termination_count = 10000;
    unsigned char methods[5];
    methods[0] = 0;//vm["method"].as<int>();
    methods[1] = 0;//vm["initial_dir"].as<int>();
    methods[2] = 0;//vm["interpolation"].as<int>();
    methods[3] = 0;//stop_by_seed;
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
    thread_handle->track_buffer.swap(tracts);
}

void vbc::run_thread(unsigned int thread_count,unsigned int thread_id,unsigned int permutation_num,
                     float qa_threshold,float t_threshold)
{

    for(unsigned int per_index = thread_id+1;!terminated &&
            per_index <= permutation_num;per_index += thread_count,++cur_prog)
    {
        std::vector<std::vector<float> > data;
        calculate_statistics(qa_threshold,data,per_index);

        // do tracking
        {
            boost::mutex::scoped_lock lock(lock_function);
            std::vector<std::vector<float> > tracts;
            set_fib(true,data);
            run_tracking(t_threshold,tracts);
            for(unsigned int index = 0;index < tracts.size();++index)
                length_dist[std::min<unsigned int>(tracts[index].size()/3,max_length-1)]++;
            set_fib(false,data);
            run_tracking(t_threshold,tracts);
            for(unsigned int index = 0;index < tracts.size();++index)
                length_dist[std::min<unsigned int>(tracts[index].size()/3,max_length-1)]++;

        }
    }

}

void vbc::calculate_null(unsigned int thread_count,
                                unsigned int permutation_num,
                                float qa_threshold,float t_threshold)
{
    length_dist.clear();
    length_dist.resize(max_length);
    // if single subject
    if(num_files1 == 1)
        permutation_num = num_files2;
    total_prog = permutation_num;
    cur_prog = 0;
    terminated = false;
    threads.reset(new boost::thread_group);
    for (unsigned int thread_id = 0;thread_id < thread_count;++thread_id)
        threads->add_thread(new boost::thread(
            &vbc::run_thread,this,thread_count,thread_id,permutation_num,qa_threshold,t_threshold));
}
void vbc::fdr_select_tracts(float fdr,std::vector<std::vector<float> > &tracts)
{
    std::vector<double> p_value(length_dist.size());
    {
        double sum = std::accumulate(length_dist.begin(),length_dist.end(),0.0);
        double cdf = 0.0;
        if(sum == 0.0)
        {
            tracts.clear();
            return;
        }
        for(unsigned int index = 0;index < p_value.size();++index)
        {
            p_value[index] = (sum-cdf)/sum;
            cdf += length_dist[index];
        }
    }
    std::vector<float> tract_p_values(tracts.size());
    for(unsigned int index = 0;index < tracts.size();++index)
        tract_p_values[index] = p_value[std::min<unsigned int>(tracts[index].size(),p_value.size()-1)];

    std::sort(tract_p_values.begin(),tract_p_values.end(),std::greater<float>());
    unsigned int k = 0;
    float critical_p_value = 0.0;
    for(;k < tract_p_values.size();++k)
    {
        std::cout << " k=" << k <<
                     " p_value=" << tract_p_values[k] <<
                     " fdr=" << fdr*(float)(tract_p_values.size()-k)/(float)tracts.size() << std::endl;
        if(tract_p_values[k] < fdr*(float)(tract_p_values.size()-k)/(float)tracts.size())
        {
            critical_p_value = tract_p_values[k];
            break;
        }
    }
    if(critical_p_value == 0.0)
    {
        tracts.clear();
        return;
    }
    std::vector<std::vector<float> > selected_tracts;
    for(unsigned int index = 0;index < tracts.size();++index)
        if(p_value[std::min<unsigned int>(tracts[index].size(),p_value.size()-1)] <= critical_p_value)
        {
            selected_tracts.push_back(std::vector<float>());
            selected_tracts.back().swap(tracts[index]);
        }
    selected_tracts.swap(tracts);
}

bool vbc::fdr_tracking(const char* file_name,float qa_threshold,float t_threshold,float fdr,bool greater)
{
    std::vector<std::vector<float> > data;
    calculate_statistics(qa_threshold,data,0);
    std::vector<std::vector<float> > tracts;
    set_fib(greater,data);
    run_tracking(t_threshold,tracts);
    fdr_select_tracts(fdr,tracts);
    if(tracts.empty())
        return false;
    TractModel tract_model(fib_file.get(),fib_file->fib_data.dim,fib_file->fib_data.vs);
    tract_model.add_tracts(tracts);
    tract_model.save_tracts_to_file(file_name);
    return true;
}
