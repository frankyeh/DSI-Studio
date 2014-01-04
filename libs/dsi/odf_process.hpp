#ifndef ODF_TRANSFORMATION_PROCESS_HPP
#define ODF_TRANSFORMATION_PROCESS_HPP
#include <boost/lambda/lambda.hpp>
#include "basic_process.hpp"
#include "basic_voxel.hpp"

class ReadDWIData : public BaseProcess{
public:
    virtual void init(Voxel&) {}
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        data.space.resize(voxel.image_model->dwi_data.size());
        for (unsigned int index = 0; index < data.space.size(); ++index)
            data.space[index] = voxel.image_model->dwi_data[index][data.voxel_index];
    }
    virtual void end(Voxel&,gz_mat_write& mat_writer) {}
};

class BalanceScheme : public BaseProcess{
    std::vector<float> trans;
    unsigned int new_q_count;
private:
    Voxel* stored_voxel;
    unsigned int old_q_count;
    std::vector<image::vector<3,float> > old_bvectors;
    std::vector<float> old_bvalues;
public:
    BalanceScheme(void):stored_voxel(0){}

    virtual void init(Voxel& voxel)
    {
        if(!voxel.scheme_balance)
            return;
        std::vector<unsigned int> shell;
        shell.push_back(0);
        unsigned int b_count = voxel.q_count;
        for(unsigned int index = 1;index < b_count;++index)
        {
            if(std::abs(voxel.bvalues[index]-voxel.bvalues[index-1]) > 100)
                shell.push_back(index);
        }
        if(shell.size() > 4) // more than 3 shell
        {
            voxel.scheme_balance = false;
            return;
        }

        unsigned int total_signals = 0;

        tessellated_icosahedron new_dir;
        new_dir.init(6);

        std::vector<image::vector<3,float> > new_bvectors;
        std::vector<float> new_bvalues;

        for(unsigned int shell_index = 0;shell_index < shell.size();++shell_index)
        {
            unsigned int from = shell[shell_index];
            unsigned int to = (shell_index + 1 == shell.size() ? b_count:shell[shell_index+1]);
            unsigned int num = to-from;
            // if there is b0, average them
            if(shell_index == 0 && voxel.bvalues.front() < 100)
            {
                trans.resize(num*b_count);
                new_bvectors.resize(num);
                new_bvalues.resize(num);
                float weighting = 1.0/(float)num;
                for(unsigned int index = 0;index < num;++index)
                    trans[index+index*b_count] = weighting;
                total_signals += num;

                continue;
            }

            //calculate averaged angle distance
            double averaged_angle = 0.0;
            for(unsigned int i = from;i < to;++i)
            {
                double max_cos = 0.0;
                for(unsigned int j = from;j < to;++j)
                {
                    if(i == j)
                        continue;
                    double cur_cos = std::fabs(std::cos(voxel.bvectors[i]*voxel.bvectors[j]));
                    if(cur_cos > 0.998)
                        continue;
                    max_cos = std::max<double>(max_cos,cur_cos);
                }
                averaged_angle += std::acos(max_cos);
            }
            averaged_angle /= num;

            //calculate averaged b_value
            double avg_b = image::mean(voxel.bvalues.begin()+from,voxel.bvalues.begin()+to);
            unsigned int trans_old_size = trans.size();
            trans.resize(trans.size() + new_dir.half_vertices_count*b_count);
            for(unsigned int i = 0; i < new_dir.half_vertices_count;++i)
            {
                std::vector<double> t(b_count);
                for(unsigned int j = from;j < to;++j)
                {
                    double angle = std::acos(std::min<double>(1.0,std::fabs(new_dir.vertices[i]*voxel.bvectors[j])));
                    angle/=averaged_angle;
                    t[j] = std::exp(-2.0*angle*angle); // if the angle == 1, then weighting = 0.135
                }
                double w = avg_b/1000.0;
                w /= std::accumulate(t.begin(),t.end(),0.0);
                std::for_each(t.begin(),t.end(),boost::lambda::_1 *= w);
                std::copy(t.begin(),t.end(),trans.begin() + trans_old_size + i * b_count);
                new_bvalues.push_back(avg_b);
                new_bvectors.push_back(new_dir.vertices[i]);
            }
            total_signals += new_dir.half_vertices_count;
        }

        old_q_count = voxel.q_count;
        voxel.q_count = new_q_count = total_signals;
        voxel.bvalues.swap(new_bvalues);
        voxel.bvectors.swap(new_bvectors);
        new_bvalues.swap(old_bvalues);
        new_bvectors.swap(old_bvectors);
        stored_voxel = &voxel;

    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {

        if(!voxel.scheme_balance)
            return;
        if(stored_voxel)
        {
            stored_voxel = 0;
            voxel.q_count = old_q_count;
            voxel.bvalues = old_bvalues;
            voxel.bvectors = old_bvectors;
        }
        std::vector<float> new_data(new_q_count);
        data.space.swap(new_data);
        image::matrix::vector_product(trans.begin(),new_data.begin(),data.space.begin(),image::dyndim(new_q_count,old_q_count));
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
    }
};

struct GeneralizedFA
{
    float operator()(const std::vector<float>& odf)
    {
        float m1 = 0.0;
        float m2 = 0.0;
        std::vector<float>::const_iterator iter = odf.begin();
        std::vector<float>::const_iterator end = odf.end();
        for (;iter != end; ++iter)
        {
            float t = *iter;
            m1 += t;
            m2 += t*t;
        }
        m1 *= m1;
        m1 /= ((float)odf.size());
        if (m2 == 0.0)
            return 0.0;
        return std::sqrt(((float)odf.size())/((float)odf.size()-1.0)*(m2-m1)/m2);
    }
};

const unsigned int odf_block_size = 20000;
struct OutputODF : public BaseProcess
{
protected:
    std::vector<std::vector<float> > odf_data;
    std::vector<unsigned int> odf_index_map;
public:
    virtual void init(Voxel& voxel)
    {
        odf_data.clear();
        if (voxel.need_odf)
        {
            unsigned int total_count = 0;
            odf_index_map.resize(voxel.image_model->mask.size());
            for (unsigned int index = 0;index < voxel.image_model->mask.size();++index)
                if (voxel.image_model->mask[index])
                {
                    odf_index_map[index] = total_count;
                    ++total_count;
                }
            try
            {
                std::vector<unsigned int> size_list;
                while (1)
                {

                    if (total_count > odf_block_size)
                    {
                        size_list.push_back(odf_block_size);
                        total_count -= odf_block_size;
                    }
                    else
                    {
                        size_list.push_back(total_count);
                        break;
                    }
                }
                odf_data.resize(size_list.size());
                for (unsigned int index = 0;index < odf_data.size();++index)
                    odf_data[index].resize(size_list[index]*(voxel.ti.half_vertices_count));
            }
            catch (...)
            {
                odf_data.clear();
                voxel.need_odf = false;
            }
        }

    }
    virtual void run(Voxel& voxel,VoxelData& data)
    {

        if (voxel.need_odf && data.fa[0] + 1.0 != 1.0)
        {
            unsigned int odf_index = odf_index_map[data.voxel_index];
            std::copy(data.odf.begin(),data.odf.end(),
                      odf_data[odf_index/odf_block_size].begin() + (odf_index%odf_block_size)*(voxel.ti.half_vertices_count));
        }

    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {

        if (!voxel.need_odf)
            return;
        {
            set_title("output odfs");
            for (unsigned int index = 0;index < odf_data.size();++index)
            {
                if (!voxel.odf_deconvolusion)
                    std::for_each(odf_data[index].begin(),odf_data[index].end(),boost::lambda::_1 /= voxel.z0);
                std::ostringstream out;
                out << "odf" << index;
                mat_writer.write(out.str().c_str(),&*odf_data[index].begin(),
                                      voxel.ti.half_vertices_count,
                                      odf_data[index].size()/(voxel.ti.half_vertices_count));
            }
            odf_data.clear();
        }

    }
};



struct AccumulateODF : public BaseProcess
{
    std::vector<unsigned int> odf_index_map;
public:
    virtual void init(Voxel& voxel)
    {
        std::vector<std::vector<float> >& odf_data = voxel.template_odfs;

        unsigned int total_count = 0;
        odf_index_map.resize(voxel.image_model->mask.size());
        for (unsigned int index = 0;index < voxel.image_model->mask.size();++index)
            if (voxel.image_model->mask[index])
            {
                odf_index_map[index] = total_count;
                ++total_count;
            }
        if(odf_data.empty())
        {
            std::vector<unsigned int> size_list;
            while (1)
            {
                if (total_count > odf_block_size)
                {
                    size_list.push_back(odf_block_size);
                    total_count -= odf_block_size;
                }
                else
                {
                    size_list.push_back(total_count);
                    break;
                }
            }
            odf_data.resize(size_list.size());
            for (unsigned int index = 0;index < odf_data.size();++index)
                odf_data[index].resize(size_list[index]*(voxel.ti.half_vertices_count));
        }

    }
    virtual void run(Voxel& voxel,VoxelData& data)
    {
        std::for_each(data.odf.begin(),data.odf.end(),boost::lambda::_1 /= voxel.z0);
        unsigned int odf_index = odf_index_map[data.voxel_index];
        image::add(voxel.template_odfs[odf_index/odf_block_size].begin() + (odf_index%odf_block_size)*(voxel.ti.half_vertices_count),
                   voxel.template_odfs[odf_index/odf_block_size].begin() + (odf_index%odf_block_size+1)*(voxel.ti.half_vertices_count),
                   data.odf.begin());
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
    }
};

struct ODFLoader : public BaseProcess
{
    std::vector<unsigned char> index_mapping1;
    std::vector<unsigned int> index_mapping2;
public:
    virtual void init(Voxel& voxel)
    {
        //voxel.qa_scaling must be 1
        voxel.z0 = 0.0;
        index_mapping1.resize(voxel.image_model->mask.size());
        index_mapping2.resize(voxel.image_model->mask.size());
        int voxel_index = 0;
        for(unsigned char i = 0;i < voxel.template_odfs.size();++i)
        {
            for(unsigned int j = 0;j < voxel.template_odfs[i].size();j += voxel.ti.half_vertices_count)
            {
                int k_end = j + voxel.ti.half_vertices_count;
                bool is_odf_zero = true;
                for(int k = j;k < k_end;++k)
                    if(voxel.template_odfs[i][k] != 0.0)
                    {
                        is_odf_zero = false;
                        break;
                    }
                if(!is_odf_zero)
                    for(;voxel_index < index_mapping1.size();++voxel_index)
                        if(voxel.image_model->mask[voxel_index] != 0)
                            break;
                if(voxel_index >= index_mapping1.size())
                    break;
                index_mapping1[voxel_index] = i;
                index_mapping2[voxel_index] = j;
                ++voxel_index;
            }
        }
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        int cur_index = data.voxel_index;
        std::copy(voxel.template_odfs[index_mapping1[cur_index]].begin() +
                  index_mapping2[cur_index],
                  voxel.template_odfs[index_mapping1[cur_index]].begin() +
                  index_mapping2[cur_index]+data.odf.size(),data.odf.begin());

    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        if (voxel.need_odf)
        {
            set_title("output odfs");
            for (unsigned int index = 0;index < voxel.template_odfs.size();++index)
            {
                std::ostringstream out;
                out << "odf" << index;
                mat_writer.write(out.str().c_str(),&*voxel.template_odfs[index].begin(),
                                      voxel.ti.half_vertices_count,
                                      voxel.template_odfs[index].size()/(voxel.ti.half_vertices_count));
            }
        }
        mat_writer.write("trans",&*voxel.param,4,4);
    }
};

// for normalization
class RecordQA  : public BaseProcess
{
public:
    virtual void init(Voxel& voxel)
    {
        voxel.qa_map.resize(voxel.dim);
        std::fill(voxel.qa_map.begin(),voxel.qa_map.end(),0.0);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        voxel.qa_map[data.voxel_index] = data.fa[0];
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {

    }
};

struct SaveFA : public BaseProcess
{
protected:
    std::vector<float> gfa,iso,sum;
    std::vector<std::vector<float> > fa;
public:
    virtual void init(Voxel& voxel)
    {

        fa.resize(voxel.max_fiber_number);
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            fa[index].resize(voxel.dim.size());
        gfa.clear();
        gfa.resize(voxel.dim.size());
        iso.clear();
        iso.resize(voxel.dim.size());
        sum.clear();
        sum.resize(voxel.dim.size());
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        iso[data.voxel_index] = data.min_odf;
        sum[data.voxel_index] = data.sum_odf;
        gfa[data.voxel_index] = GeneralizedFA()(data.odf);
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            fa[index][data.voxel_index] = data.fa[index];
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        set_title("output gfa");
        mat_writer.write("gfa",&*gfa.begin(),1,gfa.size());

        if(!voxel.odf_deconvolusion && voxel.z0 + 1.0 != 1.0)
        {
            mat_writer.write("z0",&voxel.z0,1,1);
            std::for_each(iso.begin(),iso.end(),boost::lambda::_1 /= voxel.z0);
            std::for_each(sum.begin(),sum.end(),boost::lambda::_1 /= voxel.z0);
            for (unsigned int i = 0;i < voxel.max_fiber_number;++i)
                std::for_each(fa[i].begin(),fa[i].end(),boost::lambda::_1 /= voxel.z0);
        }

        mat_writer.write("iso",&*iso.begin(),1,iso.size());
        mat_writer.write("sum",&*sum.begin(),1,sum.size());

        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
        {
            std::ostringstream out;
            out << index;
            std::string num = out.str();
            std::string fa_str = "fa";
            fa_str += num;
            set_title(fa_str.c_str());
            mat_writer.write(fa_str.c_str(),&*fa[index].begin(),1,fa[index].size());
        }

        // output normalized qa
        {
            float max_qa = 0.0;
            for (unsigned int i = 0;i < voxel.max_fiber_number;++i)
                max_qa = std::max<float>(*std::max_element(fa[i].begin(),fa[i].end()),max_qa);

            if(max_qa != 0.0)
            for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            {
                std::for_each(fa[index].begin(),fa[index].end(),boost::lambda::_1 /= max_qa);
                std::ostringstream out;
                out << index;
                std::string num = out.str();
                std::string fa_str = "nqa";
                fa_str += num;
                set_title(fa_str.c_str());
                mat_writer.write(fa_str.c_str(),&*fa[index].begin(),1,fa[index].size());
            }
        }

    }
};



struct SaveDirIndex : public BaseProcess
{
protected:
    std::vector<std::vector<short> > findex;
public:
    virtual void init(Voxel& voxel)
    {

        findex.resize(voxel.max_fiber_number);
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            findex[index].resize(voxel.dim.size());
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {

        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            findex[index][data.voxel_index] = data.dir_index[index];
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
        {
            std::ostringstream out;
            out << index;
            std::string num = out.str();
            std::string index_str = "index";
            index_str += num;
            set_title(index_str.c_str());
            mat_writer.write(index_str.c_str(),&*findex[index].begin(),1,findex[index].size());
        }
    }
};


struct SaveDir : public BaseProcess
{
protected:
    std::vector<std::vector<float> > dir;
public:
    virtual void init(Voxel& voxel)
    {

        dir.resize(voxel.max_fiber_number);
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            dir[index].resize(voxel.dim.size()*3);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {

        unsigned int dir_index = data.voxel_index;
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            std::copy(data.dir[index].begin(),data.dir[index].end(),dir[index].begin() + dir_index + dir_index + dir_index);
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
        {
            std::ostringstream out;
            out << index;
            std::string num = out.str();
            std::string index_str = "dir";
            index_str += num;
            set_title(index_str.c_str());
            mat_writer.write(index_str.c_str(),&*dir[index].begin(),1,dir[index].size());
        }
    }
};




struct SearchLocalMaximum
{
    std::vector<std::vector<unsigned short> > neighbor;
    std::map<float,unsigned short,std::greater<float> > max_table;
    void init(Voxel& voxel)
    {

        unsigned int half_odf_size = voxel.ti.half_vertices_count;
        unsigned int faces_count = voxel.ti.faces.size();
        neighbor.resize(voxel.ti.half_vertices_count);
        for (unsigned int index = 0;index < faces_count;++index)
        {
            short i1 = voxel.ti.faces[index][0];
            short i2 = voxel.ti.faces[index][1];
            short i3 = voxel.ti.faces[index][2];
            if (i1 >= half_odf_size)
                i1 -= half_odf_size;
            if (i2 >= half_odf_size)
                i2 -= half_odf_size;
            if (i3 >= half_odf_size)
                i3 -= half_odf_size;
            neighbor[i1].push_back(i2);
            neighbor[i1].push_back(i3);
            neighbor[i2].push_back(i1);
            neighbor[i2].push_back(i3);
            neighbor[i3].push_back(i1);
            neighbor[i3].push_back(i2);
        }
    }
    void search(const std::vector<float>& old_odf)
    {
        max_table.clear();
        for (unsigned int index = 0;index < neighbor.size();++index)
        {
            float value = old_odf[index];
            bool is_max = true;
            std::vector<unsigned short>& nei = neighbor[index];
            for (unsigned int j = 0;j < nei.size();++j)
            {
                if (value < old_odf[nei[j]])
                {
                    is_max = false;
                    break;
                }
            }
            if (is_max)
                max_table[value] = (unsigned short)index;
        }
    }
};


struct DetermineFiberDirections : public BaseProcess
{
    SearchLocalMaximum lm;
    boost::mutex mutex;
public:
    virtual void init(Voxel& voxel)
    {
        lm.init(voxel);
    }

    virtual void run(Voxel& voxel,VoxelData& data)
    {
        data.min_odf = *std::min_element(data.odf.begin(),data.odf.end());
        data.sum_odf = image::mean(data.odf.begin(),data.odf.end());
        boost::mutex::scoped_lock lock(mutex);
        lm.search(data.odf);
        std::map<float,unsigned short,std::greater<float> >::const_iterator iter = lm.max_table.begin();
        std::map<float,unsigned short,std::greater<float> >::const_iterator end = lm.max_table.end();
        for (unsigned int index = 0;iter != end && index < voxel.max_fiber_number;++index,++iter)
        {
            data.dir_index[index] = iter->second;
            data.fa[index] = iter->first - data.min_odf;
        }
    }
};

struct ScaleZ0ToMinODF : public BaseProcess
{
    float max_min_odf;
public:
    virtual void init(Voxel& voxel)
    {
        voxel.z0 = 0.0;
    }

    virtual void run(Voxel& voxel,VoxelData& data)
    {
        if(data.min_odf > voxel.z0)
            voxel.z0 = data.min_odf;
    }
};




#endif//ODF_TRANSFORMATION_PROCESS_HPP
