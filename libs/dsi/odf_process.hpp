#ifndef ODF_TRANSFORMATION_PROCESS_HPP
#define ODF_TRANSFORMATION_PROCESS_HPP
#include "basic_process.hpp"
#include "basic_voxel.hpp"

class ReadDWIData : public BaseProcess{
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        data.space.resize(voxel.dwi_data.size());
        for (unsigned int index = 0; index < data.space.size(); ++index)
            data.space[index] = voxel.dwi_data[index][data.voxel_index];
    }
};
class BalanceScheme : public BaseProcess{
    std::vector<float> trans;
    unsigned int new_q_count;
    unsigned int old_q_count;
public:
    BalanceScheme(void){}
    virtual bool needed(Voxel& voxel)
    {
        return voxel.scheme_balance;
    }
    virtual void init(Voxel& voxel)
    {
        unsigned int b_count = uint32_t(voxel.bvalues.size());
        unsigned int total_signals = 0;

        tessellated_icosahedron new_dir(6);

        std::vector<tipl::vector<3,float> > new_bvectors;
        std::vector<float> new_bvalues;

        // if b0
        if(voxel.bvalues.front() == 0.0f)
        {
            trans.resize(b_count);
            trans[0] = 1;
            new_bvectors.resize(1);
            new_bvalues.resize(1);
            total_signals += 1;
        }
        for(unsigned int shell_index = 0;shell_index < voxel.shell.size();++shell_index)
        {
            unsigned int from = voxel.shell[shell_index];
            unsigned int to = (shell_index + 1 == voxel.shell.size() ? b_count : voxel.shell[shell_index+1]);
            unsigned int num = to-from;


            //calculate averaged angle distance
            double averaged_angle = 0.0;
            for(unsigned int i = from;i < to;++i)
            {
                double max_cos = 0.0;
                for(unsigned int j = from;j < to;++j)
                {
                    if(i == j)
                        continue;
                    double cur_cos = std::abs(std::cos(double(voxel.bvectors[i]*voxel.bvectors[j])));
                    if(cur_cos > 0.998)
                        continue;
                    max_cos = std::max<double>(max_cos,cur_cos);
                }
                averaged_angle += std::acos(max_cos);
            }
            averaged_angle /= num;

            //calculate averaged b_value
            double avg_b = tipl::mean(voxel.bvalues.begin()+from,voxel.bvalues.begin()+to);
            size_t trans_old_size = trans.size();
            trans.resize(trans.size() + size_t(new_dir.half_vertices_count)*size_t(b_count));
            for(unsigned int i = 0; i < new_dir.half_vertices_count;++i)
            {
                std::vector<double> t(b_count);
                double effective_b = 0.0;
                for(unsigned int j = from;j < to;++j)
                {
                    double angle = std::acos(std::min<double>(1.0,std::abs(double(new_dir.vertices[i]*voxel.bvectors[j]))));
                    angle/=averaged_angle;
                    t[j] = std::exp(-2.0*angle*angle); // if the angle == 1, then weighting = 0.135
                    effective_b += t[j]*double(voxel.bvalues[j]);
                }
                double sum_t = std::accumulate(t.begin(),t.end(),0.0);
                tipl::multiply_constant(t,avg_b/1000.0/sum_t);
                if(std::isnan(t[0])) // the sampling is too sparse
                    throw "Scheme balance failed due to insufficient sampling";
                std::copy(t.begin(),t.end(),trans.begin() + int64_t(trans_old_size + size_t(i) * size_t(b_count)));
                new_bvalues.push_back(float(effective_b/sum_t));
                new_bvectors.push_back(new_dir.vertices[i]);
            }
            total_signals += new_dir.half_vertices_count;
        }

        old_q_count = uint32_t(voxel.bvalues.size());
        new_q_count = total_signals;
        voxel.bvalues.swap(new_bvalues);
        voxel.bvectors.swap(new_bvectors);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        std::vector<float> new_data(new_q_count);
        data.space.swap(new_data);
        tipl::mat::vector_product(trans.begin(),new_data.begin(),data.space.begin(),tipl::shape<2>(new_q_count,old_q_count));
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
        m1 /= float(odf.size());
        if (m2 == 0.0f)
            return 0.0f;
        return std::sqrt((float(odf.size()))/(float(odf.size())-1.0f)*(m2-m1)/m2);
    }
};

const unsigned int odf_block_size = 20000;
struct OutputODF : public BaseProcess
{
protected:
    std::vector<std::vector<float> > odf_data;
    std::vector<size_t> odf_index_map;
public:
    virtual bool needed(Voxel& voxel)
    {
        return voxel.output_odf;
    }
    virtual void init(Voxel& voxel)
    {
        odf_data.clear();
        {
            voxel.step_report << "[Step T2b(2)][ODFs]=checked" << std::endl;
            size_t total_count = 0;
            odf_index_map.resize(voxel.mask.size());
            for (size_t index = 0;index < voxel.mask.size();++index)
                if (voxel.mask[index])
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
                    odf_data[index].resize(size_t(size_list[index])*size_t(voxel.ti.half_vertices_count));
            }
            catch (...)
            {
                odf_data.clear();
                throw std::runtime_error("Memory not enough for creating an ODF containing fib file.");
            }
        }

    }
    virtual void run(Voxel& voxel,VoxelData& data)
    {
        if (data.fa[0] != 0.0f)
        {
            size_t odf_index = odf_index_map[data.voxel_index];
            std::copy(data.odf.begin(),data.odf.end(),
                      odf_data[odf_index/odf_block_size].begin() + (odf_index%odf_block_size)*(voxel.ti.half_vertices_count));
        }

    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        tipl::par_for (odf_data.size(),[&](unsigned int index)
        {
            tipl::multiply_constant(odf_data[index],voxel.z0);
        });
        tipl::progress prog("odf",true);
        for (unsigned int index = 0;prog(index,odf_data.size());++index)
            mat_writer.write((std::string("odf")+std::to_string(index)).c_str(),odf_data[index],voxel.ti.half_vertices_count);
        odf_data.clear();
    }
};


struct ODFLoader : public BaseProcess
{
    std::vector<size_t> index_mapping1;
    std::vector<size_t> index_mapping2;
public:
    virtual void init(Voxel& voxel)
    {
        //voxel.qa_scaling must be 1
        index_mapping1.resize(voxel.mask.size());
        index_mapping2.resize(voxel.mask.size());
        size_t voxel_index = 0;
        for(size_t i = 0;i < voxel.template_odfs.size();++i)
        {
            for(size_t j = 0;j < voxel.template_odfs[i].size();j += voxel.ti.half_vertices_count)
            {
                size_t k_end = j + voxel.ti.half_vertices_count;
                bool is_odf_zero = true;
                for(size_t k = j;k < k_end;++k)
                    if(voxel.template_odfs[i][k] != 0.0f)
                    {
                        is_odf_zero = false;
                        break;
                    }
                if(!is_odf_zero)
                    for(;voxel_index < index_mapping1.size();++voxel_index)
                        if(voxel.mask[voxel_index])
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
        size_t cur_index = data.voxel_index;
        std::copy(voxel.template_odfs[index_mapping1[cur_index]].begin() +
                  int64_t(index_mapping2[cur_index]),
                  voxel.template_odfs[index_mapping1[cur_index]].begin() +
                  int64_t(index_mapping2[cur_index])+int64_t(data.odf.size()),
                  data.odf.begin());

    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        if (voxel.output_odf)
        {
            for (unsigned int index = 0;index < voxel.template_odfs.size();++index)
            {
                std::ostringstream out;
                out << "odf" << index;
                mat_writer.write(out.str().c_str(),voxel.template_odfs[index],voxel.ti.half_vertices_count);
            }
        }
        mat_writer.write("trans",voxel.trans_to_mni.begin(),4,4);
        for(size_t i = 0;i < voxel.template_metrics.size();++i)
            mat_writer.write(voxel.template_metrics_name[i].c_str(),voxel.template_metrics[i]);
    }
};

// for normalization
class RecordQA  : public BaseProcess
{
public:
    virtual void init(Voxel& voxel)
    {
        voxel.qa_map.resize(voxel.dim);
        voxel.iso_map.resize(voxel.dim);
        voxel.qa_map = 0;
        voxel.iso_map = 0;
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        auto min_v = data.odf[0];
        auto max_v = data.odf[0];
        tipl::minmax_value(data.odf.begin(),data.odf.end(),min_v,max_v);
        voxel.qa_map[data.voxel_index] = max_v-min_v;
        voxel.iso_map[data.voxel_index] = min_v;
    }
};

struct SearchLocalMaximum
{
    std::vector<std::vector<unsigned short> > neighbor;
    void init(Voxel& voxel)
    {
        unsigned int half_odf_size = voxel.ti.half_vertices_count;
        unsigned int faces_count = uint32_t(voxel.ti.faces.size());
        neighbor.resize(voxel.ti.half_vertices_count);
        for (unsigned int index = 0;index < faces_count;++index)
        {
            unsigned short i1 = voxel.ti.faces[index][0];
            unsigned short i2 = voxel.ti.faces[index][1];
            unsigned short i3 = voxel.ti.faces[index][2];
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
    void search(const std::vector<float>& old_odf,std::map<float,unsigned short,std::greater<float> >& max_table)
    {
        max_table.clear();
        for (uint32_t index = 0;index < neighbor.size();++index)
        {
            float value = old_odf[index];
            bool is_max = true;
            std::vector<uint16_t>& nei = neighbor[index];
            for (unsigned int j = 0;j < nei.size();++j)
            {
                if (value < old_odf[nei[j]])
                {
                    is_max = false;
                    break;
                }
            }
            if (is_max)
                max_table[value] = index;
        }
    }
};

struct ODFShaping
{

    std::vector<std::vector<unsigned int> > shape_list;
    void init(tessellated_icosahedron& ti)
    {
        unsigned int half_odf_size = ti.half_vertices_count;
        shape_list.clear();
        for (unsigned int i = 0;i < half_odf_size;++i)
        {
            std::vector<float> cos_value(half_odf_size);
            for(unsigned int j = 0;j < half_odf_size;++j)
                cos_value[j] = std::fabs(ti.vertices_cos(i,j));
            shape_list.push_back(tipl::arg_sort(cos_value,std::greater<float>()));
        }
    }
    void shape(std::vector<float>& odf,uint16_t dir)
    {
        float cur_max = odf[dir];
        odf[dir] = 0.0f;
        const std::vector<unsigned int>& remove_list = shape_list[dir];
        for (unsigned int index = 1;index < remove_list.size();++index)
        {
            unsigned int pos = remove_list[index];
            cur_max = std::min<float>(odf[pos],cur_max);
            odf[pos] -= cur_max;
        }
    }
    void reshape(std::vector<float>& odf,uint16_t dir)
    {
        const std::vector<unsigned int>& remove_list = shape_list[dir];
        for (unsigned int index = 1;index < remove_list.size();++index)
        {
            float back_value = odf[remove_list[index]];
            if(back_value > odf[remove_list[index-1]])
            {
                unsigned int j = index -1;
                do{
                    odf[remove_list[j]] = back_value;
                }
                while(back_value > odf[remove_list[j]] && j);
            }
        }
    }
    /*
    std::vector<std::vector<std::vector<unsigned int> > > shape_list;
    void init(Voxel& voxel)
    {
        unsigned int half_odf_size = voxel.ti.half_vertices_count;
        shape_list.resize(half_odf_size);
        for (unsigned int i = 0;i < half_odf_size;++i)
        {
            std::vector<std::vector<unsigned int> > cos_table(6);
            for(unsigned int j = 0;j < half_odf_size;++j)
                if(j != i)
                    cos_table[uint32_t(std::floor(float(cos_table.size())*std::fabs(voxel.ti.vertices_cos(i,j))))].push_back(j);
            std::reverse(cos_table.begin(),cos_table.end());
            shape_list[i] = std::move(cos_table);
        }
    }
    void shape(std::vector<float>& odf,uint16_t dir)
    {
        float cur_max = odf[dir];
        odf[dir] = 0.0f;
        const auto& cos_table = shape_list[dir];
        for (size_t index = 0;index < cos_table.size();++index)
        {
            for(auto sym_dir : cos_table[index])
                cur_max = std::min<float>(odf[sym_dir],cur_max);
            for(auto sym_dir : cos_table[index])
                odf[sym_dir] -= cur_max;
        }
    }*/
};

double base_function(double theta);
struct SaveMetrics : public BaseProcess
{
protected:
    SearchLocalMaximum lm;
    std::mutex mutex;
    ODFShaping shaping;
    std::vector<std::vector<short> > findex;
protected:
    std::vector<float> iso,gfa;
    std::vector<std::vector<float> > fa,rdi;
public:
    virtual void init(Voxel& voxel)
    {
        if(voxel.odf_resolving)
            shaping.init(voxel.ti);
        lm.init(voxel);

        voxel.z0 = 1.0f;
        fa = std::vector<std::vector<float> >(voxel.max_fiber_number,std::vector<float>(voxel.dim.size()));
        if(voxel.needs("gfa"))
            gfa = std::vector<float>(voxel.dim.size());
        iso = std::vector<float>(voxel.dim.size());
        if(voxel.needs("rdi"))
        {
            float sigma = voxel.param[0]; //optimal 1.24
            for(float L = 0.2f;L <= sigma;L+= 0.2f)
                rdi.push_back(std::vector<float>(voxel.dim.size()));
        }
        findex.resize(voxel.max_fiber_number);
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            findex[index].resize(voxel.dim.size());
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        data.min_odf = tipl::min_value(data.odf.begin(),data.odf.end());
        if(std::isnan(data.min_odf) || std::isinf(data.min_odf) || data.min_odf < 0.0f)
            return;
        if(voxel.odf_resolving)
        {
            std::vector<float> odf(data.odf);
            tipl::minus_constant(odf.begin(),odf.end(),data.min_odf);
            float sum = tipl::sum(odf.begin(),odf.end());
            float last_fiber_sum = 0.0f;
            for(unsigned int i = 0;i < voxel.max_fiber_number;++i)
            {
                uint16_t peak = uint16_t(std::max_element(odf.begin(),odf.end())-odf.begin());
                float qa = odf[peak];
                shaping.shape(odf,peak);
                float new_sum = tipl::sum(odf.begin(),odf.end());
                if(i && last_fiber_sum*0.2f > sum-new_sum)
                    break;
                last_fiber_sum = sum-new_sum;
                sum = new_sum;
                data.dir_index[i] = peak;
                data.fa[i] = qa;
            }

            /*
            std::vector<float> second_fiber_odf(data.odf.size());
            for(int i = 0;i < 3;++i)
            {
                std::vector<float> odf(data.odf);
                if(i)
                    tipl::minus(odf,second_fiber_odf);
                shaping.shape(odf,first_peak);
                if(i)
                    tipl::add(odf,second_fiber_odf);
                second_peak = uint16_t(std::max_element(odf.begin(),odf.end())-odf.begin());
                shaping.reshape(odf,second_peak);
                second_fiber_odf = odf;

                std::vector<float> new_odf(data.odf);
                tipl::minus(new_odf,second_fiber_odf);
                uint16_t first_peak2 = uint16_t(std::max_element(odf.begin(),odf.end())-odf.begin());
                if(first_peak != first_peak2)
                    iter->second = first_peak = first_peak2;
                else
                    break;
            }
            */

        }
        else
        {
            std::map<float,unsigned short,std::greater<float> > max_table;
            lm.search(data.odf,max_table);
            auto iter = max_table.begin();
            auto end = max_table.end();
            for (unsigned int index = 0;iter != end && index < voxel.max_fiber_number;++index,++iter)
            {
                data.dir_index[index] = iter->second;
                data.fa[index] = iter->first - data.min_odf;
            }
        }

        iso[data.voxel_index] = data.min_odf;

        if(!gfa.empty())
            gfa[data.voxel_index] = GeneralizedFA()(data.odf);

        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            fa[index][data.voxel_index] = data.fa[index];

        if(!rdi.empty())
            for (unsigned int index = 0;index < data.rdi.size();++index)
                rdi[index][data.voxel_index] = data.rdi[index];

        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            findex[index][data.voxel_index] = short(data.dir_index[index]);
    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        write_image_to_mat(mat_writer,"gfa",gfa.data(),voxel.dim);
        // output normalized qa
        {
            float max_qa = 0.0;
            for (unsigned int i = 0;i < voxel.max_fiber_number;++i)
                max_qa = std::max<float>(tipl::max_value(fa[i]),max_qa);
            if(max_qa == 0.0f)
                max_qa = 1.0f;
            voxel.z0 = float(1.0/double(max_qa));
        }
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
        {
            tipl::multiply_constant(fa[index],voxel.z0);
            write_image_to_mat(mat_writer,("fa" + std::to_string(index)).c_str(),fa[index].data(),voxel.dim);
        }

        tipl::multiply_constant(iso,voxel.z0);
        write_image_to_mat(mat_writer,"iso",iso.data(),voxel.dim);

        if(!rdi.empty())
        {
            for(unsigned int i = 0;i < rdi.size();++i)
                tipl::multiply_constant(rdi[i],voxel.z0);
            float L = 0.2f;
            write_image_to_mat(mat_writer,"rdi",rdi[0].data(),voxel.dim);

            if(voxel.shell.size() > 1)
            {
                for(unsigned int i = 0;i < rdi[0].size();++i)
                for(unsigned int j = 0;j < rdi.size();++j)
                    rdi[j][i] = rdi.back()[i]-rdi[j][i];
                L = 0.2f;
                for(unsigned int i = 0;i+1 < rdi.size();++i,L += 0.2f)
                {
                    if(tipl::max_value(rdi[i]) < 0.1f)
                        break;
                    std::ostringstream out2;
                    out2.precision(2);
                    out2 << "nrdi" << std::setfill('0') << std::setw(2) << int(L*10) << "L";
                    write_image_to_mat(mat_writer,out2.str().c_str(),rdi[i].data(),voxel.dim);
                }
            }
        }
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            mat_writer.write(("index" + std::to_string(index)).c_str(),findex[index],uint32_t(voxel.dim.plane_size()));
    }
};



#endif//ODF_TRANSFORMATION_PROCESS_HPP
