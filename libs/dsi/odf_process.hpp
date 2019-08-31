#ifndef ODF_TRANSFORMATION_PROCESS_HPP
#define ODF_TRANSFORMATION_PROCESS_HPP
#include <boost/math/special_functions/sinc.hpp>
#include "basic_process.hpp"
#include "basic_voxel.hpp"

class ReadDWIData : public BaseProcess{
public:
    virtual void init(Voxel&) {}
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        data.space.resize(voxel.dwi_data.size());
        for (unsigned int index = 0; index < data.space.size(); ++index)
            data.space[index] = voxel.dwi_data[index][data.voxel_index];
    }
    virtual void end(Voxel&,gz_mat_write&) {}
};


void calculate_shell(const std::vector<float>& sorted_bvalues,
                     std::vector<unsigned int>& shell);
class BalanceScheme : public BaseProcess{
    std::vector<float> trans;
    unsigned int new_q_count;
    unsigned int old_q_count;
private:
    Voxel* stored_voxel;
    std::vector<tipl::vector<3,float> > old_bvectors;
    std::vector<float> old_bvalues;
public:
    BalanceScheme(void):stored_voxel(nullptr){}

    virtual void init(Voxel& voxel)
    {
        if(!voxel.scheme_balance)
            return;
        std::vector<unsigned int> shell;
        calculate_shell(voxel.bvalues,shell);
        unsigned int b_count = uint32_t(voxel.bvalues.size());
        unsigned int total_signals = 0;

        tessellated_icosahedron new_dir;
        new_dir.init(6);

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

        for(unsigned int shell_index = 0;shell_index < shell.size();++shell_index)
        {
            unsigned int from = shell[shell_index];
            unsigned int to = (shell_index + 1 == shell.size() ? b_count:shell[shell_index+1]);
            unsigned int num = to-from;


            //calculate averaged angle distance
            float averaged_angle = 0.0;
            for(unsigned int i = from;i < to;++i)
            {
                float max_cos = 0.0;
                for(unsigned int j = from;j < to;++j)
                {
                    if(i == j)
                        continue;
                    float cur_cos = std::fabs(std::cos(voxel.bvectors[i]*voxel.bvectors[j]));
                    if(cur_cos > 0.998f)
                        continue;
                    max_cos = std::max<float>(max_cos,cur_cos);
                }
                averaged_angle += std::acos(max_cos);
            }
            averaged_angle /= num;

            //calculate averaged b_value
            float avg_b = float(tipl::mean(voxel.bvalues.begin()+from,voxel.bvalues.begin()+to));
            unsigned int trans_old_size = uint32_t(trans.size());
            trans.resize(trans.size() + new_dir.half_vertices_count*b_count);
            for(unsigned int i = 0; i < new_dir.half_vertices_count;++i)
            {
                std::vector<float> t(b_count);
                float effective_b = 0.0;
                for(unsigned int j = from;j < to;++j)
                {
                    float angle = std::acos(std::min<float>(1.0f,std::fabs(new_dir.vertices[i]*voxel.bvectors[j])));
                    angle/=averaged_angle;
                    t[j] = std::exp(-2.0f*angle*angle); // if the angle == 1, then weighting = 0.135
                    effective_b += t[j]*voxel.bvalues[j];
                }
                float sum_t = std::accumulate(t.begin(),t.end(),0.0f);
                tipl::multiply_constant(t,avg_b/1000.0f/sum_t);
                std::copy(t.begin(),t.end(),trans.begin() + trans_old_size + i * b_count);
                new_bvalues.push_back(effective_b/sum_t);
                new_bvectors.push_back(new_dir.vertices[i]);
            }
            total_signals += new_dir.half_vertices_count;
        }

        old_q_count = uint32_t(voxel.bvalues.size());
        new_q_count = total_signals;
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
        if(stored_voxel)// restored btalbe here in case user terminate the recon
        {
            stored_voxel = nullptr;
            voxel.bvalues = old_bvalues;
            voxel.bvectors = old_bvectors;
        }
        std::vector<float> new_data(new_q_count);
        data.space.swap(new_data);
        tipl::mat::vector_product(trans.begin(),new_data.begin(),data.space.begin(),tipl::dyndim(new_q_count,old_q_count));
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
    std::vector<unsigned int> odf_index_map;
public:
    virtual void init(Voxel& voxel)
    {
        odf_data.clear();
        if (voxel.output_odf)
        {
            voxel.step_report << "[Step T2b(2)][ODFs]=checked" << std::endl;
            unsigned int total_count = 0;
            odf_index_map.resize(voxel.mask.size());
            for (unsigned int index = 0;index < voxel.mask.size();++index)
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
                    odf_data[index].resize(size_list[index]*(voxel.ti.half_vertices_count));
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

        if (voxel.output_odf && data.fa[0] + 1.0f != 1.0f)
        {
            unsigned int odf_index = odf_index_map[data.voxel_index];
            std::copy(data.odf.begin(),data.odf.end(),
                      odf_data[odf_index/odf_block_size].begin() + (odf_index%odf_block_size)*(voxel.ti.half_vertices_count));
        }

    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {

        if (!voxel.output_odf)
            return;
        {
            set_title("Output ODFs");
            for (unsigned int index = 0;index < odf_data.size();++index)
            {
                tipl::divide_constant(odf_data[index],voxel.z0);
                std::ostringstream out;
                out << "odf" << index;
                mat_writer.write(out.str().c_str(),odf_data[index],voxel.ti.half_vertices_count);
            }
            odf_data.clear();
        }

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
        voxel.z0 = 0.0;
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
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        if (voxel.output_odf)
        {
            set_title("Output ODFs");
            for (unsigned int index = 0;index < voxel.template_odfs.size();++index)
            {
                std::ostringstream out;
                out << "odf" << index;
                mat_writer.write(out.str().c_str(),voxel.template_odfs[index],voxel.ti.half_vertices_count);
            }
        }
        mat_writer.write("trans",voxel.trans_to_mni,4,4);
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
        std::fill(voxel.qa_map.begin(),voxel.qa_map.end(),0.0);
        std::fill(voxel.iso_map.begin(),voxel.iso_map.end(),0.0);
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        voxel.qa_map[data.voxel_index] = data.fa[0];
        voxel.iso_map[data.voxel_index] = data.min_odf;
    }
    virtual void end(Voxel&,gz_mat_write&)
    {

    }
};

double base_function(double theta);
struct SaveMetrics : public BaseProcess
{
protected:
    std::vector<float> iso,gfa;
    std::vector<std::vector<float> > fa,rdi,qa_inc,qa_dec;
    tipl::geometry<3> dim;

    void output_anisotropy(gz_mat_write& mat_writer,
                           const char* name,const std::vector<std::vector<float> >& metrics)
    {
        for (unsigned int index = 0;index < metrics.size();++index)
        {
            std::ostringstream out;
            out << index;
            std::string num = out.str();
            std::string str = name + num;
            set_title(str.c_str());
            mat_writer.write(str.c_str(),metrics[index],uint32_t(dim.plane_size()));
        }
    }

protected:
    float z0;
public:
    virtual void init(Voxel& voxel)
    {
        dim = voxel.dim;
        fa = std::vector<std::vector<float> >(voxel.max_fiber_number,std::vector<float>(dim.size()));
        gfa = std::vector<float>(dim.size());
        iso= std::vector<float>(dim.size());
        if(voxel.compare_voxel) // DDI
        {
            qa_inc = std::vector<std::vector<float> >(voxel.max_fiber_number,std::vector<float>(dim.size()));
            qa_dec = std::vector<std::vector<float> >(voxel.max_fiber_number,std::vector<float>(dim.size()));
        }
        if(voxel.output_rdi)
        {
            float sigma = voxel.param[0]; //optimal 1.24
            for(float L = 0.2f;L <= sigma;L+= 0.2f)
                rdi.push_back(std::vector<float>(dim.size()));
        }

        if(voxel.csf_calibration)
        {
            std::vector<unsigned short> data(voxel.dwi_data[0],voxel.dwi_data[0]+dim.size());
            std::sort(data.begin(),data.end());
            // CSF selected at 125,000 mm^3
            int size = int(125000.0f/(voxel.vs[0]*voxel.vs[1]*voxel.vs[2]));
            float water_b0 = *(data.end()-size);

            // numerically estimate free water ODF
            unsigned int odf_size = voxel.ti.half_vertices_count;
            std::vector<float> dwi(voxel.bvalues.size()),odf(odf_size);
            for(size_t i = 0;i < dwi.size();++i)
                dwi[i] = water_b0*std::exp(-voxel.bvalues[i]*0.003f); // free water diffusivity
            std::vector<float> sinc_ql;
            voxel.calculate_sinc_ql(sinc_ql);
            tipl::mat::vector_product(&*sinc_ql.begin(),&*dwi.begin(),&*odf.begin(),
                                    tipl::dyndim(uint32_t(odf.size()),uint32_t(dwi.size())));
            z0 = float(tipl::mean(odf.begin(),odf.end()));
        }

        voxel.z0 = 0.0;
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        iso[data.voxel_index] = data.min_odf;
        gfa[data.voxel_index] = GeneralizedFA()(data.odf);
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            fa[index][data.voxel_index] = data.fa[index];
        if(voxel.output_rdi)
            for (unsigned int index = 0;index < data.rdi.size();++index)
                rdi[index][data.voxel_index] = data.rdi[index];
        if(data.min_odf > voxel.z0)
            voxel.z0 = data.min_odf;
        if(voxel.compare_voxel) // DDI
        {
            for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
                if(data.fa[index] > 0.0f)
                {
                    float value1 = data.odf1[data.dir_index[index]];
                    float value2 = data.odf2[data.dir_index[index]];
                    float change = value2-value1;
                    if(change > 0.0f)
                        qa_inc[index][data.voxel_index] = change/data.fa[index];
                    else
                        qa_dec[index][data.voxel_index] = -change/data.fa[index];
                }
            data.odf = data.odf2;
            tipl::minus(data.odf,data.odf1);
        }
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        set_title("output data");
        mat_writer.write("gfa",gfa,uint32_t(voxel.dim.plane_size()));
        if(voxel.csf_calibration)
            voxel.z0 = z0;
        if(voxel.z0 + 1.0f == 1.0f)
            voxel.z0 = 1.0f;
        mat_writer.write("z0",&voxel.z0,1,1);



        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            tipl::divide_constant(fa[index],voxel.z0);
        output_anisotropy(mat_writer,"fa",fa);

        tipl::divide_constant(iso,voxel.z0);
        mat_writer.write("iso",iso,uint32_t(voxel.dim.plane_size()));


        // output normalized qa
        {
            float max_qa = 0.0;
            for (unsigned int i = 0;i < voxel.max_fiber_number;++i)
                max_qa = std::max<float>(*std::max_element(fa[i].begin(),fa[i].end()),max_qa);

            if(max_qa != 0.0f)
            {
                for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
                    tipl::divide_constant(fa[index],max_qa);
                output_anisotropy(mat_writer,"nqa",fa);
            }
            if(voxel.compare_voxel) // DDI
            {
                output_anisotropy(mat_writer,"inc_qa",qa_inc);
                output_anisotropy(mat_writer,"dec_qa",qa_dec);

                mat_writer.write("base_fa",voxel.fib_fa,uint32_t(voxel.dim.plane_size()));
                mat_writer.write("study_fa",voxel.compare_voxel->fib_fa,uint32_t(voxel.dim.plane_size()));
                for(size_t i = 0;i < voxel.dim.size();++i)
                    if(voxel.compare_voxel->fib_fa[i] > voxel.fib_fa[i])
                    {
                        qa_inc[0][i] = voxel.compare_voxel->fib_fa[i] - voxel.fib_fa[i];
                        qa_dec[0][i] = 0;
                    }
                else
                    {
                        qa_dec[0][i] = voxel.fib_fa[i]-voxel.compare_voxel->fib_fa[i];
                        qa_inc[0][i] = 0;
                    }

                for (unsigned int index = 1;index < voxel.max_fiber_number;++index)
                {
                    std::fill(qa_inc[index].begin(),qa_inc[index].end(),0.0f);
                    std::fill(qa_dec[index].begin(),qa_dec[index].end(),0.0f);
                }
                output_anisotropy(mat_writer,"inc_fa",qa_inc);
                output_anisotropy(mat_writer,"dec_fa",qa_dec);

            }
        }


        if(voxel.output_rdi)
        {
            for(unsigned int i = 0;i < rdi.size();++i)
                tipl::divide_constant(rdi[i],voxel.z0);
            float L = 0.2f;
            for(unsigned int i = 0;i < rdi.size();++i,L += 0.2f)
            {
                std::ostringstream out;
                out.precision(2);
                out << "rdi" << std::setfill('0') << std::setw(2) << int(L*10) << "L";
                mat_writer.write(out.str().c_str(),rdi[i],uint32_t(voxel.dim.plane_size()));
            }
            for(unsigned int i = 0;i < rdi[0].size();++i)
            for(unsigned int j = 0;j < rdi.size();++j)
                rdi[j][i] = rdi[rdi.size()-1][i]-rdi[j][i];
            L = 0.2f;
            for(unsigned int i = 0;i < rdi.size();++i,L += 0.2f)
            {
                std::ostringstream out2;
                out2.precision(2);
                out2 << "nrdi" << std::setfill('0') << std::setw(2) << int(L*10) << "L";
                mat_writer.write(out2.str().c_str(),rdi[i],uint32_t(voxel.dim.plane_size()));
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
            mat_writer.write(index_str.c_str(),findex[index]);
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
        int64_t dir_index = int64_t(data.voxel_index);
        for (size_t index = 0;index < voxel.max_fiber_number;++index)
            std::copy(data.dir[index].begin(),data.dir[index].end(),dir[index].begin() +
                      dir_index + dir_index + dir_index);
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
            mat_writer.write(index_str.c_str(),dir[index]);
        }
    }
};



struct ODFShaping
{
    std::vector<std::vector<unsigned int> > shape_list;
    void init(Voxel& voxel)
    {
        unsigned int half_odf_size = voxel.ti.half_vertices_count;
        shape_list.clear();
        for (unsigned int i = 0;i < half_odf_size;++i)
        {
            std::vector<float> cos_value(half_odf_size);
            for(unsigned int j = 0;j < half_odf_size;++j)
                cos_value[i] = std::fabs(voxel.ti.vertices_cos(i,j));
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
        for (uint16_t index = 0;index < neighbor.size();++index)
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


struct DetermineFiberDirections : public BaseProcess
{
    SearchLocalMaximum lm;
    std::mutex mutex;
    ODFShaping shaping;
public:
    virtual void init(Voxel& voxel)
    {
        if(voxel.odf_resolving)
            shaping.init(voxel);
        lm.init(voxel);
    }

    virtual void run(Voxel& voxel,VoxelData& data)
    {
        data.min_odf = *std::min_element(data.odf.begin(),data.odf.end());
        std::map<float,unsigned short,std::greater<float> > max_table;
        lm.search(data.odf,max_table);
        if(voxel.odf_resolving)
        {
            std::vector<float> odf(data.odf);
            for (unsigned int index = 0;index < 3;++index)
            {
                uint16_t max_dir = uint16_t(std::max_element(odf.begin(),odf.end())-odf.begin());
                if(odf[max_dir] == 0.0f)
                    break;
                if(index)
                    max_table[data.odf[max_dir]] = max_dir;
                shaping.shape(odf,max_dir);
            }
        }
        auto iter = max_table.begin();
        auto end = max_table.end();
        for (unsigned int index = 0;iter != end && index < voxel.max_fiber_number;++index,++iter)
        {
            data.dir_index[index] = iter->second;
            data.fa[index] = iter->first - data.min_odf;
        }
    }
};




#endif//ODF_TRANSFORMATION_PROCESS_HPP
