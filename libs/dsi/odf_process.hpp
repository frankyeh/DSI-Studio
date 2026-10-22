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
    std::vector<float> scale; // the gqi weight of each balanced signal
    std::vector<unsigned int> new_shell; // the shells of the balanced b-table
public:
    std::vector<float> trans; // the interpolation (new x old dwi, each row sums to one), also used by src_data::scheme_balance
private: // the dwi output of qsdr: template-space dwi under the balanced voxel b-table, written into the fz
    std::vector<tipl::vector<3,float> > src_bvectors; // native directions; voxel.shell keeps the native shells
    std::vector<tipl::image<3> > dwi;
private:
    // normalized angular weights of the dwi [from,to) of a shell at direction dir, a gaussian of 0.08 rad width
    std::vector<double> shell_weights(const std::vector<tipl::vector<3,float> >& bvectors,unsigned int from,unsigned int to,const tipl::vector<3,float>& dir) const
    {
        std::vector<double> t;
        for(unsigned int j = from;j < to;++j)
        {
            double angle = std::acos(std::min<double>(1.0,std::abs(double(dir*bvectors[j]))))/0.08;
            t.push_back(std::exp(-2.0*angle*angle));
        }
        tipl::multiply_constant(t.begin(),t.end(),1.0/std::accumulate(t.begin(),t.end(),0.0));
        return t;
    }
public:
    virtual bool needed(Voxel& voxel)
    {
        return voxel.shell.size() < 5; // shell schemes, including those already balanced by scheme_balance
    }
    // balances the voxel b-table (unless already done by scheme_balance) and gets the gqi scale of the balanced signals
    virtual void init(Voxel& voxel)
    {
        unsigned int old_q_count = uint32_t(voxel.dwi_data.size());
        if(voxel.qsdr && voxel.needs("dwi") && !voxel.shell.empty())
            src_bvectors = voxel.bvectors;
        new_shell = voxel.shell;
        if(!tipl::contains(voxel.steps,"scheme_balance"))
        {
            bool b0 = voxel.bvalues.front() == 0.0f; // the b0 keeps the first row
            std::vector<float> new_bvalues(b0);
            std::vector<tipl::vector<3,float> > new_bvectors(b0);
            if(b0)
            {
                trans.resize(old_q_count);
                trans[0] = 1.0f;
            }
            for(unsigned int k = 0;k < voxel.shell.size();++k)
            {
                unsigned int from = voxel.shell[k],to = (k + 1 == voxel.shell.size() ? old_q_count : voxel.shell[k+1]);
                new_shell[k] = uint32_t(new_bvalues.size());
                double avg_b = tipl::mean(voxel.bvalues.begin()+from,voxel.bvalues.begin()+to);
                tessellated_icosahedron new_dir(avg_b > 2500.0 ? 6 : avg_b > 1500.0 ? 5 : avg_b > 500.0 ? 4 : 3); // 181, 126, 81, or 46 directions
                for(unsigned int i = 0;i < new_dir.half_vertices_count;++i)
                {
                    auto t = shell_weights(voxel.bvectors,from,to,new_dir.vertices[i]);
                    trans.resize(trans.size()+old_q_count);
                    std::copy(t.begin(),t.end(),trans.end()-old_q_count+from);
                    new_bvalues.push_back(float(std::inner_product(t.begin(),t.end(),voxel.bvalues.begin()+from,0.0)));
                    new_bvectors.push_back(new_dir.vertices[i]);
                }
            }
            voxel.bvalues.swap(new_bvalues);
            voxel.bvectors.swap(new_bvectors);
        }
        // the gqi scale b_k/(n_k*B) (b0 none): each shell sampled uniformly on its q-space surface, B the sum of shell b-values
        {
            double sum_b = 0.0;
            scale.resize(voxel.bvalues.size());
            for(unsigned int k = 0;k < new_shell.size();++k)
            {
                unsigned int from = new_shell[k],to = (k + 1 == new_shell.size() ? uint32_t(scale.size()) : new_shell[k+1]);
                double b = tipl::mean(voxel.bvalues.begin()+from,voxel.bvalues.begin()+to);
                sum_b += b;
                std::fill(scale.begin()+from,scale.begin()+to,float(b/double(to-from)));
            }
            tipl::multiply_constant(scale.begin(),scale.end(),float(1.0/sum_b));
        }
        if(!src_bvectors.empty())
            dwi.resize(voxel.bvalues.size(),tipl::image<3>(voxel.dim));
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        if(!dwi.empty() && data.jacobian.det() != 0.0f) // zero outside the native fov
        {
            for(unsigned int j = 0;j < new_shell[0];++j) // b0
                dwi[j][data.voxel_index] = data.space[j];
            for(unsigned int k = 0;k < new_shell.size();++k)
                for(unsigned int j = new_shell[k],end = (k + 1 == new_shell.size() ? uint32_t(dwi.size()) : new_shell[k+1]);j < end;++j)
                {
                    auto dir = voxel.bvectors[j]; // the qsdr convention: template direction t evaluates native normalize(J t)
                    dir.rotate(data.jacobian);
                    dir.normalize();
                    auto w = shell_weights(src_bvectors,voxel.shell[k],k + 1 == voxel.shell.size() ? uint32_t(voxel.dwi_data.size()) : voxel.shell[k+1],dir);
                    dwi[j][data.voxel_index] = float(std::inner_product(w.begin(),w.end(),data.space.begin()+voxel.shell[k],0.0));
                }
        }
        if(!trans.empty())
        {
            std::vector<float> new_data(scale.size());
            data.space.swap(new_data);
            tipl::mat::vector_product(trans.begin(),new_data.begin(),data.space.begin(),tipl::shape<2>(scale.size(),voxel.dwi_data.size()));
        }
        for(size_t i = 0;i < scale.size();++i)
            data.space[i] *= scale[i];
    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        if(dwi.empty())
            return;
        std::vector<float> b_table;
        for(size_t i = 0;i < voxel.bvalues.size();++i)
        {
            b_table.push_back(voxel.bvalues[i]);
            b_table.insert(b_table.end(),voxel.bvectors[i].begin(),voxel.bvectors[i].end());
        }
        mat_writer.write("b_table",b_table,4);
        for(size_t i = 0;i < dwi.size();++i)
            mat_writer.write<tipl::io::masked_sloped>("image"+std::to_string(i),dwi[i].data(),voxel.dim.plane_size(),voxel.dim.depth());
        if(!tipl::contains(voxel.steps,"scheme_balance")) // the fz steps then mark the embedded dwi as balanced
            voxel.step_report << "scheme_balance" << std::endl;
        dwi = std::vector<tipl::image<3> >();
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
        return voxel.needs("odf");
    }
    virtual void init(Voxel& voxel)
    {
        odf_data.clear();
        {
            voxel.step_report << "[Step T2b(2)][ODFs]=checked" << std::endl;
            size_t total_count = 0;
            odf_index_map.resize(voxel.mask.size());
            for (size_t index = 0,sz = voxel.mask.size();index < sz;++index)
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
        for(auto& each : odf_data)
            tipl::multiply_constant(each.begin(),each.end(),voxel.z0);
        tipl::progress prog("odf",true);
        for (unsigned int index = 0;prog(index,odf_data.size());++index)
            mat_writer.write<tipl::io::sloped>((std::string("odf")+std::to_string(index)).c_str(),odf_data[index],voxel.ti.half_vertices_count);
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
        std::copy_n(voxel.template_odfs[index_mapping1[cur_index]].begin() +
                  int64_t(index_mapping2[cur_index]),int64_t(data.odf.size()),
                  data.odf.begin());

    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        if (voxel.needs("odf"))
        {
            for (unsigned int index = 0;index < voxel.template_odfs.size();++index)
            {
                std::ostringstream out;
                out << "odf" << index;
                mat_writer.write<tipl::io::sloped>(out.str().c_str(),voxel.template_odfs[index],voxel.ti.half_vertices_count);
            }
        }
        mat_writer.write("trans",voxel.trans_to_mni);
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
};

struct SaveMetrics : public BaseProcess
{
protected:
    SearchLocalMaximum lm;
    std::vector<tipl::image<3,unsigned short> > fixel; // refined fiber directions when "fixel" is requested, 0 where there is no fiber
    std::vector<tipl::image<3,short> > findex; // otherwise the odf vertex of each fiber
protected:
    tipl::image<3> iso,gfa;
    std::vector<tipl::image<3> > fa,rdi;
public:
    virtual void init(Voxel& voxel)
    {
        lm.init(voxel);

        voxel.z0 = 1.0f;
        fa.resize(voxel.max_fiber_number);
        for(auto& each : fa)
            each = std::move(tipl::image<3>(voxel.dim));

        if(voxel.needs("gfa"))
            gfa = std::move(tipl::image<3>(voxel.dim));
        iso = std::move(tipl::image<3>(voxel.dim));
        if(voxel.needs("rdi"))
        {
            float sigma = voxel.param[0]; //optimal 1.24
            for(float L = 0.2f;L <= sigma;L+= 0.2f)
                rdi.emplace_back(voxel.dim);
        }
        if(voxel.needs("fixel"))
            fixel.resize(voxel.max_fiber_number,tipl::image<3,unsigned short>(voxel.dim));
        else
            findex.resize(voxel.max_fiber_number,tipl::image<3,short>(voxel.dim));
    }
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        float min_odf = tipl::min_value(data.odf.begin(),data.odf.end());
        if(!std::isfinite(min_odf) || min_odf < 0.0f)
            return;
        std::map<float,unsigned short,std::greater<float> > max_table;
        lm.search(data.odf,max_table);
        auto iter = max_table.begin();
        auto end = max_table.end();
        for (unsigned int index = 0;iter != end && index < voxel.max_fiber_number;++index,++iter)
        {
            data.fa[index] = iter->first - min_odf;
            if(fixel.empty())
            {
                findex[index][data.voxel_index] = short(iter->second);
                continue;
            }
            tipl::vector<3,float> d(voxel.ti.vertices[iter->second]);
            fixel[index][data.voxel_index] = encode_fixel(d);
            if(voxel.q_vectors.empty()) // an odf template: no diffusion signals to refine on
                continue;
            // refine the peak on the gqi odf: a quadratic fit of 9 tangent-plane samples (spacing h) around the seed d,
            // accepted only as a maximum within the seed's one-ring (radius r) that does not lower the odf
            tipl::vector<3,float> e1(d.normal()),e2(d.cross_product(e1));
            auto f = [&](float a,float b)
            {
                auto u = d+e1*a+e2*b;
                u.normalize();
                return voxel.odf_at(data,u);
            };
            float r = std::sqrt(4.0f*float(M_PI)/float(voxel.ti.vertices_count)),h = r*0.5f,f0 = data.odf[iter->second];
            float p = f(h,0.0f),m = f(-h,0.0f),gx = (p-m)/(2.0f*h),hxx = (p+m-2.0f*f0)/(h*h);
            p = f(0.0f,h);
            m = f(0.0f,-h);
            float gy = (p-m)/(2.0f*h),hyy = (p+m-2.0f*f0)/(h*h),hxy = (f(h,h)-f(h,-h)-f(-h,h)+f(-h,-h))/(4.0f*h*h);
            float det = hxx*hyy-hxy*hxy,a = (hxy*gy-hyy*gx)/det,b = (hxy*gx-hxx*gy)/det;
            auto u = d+e1*a+e2*b;
            u.normalize();
            if(hxx < 0.0f && det > 0.0f && a*a+b*b <= r*r && voxel.odf_at(data,u) >= f0)
                fixel[index][data.voxel_index] = encode_fixel(u);
        }

        iso[data.voxel_index] = min_odf;

        if(!gfa.empty())
            gfa[data.voxel_index] = GeneralizedFA()(data.odf);

        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            fa[index][data.voxel_index] = data.fa[index];

        if(!rdi.empty())
            for (unsigned int index = 0;index < data.rdi.size();++index)
                rdi[index][data.voxel_index] = data.rdi[index];
    }
    virtual void end(Voxel& voxel,tipl::io::gz_mat_write& mat_writer)
    {
        {
            float max_qa = tipl::max_value(fa[0]);
            if(max_qa != 0.0f)
                voxel.z0 = float(1.0/double(max_qa));
        }

        mat_writer.write<tipl::io::masked_sloped>("iso",iso*=voxel.z0,voxel.dim.plane_size());
        for (unsigned int index = 0;index < voxel.max_fiber_number;++index)
            mat_writer.write<tipl::io::masked_sloped>("fa" + std::to_string(index),fa[index]*=voxel.z0,voxel.dim.plane_size());
        mat_writer.write<tipl::io::masked_sloped>("gfa",gfa,voxel.dim.plane_size());

        if(!rdi.empty())
        {
            for(auto& each : rdi)
                each *= voxel.z0;

            float L = 0.2f;
            mat_writer.write<tipl::io::masked_sloped>("rdi",rdi[0],voxel.dim.plane_size());
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
                    mat_writer.write<tipl::io::masked_sloped>(out2.str(),rdi[i],voxel.dim.plane_size());
                }
            }
        }
        for (unsigned int index = 0;index < fixel.size();++index)
            mat_writer.write<tipl::io::masked>("fixel" + std::to_string(index),fixel[index],voxel.dim.plane_size());
        for (unsigned int index = 0;index < findex.size();++index)
            mat_writer.write<tipl::io::masked>("index" + std::to_string(index),findex[index],voxel.dim.plane_size());

    }
};



#endif//ODF_TRANSFORMATION_PROCESS_HPP
