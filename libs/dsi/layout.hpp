#ifndef LAYOUT_HPP
#define LAYOUT_HPP
#include <iterator>
#include "mix_gaussian_model.hpp"
#include "tessellated_icosahedron.hpp"
#include "racian_noise.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"

class Layout
{

private:
    std::vector<basic_sigal_model*> models;
    std::vector<std::shared_ptr<RacianNoise> > rician_gen;
    float discrete_scale;
    float spin_density;
    float mean_dif;
    float encodeNoise(float signal)
    {
        return (*rician_gen[signal/discrete_scale])();
    }
private:
    tessellated_icosahedron ti;
    tipl::geometry<3> dim;
    std::vector<tipl::vector<3,float> > bvectors;
    std::vector<float> bvalues;
private:

public:
    Layout(float s0_snr,float mean_dif_,unsigned char odf_fold = 8,float discrete_scale_ = 0.5,float spin_density_ = 2000.0)
            :mean_dif(mean_dif_),discrete_scale(discrete_scale_),spin_density(spin_density_)
    {
        ti.init(odf_fold);
        for (unsigned int index = 0; index * discrete_scale <= spin_density+1.0; ++index)
            rician_gen.push_back(std::make_shared<RacianNoise>((float)index * discrete_scale,spin_density/s0_snr));
    }
    ~Layout(void)
    {
        for (unsigned int index = 0; index < models.size(); ++index)
            delete models[index];
    }
    bool load_b_table(const char* file_name)
    {
        bvalues.clear();
        bvectors.clear();

        std::ifstream in(file_name);
        if (!in)
            return false;

        std::vector<float> data;
        std::copy(std::istream_iterator<float>(in),
                  std::istream_iterator<float>(),
                  std::back_inserter(data));

        for (unsigned int index = 0; index + 3 < data.size(); index += 4)
        {
            tipl::vector<3,float> v(data[index+1],data[index+2],data[index+3]);
            v.normalize();
            bvalues.push_back(data[index]);
            bvectors.push_back(v);
        }
        return true;
    }

    void createLayout(const char* file_name,
                      float fa_value,
                      const std::vector<float>& angle_iteration,
                      unsigned int repeat_num,
                      unsigned int phantom_width,
                      unsigned int boundary)
    {
        float iso_fraction = 0.2f;
        float fiber_fraction = 1.0f-iso_fraction;
        dim[0] = phantom_width+boundary+boundary;
        dim[1] = phantom_width+boundary+boundary;
        dim[2] = std::max<int>(1,angle_iteration.size())*repeat_num;

        unsigned int total_size = dim.size();
        std::vector<float> fa[2];
        std::vector<float> gfa;
        std::vector<short> findex[2];

        models.resize(total_size);
        fa[0].resize(total_size);
        fa[1].resize(total_size);

        gfa.resize(total_size);
        findex[0].resize(total_size);
        findex[1].resize(total_size);


        unsigned int main_fiber_index = ti.discretize(tipl::vector<3>(1.0,0.0,0.0));

        std::fill(models.begin(),models.end(),(MixGaussianModel*)0);
        begin_prog("creating layout");

        if(angle_iteration.empty()) // use 0 to 90 degrees crossing
            for (unsigned int n = 0,index = 0; n < repeat_num; ++n)
                    {
                        if (!check_prog(index,total_size))
                            break;
                        float fa2 = fa_value*fa_value;
                        //fa*fa = (r*r-2*r+1)/(r*r+2)
                        float r = (1.0+fa_value*std::sqrt(3-2*fa2))/(1-fa2);
                        float l2 = mean_dif*3.0/(2.0+r);
                        float l1 = r*l2;
                        for (unsigned int y = 0; y < dim[1]; ++y)
                        {
                            for (unsigned int x = 0; x < dim[0]; ++x,++index)
                            {
                                if (x >= boundary &&
                                    x < boundary+phantom_width &&
                                    y >= boundary &&
                                    y < boundary+phantom_width)
                                {
                                    float xf = ((float)x - boundary + 1)/((float)phantom_width);//from 0.02 to 1.00
                                    xf = 1.0f-xf;//0.00 to 0.98
                                    xf = 0.5f+0.5f*xf;//0.50 to 0.99
                                    float angle = ((float)y - boundary)/((float)phantom_width);//0.00 to 0.98
                                    angle = 1.0f-angle;//0.02 to 1.00
                                    angle *= float(M_PI*0.5f);//1.8 degrees 90 degrees
                                    models[index] = new MixGaussianModel(l1,l2,mean_dif,angle,
                                                                         fiber_fraction*xf,
                                                                         fiber_fraction*(1.0-xf));
                                    fa[0][index] = fiber_fraction*xf;
                                    fa[1][index] = fiber_fraction*(1.0-xf);
                                    gfa[index] = fa_value;
                                    findex[0][index] = main_fiber_index;
                                    findex[1][index] = ti.discretize(tipl::vector<3>(std::cos(angle),std::sin(angle),0.0));
                                }
                            }
                        }
                    }
        else
            for (unsigned int j = 0,index = 0; j < angle_iteration.size(); ++j)
                for (unsigned int n = 0; n < repeat_num; ++n)
                {
                    if (!check_prog(index,total_size))
                        break;
                    float inner_angle = angle_iteration[j]*M_PI/180.0;
                    float fa2 = fa_value*fa_value;
                    //fa*fa = (r*r-2*r+1)/(r*r+2)
                    float r = (1.0+fa_value*std::sqrt(3-2*fa2))/(1-fa2);
                    float l2 = mean_dif*3.0/(2.0+r);
                    float l1 = r*l2;
                    for (unsigned int y = 0; y < dim[1]; ++y)
                    {
                        for (unsigned int x = 0; x < dim[0]; ++x,++index)
                        {
                            if (x >= boundary &&
                                x < boundary+phantom_width &&
                                y >= boundary &&
                                y < boundary+phantom_width)
                            {
                                if(inner_angle >= 0.0)
                                    models[index] = new MixGaussianModel(l1,l2,mean_dif,inner_angle,0.5,0.5);
                                else
                                    models[index] = new GaussianDispersion(l1,l2,mean_dif,inner_angle,1.0);
                                fa[0][index] = fiber_fraction/2.0;
                                fa[1][index] = fiber_fraction/2.0;
                                gfa[index] = fa_value;
                                findex[0][index] = main_fiber_index;
                                findex[1][index] = ti.discretize(tipl::vector<3>(std::cos(inner_angle),std::sin(inner_angle),0.0));
                            }
                        }
                    }
                }

        set_title("Generating images");

        std::string fib_file_name(file_name);
        fib_file_name += ".layout.fib";
        gz_mat_write mat_writer(file_name),mat_layout(fib_file_name.c_str());
        // output dimension
        {
            mat_writer.write("dimension",&*dim.begin(),1,3);
            mat_layout.write("dimension",&*dim.begin(),1,3);
        }
        // output vexol size
        {
            float vs[3] = {1.0,1.0,1.0};
            mat_writer.write("voxel_size",vs,1,3);
            mat_layout.write("voxel_size",vs,1,3);
        }
        // output b_table
        {
            std::vector<float> buffer;
            buffer.reserve(bvalues.size()*4);
            for (unsigned int index = 0; index < bvalues.size(); ++index)
            {
                buffer.push_back(bvalues[index]);
                std::copy(bvectors[index].begin(),bvectors[index].end(),std::back_inserter(buffer));
            }
            mat_writer.write("b_table",&*buffer.begin(),4,bvalues.size());
        }
        // output images
        {
            std::vector<short> buffer(models.size());
            begin_prog("generating images");
            for (unsigned int index = 0; check_prog(index,bvectors.size()); ++index)
            {
                for (unsigned int i = 0; i < models.size(); ++i)
                {
                    if (models[i])
                        buffer[i] = encodeNoise((*models[i])(bvalues[index]/1000.0,bvectors[index])*spin_density*0.5); // 0.5 volume of water
                    else
                        buffer[i] = encodeNoise(spin_density*exp(-bvalues[index]*0.0016));
                    // water its coefficient is 0.0016 mm?/s
                }
                std::ostringstream out;
                out << "image" << index;
                mat_writer.write(out.str().c_str(),&*buffer.begin(),1,buffer.size());
            }
        }
        // output layout
        {
            std::vector<float> float_data;
            std::vector<short> short_data;
            ti.save_to_buffer(float_data,short_data);
            mat_layout.write("odf_vertices",&*float_data.begin(),3,ti.vertices_count);
            mat_layout.write("odf_faces",&*short_data.begin(),3,ti.faces.size());
            mat_layout.write("fa0",&*fa[0].begin(),1,fa[0].size());
            mat_layout.write("fa1",&*fa[1].begin(),1,fa[1].size());
            mat_layout.write("gfa",&*gfa.begin(),1,gfa.size());
            mat_layout.write("index0",&*findex[0].begin(),1,findex[0].size());
            mat_layout.write("index1",&*findex[1].begin(),1,findex[1].size());

        }


    }
};




#endif//LAYOUT_HPP
