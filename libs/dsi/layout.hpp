#ifndef LAYOUT_HPP
#define LAYOUT_HPP
#include <iterator>
#include "tessellated_icosahedron.hpp"

class Layout
{

private:
    std::vector<MixGaussianModel*> models;
    boost::ptr_vector<RacianNoise> rician_gen;
    float discrete_scale;
    float spin_density;
    float mean_dif;
    float encodeNoise(float signal)
    {
        return rician_gen[signal/discrete_scale]();
    }
private:
    tessellated_icosahedron ti;
    short dim[3];
    std::vector<image::vector<3,float> > bvectors;
    std::vector<float> bvalues;
    std::vector<float> fa[2];
    std::vector<float> gfa;
    std::vector<float> vf[3];
    std::vector<short> findex[2];
private:

public:
    Layout(float s0_snr,float mean_dif_,float discrete_scale_ = 0.5,float spin_density_ = 2000.0)
            :mean_dif(mean_dif_),discrete_scale(discrete_scale_),spin_density(spin_density_)
    {
        for (unsigned int index = 0; index * discrete_scale <= spin_density+1.0; ++index)
            rician_gen.push_back(new RacianNoise((float)index * discrete_scale,spin_density/s0_snr));
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
            image::vector<3,float> v(data[index+1],data[index+2],data[index+3]);
            v.normalize();
            bvalues.push_back(data[index]);
            bvectors.push_back(v);
        }
        return true;
    }

    void createLayout(const std::vector<float>& fa_iteration,
                      const std::vector<float>& angle_iteration,
                      unsigned int repeat_num)
    {
        dim[0] = 128;
        dim[1] = 128;
        dim[2] = fa_iteration.size()*angle_iteration.size()*repeat_num;

        unsigned int total_size = 128*128*fa_iteration.size()*angle_iteration.size()*repeat_num;
        models.resize(total_size);
        fa[0].resize(total_size);
        fa[1].resize(total_size);
        vf[0].resize(total_size);
        vf[1].resize(total_size);
        vf[2].resize(total_size);

        gfa.resize(total_size);
        findex[0].resize(total_size);
        findex[1].resize(total_size);


        unsigned int main_fiber_index = ti.discretize(1.0,0.0,0.0);

        std::fill(models.begin(),models.end(),(MixGaussianModel*)0);
        begin_prog("creating layout");

        for (unsigned int i = 0,index = 0; i < fa_iteration.size(); ++i)
            for (unsigned int j = 0; j < angle_iteration.size(); ++j)
                for (unsigned int n = 0; n < repeat_num; ++n)
                {
                    if (!check_prog(index,total_size))
                        break;
                    float fa_value = fa_iteration[i];
                    float inner_angle = angle_iteration[j]*M_PI/180.0;
                    float fa2 = fa_value*fa_value;
                    //fa*fa = (r*r-2*r+1)/(r*r+2)
                    float r = (1.0+fa_value*std::sqrt(3-2*fa2))/(1-fa2);
                    float l2 = mean_dif*3.0/(2.0+r);
                    float l1 = r*l2;
                    float l0 = mean_dif;
                    for (unsigned int y = 0; y < 128; ++y)
                    {
                        float fraction = 0.5+0.5*((float)y - 32.0)/64.0;//from 0.5 to 1.0
                        for (unsigned int x = 0; x < 128; ++x,++index)
                        {
                            if (x >= 32 && x <= 96 && y >= 32 && y <= 96)
                            {
                                float fiber_fraction = 0.5+0.5*((float)x - 32.0)/64.0;//from 0.5 to 1.0
                                //models[index] = new MixGaussianModel(l1,l2,mean_dif,inner_angle,fiber_fraction*fraction,fiber_fraction*(1.0-fraction));
                                models[index] = new MixGaussianModel(l1,l2,mean_dif,inner_angle,0.5,0.5);
                                fa[0][index] = fraction;
                                fa[1][index] = 1.0-fraction;
                                gfa[index] = fa_value;
                                findex[0][index] = main_fiber_index;
                                findex[1][index] = ti.discretize(std::cos(inner_angle),std::sin(inner_angle),0.0);
                                vf[1][index] = fiber_fraction*fraction;
                                vf[2][index] = fiber_fraction*(1.0-fraction);
                                vf[0][index] = 1.0-fiber_fraction;
                            }
                        }
                    }
                }
    }
    void generate(const char* file_name)
    {
        set_title("Generating images");

        std::string fib_file_name(file_name);
        fib_file_name += ".layout.fib";
        MatFile mat_writer(file_name),mat_layout(fib_file_name.c_str());
        // output dimension
        {
            mat_writer.add_matrix("dimension",dim,1,3);
            mat_layout.add_matrix("dimension",dim,1,3);
        }
        // output vexol size
        {
            float vs[3] = {1.0,1.0,1.0};
            mat_writer.add_matrix("voxel_size",vs,1,3);
            mat_layout.add_matrix("voxel_size",vs,1,3);
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
            mat_writer.add_matrix("b_table",&*buffer.begin(),4,bvalues.size());
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
                mat_writer.add_matrix(out.str().c_str(),&*buffer.begin(),1,buffer.size());
            }
        }
        // output layout
        {
            std::vector<float> float_data;
            std::vector<short> short_data;
            ti.save_to_buffer(float_data,short_data);
            mat_layout.add_matrix("odf_vertices",&*float_data.begin(),3,ti.vertices_count);
            mat_layout.add_matrix("odf_faces",&*short_data.begin(),3,ti.faces.size());

            mat_layout.add_matrix("fa0",&*fa[0].begin(),1,fa[0].size());
            mat_layout.add_matrix("fa1",&*fa[1].begin(),1,fa[1].size());
            std::fill(fa[0].begin(),fa[0].end(),0);
            mat_layout.add_matrix("fa2",&*fa[0].begin(),1,fa[0].size());

            mat_layout.add_matrix("gfa",&*gfa.begin(),1,gfa.size());

            mat_layout.add_matrix("index0",&*findex[0].begin(),1,findex[0].size());
            mat_layout.add_matrix("index1",&*findex[1].begin(),1,findex[1].size());
            std::fill(findex[0].begin(),findex[0].end(),0);
            mat_layout.add_matrix("index2",&*findex[0].begin(),1,findex[0].size());

            mat_layout.add_matrix("vf0",&*vf[0].begin(),1,vf[0].size());
            mat_layout.add_matrix("vf1",&*vf[1].begin(),1,vf[1].size());
            mat_layout.add_matrix("vf2",&*vf[2].begin(),1,vf[2].size());
        }


    }
};




#endif//LAYOUT_HPP
