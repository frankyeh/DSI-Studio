#include <boost/thread.hpp>
#include <QFileInfo>
#include <QApplication>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include "mapping/fa_template.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/atlas.hpp"

namespace po = boost::program_options;
extern fa_template fa_template_imp;
extern std::vector<atlas> atlas_list;
std::string get_fa_template_path(void);

bool atl_load_atlas(std::string atlas_name)
{
    std::cout << "loading atlas..." << std::endl;
    std::replace(atlas_name.begin(),atlas_name.end(),',',' ');
    std::istringstream in(atlas_name);
    std::vector<std::string> name_list;
    std::copy(std::istream_iterator<std::string>(in),
              std::istream_iterator<std::string>(),std::back_inserter(name_list));

    for(unsigned int index = 0;index < name_list.size();++index)
    {
        std::string atlas_path = QCoreApplication::applicationDirPath().toLocal8Bit().begin();
        atlas_path += "/atlas/";
        atlas_path += name_list[index];
        atlas_path += ".nii.gz";
        atlas_list.push_back(atlas());
        if(!atlas_list.back().load_from_file(atlas_path.c_str()))
        {
            std::cout << "Cannot load atlas " << atlas_path << std::endl;
            return false;
        }
        std::cout << name_list[index] << " loaded." << std::endl;
        atlas_list.back().name = name_list[index];

    }
    return true;
}
void atl_get_mapping(image::basic_image<float,3>& from,
                     image::basic_image<float,3>& to,
                     const image::vector<3>& vs,
                     unsigned int factor,
                     unsigned int thread_count,
                     image::basic_image<image::vector<3>,3>& mapping,
                     float* out_trans)
{
    std::cout << "perform image registration..." << std::endl;
    image::affine_transform<3,float> arg;
    arg.scaling[0] = vs[0] / std::fabs(fa_template_imp.tran[0]);
    arg.scaling[1] = vs[1] / std::fabs(fa_template_imp.tran[5]);
    arg.scaling[2] = vs[2] / std::fabs(fa_template_imp.tran[10]);
    image::reg::align_center(from,to,arg);

    image::filter::gaussian(from);
    from -= image::segmentation::otsu_threshold(from);
    image::lower_threshold(from,0.0);

    image::normalize(from,1.0);
    image::normalize(to,1.0);

    bool terminated = false;
    std::cout << "perform linear registration..." << std::endl;
    image::reg::linear(from,to,arg,image::reg::affine,image::reg::mutual_information(),terminated);
    image::transformation_matrix<3,float> T(arg,from.geometry(),to.geometry()),iT(arg,from.geometry(),to.geometry());
    iT.inverse();


    // output linear registration
    float T_buf[16];
    T.save_to_transform(T_buf);
    T_buf[15] = 1.0;
    std::copy(T_buf,T_buf+4,std::ostream_iterator<float>(std::cout," "));
    std::cout << std::endl;
    std::copy(T_buf+4,T_buf+8,std::ostream_iterator<float>(std::cout," "));
    std::cout << std::endl;
    std::copy(T_buf+8,T_buf+12,std::ostream_iterator<float>(std::cout," "));
    std::cout << std::endl;


    image::basic_image<float,3> new_from(to.geometry());
    image::resample(from,new_from,iT);


    std::cout << "perform nonlinear registration..." << std::endl;
    //image::reg::bfnorm(new_from,to,*bnorm_data,*terminated);

    std::cout << "order=" << factor << std::endl;
    std::cout << "thread count=" << thread_count << std::endl;

    image::reg::bfnorm_mapping<float,3> mni(new_from.geometry(),image::geometry<3>(factor*7,factor*9,factor*7));
    multi_thread_reg(mni,new_from,to,thread_count,terminated);
    mapping.resize(from.geometry());
    for(image::pixel_index<3> index;from.geometry().is_valid(index);index.next(from.geometry()))
        if(from[index.index()] > 0)
        {
            image::vector<3,float> pos;
            T(index,pos);// from -> new_from
            mni(pos,mapping[index.index()]); // new_from -> to
            fa_template_imp.to_mni(mapping[index.index()]);
        }
    image::matrix::product(fa_template_imp.tran.begin(),T_buf,out_trans,image::dyndim(4,4),image::dyndim(4,4));
}

void atl_save_mapping(const std::string& file_name,const image::geometry<3>& geo,
                      const image::basic_image<image::vector<3>,3>& mapping,const float* trans,const float* vs,
                      bool multiple)
{
    for(unsigned int i = 0;i < atlas_list.size();++i)
    {
        std::string base_name = file_name;
        base_name += ".";
        base_name += atlas_list[i].name;
        image::basic_image<short,3> all_roi(geo);
        for(unsigned int j = 0;j < atlas_list[i].get_list().size();++j)
        {
            std::string output = base_name;
            output += ".";
            output += atlas_list[i].get_list()[j];
            output += ".nii.gz";

            image::basic_image<unsigned char,3> roi(geo);
            for(unsigned int k = 0;k < mapping.size();++k)
                if (atlas_list[i].is_labeled_as(mapping[k], j))
                {
                    roi[k] = 1;
                    all_roi[k] = atlas_list[i].get_label_at(mapping[k]);
                }
            if(multiple)
            {
                image::io::nifti out;
                out.set_voxel_size(vs);
                if(trans)
                    out.set_image_transformation(trans);
                else
                    image::flip_xy(roi);
                out << roi;
                out.save_to_file(output.c_str());
                std::cout << "save " << output << std::endl;
            }
        }
        {
            std::string label_name = base_name;
            label_name += ".txt";
            std::ofstream txt_out(label_name.c_str());
            for(unsigned int j = 0;j < atlas_list[i].get_list().size();++j)
                txt_out << atlas_list[i].get_num()[j] << " " << atlas_list[i].get_list()[j] << std::endl;
        }
        base_name += ".nii.gz";
        image::io::nifti out;
        out.set_voxel_size(vs);
        if(trans)
            out.set_image_transformation(trans);
        else
            image::flip_xy(all_roi);
        out << all_roi;
        out.save_to_file(base_name.c_str());
        std::cout << "save " << base_name << std::endl;
    }
}

int atl(int ac, char *av[])
{
    po::options_description norm_desc("fiber tracking options");
    norm_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "atl: output atlas")
    ("source", po::value<std::string>(), "assign the .fib file name")
    ("order", po::value<int>()->default_value(0), "normalization order (0~3)")
    ("thread_count", po::value<int>()->default_value(4), "thread count")
    ("atlas", po::value<std::string>(), "atlas name")
    ("output", po::value<std::string>()->default_value("multiple"), "output files")
    ;

    if(!ac)
    {
        std::cout << norm_desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(norm_desc).run(), vm);
    po::notify(vm);


    gz_mat_read mat_reader;
    std::string file_name = vm["source"].as<std::string>();
    std::cout << "loading " << file_name << "..." <<std::endl;
    if(!QFileInfo(file_name.c_str()).exists())
    {
        std::cout << file_name << " does not exist. terminating..." << std::endl;
        return 0;
    }
    if (!mat_reader.load_from_file(file_name.c_str()))
    {
        std::cout << "Invalid MAT file format" << std::endl;
        return 0;
    }

    unsigned int col,row;
    const unsigned short* dim = 0;
    const float* vs = 0;
    const float* fa0 = 0;
    if(!mat_reader.read("dimension",row,col,dim) ||
       !mat_reader.read("voxel_size",row,col,vs) ||
       !mat_reader.read("fa0",row,col,fa0))
    {
        std::cout << "Invalid file format" << std::endl;
        return 0;
    }
    image::geometry<3> geo(dim);


    if(!fa_template_imp.load_from_file(get_fa_template_path().c_str()) ||
       !atl_load_atlas(vm["atlas"].as<std::string>()))
        return -1;


    const float* trans = 0;
    //QSDR
    if(mat_reader.read("trans",row,col,trans))
    {
        std::cout << "Transformation matrix found." << std::endl;
        image::basic_image<image::vector<3>,3> mapping(geo);
        for(image::pixel_index<3> index;geo.is_valid(index);index.next(geo))
        {
            image::vector<3,float> pos(index),mni;
            image::vector_transformation(pos.begin(),mni.begin(),trans,image::vdim<3>());
            mapping[index.index()] = mni;
        }
        atl_save_mapping(file_name,geo,mapping,trans,vs,vm["output"].as<std::string>() == "multiple");
        return 0;
    }

    image::basic_image<float,3> from(fa0,geo);
    image::basic_image<image::vector<3>,3> mapping;
    unsigned int factor = vm["order"].as<int>() + 1;
    unsigned int thread_count = vm["thread_count"].as<int>();
    image::vector<3> vs_(vs);
    float out_trans[16];
    atl_get_mapping(from,fa_template_imp.I,vs_,factor,thread_count,mapping,out_trans);
    atl_save_mapping(file_name,geo,mapping,0,vs,vm["output"].as<std::string>() == "multiple");
}
