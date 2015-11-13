#include <boost/thread.hpp>
#include <QFileInfo>
#include <QApplication>
#include "image/image.hpp"
#include "boost/program_options.hpp"
#include "mapping/fa_template.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/atlas.hpp"
#include "manual_alignment.h"
namespace po = boost::program_options;
extern fa_template fa_template_imp;
extern std::vector<atlas> atlas_list;
std::string get_fa_template_path(void);

bool atl_load_atlas(std::string atlas_name)
{
    std::cout << "loading atlas..." << std::endl;
    QStringList name_list = QString(atlas_name.c_str()).split(",");


    for(unsigned int index = 0;index < name_list.size();++index)
    {
        bool has_atlas = false;
        for(unsigned int i = 0;i < atlas_list.size();++i)
            if(atlas_list[i].name == name_list[index].toStdString())
                has_atlas = true;
        if(has_atlas)
            continue;
        std::string file_path;
        if(QFileInfo(name_list[index]).exists())
        {
            file_path = name_list[index].toStdString();
            name_list[index] = QFileInfo(name_list[index]).baseName();
        }
        else
        {
            std::string atlas_path = QCoreApplication::applicationDirPath().toStdString();
            atlas_path += "/atlas/";
            atlas_path += name_list[index].toStdString();
            atlas_path += ".nii.gz";
            if(QFileInfo(atlas_path.c_str()).exists())
                file_path = atlas_path;
            else
            {
                std::cout << "Load " << name_list[index].toStdString() << " failed. Cannot find file in " << atlas_path << std::endl;
                return false;
            }
        }

        {
            std::cout << "loading " << name_list[index].toStdString() << "..." << std::endl;
            atlas_list.push_back(atlas());
            atlas_list.back().filename = file_path;
            atlas_list.back().name = name_list[index].toStdString();
            if(atlas_list.back().get_num().empty())
            {
                std::cout << "Invalid file format. No ROI found in " << name_list[index].toStdString() << "." << std::endl;
                return false;
            }
            continue;
        }
    }
    return true;
}
bool atl_get_mapping(gz_mat_read& mat_reader,
                     unsigned int factor,
                     unsigned int thread_count,
                     image::basic_image<image::vector<3>,3>& mapping)
{
    unsigned int col,row;
    const unsigned short* dim = 0;
    const float* vs = 0;
    const float* fa0 = 0;
    if(!mat_reader.read("dimension",row,col,dim) ||
       !mat_reader.read("voxel_size",row,col,vs) ||
       !mat_reader.read("fa0",row,col,fa0))
    {
        std::cout << "Invalid file format" << std::endl;
        return false;
    }
    image::geometry<3> geo(dim);
    image::vector<3> voxel_size(vs);

    if(fa_template_imp.I.empty() && !fa_template_imp.load_from_file())
        return false;
    //QSDR
    const float* trans = 0;
    mapping.resize(geo);
    if(mat_reader.read("trans",row,col,trans))
    {
        std::cout << "Transformation matrix found." << std::endl;
        for(image::pixel_index<3> index;geo.is_valid(index);index.next(geo))
        {
            image::vector<3> pos(index);
            pos.to(trans);
            mapping[index.index()] = pos;
        }
    }
    else
    {
        std::cout << "Conduct spatial warping: " << thread_count << "-thread, " << factor << "-factor" << std::endl;
        image::basic_image<float,3> from(fa0,geo),to(fa_template_imp.I);
        reg_data data(fa_template_imp.I.geometry(),image::reg::affine,factor);

        image::filter::gaussian(from);
        from -= image::segmentation::otsu_threshold(from);
        image::lower_threshold(from,0.0);
        image::normalize(from,1.0);
        image::normalize(to,1.0);
        run_reg(from,voxel_size,fa_template_imp.I,fa_template_imp.vs,data,thread_count);
        image::transformation_matrix<float> T(data.arg,from.geometry(),voxel_size,fa_template_imp.I.geometry(),fa_template_imp.vs);
        mapping.resize(from.geometry());
        for(image::pixel_index<3> index;from.geometry().is_valid(index);index.next(from.geometry()))
            if(from[index.index()] > 0)
            {
                image::vector<3,float> pos;
                T(index,pos);// from -> new_from
                data.bnorm_data(pos,mapping[index.index()]);
                fa_template_imp.to_mni(mapping[index.index()]);
            }
    }
    return true;
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
    ("thread_count", po::value<int>()->default_value(boost::thread::hardware_concurrency()), "thread count")
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

    if(!atl_load_atlas(vm["atlas"].as<std::string>()))
        return 0;

    unsigned int factor = vm["order"].as<int>() + 1;
    unsigned int thread_count = vm["thread_count"].as<int>();
    std::cout << "Reg order = " << factor << std::endl;
    std::cout << "Thread count = " << thread_count << std::endl;

    const float *trans = 0;
    unsigned int col,row;
    const unsigned short* dim = 0;
    const float* vs = 0;
    if(!mat_reader.read("dimension",row,col,dim) ||
       !mat_reader.read("voxel_size",row,col,vs))
    {
        std::cout << "Invalid file format" << std::endl;
        return 0;
    }
    mat_reader.read("trans",row,col,trans);
    image::basic_image<image::vector<3>,3> mapping;
    if(!atl_get_mapping(mat_reader,factor,thread_count,mapping))
        return 0;
    atl_save_mapping(file_name,image::geometry<3>(dim),mapping,trans,vs,vm["output"].as<std::string>() == "multiple");
    return 0;
}
