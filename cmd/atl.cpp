
#include <QFileInfo>
#include <QApplication>
#include <QDir>
#include "image/image.hpp"
#include "mapping/fa_template.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/atlas.hpp"
#include "program_option.hpp"
#include "fib_data.hpp"

extern fa_template fa_template_imp;
extern std::vector<atlas> atlas_list;
std::string get_fa_template_path(void);
const char* odf_average(const char* out_name,std::vector<std::string>& file_names);
bool atl_load_atlas(std::string atlas_name)
{
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
bool atl_get_mapping(std::shared_ptr<fib_data> handle,
                     unsigned int factor,
                     image::basic_image<image::vector<3>,3>& mapping)
{
    if(fa_template_imp.I.empty() && !fa_template_imp.load_from_file())
        return false;
    if(!handle->is_qsdr)
    {
        std::cout << "Conduct spatial warping with norm factor of " << factor << std::endl;
        handle->run_normalization(factor,false/*not background*/);
    }
    handle->get_mni_mapping(mapping);
    return true;
}

void atl_save_mapping(const std::string& file_name,const image::geometry<3>& geo,
                      const image::basic_image<image::vector<3>,3>& mapping,
                      const std::vector<float>& trans,
                      const image::vector<3>& vs,
                      bool multiple)
{
    for(unsigned int i = 0;i < atlas_list.size();++i)
    {
        std::string base_name = file_name;
        base_name += ".";
        base_name += atlas_list[i].name;
        for(unsigned int j = 0;j < atlas_list[i].get_list().size();++j)
        {
            std::string output = base_name;
            output += ".";
            output += atlas_list[i].get_list()[j];
            output += ".nii.gz";

            image::basic_image<unsigned char,3> roi(geo);
            for(unsigned int k = 0;k < mapping.size();++k)
                if (atlas_list[i].is_labeled_as(mapping[k], j))
                    roi[k] = 1;
            if(multiple)
            {
                image::io::nifti out;
                out.set_voxel_size(vs);
                if(!trans.empty())
                    out.set_image_transformation(trans.begin());
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
    }
}
std::shared_ptr<fib_data> cmd_load_fib(const std::string file_name);

int atl(void)
{
    // construct an atlas
    if(po.get("order",int(0)) == -1)
    {
        std::string dir = po.get("source");
        std::cout << "Constructing an atlas" << std::endl;
        std::cout << "Loading fib file in " << dir << std::endl;
        QDir directory(QString(dir.c_str()));
        QStringList file_list = directory.entryList(QStringList("*.fib.gz"),QDir::Files);
        if(file_list.empty())
        {
            std::cout << "Cannot find fib file to construct an atlas" << std::endl;
            return 0;
        }
        std::vector<std::string> name_list;
        for (unsigned int index = 0;index < file_list.size();++index)
        {
            std::string file_name = dir;
            file_name += "/";
            file_name += file_list[index].toStdString();
            if(!file_list[index].contains("rec"))
            {
                std::cout << file_list[index].toStdString() << " seems not containing QSDR ODF information. Skipping..." << std::endl;
                continue;
            }
            name_list.push_back(file_name);
        }
        dir += "/";
        dir += "atlas";
        const char* msg = odf_average(dir.c_str(),name_list);
        if(msg)
            std::cout << msg << std::endl;
        return 0;
    }


    std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
    if(!handle.get())
    {
        std::cout << handle->error_msg << std::endl;
        return 0;
    }
    if(!atl_load_atlas(po.get("atlas")))
        return 0;

    image::basic_image<image::vector<3>,3> mapping;
    if(!atl_get_mapping(handle,po.get("order",int(0)) + 1,mapping))
        return 0;
    atl_save_mapping(po.get("source"),handle->dim,
                     mapping,handle->trans_to_mni,handle->vs,
                     po.get("output","multiple") == "multiple");
    return 0;
}
