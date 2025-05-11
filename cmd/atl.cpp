#include <QFileInfo>
#include <QApplication>
#include <QDir>
#include "fib_data.hpp"
#include "mapping/atlas.hpp"
#include "connectometry/group_connectometry_analysis.h"

#include <filesystem>

extern std::vector<std::string> fa_template_list;
bool atl_load_atlas(std::shared_ptr<fib_data> handle,std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list)
{
    QStringList name_list = QString(atlas_name.c_str()).split(",");
    for(int index = 0;index < name_list.size();++index)
    {
        auto at = handle->get_atlas(name_list[index].toStdString());
        if(!at.get())
        {
            tipl::error() << handle->error_msg << std::endl;
            return false;
        }
        atlas_list.push_back(at);
    }
    return true;
}

std::vector<std::string> get_source_list(tipl::program_option<tipl::out>& po)
{
    std::string source = po.get("source");
    std::vector<std::string> name_list;
    if(QFileInfo(source.c_str()).isDir())
    {
        tipl::out() << "Searching all fib files in directory " << source << std::endl;
        tipl::search_filesystem<tipl::out,tipl::error>(source + "/*.fib.gz",name_list);
    }
    else
        po.get_files("source",name_list);
    return name_list;
}
int db(tipl::program_option<tipl::out>& po)
{
    std::vector<std::string> name_list = get_source_list(po);
    if(name_list.empty())
    {
        tipl::error() << "no file found in " << po.get("source") << std::endl;
        return 1;
    }

    if(po.has("demo") && !std::filesystem::exists(po.get("demo")))
    {
        tipl::error() << "cannot find demo file " << po.get("demo") <<std::endl;
        return 1;
    }


    std::vector<std::string> index_name;

    // get the name of all metrics from the first file
    std::vector<std::string> item_list;
    float template_reso = 1.0f;
    {
        fib_data fib;
        if(!fib.load_from_file(name_list[0].c_str()))
        {
            tipl::error() << "cannot load subject fib " << name_list[0] << std::endl;
            return 1;
        }
        std::ostringstream out;
        for(const auto& each: fib.get_index_list())
            out << each << " ";
        tipl::out() << "available metrics: " << out.str() << std::endl;
        template_reso = fib.vs[0];
    }

    {
        fib_data fib;
        if(!fib.load_template_fib(po.get("template",0),template_reso) ||
           !fib.db.create_db(name_list,tipl::split(po.get("index_name","dti_fa,qa,rdi,iso"),',')) ||
           (po.has("demo") && !fib.db.parse_demo(po.get("demo"))))
        {
            tipl::error() <<  fib.error_msg << std::endl;
            return 1;
        }

        if(po.has("intro"))
        {
            std::ifstream file(po.get("intro"));
            fib.intro = std::string(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());
        }


        std::string output = std::string(name_list.front().begin(),
                                         std::mismatch(name_list.front().begin(),name_list.front().begin()+
                                         int64_t(std::min(name_list.front().length(),name_list.back().length())),
                                           name_list.back().begin()).first) + ".dz";
        tipl::out() << "saving " << po.get("output",output);
        if(!fib.save_to_file(po.get("output",output)))
        {
            tipl::error() << "cannot save db file " << fib.error_msg << std::endl;
            return 1;
        }
        tipl::out() << "saving " << (po.get("output",output)+".R2.tsv");
        std::ofstream out((po.get("output",output)+".R2.tsv").c_str());
        out << "name\tR2" << std::endl;
        for(size_t i = 0;i < fib.db.subject_names.size();++i)
            out << fib.db.subject_names[i] << "\t" << fib.db.R2[i] << std::endl;
    }
    return 0;
}
bool odf_average(const char* out_name,std::vector<std::string>& file_names,std::string& error_msg);
int tmp(tipl::program_option<tipl::out>& po)
{
    std::vector<std::string> name_list = get_source_list(po);
    if(name_list.empty())
    {
        tipl::error() << "no file found in " << po.get("source") << std::endl;
        return 1;
    }
    std::string error_msg;
    tipl::out() << "constructing a group average template" << std::endl;
    if(tipl::ends_with(name_list[0],".fib.gz") ||
       tipl::ends_with(name_list[0],".fz"))
    {
        if(!odf_average(po.get("output",(QFileInfo(name_list[0].c_str()).absolutePath()+"/template").toStdString()).c_str(),name_list,error_msg))
        {
            tipl::error() << error_msg;
            return 1;
        }
        return 0;
    }
    if(tipl::ends_with(name_list[0],".nii.gz"))
    {
        if(!odf_average(po.get("output",name_list[0] + ".avg.nii.gz").c_str(),name_list,error_msg))
        {
            tipl::error() << error_msg;
            return 1;
        }
        return 0;
    }
    tipl::error() << "unsupported format" << std::endl;
    return 1;
}
std::shared_ptr<fib_data> cmd_load_fib(tipl::program_option<tipl::out>& po);
int atl(tipl::program_option<tipl::out>& po)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po);
    if(!handle.get())
    {
        tipl::error() << handle->error_msg << std::endl;
        return 1;
    }
    std::vector<std::shared_ptr<atlas> > atlas_list;
    if(!atl_load_atlas(handle,po.get("atlas"),atlas_list))
        return 1;
    if(handle->get_sub2temp_mapping().empty())
    {
        tipl::error() << "cannot output connectivity: no mni mapping" << std::endl;
        return 1;
    }
    std::string file_name = po.get("source");
    const auto& mapping = handle->get_sub2temp_mapping();
    bool multiple = (po.get("output","multiple") == "multiple");

    for(unsigned int i = 0;i < atlas_list.size();++i)
    {
        std::string base_name = file_name;
        base_name += ".";
        base_name += atlas_list[i]->name;
        for(unsigned int j = 0;j < atlas_list[i]->get_list().size();++j)
        {
            std::string output = base_name;
            output += ".";
            output += atlas_list[i]->get_list()[j];
            output += ".nii.gz";
            tipl::image<3,unsigned char> roi(handle->dim);
            for(unsigned int k = 0;k < mapping.size();++k)
                if (atlas_list[i]->is_labeled_as(mapping[k], j))
                    roi[k] = 1;
            if(multiple)
            {
                tipl::out() << "saving " << output << std::endl;
                if(!tipl::io::gz_nifti::save_to_file(output.c_str(),roi,handle->vs,handle->trans_to_mni,handle->is_mni))
                    tipl::out() << "cannot write output to " << output << std::endl;
            }
        }
        {
            std::string label_name = base_name;
            label_name += ".txt";
            std::ofstream txt_out(label_name.c_str());
            for(unsigned int j = 0;j < atlas_list[i]->get_list().size();++j)
                txt_out << atlas_list[i]->get_num()[j] << " " << atlas_list[i]->get_list()[j] << std::endl;
        }
    }
    return 0;
}
