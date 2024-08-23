#include <QFileInfo>
#include <QApplication>
#include <QDir>
#include "fib_data.hpp"
#include "mapping/atlas.hpp"
#include "connectometry/group_connectometry_analysis.h"

#include <filesystem>

extern std::vector<std::string> fa_template_list,fib_template_list;
const char* odf_average(const char* out_name,std::vector<std::string>& file_names);
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

std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);
int atl(tipl::program_option<tipl::out>& po)
{
    // construct an atlas
    std::string cmd = po.get("cmd");

    std::string source = po.get("source");
    std::vector<std::string> name_list;
    if(QFileInfo(source.c_str()).isDir())
    {
        tipl::out() << "Searching all fib files in directory " << source << std::endl;
        if(!tipl::search_filesystem<tipl::out,tipl::error>(source + "/*.fib.gz",name_list))
            return 1;
    }
    else
    {
        if(!po.get_files("source",name_list))
            return 1;
    }

    if(name_list.empty())
    {
        tipl::error() << "no file found in " << source << std::endl;
        return 1;
    }

    if(cmd=="template")
    {
        tipl::out() << "constructing a group average template" << std::endl;
        if(tipl::ends_with(name_list[0],".fib.gz") ||
           tipl::ends_with(name_list[0],".fz"))
        {
            const char* msg = odf_average(po.get("output",(QFileInfo(name_list[0].c_str()).absolutePath()+"/template").toStdString()).c_str(),name_list);
            if(msg)
            {
                tipl::error() << msg << std::endl;
                return 1;
            }
        }
        if(tipl::ends_with(name_list[0],".nii.gz"))
        {
            tipl::image<3,double> sum;
            tipl::image<3> each;
            tipl::vector<3> vs;
            tipl::matrix<4,4> T;
            bool is_mni;
            for(auto name : name_list)
            {
                tipl::out() << "adding " << name;
                if(!tipl::io::gz_nifti::load_from_file(name.c_str(),each,vs,T,is_mni))
                {
                    tipl::error() << "cannot load file" << name_list[0] << std::endl;
                    return 1;
                }
                if(sum.empty())
                    sum.resize(each.shape());
                sum += each;
            }
            sum /= name_list.size();
            auto output = po.get("output",name_list[0] + ".avg.nii.gz");
            tipl::out() << "saving " << output;
            if(!tipl::io::gz_nifti::save_to_file(output.c_str(),sum,vs,T,is_mni))
            {
                tipl::error() << "cannot save file" << output << std::endl;
                return 1;
            }
            return 0;
        }
        tipl::error() << "unsupported format" << std::endl;
        return 1;
    }
    if(cmd=="db")
    {        
        for(size_t id = 0;id < fib_template_list.size();++id)
            if(!fib_template_list[id].empty())
                tipl::out() << "template " << id << ": " << std::filesystem::path(fib_template_list[id]).stem() << std::endl;

        auto template_id = po.get("template",0);
        if(template_id >= fib_template_list.size())
        {
            tipl::error() << "invalid template value" << std::endl;
            return 1;
        }
        if(fib_template_list[template_id].empty())
        {
            tipl::error() << "no FIB template for " <<  std::filesystem::path(fa_template_list[template_id]).stem().stem().stem() << std::endl;
            return 1;
        }

        std::shared_ptr<fib_data> template_fib(new fib_data);
        if(!template_fib->load_from_file(fib_template_list[template_id].c_str()))
        {
            tipl::error() <<  template_fib->error_msg << std::endl;
            return 1;
        }
        template_fib->set_template_id(template_id);


        tipl::out() << "constructing a connectometry db" << std::endl;
        std::vector<std::string> index_name;
        float reso = template_fib->vs[0];

        // get the name of all metrics from the first file
        std::vector<std::string> item_list;
        {
            fib_data fib;
            if(!fib.load_from_file(name_list[0].c_str()))
            {
                tipl::error() << "cannot load subject fib " << name_list[0] << std::endl;
                return 1;
            }
            reso = po.get("resolution",std::floor((fib.vs[0] + fib.vs[2])*0.5f*100.0f)/100.0f);
            std::ostringstream out;
            for(const auto& each: fib.get_index_list())
                out << each << " ";
            tipl::out() << "available metrics: " << out.str() << std::endl;
        }

        if(po.get("index_name","qa") == std::string("*"))
        {
            for(size_t i = 0;i < item_list.size();++i)
                index_name.push_back(item_list[i]);
        }
        else
        {
            std::istringstream in(po.get("index_name","qa"));
            std::string line;
            while(std::getline(in,line,','))
                index_name.push_back(line);
        }

        if(reso > template_fib->vs[0] && !template_fib->resample_to(reso))
        {
            tipl::error() << template_fib->error_msg << std::endl;
            return 1;
        }

        for(size_t i = 0; i < index_name.size();++i)
        {
            std::shared_ptr<group_connectometry_analysis> data(new group_connectometry_analysis);
            if(!data->create_database(template_fib))
            {
                tipl::error() << data->error_msg << std::endl;
                return 1;
            }
            tipl::out() << "extracting " << index_name[i] << std::endl;
            data->handle->db.index_name = index_name[i];
            for (unsigned int index = 0;index < name_list.size();++index)
            {
                if(tipl::ends_with(name_list[index],".db.fib.gz") ||
                   tipl::ends_with(name_list[index],".db.fz"))
                    continue;
                tipl::out() << "reading " << name_list[index] << std::endl;
                if(!data->handle->db.add(name_list[index],
                    QFileInfo(name_list[index].c_str()).baseName().toStdString()))
                {
                    tipl::error() << "failed to load subject fib file " << data->handle->db.error_msg << std::endl;
                    return 1;
                }
            }
            // Output

            if(po.has("demo") && !data->handle->db.parse_demo(po.get("demo")))
            {
                tipl::error() << data->handle->db.error_msg <<std::endl;
                return 1;
            }
            std::string output = std::string(name_list.front().begin(),
                                             std::mismatch(name_list.front().begin(),name_list.front().begin()+
                                             int64_t(std::min(name_list.front().length(),name_list.back().length())),
                                               name_list.back().begin()).first) + "." + index_name[i] + ".db.fz";
            tipl::out() << "saving " << po.get("output",output);
            if(!data->handle->db.save_db(po.get("output",output).c_str()))
            {
                tipl::error() << "cannot save db file " << data->handle->db.error_msg << std::endl;
                return 1;
            }
            tipl::out() << "saving " << (po.get("output",output)+".R2.tsv");
            std::ofstream out((po.get("output",output)+".R2.tsv").c_str());
            out << "name\tR2" << std::endl;
            for(size_t i = 0;i < data->handle->db.subject_names.size();++i)
                out << data->handle->db.subject_names[i] << "\t" << data->handle->db.R2[i] << std::endl;
        }
        return 0;
    }

    for(auto& source : name_list)
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(source);
        if(!handle.get())
        {
            tipl::error() << handle->error_msg << std::endl;
            return 1;
        }
        if(cmd=="roi")
        {

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
    }
    tipl::error() << "unknown command: " << cmd << std::endl;
    return 1;
}
