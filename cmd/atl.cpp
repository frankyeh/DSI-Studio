#include <QFileInfo>
#include <QApplication>
#include <QDir>
#include "tipl/tipl.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/atlas.hpp"
#include "program_option.hpp"
#include "fib_data.hpp"
#include "connectometry/group_connectometry_analysis.h"

#include "mac_filesystem.hpp"

extern std::string fib_template_file_name_2mm;
extern std::vector<std::vector<std::string> > template_atlas_list;
const char* odf_average(const char* out_name,std::vector<std::string>& file_names);
bool atl_load_atlas(std::shared_ptr<fib_data> handle,std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list)
{
    QStringList name_list = QString(atlas_name.c_str()).split(",");


    for(int index = 0;index < name_list.size();++index)
    {
        bool has_atlas = false;
        for(unsigned int i = 0;i < atlas_list.size();++i)
            if(atlas_list[i]->name == name_list[index].toStdString())
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
            const auto& atlas_list = template_atlas_list[handle->template_id];
            for(size_t j = 0;j < atlas_list.size();++j)
            {
                if(QFileInfo(atlas_list[j].c_str()).baseName().toLower() == name_list[index].toLower())
                {
                    file_path = atlas_list[j];
                    break;
                }
            }
            if(!std::filesystem::exists(file_path))
            {
                std::cout << "ERROR: did not find atlas at " << QFileInfo(template_atlas_list[handle->template_id][0].c_str()).absolutePath().toStdString() << std::endl;
                return false;
            }
        }

        {
            std::cout << "loading " << name_list[index].toStdString() << "..." << std::endl;
            atlas_list.push_back(std::make_shared<atlas>());
            atlas_list.back()->filename = file_path;
            atlas_list.back()->name = name_list[index].toStdString();
            if(atlas_list.back()->get_num().empty())
            {
                std::cout << "ERROR: fail to open " << name_list[index].toStdString() << ":" << atlas_list.back()->error_msg << std::endl;
                return false;
            }
            continue;
        }
    }
    handle->atlas_list = atlas_list;
    return !atlas_list.empty();
}

std::shared_ptr<fib_data> cmd_load_fib(const std::string file_name);

void get_files_in_folder(std::string dir,std::string file,std::vector<std::string>& files)
{
    QDir directory(QString(dir.c_str()));
    QStringList file_list = directory.entryList(QStringList(file.c_str()),QDir::Files);
    if(file_list.empty())
        return;
    std::vector<std::string> name_list;
    for (int index = 0;index < file_list.size();++index)
    {
        std::string file_name = dir;
        file_name += "/";
        file_name += file_list[index].toStdString();
        name_list.push_back(file_name);
    }
    files = std::move(name_list);
}

int atl(program_option& po)
{
    // construct an atlas
    std::string cmd = po.get("cmd");
    if(cmd=="template")
    {
        std::string dir = po.get("source");
        std::cout << "constructing a group average template" << std::endl;
        std::cout << "loading fib file in " << dir << std::endl;
        std::vector<std::string> name_list;
        get_files_in_folder(dir,"*.fib.gz",name_list);
        if(name_list.empty())
        {
            std::cout << "no FIB file found in the directory." << std::endl;
            return 1;
        }
        dir += "/template";
        const char* msg = odf_average(dir.c_str(),name_list);
        if(msg)
            std::cout << msg << std::endl;
        return 0;
    }
    if(cmd=="db")
    {
        std::cout << "constructing a connectometry db" << std::endl;
        // Find all the FIB files
        std::string dir = po.get("source");
        std::cout << "loading fib file in " << dir << std::endl;
        std::vector<std::string> name_list;
        get_files_in_folder(dir,"*.fib.gz",name_list);
        if(name_list.empty())
        {
            std::cout << "no FIB file found in the directory." << std::endl;
            return 1;
        }
        // Determine the template
        std::string tm;
        if(po.has("template"))
            tm = po.get("template");
        else
            tm = fib_template_file_name_2mm.c_str();
        // Initialize the DB
        std::cout << "loading template" << tm << std::endl;
        std::vector<std::string> index_name;
        if(po.get("index_name","qa") == std::string("all"))
        {
            fib_data fib;
            if(!fib.load_from_file(name_list[0].c_str()))
            {
                std::cout << "ERROR loading subject fib files:" << name_list[0] << std::endl;
                return 1;
            }
            std::vector<std::string> item_list;
            fib.get_index_list(item_list);
            index_name.push_back("qa");
            for(size_t i = fib.dir.index_name.size();i < item_list.size();++i)
                index_name.push_back(item_list[i]);
        }
        else
            index_name.push_back(po.get("index_name","qa"));

        for(size_t i = 0; i < index_name.size();++i)
        {
            std::shared_ptr<group_connectometry_analysis> data(new group_connectometry_analysis);
            if(!data->create_database(tm.c_str()))
            {
                std::cout << "ERROR in initializing the database:" << data->error_msg << std::endl;
                return 1;
            }
            // Extracting metrics
            std::cout << "extracting index:" << index_name[i] << std::endl;
            data->handle->db.index_name = index_name[i];
            for (unsigned int index = 0;index < name_list.size();++index)
            {
                std::cout << "reading " << name_list[index] << std::endl;
                if(!data->handle->db.add_subject_file(name_list[index],
                    QFileInfo(name_list[index].c_str()).baseName().toStdString()))
                {
                    std::cout << "ERROR loading subject fib files:" << data->handle->error_msg << std::endl;
                    return 1;
                }
            }
            // Output
            std::string output = dir;
            output += "/";
            output += "connectometry.";
            output += index_name[i];
            output += ".db.fib.gz";
            if(!data->handle->db.save_subject_data(output.c_str()))
            {
                std::cout << "ERROR saving the db file:" << data->handle->error_msg << std::endl;
                return 1;
            }
            std::cout << "connectometry db created:" << output << std::endl;
        }
        return 0;
    }
    if(cmd=="roi")
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
        if(!handle.get())
        {
            std::cout << handle->error_msg << std::endl;
            return 1;
        }
        std::vector<std::shared_ptr<atlas> > atlas_list;
        if(!atl_load_atlas(handle,po.get("atlas"),atlas_list))
            return 1;
        if(!handle->can_map_to_mni() || handle->get_sub2temp_mapping().empty())
        {
            std::cout << "cannot output connectivity: no mni mapping" << std::endl;
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
                    std::cout << "save " << output << std::endl;
                    if(!gz_nifti::save_to_file(output.c_str(),roi,handle->vs,handle->trans_to_mni))
                        std::cout << "cannot write output to file:" << output << std::endl;
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
    if(cmd=="trk")
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
        if(!handle.get())
        {
            std::cout << handle->error_msg << std::endl;
            return 1;
        }
        if(!handle->is_qsdr)
        {
            std::cout << "only QSDR reconstructed FIB file is supported." << std::endl;
            return 1;
        }
        if(handle->get_native_position().empty())
        {
            std::cout << "no mapping information found. Please reconstruct QSDR with 'mapping' included in the output." << std::endl;
            return 1;
        }
        TractModel tract_model(handle);
        std::string file_name = po.get("tract");
        {
            std::cout << "loading " << file_name << "..." <<std::endl;
            if (!tract_model.load_from_file(file_name.c_str()))
            {
                std::cout << "cannot open file " << file_name << std::endl;
                return 1;
            }
            std::cout << file_name << " loaded" << std::endl;
        }
        file_name += "native.tt.gz";
        tract_model.save_tracts_in_native_space(handle,file_name.c_str());
        std::cout << "native tracks saved to " << file_name << " loaded" << std::endl;
        return 0;
    }
    std::cout << "unknown command:" << cmd << std::endl;
    return 1;
}
