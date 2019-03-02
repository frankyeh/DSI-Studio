
#include <QFileInfo>
#include <QApplication>
#include <QDir>
#include "tipl/tipl.hpp"
#include "libs/gzip_interface.hpp"
#include "mapping/atlas.hpp"
#include "program_option.hpp"
#include "fib_data.hpp"
#include "vbc/vbc_database.h"

extern std::string fib_template_file_name_1mm,fib_template_file_name_2mm;
const char* odf_average(const char* out_name,std::vector<std::string>& file_names);
bool atl_load_atlas(std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list)
{
    QStringList name_list = QString(atlas_name.c_str()).split(",");


    for(unsigned int index = 0;index < name_list.size();++index)
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
            atlas_list.push_back(std::make_shared<atlas>());
            atlas_list.back()->filename = file_path;
            atlas_list.back()->name = name_list[index].toStdString();
            if(atlas_list.back()->get_num().empty())
            {
                std::cout << "Invalid file format. No ROI found in " << name_list[index].toStdString() << "." << std::endl;
                return false;
            }
            continue;
        }
    }
    return !atlas_list.empty();
}

void atl_save_mapping(std::vector<std::shared_ptr<atlas> >& atlas_list,
                      const std::string& file_name,const tipl::geometry<3>& geo,
                      const tipl::image<tipl::vector<3>,3>& mapping,
                      const std::vector<float>& trans,
                      const tipl::vector<3>& vs,
                      bool multiple)
{
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

            tipl::image<unsigned char,3> roi(geo);
            for(unsigned int k = 0;k < mapping.size();++k)
                if (atlas_list[i]->is_labeled_as(mapping[k], j))
                    roi[k] = 1;
            if(multiple)
            {
                tipl::io::nifti out;
                out.set_voxel_size(vs);
                if(!trans.empty())
                    out.set_LPS_transformation(trans.begin(),roi.geometry());
                tipl::flip_xy(roi);
                out << roi;
                out.save_to_file(output.c_str());
                std::cout << "save " << output << std::endl;
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
}
std::shared_ptr<fib_data> cmd_load_fib(const std::string file_name);

void get_files_in_folder(std::string dir,std::string file,std::vector<std::string>& files)
{
    QDir directory(QString(dir.c_str()));
    QStringList file_list = directory.entryList(QStringList(file.c_str()),QDir::Files);
    if(file_list.empty())
        return;
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
    files = std::move(name_list);
}

int atl(void)
{
    // construct an atlas
    std::string cmd = po.get("cmd");
    if(cmd=="template")
    {
        std::string dir = po.get("source");
        std::cout << "Constructing a group average template" << std::endl;
        std::cout << "Loading fib file in " << dir << std::endl;
        std::vector<std::string> name_list;
        get_files_in_folder(dir,"*.fib.gz",name_list);
        if(name_list.empty())
        {
            std::cout << "No FIB file found in the directory." << std::endl;
            return 0;
        }
        dir += "/template";
        const char* msg = odf_average(dir.c_str(),name_list);
        if(msg)
            std::cout << msg << std::endl;
        return 0;
    }
    if(cmd=="db")
    {
        std::cout << "Constructing a connectometry db" << std::endl;
        // Find all the FIB files
        std::string dir = po.get("source");
        std::cout << "Loading fib file in " << dir << std::endl;
        std::vector<std::string> name_list;
        get_files_in_folder(dir,"*.fib.gz",name_list);
        if(name_list.empty())
        {
            std::cout << "No FIB file found in the directory." << std::endl;
            return 0;
        }
        // Determine the template
        std::string tm;
        if(po.has("template"))
            tm = po.get("template");
        {
            fib_data fib;
            if(!fib.load_from_file(name_list[0].c_str()))
            {
                std::cout << "Invalid FIB file format:" << name_list[0] << std::endl;
                return 0;
            }
            if(fib.vs[0] < 1.5f)
                tm = fib_template_file_name_1mm.c_str();
            else
                tm = fib_template_file_name_2mm.c_str();

        }
        // Initialize the DB
        std::cout << "Loading template" << tm << std::endl;
        std::auto_ptr<vbc_database> data(new vbc_database);
        if(!data->create_database(tm.c_str()))
        {
            std::cout << "Error in initializing the database:" << data->error_msg << std::endl;
            return 0;
        }
        // Extracting metrics
        std::string index_name = po.get("index_name","sdf");
        std::cout << "Extracting index:" << index_name << std::endl;
        data->handle->db.index_name = index_name;
        for (unsigned int index = 0;index < name_list.size();++index)
        {
            std::cout << "Reading " << name_list[index] << std::endl;
            if(!data->handle->db.add_subject_file(name_list[index],
                QFileInfo(name_list[index].c_str()).baseName().toStdString()))
            {
                std::cout << "Error loading subject fib files:" << data->handle->error_msg << std::endl;
                return 0;
            }
        }
        // Output
        std::string output = dir;
        output += "/";
        output += "connectometry.db.fib.gz";
        if(!data->handle->db.save_subject_data(output.c_str()))
        {
            std::cout << "Error saving the db file:" << data->handle->error_msg << std::endl;
            return 0;
        }
        std::cout << "Connectometry db created:" << output << std::endl;
        return 0;
    }
    if(cmd=="roi")
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
        if(!handle.get())
        {
            std::cout << handle->error_msg << std::endl;
            return 0;
        }
        std::vector<std::shared_ptr<atlas> > atlas_list;
        if(!atl_load_atlas(po.get("atlas"),atlas_list))
            return 0;
        if(!handle->can_map_to_mni())
        {
            std::cout << "Cannot output connectivity: no mni mapping" << std::endl;
            return 0;
        }
        atl_save_mapping(atlas_list,
                         po.get("source"),handle->dim,
                         handle->get_mni_mapping(),handle->trans_to_mni,handle->vs,
                         po.get("output","multiple") == "multiple");
        return 0;
    }
    if(cmd=="trk")
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
        if(!handle.get())
        {
            std::cout << handle->error_msg << std::endl;
            return 0;
        }
        if(!handle->is_qsdr)
        {
            std::cout << "Only QSDR reconstructed FIB file is supported." << std::endl;
            return 0;
        }
        if(handle->native_position.empty())
        {
            std::cout << "No mapping information found. Please reconstruct QSDR with mapping checked in advanced option." << std::endl;
            return 0;
        }
        TractModel tract_model(handle);
        std::string file_name = po.get("tract");
        {
            std::cout << "loading " << file_name << "..." <<std::endl;
            if (!tract_model.load_from_file(file_name.c_str()))
            {
                std::cout << "Cannot open file " << file_name << std::endl;
                return 0;
            }
            std::cout << file_name << " loaded" << std::endl;
        }
        file_name += "native.trk.gz";
        tract_model.save_tracts_in_native_space(file_name.c_str(),handle->native_position);
        std::cout << "Native tracks saved to " << file_name << " loaded" << std::endl;
        return 0;
    }
    std::cout << "Unknown command:" << cmd << std::endl;
    return 0;
}
