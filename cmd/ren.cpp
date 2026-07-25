#include "TIPL/tipl.hpp"


void check_name(std::string& name)
{
    for(unsigned int index = 0;index < name.size();++index)
        if((name[index] < '0' || name[index] > '9') &&
            (name[index] < 'a' || name[index] > 'z') &&
            (name[index] < 'A' || name[index] > 'Z') &&
            name[index] != '.')
            name[index] = '_';
}

std::filesystem::path rename_dicom(const std::filesystem::path& file_name,
                                   std::filesystem::path output)
{
    std::string person, sequence, imagename;
    {
        tipl::io::dicom header;
        if (!header.load_from_file(file_name))
        {
            tipl::out() << "not a DICOM file. Skipping";
            return std::string();
        }
        header.get_patient(person);
        header.get_sequence(sequence);
        header.get_image_name(imagename);
    }
    check_name(person);
    check_name(sequence);
    check_name(imagename);
    output = output/person;
    output = output/sequence;
    output = output/imagename;
    if(file_name != output)
    {
        tipl::out() << file_name << "->" << output;
        std::error_code ec;
        if (!std::filesystem::exists(output.parent_path()) && !std::filesystem::create_directories(output.parent_path()))
        {
            if(!std::filesystem::exists(output.parent_path()))
                tipl::error() << "cannot create dir " << output;
        }
        std::filesystem::rename(file_name,output,ec);
        if(ec)
        {
            tipl::error() << "cannot rename " << file_name << " to " << output << ": " << ec.message();
            return {};
        }
    }
    return output;
}

std::vector<std::filesystem::path> rename_dicom_at_dir(std::filesystem::path path,
                                                       std::filesystem::path output)
{
    tipl::progress prog("Renaming DICOM");
    tipl::out() << "current directory is " << std::filesystem::current_path();
    tipl::out() << "source directory is " << path;
    tipl::out() << "output directory is " << output;

    auto files = tipl::search_files(path,"",true);
    tipl::par_for(files.size(),[&](size_t i)
    {
        auto renamed = rename_dicom(files[i],output);
        files[i] = renamed.empty() ? std::filesystem::path() : renamed.parent_path().parent_path();
    });
    files.erase(std::remove_if(files.begin(),files.end(),
                               [](const auto& p){return p.empty();}),files.end());
    std::sort(files.begin(),files.end());
    files.erase(std::unique(files.begin(),files.end()),files.end());
    return files;
}

void dicom2src_and_nii(const std::filesystem::path& dir,bool overwrite);
int ren(tipl::program_option<tipl::out>& po)
{
    tipl::progress prog("run ren");
    auto subject_dir = rename_dicom_at_dir(po.get("source"),po.get("output",po.get("source")));
    if(po.get("to_src_nii",0))
        for(const auto& dir : subject_dir)
        {
            tipl::progress prog("DICOM to SRC/NII",dir.u8string());
            dicom2src_and_nii(dir,po.get("overwrite",0));
        }
    return 0;
}
