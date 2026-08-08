#include <cctype>
#include "TIPL/tipl.hpp"

void check_name(std::string& name)
{
    for(auto& ch : name)
        if(!std::isalnum(static_cast<unsigned char>(ch)) && ch != '.')
            ch = '_';
    if(name.empty() || name == "." || name == "..")
        name = "unknown";
}

std::filesystem::path rename_dicom(const std::filesystem::path& file_name,
                                   std::filesystem::path output,bool* had_error = nullptr)
{
    std::string person,sequence,imagename;
    {
        tipl::io::dicom header;
        if(!header.load_from_file(file_name))
            return tipl::out() << "not a DICOM file. Skipping",
                   std::filesystem::path();

        header.get_patient(person);
        header.get_sequence(sequence);
        header.get_image_name(imagename);
    } // header, and the file handle it holds, is destroyed here -- Windows cannot rename a file that is still open
    for(auto* name : {&person,&sequence,&imagename})
        check_name(*name);

    output = output/person/sequence/imagename;
    if(file_name == output)
        return output;

    tipl::out() << file_name << "->" << output;
    std::error_code ec;
    std::filesystem::create_directories(output.parent_path(),ec);
    if(ec)
    {
        if(had_error)
            *had_error = true;
        return tipl::error() << "cannot create directory "
               << output.parent_path() << ": " << ec.message(),
               std::filesystem::path();
    }
    if(std::filesystem::exists(output))
    {
        if(had_error)
            *had_error = true;
        return tipl::error() << "destination already exists: "
               << output,std::filesystem::path();
    }
    std::filesystem::rename(file_name,output,ec);
    if(ec)
    {
        if(had_error)
            *had_error = true;
        return tipl::error() << "cannot rename " << file_name
               << " to " << output << ": " << ec.message(),
               std::filesystem::path();
    }
    return output;
}

std::vector<std::filesystem::path> rename_dicom_at_dir(std::filesystem::path path,
                                                       std::filesystem::path output,bool& had_error)
{
    tipl::progress prog("Renaming DICOM");
    tipl::out() << "current directory is " << std::filesystem::current_path();
    tipl::out() << "source directory is " << path;
    tipl::out() << "output directory is " << output;

    auto files = tipl::search_files(path,"",true);
    std::vector<char> failed(files.size(),0); // per-file flag; each par_for thread only ever writes its own index
    tipl::par_for(files.size(),[&](size_t i)
    {
        bool file_had_error = false;
        auto renamed = rename_dicom(files[i],output,&file_had_error);
        failed[i] = file_had_error;
        files[i] = renamed.empty() ? std::filesystem::path() : renamed.parent_path().parent_path();
    });
    if(std::any_of(failed.begin(),failed.end(),[](char c){return bool(c);}))
        had_error = true;
    files.erase(std::remove_if(files.begin(),files.end(),
                               [](const auto& p){return p.empty();}),files.end());
    std::sort(files.begin(),files.end());
    files.erase(std::unique(files.begin(),files.end()),files.end());
    return files;
}

bool dicom2src_and_nii(const std::filesystem::path& dir,bool overwrite);
int ren(tipl::program_option<tipl::out>& po)
{
    tipl::progress prog("run ren");
    bool had_error = false;
    auto subject_dir = rename_dicom_at_dir(po.get("source"),po.get("output",po.get("source")),had_error);
    bool result = !had_error;
    if(po.get("to_src_nii",0))
    {
        bool overwrite = po.get("overwrite",0);
        for(const auto& dir : subject_dir)
        {
            tipl::progress prog("DICOM to SRC/NII",dir.u8string());
            if(!dicom2src_and_nii(dir,overwrite))
                result = false;
        }
    }
    return result ? 0 : 1;
}
