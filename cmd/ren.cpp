#include "TIPL/tipl.hpp"

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
