#include <iostream>
#include <QDir>
#include "TIPL/tipl.hpp"
std::vector<std::filesystem::path> rename_dicom_at_dir(std::filesystem::path path,std::filesystem::path output);
void dicom2src_and_nii(const std::filesystem::path& dir,bool overwrite);
int ren(tipl::program_option<tipl::out>& po)
{
    tipl::progress prog("run ren");
    auto subject_dir = rename_dicom_at_dir(po.get("source"),po.get("output",po.get("source")));
    if(po.get("to_src_nii",0))
        for(auto dir : subject_dir)
        {
            tipl::progress prog("DICOM to SRC/NII ",dir.u8string());
            dicom2src_and_nii(dir,po.get("overwrite",0));
        }
    return 0;
}
