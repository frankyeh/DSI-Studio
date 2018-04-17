#include <QString>
#include <QStringList>
#include <QFileInfo>
#include "program_option.hpp"
#include "libs/dsi/image_model.hpp"

QStringList search_files(QString dir,QString filter);


std::string quality_check_src_files(QString dir)
{
    std::ostringstream out;
    QStringList filenames = search_files(dir,"*src.gz");
    out << "FileName\tImage dimension\tResolution\tDWI count\tMax b-value\tB-table matched\tNeighboring DWI correlation" << std::endl;
    int dwi_count = 0;
    float max_b = 0;
    for(int i = 0;check_prog(i,filenames.size());++i)
    {
        out << QFileInfo(filenames[i]).baseName().toStdString() << "\t";
        ImageModel handle;
        unique_prog(true);
        if (!handle.load_from_file(filenames[i].toStdString().c_str()))
        {
            out << "Cannot load SRC file"  << std::endl;
            continue;
        }
        unique_prog(false);
        // output image dimension
        out << image::vector<3,int>(handle.voxel.dim.begin()) << "\t";
        // output image resolution
        out << handle.voxel.vs << "\t";
        // output DWI count
        int cur_dwi_count = 0;
        out << (cur_dwi_count = handle.src_bvalues.size()) << "\t";
        if(i == 0)
            dwi_count = cur_dwi_count;

        // output max_b
        float cur_max_b = 0;
        out << (cur_max_b = *std::max_element(handle.src_bvalues.begin(),handle.src_bvalues.end())) << "\t";
        if(i == 0)
            max_b = cur_max_b;
        // check shell structure
        out << (max_b == cur_max_b && cur_dwi_count == dwi_count ? "Yes\t" : "No\t");

        // calculate neighboring DWI correlation
        out << handle.quality_control_neighboring_dwi_corr() << "\t";

        out << std::endl;
    }
    return out.str();
}

/**
 perform reconstruction
 */
int qc(void)
{
    std::string file_name = po.get("source");
    if(QFileInfo(file_name.c_str()).isDir())
    {
        file_name += "\\src_report.txt";
        std::ofstream out(file_name.c_str());
        out << quality_check_src_files(file_name.c_str());
    }
    return 0;
}
