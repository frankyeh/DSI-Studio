#include <QString>
#include <QStringList>
#include <QFileInfo>
#include "program_option.hpp"
#include "libs/dsi/image_model.hpp"

QStringList search_files(QString dir,QString filter);


extern bool has_gui;
std::string quality_check_src_files(QString dir)
{
    std::ostringstream out;
    QStringList filenames = search_files(dir,"*src.gz");
    out << "Directory:" << dir.toStdString() << std::endl;
    if(filenames.empty())
    {
        std::cout << "no SRC file found in the directory" << std::endl;
        return "no SRC file found in the directory";
    }
    out << "FileName\tImage dimension\tResolution\tDWI count\tMax b-value\tB-table matched\tNeighboring DWI correlation\t# Bad Slices" << std::endl;
    size_t dwi_count = 0;
    float max_b = 0;
    std::cout << "a total of " << filenames.size() << " SRC file(s) were found."<< std::endl;

    std::vector<std::vector<std::string> > output;
    std::vector<float> ndc;
    for(int i = 0;check_prog(i,filenames.size());++i)
    {
        std::cout << "checking " << QFileInfo(filenames[i]).baseName().toStdString() << std::endl;
        output.push_back(std::vector<std::string>());
        output.back().push_back(QFileInfo(filenames[i]).baseName().toStdString());
        ImageModel handle;
        bool restore_gui = false;
        if(has_gui)
        {
            has_gui = false;
            restore_gui = true;
        }
        if (!handle.load_from_file(filenames[i].toStdString().c_str()))
        {
            std::cout << "cannot read SRC file" << std::endl;
            out << "cannot load SRC file"  << std::endl;
            continue;
        }
        if(restore_gui)
            has_gui = true;
        // output image dimension
        {
            std::ostringstream out1;
            out1 << tipl::vector<3,int>(handle.voxel.dim.begin();
            output.back().push_back(out1.str());
        }
        // output image resolution
        {
            std::ostringstream out1;
            out1 << handle.voxel.vs;
            output.back().push_back(out1.str());
        }
        // output DWI count
        size_t cur_dwi_count = handle.src_bvalues.size();
        output.back().push_back(std::to_string(cur_dwi_count));
        if(i == 0)
            dwi_count = cur_dwi_count;

        // output max_b
        float cur_max_b = 0;
        output.back().push_back(std::to_string(cur_max_b = *std::max_element(handle.src_bvalues.begin(),handle.src_bvalues.end())));

        if(i == 0)
            max_b = cur_max_b;
        // check shell structure
        output.back().push_back(std::fabs(max_b-cur_max_b) < 1.0f && cur_dwi_count == dwi_count ? "Yes" : "No");

        // calculate neighboring DWI correlation
        ndc.push_back(handle.quality_control_neighboring_dwi_corr());
        output.back().push_back(std::to_string(ndc.back()));

        output.back().push_back(std::to_string(handle.get_bad_slices().size()));

    }
    auto ndc_copy = ndc;
    float m = tipl::median(ndc_copy.begin(),ndc_copy.end());
    float mad = float(tipl::median_absolute_deviation(ndc_copy.begin(),ndc_copy.end(),double(m)));
    float outlier_threshold = m-3.0f*mad;
    for(size_t i = 0;i < output.size();++i)
    {
        for(size_t j = 0 ;j < output[i].size();++j)
            out << output[i][j] << "\t";
        if(ndc[i] < outlier_threshold)
            out << "low quality outlier";
        out << "\t";
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
        std::cout << "quality control checking src files in " << file_name << std::endl;
        std::string report_file_name = file_name + "/src_report.txt";
        std::ofstream out(report_file_name.c_str());
        out << quality_check_src_files(file_name.c_str());
        std::cout << "report saved to " << report_file_name << std::endl;
    }
    else {
        std::cout << file_name << " is not a file folder." << std::endl;
    }
    return 0;
}
