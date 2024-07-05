#include <QString>
#include <QStringList>
#include <QFileInfo>
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"

QStringList search_files(QString dir,QString filter);
const char* src_qc_title = "FileName\tImage dimension\tResolution\tDWI count\tMax b-value\tDWI contrast\tNeighboring DWI correlation\tNeighboring DWI correlation(masked)\t# Bad Slices";
float check_src(std::string file_name,std::vector<std::string>& output) // return masked_ndc
{
    tipl::out() << "checking " << file_name << std::endl;
    output.push_back(QFileInfo(file_name.c_str()).baseName().toStdString());
    src_data handle;
    if (!handle.load_from_file(file_name.c_str()))
    {
        tipl::out() << "cannot read SRC file" << std::endl;
        return 0.0;
    }
    // output image dimension
    {
        std::ostringstream out1;
        out1 << tipl::vector<3,int>(handle.voxel.dim.begin());
        output.push_back(out1.str());
    }
    // output image resolution
    {
        std::ostringstream out1;
        out1 << handle.voxel.vs;
        output.push_back(out1.str());
    }
    // output DWI count
    size_t cur_dwi_count = handle.src_bvalues.size();
    output.push_back(std::to_string(cur_dwi_count));

    // output max_b
    output.push_back(std::to_string(tipl::max_value(handle.src_bvalues)));

    // dwi contrast
    output.push_back(std::to_string(handle.dwi_contrast()));

    // calculate neighboring DWI correlation
    auto ndc = handle.quality_control_neighboring_dwi_corr();

    output.push_back(std::to_string(ndc.first));
    output.push_back(std::to_string(ndc.second)); // masked
    output.push_back(std::to_string(handle.get_bad_slices().size()));
    return ndc.second; // masked ndc
}
std::string quality_check_src_files(QString dir)
{
    std::ostringstream out;
    QStringList filenames;
    if(QFileInfo(dir).isDir())
    {

        filenames = search_files(dir,"*src.gz");
        out << "directory: " << dir.toStdString() << std::endl;
        if(filenames.empty())
        {
            tipl::out() << "no SRC file found in " << dir.toStdString() << std::endl;
            return std::string();
        }
        tipl::out() << "a total of " << filenames.size() << " SRC file(s) were found."<< std::endl;
    }
    else
    {
        if(!QFileInfo(dir).exists())
        {
            tipl::out() << "Cannot find " << dir.toStdString() << std::endl;
            return std::string();
        }
        filenames << dir;
    }
    out << src_qc_title << std::endl;

    std::vector<std::vector<std::string> > output;
    std::vector<float> ndc;
    tipl::progress prog("checking SRC files");
    for(int i = 0;prog(i,filenames.size());++i)
    {
        std::vector<std::string> output_each;
        float mask_ndc_each = check_src(filenames[i].toStdString(),output_each);
        if(mask_ndc_each == 0.0f)
        {
            out << "cannot load SRC file " << filenames[i].toStdString() << std::endl;
            continue;
        }
        output.push_back(std::move(output_each));
        ndc.push_back(mask_ndc_each);
    }
    auto ndc_copy = ndc;
    float m = tipl::median(ndc_copy.begin(),ndc_copy.end());
    float mad = float(tipl::median_absolute_deviation(ndc_copy.begin(),ndc_copy.end(),double(m)));
    float outlier_threshold = m-3.0f*1.482602218505602f*mad;
    // 3 "scaled" MAD approach. The scale is -1/(sqrt(2)*erfcinv(3/2)) = 1.482602218505602f
    unsigned int outlier_count = 0;
    for(size_t i = 0;i < output.size();++i)
    {
        for(size_t j = 0 ;j < output[i].size();++j)
            out << output[i][j] << "\t";
        if(ndc[i] < outlier_threshold)
        {
            out << "low quality outlier";
            ++outlier_count;
        }
        out << "\t";
        out << std::endl;
    }
    out << "total scans: " << output.size() << std::endl;
    out << "total outliers: " << outlier_count << std::endl;
    return out.str();
}
std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);
std::string quality_check_fib_files(QString dir)
{
    std::ostringstream out;
    QStringList filenames = search_files(dir,"*fib.gz");
    out << "directory: " << dir.toStdString() << std::endl;
    if(filenames.empty())
    {
        tipl::out() << "no FIB file found in the directory" << std::endl;
        return std::string();
    }
    out << "FileName\tImage dimension\tResolution\tCoherence Index" << std::endl;
    tipl::out() << "a total of " << filenames.size() << " FIB file(s) were found."<< std::endl;

    std::vector<std::vector<std::string> > output;
    std::vector<float> ndc;
    tipl::progress prog("checking FIB files");
    for(int i = 0;prog(i,filenames.size());++i)
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(filenames[i].toStdString());
        if(!handle.get())
            return QString("Failed to open ").toStdString() + filenames[i].toStdString();
        std::pair<float,float> result = evaluate_fib(handle->dim,handle->dir.fa_otsu*0.6f,handle->dir.fa,
                                                     [&](int pos,char fib)
                                                     {return handle->dir.get_fib(size_t(pos),uint32_t(fib));});
        out << QFileInfo(filenames[i]).baseName().toStdString() << "\t";
        out << handle->dim << "\t";
        out << handle->vs << "\t";
        out << result.first << std::endl;

    }
    out << "total scans: " << output.size() << std::endl;
    return out.str();
}

/**
 perform reconstruction
 */
int qc(tipl::program_option<tipl::out>& po)
{
    std::string file_name = po.get("source");
    if(QFileInfo(file_name.c_str()).isDir())
    {
        {
            std::string report_file_name = po.get("output",file_name + "/qc_src.txt");
            tipl::out() << "quality control checking src files in " << file_name << std::endl;
            auto result = quality_check_src_files(file_name.c_str());
            if(result.empty())
                return 1;
            std::ofstream(report_file_name.c_str()) << result;
            tipl::out() << "report saved to " << report_file_name << std::endl;
        }
        {
            std::string report_file_name = po.get("output",file_name + "/qc_fib.txt");
            tipl::out() << "quality control checking fib files in " << file_name << std::endl;
            auto result = quality_check_fib_files(file_name.c_str());
            if(result.empty())
                return 1;
            std::ofstream(report_file_name.c_str()) << result;
            tipl::out() << "report saved to " << report_file_name << std::endl;

        }
    }
    else {
        std::string report_file_name = po.get("output",file_name.substr(0,file_name.size()-7) + ".qc.txt");
        if(QString(file_name.c_str()).endsWith("fib.gz"))
        {
            std::shared_ptr<fib_data> handle = cmd_load_fib(po.get("source"));
            if(!handle.get())
                return 1;
            std::pair<float,float> result = evaluate_fib(handle->dim,handle->dir.fa_otsu*0.6f,handle->dir.fa,
                                                         [&](int pos,char fib)
                                                         {return handle->dir.get_fib(size_t(pos),uint32_t(fib));});
            std::ofstream out(report_file_name.c_str());
            out << "Fiber coherence index: " << result.first << std::endl;
            out << "Fiber incoherent index: " << result.second << std::endl;
        }
        if(QString(file_name.c_str()).endsWith("src.gz") ||
           QString(file_name.c_str()).endsWith("nii.gz"))
        {
            auto result = quality_check_src_files(file_name.c_str());
            if(result.empty())
                return 1;

            std::ofstream(report_file_name.c_str()) << result;
            tipl::out() << "report saved to " << report_file_name << std::endl;
        }
    }
    return 0;
}
