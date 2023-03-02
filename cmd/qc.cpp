#include <QString>
#include <QStringList>
#include <QFileInfo>
#include "TIPL/tipl.hpp""
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"

QStringList search_files(QString dir,QString filter);
bool check_src(std::string file_name,std::vector<std::string>& output,float& ndc)
{
    show_progress() << "checking " << file_name << std::endl;
    output.push_back(QFileInfo(file_name.c_str()).baseName().toStdString());
    ImageModel handle;
    if (!handle.load_from_file(file_name.c_str()))
    {
        show_progress() << "cannot read SRC file" << std::endl;
        return false;
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

    // calculate neighboring DWI correlation
    ndc = handle.quality_control_neighboring_dwi_corr();
    output.push_back(std::to_string(ndc));
    output.push_back(std::to_string(handle.get_bad_slices().size()));
    return true;
}
std::string quality_check_src_files(QString dir)
{
    std::ostringstream out;
    QStringList filenames = search_files(dir,"*src.gz");
    out << "Directory:" << dir.toStdString() << std::endl;
    if(filenames.empty())
    {
        show_progress() << "no SRC file found in the directory" << std::endl;
        return "no SRC file found in the directory";
    }
    out << "FileName\tImage dimension\tResolution\tDWI count\tMax b-value\tNeighboring DWI correlation\t# Bad Slices" << std::endl;
    show_progress() << "a total of " << filenames.size() << " SRC file(s) were found."<< std::endl;

    std::vector<std::vector<std::string> > output;
    std::vector<float> ndc;
    for(int i = 0;progress::at(i,filenames.size());++i)
    {
        bool has_gui_ = has_gui;
        has_gui = false;
        std::vector<std::string> output_each;
        float ndc_each;
        if(!check_src(filenames[i].toStdString(),output_each,ndc_each))
        {
            out << "cannot load SRC file " << filenames[i].toStdString() << std::endl;
            has_gui = has_gui_;
            continue;
        }
        has_gui = has_gui_;
        output.push_back(std::move(output_each));
        ndc.push_back(ndc_each);
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
    out << "total scans:" << output.size() << std::endl;
    out << "total outliers:" << outlier_count << std::endl;
    return out.str();
}

/**
 perform reconstruction
 */
std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);
int qc(tipl::io::program_option<show_progress>& po)
{
    std::string file_name = po.get("source");
    if(QFileInfo(file_name.c_str()).isDir())
    {
        std::string report_file_name = po.get("output",file_name + "/qc.txt");
        show_progress() << "quality control checking src files in " << file_name << std::endl;
        std::ofstream out(report_file_name.c_str());
        out << quality_check_src_files(file_name.c_str());
        show_progress() << "report saved to " << report_file_name << std::endl;
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
            std::ofstream out(report_file_name.c_str());
            out << "FileName\tImage dimension\tResolution\tDWI count\tMax b-value\tNeighboring DWI correlation\t# Bad Slices" << std::endl;
            std::vector<std::string> output_each;
            float ndc_each;
            if(!check_src(file_name,output_each,ndc_each))
            {
                show_progress() << "cannot load file " << file_name << std::endl;
                return 1;
            }
            for(size_t j = 0 ;j < output_each.size();++j)
                out << output_each[j] << "\t";
        }
    }
    return 0;
}
