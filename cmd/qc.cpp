#include <QString>
#include <QStringList>
#include <QFileInfo>
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"

QStringList search_files(QString dir,QString filter);
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
    {
        size_t cur_dwi_count = handle.src_bvalues.size();
        size_t b0_count = std::count(handle.src_bvalues.begin(),handle.src_bvalues.end(),0.0f);
        output.push_back(std::to_string(b0_count)+"/" + std::to_string(cur_dwi_count-b0_count));
    }

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
std::string quality_check_src_files(const std::vector<std::string>& file_list)
{
    std::ostringstream out;
    out << "file name\tdimension\tresolution\tdwi count(b0/dwi)\tmax b-value\tDWI contrast\tneighboring DWI correlation\tneighboring DWI correlation(masked)\t#bad slices" << std::endl;
    std::vector<std::vector<std::string> > output;
    std::vector<float> ndc;
    tipl::progress prog("checking SRC files");
    for(int i = 0;prog(i,file_list.size());++i)
    {
        std::vector<std::string> output_each;
        float mask_ndc_each = check_src(file_list[i],output_each);
        if(mask_ndc_each == 0.0f)
        {
            out << "cannot load SRC file " << file_list[i] << std::endl;
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
    for(size_t i = 0;i < output.size();++i)
    {
        for(size_t j = 0 ;j < output[i].size();++j)
            out << output[i][j] << "\t";
        if(ndc[i] < outlier_threshold)
        {
            out << "low quality outlier";
        }
        out << "\t";
        out << std::endl;
    }
    return out.str();
}
std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);
std::string quality_check_fib_files(const std::vector<std::string>& file_list)
{
    std::ostringstream out;
    out << "FileName\tImage dimension\tResolution\tCoherence Index" << std::endl;
    std::vector<std::vector<std::string> > output;
    std::vector<float> ndc;
    tipl::progress prog("checking FIB files");
    for(int i = 0;prog(i,file_list.size());++i)
    {
        std::shared_ptr<fib_data> handle = cmd_load_fib(file_list[i]);
        if(!handle.get())
            return QString("Failed to open ").toStdString() + file_list[i];
        std::pair<float,float> result = evaluate_fib(handle->dim,handle->dir.fa_otsu*0.6f,handle->dir.fa,
                                                     [&](int pos,char fib)
                                                     {return handle->dir.get_fib(size_t(pos),uint32_t(fib));});
        out << file_list[i] << "\t";
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
    std::string source = po.get("source");
    bool is_fib = po.get("is_fib",source.find("fib.gz") != std::string::npos ? 1:0);

    std::vector<std::string> file_list;
    if(QFileInfo(source.c_str()).isDir())
        tipl::search_files(source,is_fib ? "*.fib.gz" : "*.src.gz",file_list);
    else
        if(!po.get_files("source",file_list))
        {
            tipl::error() << po.error_msg;
            return 1;
        }

    std::string report_file_name = po.get("output","qc.tsv");
    tipl::out() << "saving " << report_file_name << std::endl;
    std::ofstream(report_file_name.c_str()) <<
        (is_fib ? quality_check_fib_files(file_list) : quality_check_src_files(file_list));
    return 0;

}
