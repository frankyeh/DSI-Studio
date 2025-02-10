#include <QString>
#include <QStringList>
#include <QFileInfo>
#include "libs/dsi/image_model.hpp"
#include "fib_data.hpp"

QStringList search_files(QString dir,QString filter);

std::string quality_check_src_files(const std::vector<std::string>& file_list,
                                    bool check_btable,bool use_template,unsigned int template_id)
{
    std::ostringstream out;
    out << "file name\tdimension\tresolution\tdwi count(b0/dwi)\tmax b-value\tDWI contrast\tneighboring DWI correlation\tneighboring DWI correlation(masked)\t#bad slices\toutlier" << std::endl;
    std::vector<std::vector<std::string> > output;
    std::vector<float> ndc;
    tipl::progress prog("checking SRC files");
    for(int i = 0;prog(i,file_list.size());++i)
    {
        std::vector<std::string> output_each;
        std::string file_name = file_list[i];

        tipl::out() << "checking " << file_name << std::endl;
        output_each.push_back(std::filesystem::path(file_name).filename().string());
        src_data handle;
        if (!handle.load_from_file(file_name.c_str()))
        {
            out << "cannot load SRC file " << file_list[i] << std::endl;
            continue;
        }
        // output image dimension
        {
            std::ostringstream out1;
            out1 << tipl::vector<3,int>(handle.voxel.dim.begin());
            output_each.push_back(out1.str());
        }
        // output image resolution
        {
            std::ostringstream out1;
            out1 << handle.voxel.vs;
            output_each.push_back(out1.str());
        }
        // output DWI count
        {
            size_t cur_dwi_count = handle.src_bvalues.size();
            size_t b0_count = std::count(handle.src_bvalues.begin(),handle.src_bvalues.end(),0.0f);
            output_each.push_back(std::to_string(b0_count)+"/" + std::to_string(cur_dwi_count-b0_count));
            if(check_btable)
            {
                if(use_template)
                {
                    handle.voxel.template_id = template_id;
                    handle.check_b_table(true);
                }
                else
                    handle.check_b_table(false);

                size_t pos = handle.error_msg.find_last_of(' ');
                if(pos != std::string::npos)
                    output_each.back() += handle.error_msg.substr(pos + 1);

            }
        }

        // output max_b
        output_each.push_back(std::to_string(tipl::max_value(handle.src_bvalues)));

        // dwi contrast
        output_each.push_back(std::to_string(handle.dwi_contrast()));

        // calculate neighboring DWI correlation
        auto n = handle.quality_control_neighboring_dwi_corr();
        ndc.push_back(n.second);

        output_each.push_back(std::to_string(n.first));
        output_each.push_back(std::to_string(n.second)); // masked
        output_each.push_back(std::to_string(handle.get_bad_slices().size()));
        output.push_back(std::move(output_each));

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
        out << ((ndc[i] < outlier_threshold) ? "1" : "0") << std::endl;
    }
    return out.str();
}
std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);
std::string quality_check_fib_files(const std::vector<std::string>& file_list)
{
    std::ostringstream out;
    out << "FileName\tImage dimension\tResolution\tCoherence Index\tR2 (QSDR)" << std::endl;
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
        out << result.first << "\t";
        float R2 = 0.0f;
        if(handle->mat_reader.read("R2",R2))
            out << R2 << std::endl;
        else
            out << std::endl;
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
    bool is_fib = po.get("is_fib",tipl::ends_with(source,"fib.gz") || tipl::ends_with(source,".fz") ? 1:0);

    std::vector<std::string> file_list;
    if(QFileInfo(source.c_str()).isDir())
    {
        tipl::search_files(source,is_fib ? "*.fib.gz" : "*.src.gz",file_list);
        tipl::search_files(source,is_fib ? "*.fz" : "*.sz",file_list);
    }
    else
        if(!po.get_files("source",file_list))
        {
            tipl::error() << po.error_msg;
            return 1;
        }
    if(file_list.empty())
    {
        tipl::error() << "no file to run quality control";
        return 1;
    }
    std::string report_file_name = po.get("output","qc.tsv");
    tipl::out() << "saving " << report_file_name << std::endl;
    std::ofstream(report_file_name.c_str()) <<
        (is_fib ? quality_check_fib_files(file_list) :
                  quality_check_src_files(file_list,po.get("check_btable",0),po.has("template"),po.get("template",0)));
    return 0;

}
