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
    std::vector<std::vector<std::string> > output(file_list.size());
    std::vector<float> ndc(file_list.size());
    tipl::progress prog("checking SRC files");
    size_t p = 0;
    tipl::par_for(file_list.size(),[&](size_t i)
    {
        prog(++p,file_list.size());
        if(prog.aborted())
            return;
        std::vector<std::string> output_each;
        std::string file_name = file_list[i];
        tipl::out() << "checking " << file_name << std::endl;
        output_each.push_back(file_name);
        src_data handle;
        if (!handle.load_from_file(file_name.c_str()))
        {
            out << "cannot load SRC file " << file_list[i] << std::endl;
            return;
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
        ndc[i] = n.second;

        output_each.push_back(std::to_string(n.first));
        output_each.push_back(std::to_string(n.second)); // masked
        output_each.push_back(std::to_string(handle.get_bad_slices().size()));
        output[i] = std::move(output_each);
    });


    //trim_common_border
    if(output.empty() || output[0].empty())
    {
        auto trim=[&](bool front){
            while(true){
                int freq[256]={0}, total=0;
                for(auto& row:output){
                    auto& s=row[0];
                    if(!s.empty()){
                        ++freq[(unsigned char)(front?s.front():s.back())];
                        ++total;
                    }
                }
                if(!total) break;
                int best=0, best_char=0;
                for(int i=0;i<256;++i) if(freq[i]>best){ best=freq[i]; best_char=i; }
                if(best*4 <= total*3) break;
                for(auto& row:output){
                    auto& s=row[0];
                    if(!s.empty() && (unsigned char)(front?s.front():s.back())==best_char){
                        if(front) s.erase(s.begin()); else s.pop_back();
                    }
                }
            }
        };
        trim(true);
        trim(false);
    }

    float outlier_threshold = tipl::outlier_range(ndc.begin(),ndc.end()).first;
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
        auto result = evaluate_fib(handle->dim,handle->dir.fa_otsu*0.6f,handle->dir.fa,
                                                     [&](int pos,char fib)
                                                     {return handle->dir.get_fib(size_t(pos),uint32_t(fib));});
        out << std::filesystem::path(file_list[i]).filename().string() << "\t";
        out << handle->dim << "\t";
        out << handle->vs << "\t";
        out << result << "\t";
        float R2 = 0.0f;
        if(handle->mat_reader.read("R2",R2))
            out << R2 << std::endl;
        else
            out << std::endl;
    }
    out << "total scans: " << output.size() << std::endl;
    return out.str();
}

std::string quality_check_nii_files(const std::vector<std::string>& file_list)
{
    std::map<int,std::string> data_type = {
                {2,"DT_SIGNED_CHAR"},
                {4,"DT_SIGNED_SHORT"},
                {8,"DT_SIGNED_INT"},
                {16,"DT_FLOAT"},
                {64,"DT_DOUBLE"},
                {128,"DT_RGB"},
                {256,"DT_INT8"},
                {512,"DT_UINT16"},
                {768,"DT_UINT32"},
                {1024,"DT_INT64"},
                {1280,"DT_UINT64"}};
    std::vector<std::string> sform_code = {
            "NIFTI_XFORM_UNKNOWN",
            "NIFTI_XFORM_SCANNER_ANAT",
            "NIFTI_XFORM_ALIGNED_ANAT",
            "NIFTI_XFORM_TALAIRACH",
            "NIFTI_XFORM_MNI_152",
            "NIFTI_XFORM_TEMPLATE_OTHER"};

    std::ostringstream out;
    out << "FileName\tImage Dimension\tResolution\tData Type\tSlope\tInter\tSForm Code\tSRow X\tSRow Y\tSRow Z" << std::endl;
    std::vector<std::vector<std::string> > output;
    tipl::progress prog("checking nifti files");
    for(int i = 0;prog(i,file_list.size());++i)
    {
        tipl::io::gz_nifti nii;
        if(nii.open(file_list[i],std::ios::in))
        {
            out << std::filesystem::path(file_list[i]).filename().string() << "\t";
            if(nii.dim(4) == 1)
                out << nii.get_image_dimension<3>() << "\t";
            else
                out << nii.get_image_dimension<4>() << "\t";
            out << nii.get_voxel_size<3>() << "\t"
                << data_type[nii.nif_header.datatype] << "\t"
                << nii.nif_header.scl_slope << "\t"
                << nii.nif_header.scl_inter << "\t"
                << sform_code[nii.nif_header.sform_code] << "\t"
                << nii.get_transformation()[0] << " "
                << nii.get_transformation()[1] << " "
                << nii.get_transformation()[2] << " "
                << nii.get_transformation()[3] << "\t"
                << nii.get_transformation()[4] << " "
                << nii.get_transformation()[5] << " "
                << nii.get_transformation()[6] << " "
                << nii.get_transformation()[7] << "\t"
                << nii.get_transformation()[8] << " "
                << nii.get_transformation()[9] << " "
                << nii.get_transformation()[10] << " "
                << nii.get_transformation()[11];
        }
        out << std::endl;
    }
    return out.str();
}

/**
 perform reconstruction
 */
int qc(tipl::program_option<tipl::out>& po)
{
    std::string source = po.get("source");

    bool is_fib = po.get("is_fib",tipl::ends_with(source,"fib.gz") || tipl::ends_with(source,".fz") ? 1:0);
    tipl::max_thread_count = std::min<int>(12,tipl::max_thread_count);
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

    if(tipl::ends_with(source,"nii.gz"))
        return (std::ofstream(report_file_name) << quality_check_nii_files(file_list)) ? 0:1;
    if(is_fib)
        return (std::ofstream(report_file_name) << quality_check_fib_files(file_list)) ? 0:1;

    return (std::ofstream(report_file_name) << quality_check_src_files(file_list,
                    po.get("check_btable",0),po.has("template"),po.get("template",0))) ? 0:1;
}
