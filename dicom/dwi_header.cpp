#include <QFileInfo>
#include <sstream>
#include <string>
#include "dwi_header.hpp"
#include "image_model.hpp"
void get_report_from_dicom(const tipl::io::dicom& header,std::string& report)
{
    std::string manu,make,seq;
    header.get_text(0x0008,0x0070,manu);//Manufacturer
    header.get_text(0x0008,0x1090,make);
    header.get_text(0x0018,0x1030,seq);
    std::replace(manu.begin(),manu.end(),' ',(char)0);
    make.erase(std::remove(make.begin(),make.end(),' '),make.end());
    std::ostringstream out;
    out << " The diffusion images were acquired on a " << manu.c_str() << " " << make.c_str()
        << " scanner using a ";
    if(seq.find("ep2d") != std::string::npos)
        out << "2D EPI ";
    float te = header.get_float(0x0018,0x0081);
    if(te == 0)
        te = header.get_float(0x2001,0x1025); // for philips scanner;
    float tr = header.get_float(0x0018,0x0080);
    if(tr == 0)
        tr = header.get_float(0x2005,0x1030); // for philips scanner;
    out << "diffusion sequence (" << seq.c_str() << ")."
        << " TE=" << te << " ms, and TR=" << tr << " ms.";
    report += out.str();
}
void get_report_from_bruker(const tipl::io::bruker_info& header,std::string& report)
{
    std::ostringstream out;
    out << " The diffusion images were acquired on a " << header["ORIGIN"] << " scanner using a "
        << header["Method"] << " " << header["PVM_DiffPrepMode"]
        <<  " sequence. TE=" << header["PVM_EchoTime"] << " ms, and TR=" << header["PVM_RepetitionTime"] << " ms."
        << " The diffusion time was " << header["PVM_DwGradSep"] << " ms. The diffusion encoding duration was " << header["PVM_DwGradDur"] << " ms.";
    report += out.str();
}
void get_report_from_bruker2(const tipl::io::bruker_info& header,std::string& report)
{
    std::ostringstream out;
    out << " The diffusion images were acquired on a " << header["ORIGIN"]
        << " scanner using a " << header["IMND_method"]
        <<  " sequence. TE=" << header["IMND_EffEchoTime1"] << " ms, and TR=" << header["IMND_rep_time"] << " ms."
        << " The diffusion time was " << header["IMND_big_delta"] << " ms. The diffusion encoding duration was "
        << header["IMND_diff_grad_dur"] << " ms.";
    report += out.str();
}
bool get_compressed_image(tipl::io::dicom& dicom,tipl::image<2,short>& I);
bool DwiHeader::open(const char* filename)
{
    file_name = filename;
    tipl::io::dicom header;
    if (!header.load_from_file(filename))
    {
        tipl::io::nifti nii;
        if (!nii.load_from_file(filename))
        {
            error_msg = "unsupported file format";
            return false;
        }
        nii.toLPS(image);
        nii.get_voxel_size(voxel_size);
        return true;
    }
    header >> image;
    header.get_voxel_size(voxel_size);
    slice_location = header.get_slice_location();

    if(header.is_compressed)
    {
        tipl::image<2,short> I;
        if(!get_compressed_image(header,I))
        {
            error_msg = "unsupported transfer syntax:";
            error_msg += header.encoding;
            return false;
        }
        if(I.size() == image.size())
            std::copy(I.begin(),I.end(),image.begin());
    }
    header.get_voxel_size(voxel_size);
    get_report_from_dicom(header,report);

    float orientation_matrix[9];
    unsigned char dim_order[3] = {0,1,2};
    char flip[3] = {0,0,0};
    bool has_orientation_info = false;

    if(header.get_image_orientation(orientation_matrix))
    {
        tipl::get_orientation(3,orientation_matrix,dim_order,flip);
        tipl::reorient_vector(voxel_size,dim_order);
        tipl::reorient_matrix(orientation_matrix,dim_order,flip);
        tipl::reorder(image,dim_order,flip);
        has_orientation_info = true;
    }


    // get TE
    te = header.get_te();

    float bx,by,bz;
    if(!header.get_btable(bvalue,bvec[0],bvec[1],bvec[2]))
        return false;
    if(bvalue == 0.0f)
        return true;

    bvec.normalize();
    if(has_orientation_info)
    {
        {
            tipl::reorient_vector(bvec,dim_order);
            float x = bvec[dim_order[0]];
            float y = bvec[dim_order[1]];
            float z = bvec[dim_order[2]];
            bvec[0] = x;
            bvec[1] = y;
            bvec[2] = z;
        }
        bvec.rotate(orientation_matrix);
    }
    bvec.normalize();
    return true;
}

/*
if (sort_and_merge)
{

    // merge files of the same bvec
    tipl::progress prog_("Merge bvalue Files");
    for (unsigned int i = 0;prog(i,dwi_files.size());++i)
    {
        unsigned int j = i + 1;
        for (;j < dwi_files.size() && dwi_files[i] == dwi_files[j];++j)
            ;
        if (j == i + 1)
            continue;
        std::vector<unsigned int> sum(dwi_files[i]->size());
        for (unsigned int l = i; l < j;++l)
        {
            const DwiHeader& dwi_image = dwi_files[l];
            for (unsigned int index = 0;index < sum.size();++index)
                sum[index] += dwi_image[index];
        }
        unsigned int count = j-i;
        DwiHeader& dwi_image = dwi_files[i];
        for (unsigned int index = 0;index < sum.size();++index)
            dwi_image[index] = (unsigned short)(sum[index] / count);
        dwi_files.erase(dwi_files.begin()+i+1,dwi_files.begin()+j);
    }
}
*/

void sort_dwi(std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    std::sort(dwi_files.begin(),dwi_files.end(),[&]
              (const std::shared_ptr<DwiHeader>& lhs,const std::shared_ptr<DwiHeader>& rhs){return *lhs < *rhs;});
    for (int i = dwi_files.size()-1;i >= 1;--i)
        if (dwi_files[i]->bvalue == dwi_files[i-1]->bvalue &&
                dwi_files[i]->bvec == dwi_files[i-1]->bvec)
        {
            tipl::image<3> I = dwi_files[i]->image;
            I += dwi_files[i-1]->image;
            I *= 0.5f;
            dwi_files[i-1]->image = I;
            dwi_files.erase(dwi_files.begin()+i);
        }
}

bool DwiHeader::has_b_table(std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    for(size_t i = 0;i < dwi_files.size();++i)
        if(dwi_files[i]->bvalue > 0.0f)
            return true;
    return false;
}
bool DwiHeader::output_src(const std::string& file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
                           bool sort_btable,const std::string& intro_file_name,std::string& error_msg)
{
    if(dwi_files.empty())
    {
        error_msg = "no DWI data to output";
        return false;
    }
    if(sort_btable)
        sort_dwi(dwi_files);

    // removing inconsistent dwi
    for(unsigned int index = 0;index < dwi_files.size();++index)
    {
        if(dwi_files[index]->bvalue < 100.0f)
        {
            dwi_files[index]->bvalue = 0.0f;
            dwi_files[index]->bvec = tipl::vector<3>(0.0f,0.0f,0.0f);
        }
        if(dwi_files[index]->image.shape() != dwi_files[0]->image.shape())
        {
            tipl::warning() << " removing inconsistent image dimensions found at dwi " << index
                          << " size=" << dwi_files[index]->image.shape()
                          << " versus " << dwi_files[0]->image.shape();
            dwi_files.erase(dwi_files.begin() + index);
            --index;
        }
    }

    src_data src;
    src.voxel.dim = dwi_files.front()->image.shape();
    src.voxel.vs = dwi_files.front()->voxel_size;

    for (auto& each : dwi_files)
    {
        src.src_bvalues.push_back(each->bvalue);
        src.src_bvectors.push_back(each->bvec);
        src.src_dwi_data.push_back(each->begin());
    }


    std::ifstream file(std::filesystem::exists(intro_file_name) ? intro_file_name : "README");
    if(file)
    {
        src.voxel.intro = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        tipl::out() << "read intro: " << std::string(src.voxel.intro.begin(),
                                                     src.voxel.intro.begin()+std::min<size_t>(src.voxel.intro.size(),64)) << "...";
    }

    src.voxel.report = dwi_files.front()->report + src.get_report();
    src.calculate_dwi_sum(false);
    if(!src.save_to_file(file_name))
    {
        error_msg = src.error_msg;
        return false;
    }
    return true;
}
