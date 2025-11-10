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

    float slice_thickness = header.get_float(0x0018, 0x0050);
    float spacing_between = header.get_float(0x0018, 0x0088);
    if(slice_thickness > 0.0f)
        out << " The slice thickness was " << slice_thickness <<
               (spacing_between == slice_thickness ? " mm (no gap)." : " mm.");
    if(spacing_between > slice_thickness)
        out << " The spacing between slices is " << spacing_between << " mm.";

    std::string pixel_spacing;
    if(header.get_text(0x0028, 0x0030, pixel_spacing))
    {
        // DICOM pixel spacing is usually stored as two numbers separated by a backslash.
        std::replace(pixel_spacing.begin(), pixel_spacing.end(), '\\', ' ');
        std::istringstream iss(pixel_spacing);
        float res1, res2;
        if(iss >> res1 >> res2)
            out << " The in-plane resolution was " << res1 << " mm x " << res2 << " mm.";
    }

    float flip_angle = header.get_float(0x0018, 0x1314);
    if(flip_angle > 0.0f)
        out << " The flip angle was " << flip_angle << " degrees.";

    float pixel_bandwidth = header.get_float(0x0018, 0x0095);
    if(pixel_bandwidth > 0.0f)
        out << " The pixel bandwidth was " << pixel_bandwidth << " Hz.";

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
        return tipl::io::gz_nifti(filename,std::ios::in)
                >> voxel_size >> image
                >> [&](const std::string& e){tipl::error() << (error_msg = e);};
    header >> std::tie(image,voxel_size);
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
    get_report_from_dicom(header,report);

    float orientation_matrix[9];
    unsigned char dim_order[3] = {0,1,2};
    char flip[3] = {0,0,0};
    bool has_orientation_info = false;

    if(header.get_image_orientation(orientation_matrix))
    {
        tipl::get_orientation(orientation_matrix,dim_order,flip);
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



bool DwiHeader::has_b_table(std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    for(size_t i = 0;i < dwi_files.size();++i)
        if(dwi_files[i]->bvalue > 0.0f)
            return true;
    return false;
}
