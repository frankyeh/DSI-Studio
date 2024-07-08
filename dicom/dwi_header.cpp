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
    tipl::io::dicom header;
    if (!header.load_from_file(filename))
    {
        tipl::io::nifti nii;
        if (!nii.load_from_file(filename))
            return false;
        nii.toLPS(image);
        nii.get_voxel_size(voxel_size);
        file_name = filename;
        return true;
    }

    header >> image;
    if(header.is_compressed)
    {
        tipl::image<2,short> I;
        if(!get_compressed_image(header,I))
        {
            tipl::error() << "unsupported transfer syntax: " << header.encoding << std::endl;
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
// upsampling 1: upsampling 2: downsampling
extern std::string src_error_msg;
bool DwiHeader::output_src(const char* di_file,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
                           int upsampling,bool sort_btable)
{
    tipl::progress prog("save ",std::filesystem::path(di_file).filename().string().c_str());
    if(!has_b_table(dwi_files))
    {
        src_error_msg = "invalid b-table";
        return false;
    }
    if(dwi_files.empty())
    {
        src_error_msg = "no DWI data for output";
        return false;
    }
    if(sort_btable)
    {
        sort_dwi(dwi_files);
    }
    auto temp_file = std::string(di_file) + ".tmp.gz";
    {
        tipl::io::gz_mat_write write_mat(temp_file.c_str());
        if(!write_mat)
        {
            src_error_msg = "cannot output file to ";
            src_error_msg += di_file;
            return false;
        }
        tipl::shape<3> geo = dwi_files.front()->image.shape();

        //store dimension
        tipl::shape<3> output_dim(geo);
        {
            tipl::vector<3,uint16_t> dimension(geo);
            tipl::vector<3> voxel_size(dwi_files.front()->voxel_size);

            if(upsampling == 1) // upsampling 2
            {
                voxel_size /= 2.0;
                dimension *= 2;
            }
            if(upsampling == 2) // downsampling 2
            {
                voxel_size *= 2.0;
                dimension /= 2;
            }
            if(upsampling == 3) // upsampling 4
            {
                voxel_size /= 4.0;
                dimension *= 4;
            }
            if(upsampling == 4) // downsampling 4
            {
                voxel_size *= 4.0;
                dimension /= 4;
            }
            output_dim[0] = dimension[0];
            output_dim[1] = dimension[1];
            output_dim[2] = dimension[2];

            write_mat.write("dimension",output_dim);
            write_mat.write("voxel_size",voxel_size);

        }
        // store bvec file
        {
            std::vector<float> b_table;
            for (unsigned int index = 0;index < dwi_files.size();++index)
            {
                if(dwi_files[index]->bvalue < 100.0f)
                {
                    b_table.push_back(0.0f);
                    b_table.push_back(0.0f);
                    b_table.push_back(0.0f);
                    b_table.push_back(0.0f);
                }
                else
                {
                    b_table.push_back(dwi_files[index]->bvalue);
                    std::copy(dwi_files[index]->bvec.begin(),dwi_files[index]->bvec.end(),std::back_inserter(b_table));
                }
            }
            write_mat.write("b_table",b_table,4);
        }
        if(!dwi_files[0]->grad_dev.empty())
            write_mat.write("grad_dev",dwi_files[0]->grad_dev,uint32_t(dwi_files[0]->grad_dev.size()/9));
        if(!dwi_files[0]->mask.empty())
            write_mat.write("mask",dwi_files[0]->mask,dwi_files[0]->mask.plane_size());

        //store images
        for (unsigned int index = 0;prog(index,(unsigned int)(dwi_files.size()));++index)
        {
            std::ostringstream name;
            tipl::image<3,unsigned short> buffer;
            const unsigned short* ptr = 0;
            name << "image" << index;
            ptr = (const unsigned short*)dwi_files[index]->begin();
            if(upsampling)
            {
                buffer.resize(geo);
                std::copy(ptr,ptr+geo.size(),buffer.begin());
                if(upsampling == 1)
                    tipl::upsampling(buffer);
                if(upsampling == 2)
                    tipl::downsampling(buffer);
                if(upsampling == 3)
                {
                    tipl::upsampling(buffer);
                    tipl::upsampling(buffer);
                }
                if(upsampling == 4)
                {
                    tipl::downsampling(buffer);
                    tipl::downsampling(buffer);
                }
                ptr = (const unsigned short*)&*buffer.begin();
            }
            write_mat.write(name.str().c_str(),ptr,output_dim.plane_size(),output_dim.depth());
        }

        if(prog.aborted())
        {
            src_error_msg = "output aborted";
            goto delete_file;
        }
        std::string report1 = dwi_files.front()->report;
        std::string report2;
        {
            src_data image_model;
            for (unsigned int index = 0;index < dwi_files.size();++index)
                image_model.src_bvalues.push_back(dwi_files[index]->bvalue);
            image_model.voxel.vs = tipl::vector<3>(dwi_files.front()->voxel_size);
            image_model.get_report(report2);
        }
        report1 += report2;
        write_mat.write("report",report1);
    }
    if(std::filesystem::exists(di_file))
        std::filesystem::remove(di_file);
    std::filesystem::rename(temp_file,di_file);
    return true;

    delete_file:
    std::filesystem::remove(temp_file);
    return false;
}
