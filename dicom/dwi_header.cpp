#include <QFileInfo>
#include <sstream>
#include <string>
#include "TIPL/tipl.hpp"
#include "dwi_header.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"
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
            show_progress() << "Unsupported transfer syntax:" << header.encoding << std::endl;
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

    unsigned char man_id = 0;
    {
        std::string manu;
        header.get_text(0x0008,0x0070,manu);//Manufacturer
        if (manu.size() > 2)
        {
            std::string name(manu.begin(),manu.begin()+2);
            name[0] = ::toupper(name[0]);
            name[1] = ::toupper(name[1]);
            if (name == std::string("SI"))
                man_id = 1;
            if (name == std::string("GE"))
                man_id = 2;
            if (name == std::string("PH"))
                man_id = 3;
            if (name == std::string("TO"))
                man_id = 4;
            if (name == std::string("UI"))
                man_id = 5;
        }
    }
    // get TE
    header.get_value(0x0018,0x0081,te);


    switch (man_id)
    {
    case 1://SIEMENS
        // get b-table
        /*
        0019;XX0C;SIEMENS MR HEADER ;B_value ;1;IS;1
        0019;XX0D;SIEMENS MR HEADER ;DiffusionDirectionality ;1;CS;1
        0019;XX0E;SIEMENS MR HEADER ;DiffusionGradientDirection ;1;FD;3
        0019;XX27;SIEMENS MR HEADER ;B_matrix ;1;FD;6

        (0018,9075) DiffusionDirectionality
        (0018,9087) DiffusionBValue
         */
    {
        bool has_b = false;
        if(header.get_value(0x0018,0x9087,bvalue))
        {
            has_b = true;
            if(bvalue)
            {
                unsigned int gev_length = 0;
                const double* gvec = (const double*)header.get_data(0x0018,0x9089,gev_length);// B-vector
                if(gvec)
                {
                    bvec[0] = float(gvec[0]);
                    bvec[1] = float(gvec[1]);
                    bvec[2] = float(gvec[2]);
                    bvec.normalize();
                }
                else
                    has_b = false;
            }

        }
        else
        {
            header.get_value(0x0019,0x100C,bvalue);
            unsigned int gev_length = 0;
            const double* gvec = (const double*)header.get_data(0x0019,0x100E,gev_length);// B-vector
            if(gvec)
            {
                bvec[0] = float(gvec[0]);
                bvec[1] = float(gvec[1]);
                bvec[2] = float(gvec[2]);
            }
            else
            // from b-matrix
            {
                gvec = (const double*)header.get_data(0x0019,0x1027,gev_length);// B-vector
                if (gvec)
                {
                    if (gvec[0] != 0.0)
                    {
                        bvec[0] = float(gvec[0]);
                        bvec[1] = float(gvec[1]);
                        bvec[2] = float(gvec[2]);
                    }
                    else
                        if (gvec[3] != 0.0)
                        {
                            bvec[0] = 0.0;
                            bvec[1] = gvec[3];
                            bvec[2] = gvec[4];
                        }
                        else
                        {
                            bvec[0] = 0;
                            bvec[1] = 0;
                            bvec[2] = gvec[5];
                        }
                }
            }
            has_b = gvec != 0;
        }
        if(!has_b)// get csa data
        {
            const char* b_value = header.get_csa_data("B_value",0);
            const char* bx = header.get_csa_data("DiffusionGradientDirection",0);
            const char* by = header.get_csa_data("DiffusionGradientDirection",1);
            const char* bz = header.get_csa_data("DiffusionGradientDirection",2);
            if (b_value && bx && by && bz)
            {
                std::istringstream(std::string(b_value)) >> bvalue;
                std::istringstream(std::string(bx)) >> bvec[0];
                std::istringstream(std::string(by)) >> bvec[1];
                std::istringstream(std::string(bz)) >> bvec[2];
            }
        }
    }
    break;
    default:
    case 2://GE
        // get b-table
        header.get_value(0x0019,0x10BB,bvec[0]);
        header.get_value(0x0019,0x10BC,bvec[1]);
        header.get_value(0x0019,0x10BD,bvec[2]);
        bvec[2] = -bvec[2];
        header.get_value(0x0043,0x1039,bvalue);
        bvalue *= bvec.length();
        bvec.normalize();
        break;
    case 3://Phillips
        // get b-table
    {
        unsigned int gvalue_length = 0;
        // GE header
        const double* gvalue = (const double*)header.get_data(0x0018,0x9089,gvalue_length);// B-vector
        if(gvalue && gvalue_length == 24)
        {
            bvec[0] = float(gvalue[0]);
            bvec[1] = float(gvalue[1]);
            bvec[2] = float(gvalue[2]);
        }
        gvalue = (const double*)header.get_data(0x0018,0x9087,gvalue_length);//B-Value
        if(gvalue && gvalue_length == 8)
            bvalue = gvalue[0];

    }

    break;
    case 4://TOSHIBA
    {
        unsigned int gvalue_length = 0;
        const double* gvalue = (const double*)header.get_data(0x0018,0x9087,gvalue_length);//B-Value
        if(gvalue && gvalue_length == 8)
            bvalue = gvalue[0];
    }
    {
        unsigned int gvalue_length = 0;
        const char* str = (const char*)header.get_data(0x0020,0x4000,gvalue_length);
        //B-Value string e.g.  b=2000(0.140,0.134,-0.981)
        if(str)
        {
            std::string b_str(str+2);
            std::replace(b_str.begin(),b_str.end(),'(',' ');
            std::replace(b_str.begin(),b_str.end(),')',' ');
            std::replace(b_str.begin(),b_str.end(),',',' ');
            std::istringstream in(b_str);
            in >> bvalue >> bvec[0] >> bvec[1] >> bvec[2];
        }
    }
        break;
    case 5://UIH
    {
        unsigned int gvalue_length = 0;
        // GE header
        const double* gvalue = (const double*)header.get_data(0x0065,0x1037,gvalue_length);// B-vector
        if(gvalue && gvalue_length == 24)
        {
            bvec[0] = float(gvalue[0]);
            bvec[1] = float(gvalue[1]);
            bvec[2] = float(gvalue[2]);
        }
        gvalue = (const double*)header.get_data(0x0065,0x1009,gvalue_length);//B-Value
        if(gvalue && gvalue_length == 8)
            bvalue = gvalue[0];
    }
        break;
    }

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
    if(bvalue == 0.0f)
        bvec[0] = bvec[1] = bvec[2] = 0.0f;
    return true;
}

/*
if (sort_and_merge)
{

    // merge files of the same bvec
    progress prog_("Merge bvalue Files");
    for (unsigned int i = 0;progress::at(i,dwi_files.size());++i)
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

void correct_t2(std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    tipl::shape<3> geo = dwi_files.front()->image.shape();
    //find out if there are two b0 images having different TE
    std::vector<unsigned int> b0_index;
    std::vector<float> b0_te;
    for (unsigned int index = 0;index < dwi_files.size();++index)
        if (dwi_files[index]->bvalue == 0.0)
        {
            b0_index.push_back(index);
            b0_te.push_back(dwi_files[index]->te);
        }
    if (b0_index.size() <= 1)
        return;
    // if multiple TE
    std::vector<double> spin_density(geo.size());
    // average the b0 images
    {
        for (unsigned int index = 0;index < b0_index.size();++index)
            tipl::add(spin_density.begin(),spin_density.end(),dwi_files[b0_index[index]]->begin());
        tipl::divide_constant(spin_density.begin(),spin_density.end(),b0_index.size());
    }

    // if multiple TE, then we can perform T2 correction
    if (tipl::max_value(b0_te) != tipl::min_value(b0_te))
    {
        std::vector<double> neg_inv_T2(geo.size());//-1/T2
        {
            //progress prog_("Eliminating T2 effect");
            for (tipl::pixel_index<3> index(geo);index < geo.size();++index)
            {
                std::vector<float> te_samples;
                std::vector<float> log_Mxy_samples;
                std::vector<tipl::pixel_index<3> > neighbor_index1,neighbor_index2;
                tipl::get_neighbors(index,geo,1,neighbor_index1);
                for (unsigned int i = 0;i < b0_te.size();++i)
                {
                    log_Mxy_samples.push_back(dwi_files[b0_index[i]]->image[index.index()]);
                    te_samples.push_back(b0_te[i]);
                    for (unsigned int j = 0;j < neighbor_index1.size();++j)
                    {
                        log_Mxy_samples.push_back(dwi_files[b0_index[i]]->image[neighbor_index1[j].index()]);
                        te_samples.push_back(b0_te[i]);
                    }
                }
                // if not enough b0 images, take the neighbors!
                if (b0_te.size() < 4)
                {
                    tipl::get_neighbors(index,geo,2,neighbor_index2);
                    for (unsigned int i = 0;i < b0_te.size();++i)
                    {
                        log_Mxy_samples.push_back(dwi_files[b0_index[i]]->image[index.index()]);
                        te_samples.push_back(b0_te[i]);

                        for (unsigned int j = 0;j < neighbor_index2.size();++j)
                        {
                            log_Mxy_samples.push_back(dwi_files[b0_index[i]]->image[neighbor_index2[j].index()]);
                            te_samples.push_back(b0_te[i]);
                        }
                    }
                }
                for (unsigned int i = 0;i < log_Mxy_samples.size();)
                {
                    if (log_Mxy_samples[i])
                    {
                        log_Mxy_samples[i] = std::log(log_Mxy_samples[i]);
                        ++i;
                    }
                    else
                    {
                        log_Mxy_samples[i] = log_Mxy_samples.back();
                        te_samples[i] = te_samples.back();
                        log_Mxy_samples.pop_back();
                        te_samples.pop_back();
                    }
                }
                if (log_Mxy_samples.empty())
                    continue;
                // (-1/T2,logM0);
                std::pair<double,double> T2_M0 = tipl::linear_regression(te_samples.begin(),te_samples.end(),log_Mxy_samples.begin());
                /*												T1			T2
                Cerebrospinal fluid (similar to pure water) 	2200-2400 	500-1400
                Gray matter of cerebrum 						920 		100
                White matter of cerebrum 						780 		90
                */
                if (T2_M0.first < -1.0/2000.0)
                {
                    spin_density[index.index()] = std::exp(T2_M0.second);
                    neg_inv_T2[index.index()] = T2_M0.first;
                }
                // If the T2 is too long, then exp(-TE/T2)~1, spin density is just the averaged b0 signal

            }
        }


        // perform correction for each image
        for (unsigned int index = 0;index < dwi_files.size();++index)
        {
            // b0 will be handled later
            if (dwi_files[index]->bvalue == 0.0)
                continue;
            DwiHeader& cur_image = *dwi_files[index];
            float cur_te = dwi_files[index]->te;
            for (int i = 0;i < geo.size();++i)
                if (neg_inv_T2[i] != 0.0)
                    cur_image[i] *= std::exp(-cur_te*neg_inv_T2[i]);
        }

        for (int index = 0;index < geo.size();++index)
        {
            if (neg_inv_T2[index] != 0.0)
                neg_inv_T2[index] = -1.0/neg_inv_T2[index];
            else
                neg_inv_T2[index] = 2000;
        }
    }

    // replace b0 with spin density map
    std::copy(spin_density.begin(),spin_density.end(),dwi_files[b0_index.front()]->begin());
    // remove additional b0
    for (unsigned int index = b0_index.front()+1;index < dwi_files.size();)
    {
        if (dwi_files[index]->bvalue == 0.0)
            dwi_files.erase(dwi_files.begin()+index);
        else
            ++index;
    }
}



bool DwiHeader::has_b_table(std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    for(size_t i = 0;i < dwi_files.size();++i)
        if(dwi_files[i]->bvalue > 0.0f)
            return true;
    return false;
}
bool DwiHeader::consistent_dimension(std::vector<std::shared_ptr<DwiHeader> >& dwi_files)
{
    if(dwi_files.empty())
        return true;
    for(size_t i = 1;i < dwi_files.size();++i)
        if(dwi_files[i]->image.shape() != dwi_files[0]->image.shape())
            return false;
    return true;
}

// upsampling 1: upsampling 2: downsampling
extern std::string src_error_msg;
bool DwiHeader::output_src(const char* di_file,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
                           int upsampling,bool sort_btable)
{
    progress prog("save ",QFileInfo(di_file).fileName().toStdString().c_str());
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
        correct_t2(dwi_files);
    }
    gz_mat_write write_mat(di_file);
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
    for (unsigned int index = 0;prog.at(index,(unsigned int)(dwi_files.size()));++index)
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
        return false;
    }
    std::string report1 = dwi_files.front()->report;
    std::string report2;
    {
        ImageModel image_model;
        for (unsigned int index = 0;index < dwi_files.size();++index)
            image_model.src_bvalues.push_back(dwi_files[index]->bvalue);
        image_model.voxel.vs = tipl::vector<3>(dwi_files.front()->voxel_size);
        image_model.get_report(report2);
    }
    report1 += report2;
    write_mat.write("report",report1);
    return true;
}
