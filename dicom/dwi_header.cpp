#include <sstream>
#include <string>
#include "image/image.hpp"
#include <boost/lambda/lambda.hpp>
#include "dwi_header.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"
bool DwiHeader::open(const char* filename)
{
    image::io::dicom header;
    if (header.load_from_file(filename))
    {
        header >> image;
        header.get_voxel_size(voxel_size);
    }
    else
    {
        image::io::nifti analyze_header;
        if (!analyze_header.load_from_file(filename))
            return false;
        analyze_header >> image;
        image::flip_xy(image);
        analyze_header.get_voxel_size(voxel_size);
        return true;
    }

    unsigned char man_id = 0;
    {
        std::string manu;
        header.get_text(0x0008,0x0070,manu);//Manufacturer
        if (manu.size() > 2)
        {
            std::string name(manu.begin(),manu.begin()+2);
            if (name == std::string("SI"))
                man_id = 1;
            if (name == std::string("GE"))
                man_id = 2;
            if (name == std::string("Ph"))
                man_id = 3;
            if (name == std::string("TO"))
                man_id = 4;
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
         */
    {
        header.get_value(0x0019,0x100C,bvalue);
        unsigned int gev_length = 0;
        const double* gvec = (const double*)header.get_data(0x0019,0x100E,gev_length);// B-vector
        if(gvec)
            std::copy(gvec,gvec+3,bvec.begin());
        else
        // from b-matrix
        {
            gvec = (const double*)header.get_data(0x0019,0x1027,gev_length);// B-vector
            if (gvec)
            {
                if (gvec[0] != 0.0)
                    std::copy(gvec,gvec+3,bvec.begin());
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

        if(!gvec)// get csa data
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
        break;
    case 3://Phillips
        // get b-table
    {
        unsigned int gvalue_length = 0;
        // GE header
        const double* gvalue = (const double*)header.get_data(0x0018,0x9089,gvalue_length);// B-vector
        if(gvalue && gvalue_length == 24)
            std::copy(gvalue,gvalue+3,bvec.begin());
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
    }

    // apply slices orientation
    float A[9];
    header.get_image_orientation(A);
    // corrected bvec = A'*bvec
    image::vector<3,float> cbvec;
    image::vector_rotation(bvec.begin(),cbvec.begin(),A,image::vdim<3>());
    bvec = cbvec;
    bvec.normalize();
    return true;
}

/*
if (sort_and_merge)
{

    // merge files of the same bvec
    begin_prog("Merge bvalue Files");
    for (unsigned int i = 0;check_prog(i,dwi_files.size());++i)
    {
        unsigned int j = i + 1;
        for (;j < dwi_files.size() && dwi_files[i] == dwi_files[j];++j)
            ;
        if (j == i + 1)
            continue;
        std::vector<unsigned int> sum(dwi_files[i].size());
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
int get_slice_pile(boost::ptr_vector<DwiHeader>& dwi_files)
{
    unsigned int slice_pile = 1; // pile up the slices into an image volume
    // in case of slice number = 1, it is highly possible that the images are not mosaic
    // so we have to find out the exact slice number
    if (dwi_files.front().image.geometry()[2] == 1)
    {
        float b = dwi_files[0].get_bvalue();
        float bx = dwi_files[0].get_bvec()[0];
        float by = dwi_files[0].get_bvec()[1];
        float bz = dwi_files[0].get_bvec()[2];
        for (;slice_pile < dwi_files.size();++slice_pile)
            if (b != dwi_files[slice_pile].get_bvalue() ||
                    bx != dwi_files[slice_pile].get_bvec()[0] ||
                    by != dwi_files[slice_pile].get_bvec()[1] ||
                    bz != dwi_files[slice_pile].get_bvec()[2])
                break;
    }
    return slice_pile;
}

void sort_dwi(boost::ptr_vector<DwiHeader>& dwi_files)
{
    // selection sort
    for (unsigned int i = 0;i < dwi_files.size();++i)
        for (unsigned int j = i+1;j < dwi_files.size();++j)
            if (dwi_files[j] < dwi_files[i])
                dwi_files[i].swap(dwi_files[j]);
}

void correct_t2(boost::ptr_vector<DwiHeader>& dwi_files)
{
    image::geometry<3> geo = dwi_files.front().image.geometry();
    //find out if there are two b0 images having different TE
    std::vector<unsigned int> b0_index;
    std::vector<float> b0_te;
    for (unsigned int index = 0;index < dwi_files.size();++index)
        if (dwi_files[index].bvalue == 0.0)
        {
            b0_index.push_back(index);
            b0_te.push_back(dwi_files[index].te);
        }
    if (b0_index.size() <= 1)
        return;
    // if multiple TE
    std::vector<double> spin_density(geo.size());
    // average the b0 images
    {
        for (unsigned int index = 0;index < b0_index.size();++index)
            image::add(spin_density.begin(),spin_density.end(),dwi_files[b0_index[index]].begin());
        image::divide_constant(spin_density.begin(),spin_density.end(),b0_index.size());
    }

    // if multiple TE, then we can perform T2 correction
    if (*std::max_element(b0_te.begin(),b0_te.end()) != *std::min_element(b0_te.begin(),b0_te.end()))
    {
        std::vector<double> neg_inv_T2(geo.size());//-1/T2
        {
            //begin_prog("Eliminating T2 effect");
            for (image::pixel_index<3> index;index.is_valid(geo);index.next(geo))
            {
                std::vector<float> te_samples;
                std::vector<float> log_Mxy_samples;
                std::vector<image::pixel_index<3> > neighbor_index1,neighbor_index2;
                image::get_neighbors(index,geo,1,neighbor_index1);
                for (unsigned int i = 0;i < b0_te.size();++i)
                {
                    log_Mxy_samples.push_back(dwi_files[b0_index[i]].image[index.index()]);
                    te_samples.push_back(b0_te[i]);
                    for (unsigned int j = 0;j < neighbor_index1.size();++j)
                    {
                        log_Mxy_samples.push_back(dwi_files[b0_index[i]].image[neighbor_index1[j].index()]);
                        te_samples.push_back(b0_te[i]);
                    }
                }
                // if not enough b0 images, take the neighbors!
                if (b0_te.size() < 4)
                {
                    image::get_neighbors(index,geo,2,neighbor_index2);
                    for (unsigned int i = 0;i < b0_te.size();++i)
                    {
                        log_Mxy_samples.push_back(dwi_files[b0_index[i]].image[index.index()]);
                        te_samples.push_back(b0_te[i]);

                        for (unsigned int j = 0;j < neighbor_index2.size();++j)
                        {
                            log_Mxy_samples.push_back(dwi_files[b0_index[i]].image[neighbor_index2[j].index()]);
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
                std::pair<double,double> T2_M0 = image::linear_regression(te_samples.begin(),te_samples.end(),log_Mxy_samples.begin());
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
            if (dwi_files[index].bvalue == 0.0)
                continue;
            DwiHeader& cur_image = dwi_files[index];
            float cur_te = dwi_files[index].te;
            for (unsigned int i = 0;i < geo.size();++i)
                if (neg_inv_T2[i] != 0.0)
                    cur_image[i] *= std::exp(-cur_te*neg_inv_T2[i]);
        }

        for (unsigned int index = 0;index < geo.size();++index)
        {
            if (neg_inv_T2[index] != 0.0)
                neg_inv_T2[index] = -1.0/neg_inv_T2[index];
            else
                neg_inv_T2[index] = 2000;
        }

        //write_mat.write("spin_density",&*spin_density.begin(),1,total_size);
        //write_mat.write("T2",&*neg_inv_T2.begin(),1,total_size);

    }

    // replace b0 with spin density map
    std::copy(spin_density.begin(),spin_density.end(),dwi_files[b0_index.front()].begin());
    // remove additional b0
    for (unsigned int index = b0_index.front()+1;index < dwi_files.size();)
    {
        if (dwi_files[index].bvalue == 0.0)
            dwi_files.erase(dwi_files.begin()+index);
        else
            ++index;
    }
}
// upsampling 1: upsampling 2: downsampling
bool DwiHeader::output_src(const char* di_file,boost::ptr_vector<DwiHeader>& dwi_files,int upsampling,bool topdown)
{
    sort_dwi(dwi_files);
    unsigned int slice_pile = get_slice_pile(dwi_files);
    if (slice_pile == 1) // Siemens Mosaic
        correct_t2(dwi_files);

    gz_mat_write write_mat(di_file);
    if(!write_mat)
        return false;

    image::geometry<3> geo = dwi_files.front().image.geometry();
    if (slice_pile > 1)
        geo[2] = slice_pile;

    //store dimension
    unsigned int output_size = 0;
    {
        short dimension[3];
        std::copy(geo.begin(),geo.end(),dimension);
        if(upsampling == 1)
            std::for_each(dimension,dimension+3,boost::lambda::_1 <<= 1);
        if(upsampling == 2)
            std::for_each(dimension,dimension+3,boost::lambda::_1 >>= 1);
        output_size = dimension[0]*dimension[1]*dimension[2];
        write_mat.write("dimension",dimension,1,3);
    }
    //store dimension
    {
        float voxel_size[3];
        std::copy(dwi_files.front().voxel_size,dwi_files.front().voxel_size+3,voxel_size);
        if(upsampling == 1)
            std::for_each(voxel_size,voxel_size+3,boost::lambda::_1 /= 2.0);
        if(upsampling == 2)
            std::for_each(voxel_size,voxel_size+3,boost::lambda::_1 *= 2.0);
        write_mat.write("voxel_size",voxel_size,1,3);
    }

    //store images
    begin_prog("Save Files");
    for (unsigned int index = 0,id = 0;check_prog(index,dwi_files.size());index+=slice_pile,++id)
    {
        // avoid negative values
        image::lower_threshold(dwi_files[index].image,(short)0);
        std::ostringstream name;
        image::basic_image<short,3> buffer;
        const unsigned short* ptr = 0;
        name << "image" << id;
        if (slice_pile == 1) // Siemens Mosaic
        {
            if(topdown)
                image::flip_z(dwi_files[index].image);
            ptr = (const unsigned short*)dwi_files[index].begin();
            if(upsampling)
            {
                buffer.resize(geo);
                std::copy(ptr,ptr+geo.size(),buffer.begin());
                if(upsampling == 1)
                    image::upsampling(buffer);
                else
                    image::downsampling(buffer);
                ptr = (const unsigned short*)&*buffer.begin();
            }

        }
        else
        {
            // GE
            buffer.resize(geo);
            for (unsigned int z = 0;z < slice_pile;++z)
                std::copy(dwi_files[index+z].begin(),
                          dwi_files[index+z].begin() + dwi_files[index+z].size(),
                          buffer.begin() + z * dwi_files[index+z].size());
            if(topdown)
                image::flip_z(buffer);
            if(upsampling == 1)
                image::upsampling(buffer);
            if(upsampling == 2)
                image::downsampling(buffer);
            ptr = (const unsigned short*)&*buffer.begin();
        }
        write_mat.write(name.str().c_str(),ptr,1,output_size);

    }
    // store bvec file
    {
        std::vector<float> b_table;
        for (unsigned int index = 0;index < dwi_files.size();index+=slice_pile)
        {
            b_table.push_back(dwi_files[index].get_bvalue());
            std::copy(dwi_files[index].get_bvec(),dwi_files[index].get_bvec()+3,std::back_inserter(b_table));
        }
        write_mat.write("b_table",&b_table[0],4,b_table.size()/4);
    }

    if(!dwi_files[0].grad_dev.empty())
        write_mat.write("grad_dev",&*dwi_files[0].grad_dev.begin(),dwi_files[0].grad_dev.size()/9,9);
    if(!dwi_files[0].mask.empty())
        write_mat.write("mask",&*dwi_files[0].mask.begin(),1,dwi_files[0].mask.size());
    return true;
}
