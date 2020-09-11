#include <QFileInfo>
#include <QInputDialog>
#include "image_model.hpp"
#include "odf_process.hpp"
#include "dti_process.hpp"
#include "fib_data.hpp"

void ImageModel::draw_mask(tipl::color_image& buffer,int position)
{
    if (!dwi.size())
        return;
    buffer.resize(tipl::geometry<2>(dwi.width(),dwi.height()));
    long offset = long(position)*long(buffer.size());
    std::copy(dwi.begin()+ offset,
              dwi.begin()+ offset + long(buffer.size()),buffer.begin());

    unsigned char* slice_image_ptr = &*dwi.begin() + long(buffer.size())*position;
    unsigned char* slice_mask = &*voxel.mask.begin() + long(buffer.size())*position;

    tipl::color_image buffer2(tipl::geometry<2>(dwi.width()*2,dwi.height()));
    tipl::draw(buffer,buffer2,tipl::vector<2,int>());
    for (unsigned int index = 0; index < buffer.size(); ++index)
    {
        unsigned char value = slice_image_ptr[index];
        if (slice_mask[index])
            buffer[index] = tipl::rgb(uint8_t(255), value, value);
        else
            buffer[index] = tipl::rgb(value, value, value);
    }
    tipl::draw(buffer,buffer2,tipl::vector<2,int>(dwi.width(),0));
    buffer2.swap(buffer);
}

void ImageModel::calculate_dwi_sum(bool update_mask)
{
    dwi_sum.clear();
    dwi_sum.resize(voxel.dim);
    tipl::par_for(src_dwi_data.size(),[&](unsigned int index)
    {
        if(index > 0 && src_bvalues[index] == 0.0f)
            return;
        for (size_t pos = 0;pos < dwi_sum.size();++pos)
            dwi_sum[pos] += src_dwi_data[index][pos];
    });
    float otsu = tipl::segmentation::otsu_threshold(dwi_sum);
    float max_value = std::min<float>(*std::max_element(dwi_sum.begin(),dwi_sum.end()),otsu*3.0f);
    float min_value = max_value;
    // handle 0 strip with background value condition
    {
        for (unsigned int index = 0;index < dwi_sum.size();index += uint32_t(dwi_sum.width()+1))
            if (dwi_sum[index] < min_value && dwi_sum[index] > 0)
                min_value = dwi_sum[index];
        if(min_value >= max_value)
            min_value = 0;
        else
            tipl::minus_constant(dwi_sum,min_value);
    }
    float r = max_value-min_value;
    if(r != 0.0f)
        r = 255.9f/r;
    // update dwi
    dwi.resize(voxel.dim);
    for(size_t index = 0;index < dwi.size();++index)
        dwi[index] = uint8_t(std::max<float>(0.0f,std::min<float>(255.0f,std::floor(dwi_sum[index]*r))));

    if(update_mask)
    {
        tipl::threshold(dwi_sum,voxel.mask,(max_value-min_value)*0.2f,1,0);
        if(dwi_sum.depth() < 200)
        {
            tipl::par_for(voxel.mask.depth(),[&](int i)
            {
                tipl::pointer_image<unsigned char,2> I(&voxel.mask[0]+size_t(i)*voxel.mask.plane_size(),
                        tipl::geometry<2>(voxel.mask.width(),voxel.mask.height()));
                tipl::morphology::defragment(I);
                tipl::morphology::recursive_smoothing(I,10);
                tipl::morphology::defragment(I);
                tipl::morphology::negate(I);
                tipl::morphology::defragment(I);
                tipl::morphology::negate(I);

            });
            tipl::morphology::recursive_smoothing(voxel.mask,10);
            tipl::morphology::defragment(voxel.mask);
            tipl::morphology::recursive_smoothing(voxel.mask,10);
        }
    }
}

void ImageModel::remove(unsigned int index)
{
    src_dwi_data.erase(src_dwi_data.begin()+index);
    src_bvalues.erase(src_bvalues.begin()+index);
    src_bvectors.erase(src_bvectors.begin()+index);
    shell.clear();
    voxel.dwi_data.clear();
}

typedef boost::mpl::vector<
    ReadDWIData,
    Dwi2Tensor
> check_btable_process;


void flip_fib_dir(std::vector<tipl::vector<3> >& fib_dir,const unsigned char* order)
{
    for(size_t j = 0;j < fib_dir.size();++j)
    {
        fib_dir[j] = tipl::vector<3>(fib_dir[j][order[0]],fib_dir[j][order[1]],fib_dir[j][order[2]]);
        if(order[3])
            fib_dir[j][0] = -fib_dir[j][0];
        if(order[4])
            fib_dir[j][1] = -fib_dir[j][1];
        if(order[5])
            fib_dir[j][2] = -fib_dir[j][2];
    }
}
std::vector<size_t> ImageModel::get_sorted_dwi_index(void)
{
    std::vector<size_t> sorted_index(src_bvalues.size());
    std::iota(sorted_index.begin(),sorted_index.end(),0);

    std::sort(sorted_index.begin(),sorted_index.end(),
              [&](size_t left,size_t right)
    {
        if(int(src_bvalues[left])/400 == int(src_bvalues[right])/400)
            return src_bvectors[left] < src_bvectors[right];
        return src_bvalues[left] < src_bvalues[right];
    }
    );
    return sorted_index;
}
void ImageModel::flip_b_table(const unsigned char* order)
{
    for(unsigned int index = 0;index < src_bvectors.size();++index)
    {
        float x = src_bvectors[index][order[0]];
        float y = src_bvectors[index][order[1]];
        float z = src_bvectors[index][order[2]];
        src_bvectors[index][0] = x;
        src_bvectors[index][1] = y;
        src_bvectors[index][2] = z;
        if(order[3])
            src_bvectors[index][0] = -src_bvectors[index][0];
        if(order[4])
            src_bvectors[index][1] = -src_bvectors[index][1];
        if(order[5])
            src_bvectors[index][2] = -src_bvectors[index][2];
    }
    voxel.grad_dev.clear();
}
void ImageModel::pre_dti(void)
{
    bool output_tensor = voxel.output_tensor;
    voxel.output_tensor = false;
    reconstruct<check_btable_process>("Checking b-table");
    voxel.output_tensor = output_tensor;
}

std::string ImageModel::check_b_table(void)
{
    pre_dti();
    std::vector<tipl::image<float,3> > fib_fa(1);
    std::vector<std::vector<tipl::vector<3> > > fib_dir(1);
    fib_fa[0].swap(voxel.fib_fa);
    fib_dir[0].swap(voxel.fib_dir);

    const unsigned char order[24][6] = {
                            {0,1,2,0,0,0},
                            {0,1,2,1,0,0},
                            {0,1,2,0,1,0},
                            {0,1,2,0,0,1},
                            {0,2,1,0,0,0},
                            {0,2,1,1,0,0},
                            {0,2,1,0,1,0},
                            {0,2,1,0,0,1},
                            {1,0,2,0,0,0},
                            {1,0,2,1,0,0},
                            {1,0,2,0,1,0},
                            {1,0,2,0,0,1},
                            {1,2,0,0,0,0},
                            {1,2,0,1,0,0},
                            {1,2,0,0,1,0},
                            {1,2,0,0,0,1},
                            {2,1,0,0,0,0},
                            {2,1,0,1,0,0},
                            {2,1,0,0,1,0},
                            {2,1,0,0,0,1},
                            {2,0,1,0,0,0},
                            {2,0,1,1,0,0},
                            {2,0,1,0,1,0},
                            {2,0,1,0,0,1}};
    const char txt[24][7] = {".012",".012fx",".012fy",".012fz",
                             ".021",".021fx",".021fy",".021fz",
                             ".102",".102fx",".102fy",".102fz",
                             ".120",".120fx",".120fy",".120fz",
                             ".210",".210fx",".210fy",".210fz",
                             ".201",".201fx",".201fy",".201fz"};

    float result[24] = {0};
    float otsu = tipl::segmentation::otsu_threshold(fib_fa[0])*0.6f;
    float cur_score = evaluate_fib(voxel.dim,otsu,fib_fa,[&](uint32_t pos,uint8_t fib){return fib_dir[fib][pos];}).first;
    result[0] = cur_score;
    for(int i = 1;i < 24;++i)// 0 is the current score
    {
        auto new_dir(fib_dir);
        flip_fib_dir(new_dir[0],order[i]);
        result[i] = evaluate_fib(voxel.dim,otsu,fib_fa,[&](uint32_t pos,uint8_t fib){return new_dir[fib][pos];}).first;
    }
    long best = long(std::max_element(result,result+24)-result);
    for(int i = 0;i < 24;++i)
    {
        if(i == best)
            std::cout << (txt[i]+1) << "=BEST";
        else
            std::cout << (txt[i]+1) << "=-" << int(100.0f*(result[best]-result[i])/(result[best]+1.0f)) << "%";
        if(i % 8 == 7)
            std::cout << std::endl;
        else
            std::cout << ",";
    }

    if(result[best] > cur_score)
    {
        std::cout << "b-table corrected by " << txt[best] << " for " << file_name << std::endl;
        flip_b_table(order[best]);
        voxel.load_from_src(*this);
        return txt[best];
    }
    fib_fa[0].swap(voxel.fib_fa);
    fib_dir[0].swap(voxel.fib_dir);

    return std::string();
}
std::vector<std::pair<int,int> > ImageModel::get_bad_slices(void)
{
    voxel.load_from_src(*this);
    std::vector<size_t> sorted_index(get_sorted_dwi_index()),index_mapping;
    for(size_t i = 0;i < sorted_index.size();++i)
        if(i == 0 || src_bvalues[sorted_index[i]] != 0.0f)
            index_mapping.push_back(sorted_index[i]);

    std::vector<char> skip_slice(size_t(voxel.dim.depth()));
    for(size_t i = 0,pos = 0;i < skip_slice.size();++i,pos += voxel.dim.plane_size())
        if(std::accumulate(voxel.mask.begin()+long(pos),voxel.mask.begin()+long(pos)+voxel.dim.plane_size(),0) < voxel.dim.plane_size()/16)
            skip_slice[i] = 1;
        else
            skip_slice[i] = 0
                    ;
    tipl::image<float,2> cor_values(
                tipl::geometry<2>(int(voxel.dwi_data.size()),voxel.dim.depth()));

    tipl::par_for(voxel.dwi_data.size(),[&](size_t index)
    {
        auto I = tipl::make_image(voxel.dwi_data[index],voxel.dim);
        size_t value_index = index*size_t(voxel.dim.depth());
        for(size_t z = 0,pos = 0;z < size_t(voxel.dim.depth());
                                ++z,pos += voxel.dim.plane_size())
        {
            float cor = 0.0f;
            if(z)
                cor = float(tipl::correlation(&I[pos],&I[pos]+I.plane_size(),&I[pos]-I.plane_size()));
            if(z+1 < size_t(voxel.dim.depth()))
                cor = std::max<float>(cor,float(tipl::correlation(&I[pos],&I[pos]+I.plane_size(),&I[pos]+I.plane_size())));

            if(index)
                cor = std::max<float>(cor,float(tipl::correlation(voxel.dwi_data[index]+pos,
                                        voxel.dwi_data[index]+pos+voxel.dim.plane_size(),
                                        voxel.dwi_data[index-1]+pos)));
            if(index+1 < voxel.dwi_data.size())
                cor = std::max<float>(cor,float(tipl::correlation(voxel.dwi_data[index]+pos,
                                                            voxel.dwi_data[index]+pos+voxel.dim.plane_size(),
                                                            voxel.dwi_data[index+1]+pos)));

            cor_values[value_index+z] = cor;
        }
    });
    // check the difference with neighborings
    std::vector<size_t> bad_i,bad_z;
    std::vector<float> sum;
    for(size_t i = 0,pos = 0;i < voxel.dwi_data.size();++i)
    {
        for(size_t z = 0;z < size_t(voxel.dim.depth());++z,++pos)
        if(!skip_slice[z])
        {
            // ignore the top and bottom slices
            if(z == 0 || z + 2 >= size_t(voxel.dim.depth()))
                continue;
            float v[4] = {0.0f,0.0f,0.0f,0.0f};
            if(z > 0)
                v[0] = cor_values[pos-1]-cor_values[pos];
            if(z+1 < size_t(voxel.dim.depth()))
                v[1] = cor_values[pos+1]-cor_values[pos];
            if(i > 0)
                v[2] = cor_values[pos-size_t(voxel.dim.depth())]-cor_values[pos];
            if(i+1 < voxel.dwi_data.size())
                v[3] = cor_values[pos+size_t(voxel.dim.depth())]-cor_values[pos];
            float s = 0.0;
            s = v[0]+v[1]+v[2]+v[3];
            if(s > 0.4f)
            {
                bad_i.push_back(index_mapping[i]);
                bad_z.push_back(z);
                sum.push_back(s);
            }
        }
    }

    std::vector<std::pair<int,int> > result;

    auto arg = tipl::arg_sort(sum,std::less<float>());
    //tipl::image<float,3> bad_I(tipl::geometry<3>(voxel.dim[0],voxel.dim[1],bad_i.size()));
    for(size_t i = 0,out_pos = 0;i < bad_i.size();++i,out_pos += voxel.dim.plane_size())
    {
        result.push_back(std::make_pair(bad_i[arg[i]],bad_z[arg[i]]));
        //int pos = bad_z[arg[i]]*voxel.dim.plane_size();
    //    std::copy(voxel.dwi_data[bad_i[arg[i]]]+pos,voxel.dwi_data[bad_i[arg[i]]]+pos+voxel.dim.plane_size(),bad_I.begin()+out_pos);
    }
    //bad_I.save_to_file<gz_nifti>("D:/bad.nii.gz");
    return result;
}

float ImageModel::quality_control_neighboring_dwi_corr(void)
{
    std::vector<std::pair<size_t,size_t> > corr_pairs;
    for(size_t i = 0;i < src_bvalues.size();++i)
    {
        if(src_bvalues[i] == 0.0f)
            continue;
        float min_dis = std::numeric_limits<float>::max();
        size_t min_j = 0;
        for(size_t j = i+1;j < src_bvalues.size();++j)
        {
            tipl::vector<3> v1(src_bvectors[i]),v2(src_bvectors[j]);
            v1 *= std::sqrt(src_bvalues[i]);
            v2 *= std::sqrt(src_bvalues[j]);
            float dis = std::min<float>(float((v1-v2).length()),
                                        float((v1+v2).length()));
            if(dis < min_dis)
            {
                min_dis = dis;
                min_j = j;
            }
        }
        corr_pairs.push_back(std::make_pair(i,min_j));
    }
    float self_cor = 0.0f;
    unsigned int count = 0;
    tipl::par_for(corr_pairs.size(),[&](size_t index)
    {
        size_t i1 = corr_pairs[index].first;
        size_t i2 = corr_pairs[index].second;
        std::vector<float> I1,I2;
        I1.reserve(voxel.dim.size());
        I2.reserve(voxel.dim.size());
        for(size_t i = 0;i < voxel.dim.size();++i)
            if(voxel.mask[i])
            {
                I1.push_back(src_dwi_data[i1][i]);
                I2.push_back(src_dwi_data[i2][i]);
            }
        self_cor += float(tipl::correlation(I1.begin(),I1.end(),I2.begin()));
        ++count;
    });
    self_cor/= float(count);
    return self_cor;
}
bool is_human_size(tipl::geometry<3> dim,tipl::vector<3> vs);
bool ImageModel::is_human_data(void) const
{
    return is_human_size(voxel.dim,voxel.vs);
}

bool ImageModel::command(std::string cmd,std::string param)
{
    if(cmd == "[Step T2a][Erosion]")
    {
        tipl::morphology::erosion(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Dilation]")
    {
        tipl::morphology::dilation(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Defragment]")
    {
        tipl::morphology::defragment(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Smoothing]")
    {
        tipl::morphology::smoothing(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Negate]")
    {
        tipl::morphology::negate(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Threshold]")
    {
        int threshold;
        if(param.empty())
        {
            bool ok = true;
            threshold = (voxel.dim[2] < 200) ?
                    QInputDialog::getInt(nullptr,"DSI Studio","Please assign the threshold",
                                                         int(tipl::segmentation::otsu_threshold(dwi)),
                                                         int(*std::min_element(dwi.begin(),dwi.end())),
                                                         int(*std::max_element(dwi.begin(),dwi.end()))+1,1,&ok)
                    :QInputDialog::getInt(nullptr,"DSI Studio","Please assign the threshold");
            if (!ok)
                return true;
        }
        else
            threshold = std::stoi(param);
        tipl::threshold(dwi,voxel.mask,uint8_t(threshold));
        voxel.steps += cmd + "=" + std::to_string(threshold) + "\n";
        return true;
    }
    if(cmd == "[Step T2a][Remove Background]")
    {
        for(size_t index = 0;index < voxel.mask.size();++index)
            if(voxel.mask[index] == 0)
            {
                dwi[index] = 0;
                dwi_sum[index] = 0;
            }
        for(size_t index = 0;index < src_dwi_data.size();++index)
        {
            unsigned short* buf = const_cast<unsigned short*>(src_dwi_data[index]);
            for(size_t i = 0;i < voxel.mask.size();++i)
                if(voxel.mask[i] == 0)
                    buf[i] = 0;
        }
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Trim]")
    {
        trim();
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Image flip x]")
    {
        flip_dwi(0);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Image flip y]")
    {
        flip_dwi(1);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Image flip z]")
    {
        flip_dwi(2);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Image swap xy]")
    {
        flip_dwi(3);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Image swap yz]")
    {
        flip_dwi(4);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Image swap xz]")
    {
        flip_dwi(5);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Resample]")
    {
        resample(std::stof(param));
        voxel.steps += cmd+"="+param+"\n";
        voxel.report += std::string(" The diffusion weighted images were resampled at ")+
                        param+std::string(" mm isotropic.");
        return true;
    }
    if(cmd == "[Step T2][Edit][Rotate to MNI]")
    {
        begin_prog("rotating");
        rotate_to_mni(1.0f);
        check_prog(0,0);
        voxel.steps += cmd+"\n";
        voxel.report += std::string(" The diffusion weighted images were rotated to the MNI space at 1mm.");
        return true;
    }
    if(cmd == "[Step T2][Edit][Rotate to MNI2]")
    {
        begin_prog("rotating");
        rotate_to_mni(2.0f);
        check_prog(0,0);
        voxel.steps += cmd+"\n";
        voxel.report += std::string(" The diffusion weighted images were rotated to the MNI space at 2mm.");
        return true;
    }
    if(cmd == "[Step T2][B-table][flip bx]")
    {
        for(size_t i = 0;i < src_bvectors.size();++i)
            src_bvectors[i][0] = -src_bvectors[i][0];
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][B-table][flip by]")
    {
        for(size_t i = 0;i < src_bvectors.size();++i)
            src_bvectors[i][1] = -src_bvectors[i][1];
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][B-table][flip bz]")
    {
        for(size_t i = 0;i < src_bvectors.size();++i)
            src_bvectors[i][2] = -src_bvectors[i][2];
        voxel.steps += cmd+"\n";
        return true;
    }
    std::cout << "Unknown command:" << cmd << std::endl;
    return false;
}
void ImageModel::flip_b_table(unsigned char dim)
{
    for(unsigned int index = 0;index < src_bvectors.size();++index)
        src_bvectors[index][dim] = -src_bvectors[index][dim];
    if(!voxel.grad_dev.empty())
    {
        // <Flip*Gra_dev*b_table,ODF>
        // = <(Flip*Gra_dev*inv(Flip))*Flip*b_table,ODF>
        unsigned char nindex[3][4] = {{1,2,3,6},{1,3,5,7},{2,5,6,7}};
        for(unsigned int index = 0;index < 4;++index)
        {
            // 1  0  0         1  0  0
            //[0 -1  0] *Grad*[0 -1  0]
            // 0  0  1         0  0  1
            unsigned char pos = nindex[dim][index];
            for(unsigned int i = 0;i < voxel.dim.size();++i)
                voxel.grad_dev[pos][i] = -voxel.grad_dev[pos][i];
        }
    }
}
// 0:xy 1:yz 2: xz
void ImageModel::swap_b_table(unsigned char dim)
{
    std::swap(voxel.vs[dim],voxel.vs[(dim+1)%3]);
    for (unsigned int index = 0;index < src_bvectors.size();++index)
        std::swap(src_bvectors[index][dim],src_bvectors[index][(dim+1)%3]);
    if(!voxel.grad_dev.empty())
    {
        unsigned char swap1[3][6] = {{0,3,6,0,1,2},{1,4,7,3,4,5},{0,3,6,0,1,2}};
        unsigned char swap2[3][6] = {{1,4,7,3,4,5},{2,5,8,6,7,8},{2,5,8,6,7,8}};
        for(unsigned int index = 0;index < 6;++index)
        {
            unsigned char s1 = swap1[dim][index];
            unsigned char s2 = swap2[dim][index];
            for(unsigned int i = 0;i < voxel.dim.size();++i)
                std::swap(voxel.grad_dev[s1][i],voxel.grad_dev[s2][i]);
        }
    }
}

// 0: x  1: y  2: z
// 3: xy 4: yz 5: xz
void ImageModel::flip_dwi(unsigned char type)
{
    if(type < 3)
        flip_b_table(type);
    else
        swap_b_table(type-3);
    tipl::flip(dwi_sum,type);
    tipl::flip(dwi,type);
    tipl::flip(voxel.mask,type);
    for(unsigned int i = 0;i < voxel.grad_dev.size();++i)
    {
        auto I = tipl::make_image(const_cast<float*>(&*(voxel.grad_dev[i].begin())),voxel.dim);
        tipl::flip(I,type);
    }
    begin_prog("Processing");
    tipl::par_for2(src_dwi_data.size(),[&](unsigned int index,unsigned id)
    {
        if(!id)
            check_prog(index,src_dwi_data.size());
        auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[index]),voxel.dim);
        tipl::flip(I,type);
    });
    check_prog(0,0);
    voxel.dim = dwi_sum.geometry();
    voxel.dwi_data.clear();
}
// used in eddy correction for each dwi
void ImageModel::rotate_one_dwi(unsigned int dwi_index,const tipl::transformation_matrix<double>& affine)
{
    tipl::image<float,3> tmp(voxel.dim);
    auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[dwi_index]),voxel.dim);
    tipl::resample(I,tmp,affine,tipl::cubic);
    tipl::lower_threshold(tmp,0);
    std::copy(tmp.begin(),tmp.end(),I.begin());
    // rotate b-table
    tipl::matrix<3,3,float> iT = tipl::inverse(affine.get());
    tipl::vector<3> v;
    tipl::vector_rotation(src_bvectors[dwi_index].begin(),v.begin(),iT,tipl::vdim<3>());
    v.normalize();
    src_bvectors[dwi_index] = v;
}

void ImageModel::rotate(const tipl::geometry<3>& new_geo,
                        const tipl::transformation_matrix<double>& affine,
                        const tipl::image<tipl::vector<3>,3>& cdm_dis,
                        const tipl::image<float,3>& super_reso_ref)
{
    std::vector<tipl::image<unsigned short,3> > dwi(src_dwi_data.size());
    begin_prog("rotating");
    tipl::par_for2(src_dwi_data.size(),[&](unsigned int index,unsigned int id)
    {
        if(!id)
            check_prog(index,src_dwi_data.size());
        dwi[index].resize(new_geo);
        auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[index]),voxel.dim);
        if(!super_reso_ref.empty())
            tipl::resample_with_ref(I,super_reso_ref,dwi[index],affine);
        else
        {
            if(cdm_dis.empty())
                tipl::resample(I,dwi[index],affine,tipl::cubic);
            else
                tipl::resample_dis(I,dwi[index],affine,cdm_dis,tipl::cubic);
        }
        src_dwi_data[index] = &(dwi[index][0]);
    });
    check_prog(0,0);

    tipl::image<unsigned char,3> new_mask(new_geo);
    tipl::resample(voxel.mask,new_mask,affine,tipl::linear);
    tipl::morphology::smoothing(new_mask);


    dwi.swap(new_dwi);
    // rotate b-table
    if(has_image_rotation)
    {
        tipl::matrix<3,3,float> T = tipl::inverse(affine.get());
        src_bvectors_rotate *= T;
    }
    else
        src_bvectors_rotate = tipl::inverse(affine.get());
    has_image_rotation = true;


    if(!voxel.grad_dev.empty())
    {
        // <R*Gra_dev*b_table,ODF>
        // = <(R*Gra_dev*inv(R))*R*b_table,ODF>
        float det = std::abs(src_bvectors_rotate.det());
        begin_prog("rotating grad_dev");
        for(unsigned int index = 0;check_prog(index,voxel.dim.size());++index)
        {
            tipl::matrix<3,3,float> grad_dev,G_invR;
            for(unsigned int i = 0; i < 9; ++i)
                grad_dev[i] = voxel.grad_dev[i][index];
            G_invR = grad_dev*affine.get();
            grad_dev = src_bvectors_rotate*G_invR;
            for(unsigned int i = 0; i < 9; ++i)
                voxel.grad_dev[i][index] = grad_dev[i]/det;
        }
        std::vector<tipl::image<float,3> > new_gra_dev(voxel.grad_dev.size());
        begin_prog("rotating grad_dev volume");
        for (unsigned int index = 0;check_prog(index,new_gra_dev.size());++index)
        {
            new_gra_dev[index].resize(new_geo);
            tipl::resample(voxel.grad_dev[index],new_gra_dev[index],affine,tipl::cubic);
            voxel.grad_dev[index] = tipl::make_image(const_cast<float*>(&(new_gra_dev[index][0])),voxel.dim);
        }
        new_gra_dev.swap(voxel.new_grad_dev);
    }
    voxel.dim = new_geo;
    voxel.dwi_data.clear();
    calculate_dwi_sum(false);
    voxel.mask.swap(new_mask);
}
void ImageModel::resample(float nv)
{
    tipl::vector<3,float> new_vs(nv,nv,nv);
    tipl::geometry<3> new_geo(int(std::ceil(float(voxel.dim.width())*voxel.vs[0]/new_vs[0])),
                              int(std::ceil(float(voxel.dim.height())*voxel.vs[1]/new_vs[1])),
                              int(std::ceil(float(voxel.dim.depth())*voxel.vs[2]/new_vs[2])));
    tipl::image<float,3> J(new_geo);
    if(J.empty())
        return;
    tipl::transformation_matrix<double> T;
    T.sr[0] = double(new_vs[0]/voxel.vs[0]);
    T.sr[4] = double(new_vs[1]/voxel.vs[1]);
    T.sr[8] = double(new_vs[2]/voxel.vs[2]);
    rotate(new_geo,T);
    voxel.vs = new_vs;
}
extern std::string fib_template_file_name_2mm;
extern std::vector<std::string> iso_template_list;
bool ImageModel::rotate_to_mni(float resolution)
{
    tipl::image<float,3> I;
    tipl::vector<3> vs;

    if(resolution == 2.0f)
    {
        std::string file_name = fib_template_file_name_2mm;
        gz_mat_read read;
        if(!read.load_from_file(file_name.c_str()))
        {
            error_msg = "Failed to load/find fib template.";
            return false;
        }
        if(!read.save_to_image(I,"iso"))
        {
            error_msg = "Failed to read image from fib template.";
            return false;
        }
        if(!read.get_voxel_size(vs))
        {
            error_msg = "Failed to get voxel size from fib template.";
            return false;
        }
    }
    else
    if(resolution == 1.0f)
    {
        gz_nifti nii;
        if(!nii.load_from_file(iso_template_list.front()))
        {
            error_msg = "Failed to load/find MNI template.";
            return false;
        }
        nii.toLPS(I);
        nii.get_voxel_size(vs);
    }
    else
        return false;
    tipl::transformation_matrix<double> arg;
    bool terminated = false;
    begin_prog("registering to the MNI space");
    check_prog(0,1);
    tipl::reg::two_way_linear_mr(I,vs,dwi_sum,voxel.vs,
                    arg,tipl::reg::rigid_body,tipl::reg::mutual_information(),terminated);
    begin_prog("rotating to the MNI space");
    rotate(I.geometry(),arg);
    voxel.vs = vs;
    voxel.report += " The diffusion MRI data were resampled to ";
    voxel.report += std::to_string(int(vs[0]));
    voxel.report += " mm isotropic reoslution.";
    check_prog(1,1);
    return true;
}

void ImageModel::trim(void)
{
    tipl::geometry<3> range_min,range_max;
    tipl::bounding_box(voxel.mask,range_min,range_max,0);
    begin_prog("Removing background region");
    tipl::par_for2(src_dwi_data.size(),[&](unsigned int index,unsigned int id)
    {
        if(!id)
            check_prog(index,src_dwi_data.size());
        auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[index]),voxel.dim);
        tipl::image<unsigned short,3> I0 = I;
        tipl::crop(I0,range_min,range_max);
        std::fill(I.begin(),I.end(),0);
        std::copy(I0.begin(),I0.end(),I.begin());
    });
    check_prog(0,0);
    tipl::crop(voxel.mask,range_min,range_max);
    tipl::crop(dwi_sum,range_min,range_max);
    tipl::crop(dwi,range_min,range_max);
    voxel.dim = voxel.mask.geometry();
    voxel.dwi_data.clear();
}

float interpo_pos(float v1,float v2,float u1,float u2)
{
    float w = (u2-u1-v2+v1);
    return std::max<float>(0.0,std::min<float>(1.0,w == 0.0f? 0:(v1-u1)/w));
}

template<typename image_type>
void get_distortion_map(const image_type& v1,
                        const image_type& v2,
                        tipl::image<float,3>& dis_map)
{
    int h = v1.height(),w = v1.width();
    dis_map.resize(v1.geometry());
    tipl::par_for(v1.depth()*h,[&](int z)
    {
        int base_pos = z*w;
        std::vector<float> line1(v1.begin()+base_pos,v1.begin()+base_pos+w),
                           line2(v2.begin()+base_pos,v2.begin()+base_pos+w);
        float sum1 = std::accumulate(line1.begin(),line1.end(),0.0f);
        float sum2 = std::accumulate(line1.begin(),line1.end(),0.0f);
        if(sum1 == 0.0f || sum2 == 0.0f)
            return;
        bool swap12 = sum2 > sum1;
        if(swap12)
            std::swap(line1,line2);
        // now line1 > line2
        std::vector<float> dif(line1);
        tipl::minus(dif,line2);
        tipl::lower_threshold(dif,0.0f);
        float sum_dif = std::accumulate(dif.begin(),dif.end(),0.0f);
        if(sum_dif == 0.0f)
            return;
        tipl::multiply_constant(dif,std::abs(sum1-sum2)/sum_dif);
        tipl::minus(line1,dif);
        tipl::lower_threshold(line1,0.0f);
        if(swap12)
            std::swap(line1,line2);

        std::vector<float> cdf_x1,cdf_x2;//,cdf(h);
        cdf_x1.resize(size_t(w));
        cdf_x2.resize(size_t(w));
        tipl::pdf2cdf(line1.begin(),line1.end(),&cdf_x1[0]);
        tipl::pdf2cdf(line2.begin(),line2.end(),&cdf_x2[0]);
        tipl::multiply_constant(cdf_x2,(cdf_x1.back()+cdf_x2.back())*0.5f/cdf_x2.back());
        tipl::add_constant(cdf_x2,(cdf_x1.back()-cdf_x2.back())*0.5f);
        for(int x = 0,pos = base_pos;x < w;++x,++pos)
        {
            if(cdf_x1[size_t(x)] == cdf_x2[size_t(x)])
            {
                //cdf[y] = cdf_y1[y];
                continue;
            }
            int d = 1,x1,x2;
            float v1,v2,u1,u2;
            v1 = 0.0f;
            u1 = 0.0f;
            v2 = cdf_x1[size_t(x)];
            u2 = cdf_x2[size_t(x)];
            bool positive_d = true;
            if(cdf_x1[size_t(x)] > cdf_x2[size_t(x)])
            {
                for(;d < w;++d)
                {
                    x1 = x-d;
                    x2 = x+d;
                    v1 = v2;
                    u1 = u2;
                    v2 = (x1 >=0 ? cdf_x1[size_t(x1)]:0);
                    u2 = (x2 < int(cdf_x2.size()) ? cdf_x2[size_t(x2)]:cdf_x2.back());
                    if(v2 <= u2)
                        break;
                }
            }
            else
            {
                for(;d < h;++d)
                {
                    x2 = x-d;
                    x1 = x+d;
                    v1 = v2;
                    u1 = u2;
                    v2 = (x1 < int(cdf_x1.size()) ? cdf_x1[size_t(x1)]:cdf_x1.back());
                    u2 = (x2 >= 0 ? cdf_x2[size_t(x2)]:0);
                    if(v2 >= u2)
                        break;
                }
                positive_d = false;
            }
            //cdf[y] = v1+(v2-v1)*interpo_pos(v1,v2,u1,u2);
            dis_map[pos] = interpo_pos(v1,v2,u1,u2)+d-1;
            if(!positive_d)
                dis_map[pos] = -dis_map[pos];
        }
    }
    );
}

template<typename image_type>
void apply_distortion_map2(const image_type& v1,
                          const tipl::image<float,3>& dis_map,
                          image_type& v1_out,bool positive)
{
    int w = v1.width();
    v1_out.resize(v1.geometry());
    tipl::par_for(v1.depth()*v1.height(),[&](int z)
    {
        std::vector<float> cdf_x1,cdf;
        cdf_x1.resize(size_t(w));
        cdf.resize(size_t(w));
        int base_pos = z*w;
        {
            tipl::pdf2cdf(v1.begin()+base_pos,v1.begin()+base_pos+w,&cdf_x1[0]);
            auto I1 = tipl::make_image(&cdf_x1[0],tipl::geometry<1>(w));
            for(int x = 0,pos = base_pos;x < w;++x,++pos)
                cdf[size_t(x)] = tipl::estimate(I1,positive ? x+dis_map[pos] : x-dis_map[pos]);
            for(int x = 0,pos = base_pos;x < w;++x,++pos)
                v1_out[pos] = (x? std::max<float>(0.0f,cdf[size_t(x)] - cdf[size_t(x-1)]):0);
        }
    }
    );
}




void ImageModel::distortion_correction(const ImageModel& rhs)
{
    tipl::image<float,3> v1,v2,vv1,vv2;
    v1 = tipl::make_image(src_dwi_data[0],voxel.dim);
    v2 = tipl::make_image(rhs.src_dwi_data[0],voxel.dim);


    bool swap_xy = false;
    {
        tipl::image<float,2> px1,px2,py1,py2;
        tipl::project_x(v1,px1);
        tipl::project_x(v2,px2);
        tipl::project_y(v1,py1);
        tipl::project_y(v2,py2);
        float cx = float(tipl::correlation(px1.begin(),px1.end(),px2.begin()));
        float cy = float(tipl::correlation(py1.begin(),py1.end(),py2.begin()));

        if(cx < cy)
        {
            tipl::swap_xy(v1);
            tipl::swap_xy(v2);
            swap_xy = true;
        }
    }

    tipl::image<float,3> dis_map(v1.geometry()),df,gx(v1.geometry()),v1_gx(v1.geometry()),v2_gx(v2.geometry());


    tipl::filter::gaussian(v1);
    tipl::filter::gaussian(v2);

    tipl::gradient(v1,v1_gx,1,0);
    tipl::gradient(v2,v2_gx,1,0);

    get_distortion_map(v2,v1,dis_map);
    tipl::filter::gaussian(dis_map);
    tipl::filter::gaussian(dis_map);


    for(int iter = 0;iter < 120;++iter)
    {
        apply_distortion_map2(v1,dis_map,vv1,true);
        apply_distortion_map2(v2,dis_map,vv2,false);
        df = vv1;
        df -= vv2;
        vv1 += vv2;
        df *= vv1;
        tipl::gradient(df,gx,1,0);
        gx += v1_gx;
        gx -= v2_gx;
        tipl::normalize_abs(gx,0.5f);
        tipl::filter::gaussian(gx);
        tipl::filter::gaussian(gx);
        tipl::filter::gaussian(gx);
        dis_map += gx;
    }

    std::vector<tipl::image<unsigned short,3> > dwi(src_dwi_data.size());
    for(size_t i = 0;i < src_dwi_data.size();++i)
    {
        v1 = tipl::make_image(src_dwi_data[i],voxel.dim);
        if(swap_xy)
        {
            tipl::swap_xy(v1);
        }
        apply_distortion_map2(v1,dis_map,vv1,true);
        dwi[i] = vv1;
        if(swap_xy)
            tipl::swap_xy(dwi[i]);
    }

    new_dwi.swap(dwi);
    for(size_t i = 0;i < new_dwi.size();++i)
        src_dwi_data[i] = &(new_dwi[i][0]);

    calculate_dwi_sum(false);
    voxel.report += " The phase distortion was correlated using data from an opposiate phase encoding direction.";

}

void calculate_shell(const std::vector<float>& sorted_bvalues,
                     std::vector<unsigned int>& shell)
{
    for(uint32_t i = 0;i < sorted_bvalues.size();++i)
        if(sorted_bvalues[i] > 100.0f)
            {
                shell.push_back(i);
                break;
            }
    if(shell.empty())
        return;
    for(uint32_t index = shell.back()+1;index < sorted_bvalues.size();++index)
        if(std::abs(sorted_bvalues[index]-sorted_bvalues[index-1]) > 100.0f)
            shell.push_back(index);
}

void ImageModel::calculate_shell(void)
{
    std::vector<float> sorted_bvalues(src_bvalues);
    std::sort(sorted_bvalues.begin(),sorted_bvalues.end());
    ::calculate_shell(sorted_bvalues,shell);
}
bool ImageModel::is_dsi_half_sphere(void)
{
    if(shell.empty())
        calculate_shell();
    return is_dsi() && (!shell.empty() && shell[1] - shell[0] <= 3);
}

bool ImageModel::is_dsi(void)
{
    if(shell.empty())
        calculate_shell();
    return shell.size() > 4 && (!shell.empty() && shell[1] - shell[0] <= 6);
}
bool ImageModel::need_scheme_balance(void)
{
    if(shell.empty())
        calculate_shell();
    if(is_dsi() || shell.size() > 6)
        return false;
    for(size_t i = 0;i < shell.size();++i)
    {
        size_t from = shell[i];
        size_t to = (i + 1 == shell.size() ? src_bvalues.size():shell[i+1]);
        if(to-from < 128)
            return true;
    }
    return false;
}

bool ImageModel::is_multishell(void)
{
    if(shell.empty())
        calculate_shell();
    return (shell.size() > 1) && !is_dsi();
}


void ImageModel::get_report(std::string& report)
{
    std::vector<float> sorted_bvalues(src_bvalues);
    std::sort(sorted_bvalues.begin(),sorted_bvalues.end());
    unsigned int num_dir = 0;
    for(size_t i = 0;i < src_bvalues.size();++i)
        if(src_bvalues[i] > 50)
            ++num_dir;
    std::ostringstream out;
    if(is_dsi())
    {
        out << " A diffusion spectrum imaging scheme was used, and a total of " << num_dir
            << " diffusion sampling were acquired."
            << " The maximum b-value was " << int(std::round(sorted_bvalues.back())) << " s/mm2.";
    }
    else
    if(is_multishell())
    {
        out << " A multishell diffusion scheme was used, and the b-values were ";
        for(unsigned int index = 0;index < shell.size();++index)
        {
            if(index > 0)
            {
                if(index == shell.size()-1)
                    out << " and ";
                else
                    out << " ,";
            }
            out << int(std::round(sorted_bvalues[
                index == shell.size()-1 ? (sorted_bvalues.size()+shell.back())/2 : (shell[index+1] + shell[index])/2]));
        }
        out << " s/mm2.";

        out << " The number of diffusion sampling directions were ";
        for(unsigned int index = 0;index < shell.size()-1;++index)
            out << shell[index+1] - shell[index] << (shell.size() == 2 ? " ":", ");
        out << "and " << sorted_bvalues.size()-shell.back() << ", respectively.";
    }
    else
        if(shell.size() == 1)
        {
            if(num_dir < 100)
                out << " A DTI diffusion scheme was used, and a total of ";
            else
                out << " A HARDI scheme was used, and a total of ";
            out << num_dir
                << " diffusion sampling directions were acquired."
                << " The b-value was " << sorted_bvalues.back() << " s/mm2.";
        }

    out << " The in-plane resolution was " << voxel.vs[0] << " mm."
        << " The slice thickness was " << voxel.vs[2] << " mm.";
    report = out.str();
}
bool ImageModel::save_to_file(const char* dwi_file_name)
{
    gz_mat_write mat_writer(dwi_file_name);
    if(!mat_writer)
        return false;
    {
        uint16_t dim[3];
        dim[0] = uint16_t(voxel.dim[0]);
        dim[1] = uint16_t(voxel.dim[1]);
        dim[2] = uint16_t(voxel.dim[2]);
        mat_writer.write("dimension",dim,1,3);
        mat_writer.write("voxel_size",voxel.vs);
    }
    {
        std::vector<float> b_table;
        for (unsigned int index = 0;index < src_bvalues.size();++index)
        {
            b_table.push_back(src_bvalues[index]);
            b_table.push_back(src_bvectors[index][0]);
            b_table.push_back(src_bvectors[index][1]);
            b_table.push_back(src_bvectors[index][2]);
        }
        mat_writer.write("b_table",b_table,4);
    }
    begin_prog("Saving");
    for (unsigned int index = 0;check_prog(index,src_bvalues.size());++index)
    {
        std::ostringstream out;
        out << "image" << index;
        mat_writer.write(out.str().c_str(),src_dwi_data[index],
                         uint32_t(voxel.dim.plane_size()),uint32_t(voxel.dim.depth()));
    }
    check_prog(0,0);
    mat_writer.write("mask",voxel.mask,uint32_t(voxel.dim.plane_size()));
    return true;
}
bool ImageModel::load_from_file(const char* dwi_file_name)
{
    file_name = dwi_file_name;
    if (!mat_reader.load_from_file(dwi_file_name))
    {
        gz_nifti nii;
        if(nii.load_from_file(dwi_file_name))
        {
            for(unsigned int index = 0;index < nii.dim(4);++index)
            {
                tipl::image<float,3> data;
                if(!nii.toLPS(data,index == 0))
                    break;
                tipl::lower_threshold(data,0.0f);
                tipl::image<unsigned short,3> buf = data;
                new_dwi.push_back(std::move(buf));
                src_dwi_data.push_back(&new_dwi.back()[0]);
                src_bvalues.push_back(0.0f);
                src_bvectors.push_back(tipl::vector<3>(0.0f,0.0f,0.0f));
                if(index == 0)
                {
                    nii.get_voxel_size(voxel.vs);
                    voxel.dim = new_dwi.front().geometry();
                }
            }
            calculate_dwi_sum(true);
            return !new_dwi.empty();
        }
        error_msg = "Cannot open file";
        return false;
    }

    if (!mat_reader.read("dimension",voxel.dim))
    {
        error_msg = "Cannot find dimension matrix";
        return false;
    }
    if (!mat_reader.read("voxel_size",voxel.vs))
    {
        error_msg = "Cannot find voxel_size matrix";
        return false;
    }

    if (voxel.dim[0]*voxel.dim[1]*voxel.dim[2] <= 0)
    {
        error_msg = "Invalid dimension setting";
        return false;
    }

    unsigned int row,col;
    const float* table;
    if (!mat_reader.read("b_table",row,col,table))
    {
        error_msg = "Cannot find b_table matrix";
        return false;
    }
    src_bvalues.resize(col);
    src_bvectors.resize(col);
    for (unsigned int index = 0;index < col;++index)
    {
        src_bvalues[index] = table[0];
        src_bvectors[index][0] = table[1];
        src_bvectors[index][1] = table[2];
        src_bvectors[index][2] = table[3];
        src_bvectors[index].normalize();
        table += 4;
    }

    if(!mat_reader.read("report",voxel.report))
        get_report(voxel.report);

    src_dwi_data.resize(src_bvalues.size());
    for (size_t index = 0;index < src_bvalues.size();++index)
    {
        std::ostringstream out;
        out << "image" << index;
        mat_reader.read(out.str().c_str(),row,col,src_dwi_data[index]);
        if (!src_dwi_data[index])
        {
            error_msg = "Cannot find image matrix";
            return false;
        }
    }


    {
        const float* grad_dev = nullptr;
        if(mat_reader.read("grad_dev",row,col,grad_dev) && size_t(row)*size_t(col) == voxel.dim.size()*9)
        {
            for(unsigned int index = 0;index < 9;index++)
                voxel.grad_dev.push_back(tipl::make_image(const_cast<float*>(grad_dev+index*voxel.dim.size()),voxel.dim));
            if(std::fabs(voxel.grad_dev[0][0])+std::fabs(voxel.grad_dev[4][0])+std::fabs(voxel.grad_dev[8][0]) < 1.0f)
            {
                tipl::add_constant(voxel.grad_dev[0].begin(),voxel.grad_dev[0].end(),1.0);
                tipl::add_constant(voxel.grad_dev[4].begin(),voxel.grad_dev[4].end(),1.0);
                tipl::add_constant(voxel.grad_dev[8].begin(),voxel.grad_dev[8].end(),1.0);
            }
        }

    }

    // create mask;
    calculate_dwi_sum(true);

    const unsigned char* mask_ptr = nullptr;
    if(mat_reader.read("mask",row,col,mask_ptr))
    {
        voxel.mask.resize(voxel.dim);
        if(size_t(row)*size_t(col) == voxel.dim.size())
            std::copy(mask_ptr,mask_ptr+size_t(row)*size_t(col),voxel.mask.begin());
    }
    voxel.steps += "[Step T2][Reconstruction] open ";
    voxel.steps += QFileInfo(dwi_file_name).fileName().toStdString();
    voxel.steps += "\n";
    return true;
}

bool ImageModel::save_fib(const std::string& output_name)
{
    gz_mat_write mat_writer(output_name.c_str());
    if(!mat_writer)
    {
        error_msg = "Cannot save fib file";
        return false;
    }
    {
        uint16_t dim[3];
        dim[0] = uint16_t(voxel.dim[0]);
        dim[1] = uint16_t(voxel.dim[1]);
        dim[2] = uint16_t(voxel.dim[2]);
        mat_writer.write("dimension",dim,1,3);
        mat_writer.write("voxel_size",voxel.vs);
    }

    std::vector<float> float_data;
    std::vector<short> short_data;
    voxel.ti.save_to_buffer(float_data,short_data);
    mat_writer.write("odf_vertices",float_data,3);
    mat_writer.write("odf_faces",short_data,3);

    voxel.end(mat_writer);
    std::string final_report = voxel.report;
    final_report += voxel.recon_report.str();
    mat_writer.write("report",final_report);
    std::string final_steps = voxel.steps;
    final_steps += voxel.step_report.str();
    final_steps += "[Step T2b][Run reconstruction]\n";
    mat_writer.write("steps",final_steps);
    return true;
}
bool ImageModel::save_to_nii(const char* nifti_file_name) const
{
    gz_nifti header;
    header.set_voxel_size(voxel.vs);
    header.nif_header.pixdim[0] = 4;
    header.nif_header2.pixdim[0] = 4;

    tipl::geometry<4> nifti_dim;
    std::copy(voxel.dim.begin(),voxel.dim.end(),nifti_dim.begin());
    nifti_dim[3] = int(src_bvalues.size());
    tipl::image<unsigned short,4> buffer(nifti_dim);
    for(unsigned int index = 0;index < src_bvalues.size();++index)
    {
        std::copy(src_dwi_data[index],
                  src_dwi_data[index]+voxel.dim.size(),
                  buffer.begin() + long(index*voxel.dim.size()));
    }
    tipl::flip_xy(buffer);
    header << buffer;
    return header.save_to_file(nifti_file_name);
}
bool ImageModel::save_b0_to_nii(const char* nifti_file_name) const
{
    gz_nifti header;
    header.set_voxel_size(voxel.vs);
    tipl::image<unsigned short,3> buffer(src_dwi_data[0],voxel.dim);
    tipl::flip_xy(buffer);
    header << buffer;
    return header.save_to_file(nifti_file_name);
}

bool ImageModel::save_dwi_sum_to_nii(const char* nifti_file_name) const
{
    gz_nifti header;
    header.set_voxel_size(voxel.vs);
    tipl::image<float,3> buffer(dwi_sum);
    tipl::flip_xy(buffer);
    header << buffer;
    return header.save_to_file(nifti_file_name);
}

bool ImageModel::save_b_table(const char* file_name) const
{
    std::ofstream out(file_name);
    for(unsigned int index = 0;index < src_bvalues.size();++index)
    {
        out << src_bvalues[index] << " "
            << src_bvectors[index][0] << " "
            << src_bvectors[index][1] << " "
            << src_bvectors[index][2] << std::endl;
    }
    return out.good();
}
bool ImageModel::save_bval(const char* file_name) const
{
    std::ofstream out(file_name);
    for(unsigned int index = 0;index < src_bvalues.size();++index)
    {
        if(index)
            out << " ";
        out << src_bvalues[index];
    }
    return out.good();
}
bool ImageModel::save_bvec(const char* file_name) const
{
    std::ofstream out(file_name);
    for(unsigned int index = 0;index < src_bvalues.size();++index)
    {
        out << src_bvectors[index][0] << " "
            << -src_bvectors[index][1] << " "
            << src_bvectors[index][2] << "\n";
    }
    return out.good();
}
bool ImageModel::compare_src(const char* file_name)
{
    std::shared_ptr<ImageModel> bl(new ImageModel);
    begin_prog("reading");
    if(!bl->load_from_file(file_name))
    {
        error_msg = bl->error_msg;
        return false;
    }
    study_src = bl;

    voxel.study_name = QFileInfo(file_name).baseName().toStdString();
    voxel.compare_voxel = &(study_src->voxel);

    begin_prog("Registration between longitudinal scans");
    {
        tipl::transformation_matrix<double> arg;
        bool terminated = false;
        check_prog(0,1);
        tipl::reg::two_way_linear_mr(dwi_sum,voxel.vs,
                                     study_src->dwi_sum,study_src->voxel.vs,
                                        arg,tipl::reg::rigid_body,tipl::reg::correlation(),terminated);
        // nonlinear part
        tipl::image<tipl::vector<3>,3> cdm_dis;
        if(voxel.dt_deform)
        {
            tipl::image<float,3> new_dwi_sum(dwi_sum.geometry());
            tipl::resample(study_src->dwi_sum,new_dwi_sum,arg,tipl::cubic);
            tipl::match_signal(dwi_sum,new_dwi_sum);
            bool terminated = false;
            begin_prog("Nonlinear registration between longitudinal scans");
            tipl::reg::cdm(dwi_sum,new_dwi_sum,cdm_dis,terminated);
            check_prog(0,1);

            /*
            if(1) // debug
            {
                tipl::image<float,3> result(dwi_sum.geometry());
                tipl::resample_dis(study_src->dwi_sum,result,arg,cdm_dis,tipl::cubic);
                gz_nifti o1,o2,o3;
                o1.set_voxel_size(voxel.vs);
                o1.load_from_image(dwi_sum);
                o1.save_to_file("d:/1.nii.gz");

                o2.set_voxel_size(study_src->voxel.vs);
                o2.load_from_image(new_dwi_sum);
                o2.save_to_file("d:/2.nii.gz");

                o3.set_voxel_size(study_src->voxel.vs);
                o3.load_from_image(result);
                o3.save_to_file("d:/3.nii.gz");
            }*/
        }
        study_src->rotate(dwi_sum.geometry(),arg,cdm_dis);
        study_src->voxel.vs = voxel.vs;
        study_src->voxel.mask = voxel.mask;
        check_prog(1,1);
    }


    // correct b_table first
    if(voxel.check_btable)
        study_src->check_b_table();


    for(size_t i = 0;i < voxel.mask.size();++i)
        if(study_src->src_dwi_data[0][i] == 0)
            voxel.mask[i] = 0;



    // Signal match on b0 to allow for quantitative MRI in DDI
    {
        std::vector<double> r;
        for(size_t i = 0;i < voxel.mask.size();++i)
            if(voxel.mask[i])
            {
                if(study_src->src_dwi_data[0][i] && src_dwi_data[0][i])
                    r.push_back(double(src_dwi_data[0][i])/double(study_src->src_dwi_data[0][i]));
            }

        double median_r = tipl::median(r.begin(),r.end());
        std::cout << "median_r=" << median_r << std::endl;
        tipl::par_for(study_src->new_dwi.size(),[&](size_t i)
        {
            tipl::multiply_constant(study_src->new_dwi[i].begin(),study_src->new_dwi[i].end(),median_r);
        });
        study_src->calculate_dwi_sum(false);
    }
    pre_dti();
    study_src->pre_dti();
    voxel.R2 = float(tipl::correlation(dwi_sum.begin(),dwi_sum.end(),study_src->dwi_sum.begin()));
    return true;
}
