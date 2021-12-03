#include <QFileInfo>
#include <QDir>
#include <QInputDialog>
#include <QDateTime>
#include <QImage>
#include <QProcess>
#include "image_model.hpp"
#include "odf_process.hpp"
#include "dti_process.hpp"
#include "fib_data.hpp"
#include "dwi_header.hpp"
#include "tracking/region/Regions.h"

#include "mac_filesystem.hpp"

void ImageModel::draw_mask(tipl::color_image& buffer,int position)
{
    if (!dwi.size())
        return;
    auto slice = dwi.slice_at(uint32_t(position));
    auto mask_slice = voxel.mask.slice_at(uint32_t(position));

    tipl::color_image buffer2(tipl::shape<2>(uint32_t(dwi.width())*2,uint32_t(dwi.height())));
    tipl::draw(slice,buffer2,tipl::vector<2,int>());
    buffer.resize(slice.shape());
    for (size_t index = 0; index < buffer.size(); ++index)
    {
        unsigned char value = slice[index];
        if (mask_slice[index])
            buffer[index] = tipl::rgb(uint8_t(255), value, value);
        else
            buffer[index] = tipl::rgb(value, value, value);
    }
    tipl::draw(buffer,buffer2,tipl::vector<2,int>(dwi.width(),0));
    buffer2.swap(buffer);
}

void ImageModel::calculate_dwi_sum(bool update_mask)
{
    if(src_dwi_data.empty())
        return;
    {
        tipl::image<3> dwi_sum(voxel.dim);
        tipl::par_for(src_dwi_data.size(),[&](unsigned int index)
        {
            if(index > 0 && src_bvalues[index] == 0.0f)
                return;
            dwi_sum += tipl::make_image(src_dwi_data[index],voxel.dim);
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
        tipl::upper_threshold(dwi_sum,max_value);
        dwi.resize(voxel.dim);
        tipl::normalize(dwi_sum,dwi);
    }

    if(update_mask)
    {
        tipl::threshold(dwi,voxel.mask,50,1,0);
        if(dwi.depth() < 200)
        {
            tipl::par_for(voxel.mask.depth(),[&](int i)
            {
                tipl::pointer_image<2,unsigned char> I(&voxel.mask[0]+size_t(i)*voxel.mask.plane_size(),
                        tipl::shape<2>(uint32_t(voxel.mask.width()),uint32_t(voxel.mask.height())));
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
}

extern std::string fib_template_file_name_2mm;
std::string ImageModel::check_b_table(void)
{
    // reconstruct DTI using original data and b-table
    {
        tipl::image<3,unsigned char> mask(voxel.mask.shape());
        std::fill(mask.begin(),mask.end(),1);
        mask.swap(voxel.mask);

        auto other_output = voxel.other_output;
        voxel.other_output = std::string();

        reconstruct<check_btable_process>("checking b-table");

        voxel.other_output = other_output;

        mask.swap(voxel.mask);

    }

    std::vector<tipl::image<3> > fib_fa(1);
    std::vector<std::vector<tipl::vector<3> > > fib_dir(1);
    fib_fa[0].swap(voxel.fib_fa);
    fib_dir[0].swap(voxel.fib_dir);

    const unsigned char order[24][6] = {
                            {0,1,2,0,0,0},{0,1,2,1,0,0},{0,1,2,0,1,0},{0,1,2,0,0,1},
                            {0,2,1,0,0,0},{0,2,1,1,0,0},{0,2,1,0,1,0},{0,2,1,0,0,1},
                            {1,0,2,0,0,0},{1,0,2,1,0,0},{1,0,2,0,1,0},{1,0,2,0,0,1},
                            {1,2,0,0,0,0},{1,2,0,1,0,0},{1,2,0,0,1,0},{1,2,0,0,0,1},
                            {2,1,0,0,0,0},{2,1,0,1,0,0},{2,1,0,0,1,0},{2,1,0,0,0,1},
                            {2,0,1,0,0,0},{2,0,1,1,0,0},{2,0,1,0,1,0},{2,0,1,0,0,1}};
    const char txt[24][7] = {".012",".012fx",".012fy",".012fz",
                             ".021",".021fx",".021fy",".021fz",
                             ".102",".102fx",".102fy",".102fz",
                             ".120",".120fx",".120fy",".120fz",
                             ".210",".210fx",".210fy",".210fz",
                             ".201",".201fx",".201fy",".201fz"};

    std::shared_ptr<fib_data> template_fib;
    tipl::transformation_matrix<float> T;
    tipl::matrix<3,3,float> r;

    if(is_human_data())
    {
        template_fib = std::make_shared<fib_data>();
        if(!template_fib->load_from_file(fib_template_file_name_2mm.c_str()))
            template_fib.reset();
        else
        {
            tipl::image<3> iso_fa;
            template_fib->get_iso_fa(iso_fa);
            tipl::normalize(iso_fa,255.0f);
            tipl::image<3,unsigned char> I;
            I = iso_fa;

            const float bound[8] = {1.0f,-1.0f,0.02f,-0.02f,1.2f,0.9f,0.1f,-0.1f};
            double precision[3] = {0.1,0.01,0.001};
            bool terminated = false;
            tipl::affine_transform<float> arg;
            for(int i = 0;i < 3; ++ i)
                tipl::reg::linear_mr(I,template_fib->vs,dwi,voxel.vs,
                        arg,tipl::reg::affine,tipl::reg::mutual_information(),terminated,precision[i],bound);
            tipl::rotation_matrix(arg.rotation,r.begin(),tipl::vdim<3>());
            r.inv();
            T = tipl::transformation_matrix<float>(arg,template_fib->dim,template_fib->vs,voxel.dim,voxel.vs);
        }
    }
    if(template_fib.get())
    {
        voxel.recon_report <<
        " The accuracy of b-table orientation was examined by comparing fiber orientations with those of a population-averaged template (Yeh et al. Neuroimage, 2018).";
    }
    else
    {
        voxel.recon_report <<
        " The b-table was checked by an automatic quality control routine to ensure its accuracy (Schilling et al. MRI, 2019).";
    }

    float result[24] = {0};
    float otsu = tipl::segmentation::otsu_threshold(fib_fa[0])*0.6f;
    auto subject_geo = fib_fa[0].shape();
    for(int i = 0;i < 24;++i)// 0 is the current score
    {
        auto new_dir(fib_dir);
        if(i)
            flip_fib_dir(new_dir[0],order[i]);

        if(template_fib.get()) // comparing with hcp 2mm template
        {
            double sum_cos = 0.0;
            size_t ncount = 0;
            auto template_geo = template_fib->dim;
            const float* ptr = nullptr;
            for(tipl::pixel_index<3> index(template_geo);index < template_geo.size();++index)
            {
                if(template_fib->dir.fa[0][index.index()] < 0.2f || !(ptr = template_fib->dir.get_fib(index.index(),0)))
                    continue;
                tipl::vector<3> pos(index);
                T(pos);
                pos.round();
                if(subject_geo.is_valid(pos))
                {
                    auto sub_dir = new_dir[0][tipl::pixel_index<3>(pos.begin(),subject_geo).index()];
                    sub_dir.rotate(r);
                    sum_cos += std::abs(double(sub_dir*tipl::vector<3>(ptr)));
                    ++ncount;
                }
            }
            result[i] = float(sum_cos/double(ncount));
        }
        else
        // for animal studies, use fiber coherence index
            result[i] = evaluate_fib(subject_geo,otsu,fib_fa,[&](uint32_t pos,uint8_t fib){return new_dir[fib][pos];}).first;
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

    if(result[best] > result[0])
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
    tipl::image<2,float> cor_values(
                tipl::shape<2>(int(voxel.dwi_data.size()),voxel.dim.depth()));

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
    //tipl::image<3> bad_I(tipl::shape<3>(voxel.dim[0],voxel.dim[1],bad_i.size()));
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
bool is_human_size(tipl::shape<3> dim,tipl::vector<3> vs);
bool ImageModel::is_human_data(void) const
{
    return voxel.dim[2] > 1 && is_human_size(voxel.dim,voxel.vs);
}

bool ImageModel::run_steps(std::string steps)
{
    std::istringstream in(steps);
    std::string step;
    std::getline(in,step); // ignore the first step [Step T2][Reconstruction]
    while(std::getline(in,step))
    {
        size_t pos = step.find('=');
        std::string cmd,param;
        if(pos == std::string::npos)
        {
            cmd = step.substr(0,step.find_last_of(']')+1);
        }
        else
        {
            cmd = step.substr(0,pos);
            param = step.substr(pos+1,step.size()-pos-1);
        }
        if(!command(cmd,param))
        {
            error_msg = "processing failed at ";
            error_msg += step;
            return false;
        }
    }
    return true;
}
bool ImageModel::command(std::string cmd,std::string param)
{
    std::cout << cmd << (param.empty() ? "":" param:") << param << std::endl;
    if(cmd == "[Step T2a][Open]")
    {
        if(!std::filesystem::exists(param))
        {
            error_msg = param;
            error_msg += " does not exist";
            return false;
        }
        ROIRegion region(voxel.dim,voxel.vs);
        region.LoadFromFile(param.c_str());
        region.SaveToBuffer(voxel.mask,1.0f);
        voxel.steps += std::string("[Step T2a][Open]=") + param + "\n";
        return true;
    }
    if(cmd == "[Step T2a][Erosion]")
    {
        if(voxel.mask.depth() == 1)
        {
            auto slice = voxel.mask.slice_at(0);
            tipl::morphology::erosion(slice);
        }
        else
            tipl::morphology::erosion(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Dilation]")
    {
        if(voxel.mask.depth() == 1)
        {
            auto slice = voxel.mask.slice_at(0);
            tipl::morphology::dilation(slice);
        }
        else
            tipl::morphology::dilation(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Defragment]")
    {
        if(voxel.mask.depth() == 1)
        {
            auto slice = voxel.mask.slice_at(0);
            tipl::morphology::defragment(slice);
        }
        else
            tipl::morphology::defragment(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Smoothing]")
    {
        if(voxel.mask.depth() == 1)
        {
            auto slice = voxel.mask.slice_at(0);
            tipl::morphology::smoothing(slice);
        }
        else
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
                dwi[index] = 0;
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
    if(cmd == "[Step T2b(2)][Compare SRC]")
    {
        if(param.empty()) // try to find the best match in the same directory
        {
            auto file_list = QFileInfo(file_name.c_str()).dir().entryList(QStringList("*.src.gz"),QDir::Files);
            int64_t min_dif = std::numeric_limits<int64_t>::max();
            for(int i = 0;i < file_list.size();++i)
            {
                QString cur_file_name = QFileInfo(file_name.c_str()).absolutePath() + "/" + file_list[i];
                if(file_list[i] == QFileInfo(file_name.c_str()).fileName())
                    continue;
                std::string name1 = QFileInfo(file_name.c_str()).baseName().toStdString();
                std::string name2 = QFileInfo(file_list[i]).baseName().toStdString();
                int64_t cur_dif = 0;
                for(int j = int(std::min(name1.length(),name2.length()))-1,k = 0;j >= 0 && k < 60; --j,k += 3)
                {
                    int dir = std::abs(int(name1[uint32_t(j)])-int(name2[uint32_t(j)]));
                    if(k)
                        dir <<= k;
                    cur_dif += dir;
                }
                if(cur_dif < min_dif)
                {
                    min_dif = cur_dif;
                    voxel.study_src_file_path = cur_file_name.toStdString();
                }

            }
        }
        else
            voxel.study_src_file_path = param;
        voxel.steps += cmd+" open " + std::filesystem::path(voxel.study_src_file_path).filename().string()+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Overwrite Voxel Size]")
    {
        std::istringstream in(param);
        in >> voxel.vs[0] >> voxel.vs[1] >> voxel.vs[2];
        voxel.steps += cmd+"="+param+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Crop Background]")
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
        return true;
    }
    if(cmd == "[Step T2][Edit][Rotate to MNI]")
    {
        if(!rotate_to_mni(1.0f))
            return false;
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Rotate to MNI2]")
    {
        if(!rotate_to_mni(2.0f))
            return false;
        voxel.steps += cmd+"\n";
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
    if(cmd == "[Step T2][Corrections][TOPUP EDDY]")
    {
        if(is_dsi())
        {
            if(!run_topup(param) || !run_applytopup())
                return false;
        }
        else
        {
            if(!run_topup(param) || !run_eddy())
                return false;
        }
        voxel.steps += cmd+"="+param+"\n";
        return true;
    }
    std::cout << "Unknown command:" << cmd << std::endl;
    return false;
}
void ImageModel::flip_b_table(unsigned char dim)
{
    for(unsigned int index = 0;index < src_bvectors.size();++index)
        src_bvectors[index][dim] = -src_bvectors[index][dim];
}
// 0:xy 1:yz 2: xz
void ImageModel::swap_b_table(unsigned char dim)
{
    std::swap(voxel.vs[dim],voxel.vs[(dim+1)%3]);
    for (unsigned int index = 0;index < src_bvectors.size();++index)
        std::swap(src_bvectors[index][dim],src_bvectors[index][(dim+1)%3]);
}

// 0: x  1: y  2: z
// 3: xy 4: yz 5: xz
void ImageModel::flip_dwi(unsigned char type)
{
    if(type < 3)
        flip_b_table(type);
    else
        swap_b_table(type-3);
    tipl::flip(dwi,type);
    tipl::flip(voxel.mask,type);
    prog_init p("flip image");
    if(voxel.is_histology)
        tipl::flip(voxel.hist_image,type);
    else
    tipl::par_for2(src_dwi_data.size(),[&](unsigned int index,unsigned id)
    {
        if(!id)
            check_prog(index,src_dwi_data.size());
        auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[index]),voxel.dim);
        tipl::flip(I,type);
    });
    voxel.dim = voxel.mask.shape();
}
// used in eddy correction for each dwi
void ImageModel::rotate_one_dwi(unsigned int dwi_index,const tipl::transformation_matrix<double>& T)
{
    tipl::image<3> tmp(voxel.dim);
    auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[dwi_index]),voxel.dim);
    tipl::resample(I,tmp,T,tipl::cubic);
    tipl::lower_threshold(tmp,0);
    std::copy(tmp.begin(),tmp.end(),I.begin());
    // rotate b-table
    tipl::affine_transform<double> arg;
    T.to_affine_transform(arg,voxel.dim,voxel.vs,voxel.dim,voxel.vs);
    tipl::matrix<3,3,float> r;
    tipl::rotation_matrix(arg.rotation,r.begin(),tipl::vdim<3>());
    r.inv();

    src_bvectors[dwi_index].rotate(r);
    src_bvectors[dwi_index].normalize();
}

void ImageModel::rotate(const tipl::shape<3>& new_geo,
                        const tipl::vector<3>& new_vs,
                        const tipl::transformation_matrix<double>& T,
                        const tipl::image<3,tipl::vector<3> >& cdm_dis,
                        const tipl::image<3>& super_reso_ref,double var)
{
    std::vector<tipl::image<3,unsigned short> > dwi(src_dwi_data.size());
    prog_init p("rotating");
    tipl::par_for2(src_dwi_data.size(),[&](unsigned int index,unsigned int id)
    {
        if(prog_aborted())
            return;
        if(!id)
            check_prog(index,src_dwi_data.size());
        dwi[index].resize(new_geo);
        auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[index]),voxel.dim);
        if(!super_reso_ref.empty())
            tipl::resample_with_ref(I,super_reso_ref,dwi[index],T,var);
        else
        {
            if(cdm_dis.empty())
                tipl::resample(I,dwi[index],T,tipl::cubic);
            else
                tipl::resample_dis(I,dwi[index],T,cdm_dis,tipl::cubic);
        }
        src_dwi_data[index] = &(dwi[index][0]);
    });
    if(prog_aborted())
        return;
    dwi.swap(new_dwi);
    // rotate b-table
    tipl::affine_transform<double> arg;
    T.to_affine_transform(arg,new_geo,new_vs,voxel.dim,voxel.vs);
    tipl::matrix<3,3,float> r;
    tipl::rotation_matrix(arg.rotation,r.begin(),tipl::vdim<3>());
    r.inv();

    for (auto& vec : src_bvectors)
        {
            vec.rotate(r);
            vec.normalize();
        }

    voxel.dim = new_geo;
    voxel.vs = new_vs;
    calculate_dwi_sum(true);
}
void ImageModel::resample(float nv)
{
    tipl::vector<3,float> new_vs(nv,nv,nv);
    tipl::shape<3> new_geo(uint32_t(std::ceil(float(voxel.dim.width())*voxel.vs[0]/new_vs[0])),
                              uint32_t(std::ceil(float(voxel.dim.height())*voxel.vs[1]/new_vs[1])),
                              uint32_t(std::ceil(float(voxel.dim.depth())*voxel.vs[2]/new_vs[2])));
    tipl::image<3> J(new_geo);
    if(J.empty())
        return;
    tipl::transformation_matrix<double> T;
    T.sr[0] = double(new_vs[0]/voxel.vs[0]);
    T.sr[4] = double(new_vs[1]/voxel.vs[1]);
    T.sr[8] = double(new_vs[2]/voxel.vs[2]);
    rotate(new_geo,new_vs,T);
    voxel.report += " The images were resampled to ";
    voxel.report += std::to_string(int(new_vs[0]));
    voxel.report += " mm isotropic resolution.";
}
extern std::string fib_template_file_name_2mm;
extern std::vector<std::string> iso_template_list;
bool ImageModel::arg_to_mni(float resolution,tipl::vector<3>& vs,tipl::shape<3>& new_geo,tipl::affine_transform<float>& T)
{
    tipl::image<3,unsigned char> I;
    if(resolution == 2.0f)
    {
        std::string file_name = fib_template_file_name_2mm;
        gz_mat_read read;
        if(!read.load_from_file(file_name.c_str()) || !read.get_voxel_size(vs) || !read.get_dimension(new_geo))
        {
            error_msg = "Failed to load/find fib template.";
            return false;
        }
        unsigned int row,col;
        const float* iso_ptr = nullptr;
        if(!read.read("iso",row,col,iso_ptr) || row != new_geo.size()/col)
        {
            error_msg = "Failed to read image from fib template.";
            return false;
        }
        tipl::normalize(tipl::make_image(iso_ptr,new_geo),I);
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
        nii >> I;
        nii.get_voxel_size(vs);
    }
    else
        return false;
    bool terminated = false;
    begin_prog("aligning with MNI ac-pc",true);
    check_prog(0,4);
    const float bound[8] = {1.0f,-1.0f,0.02f,-0.02f,1.2f,0.9f,0.1f,-0.1f};
    tipl::reg::linear_mr(I,vs,dwi,voxel.vs,
            T,tipl::reg::affine,tipl::reg::mutual_information(),terminated,0.1,bound);
    check_prog(1,4);
    tipl::reg::linear_mr(I,vs,dwi,voxel.vs,
            T,tipl::reg::affine,tipl::reg::mutual_information(),terminated,0.01,bound);
    check_prog(2,4);
    tipl::reg::linear_mr(I,vs,dwi,voxel.vs,
            T,tipl::reg::affine,tipl::reg::mutual_information(),terminated,0.001,bound);
    T.scaling[0] = T.scaling[1] = T.scaling[2] = 1.0f;
    T.affine[0] = T.affine[1] = T.affine[2] = 0.0f;
    check_prog(3,4);

    tipl::image<3,unsigned char> I2(I.shape());
    tipl::resample(dwi,I2,tipl::transformation_matrix<float>(T,I.shape(),vs,voxel.dim,voxel.vs),tipl::cubic);
    float r = float(tipl::correlation(I.begin(),I.end(),I2.begin()));
    std::cout << T;
    std::cout << "R2 for ac-pc alignment=" << r*r << std::endl;
    check_prog(4,4);
    if(r*r < 0.3f)
        return false;
    new_geo = I.shape();
    return true;
}

bool ImageModel::rotate_to_mni(float resolution)
{
    if(rotated_to_mni)
        return true;
    tipl::vector<3> vs;
    tipl::shape<3> geo;
    tipl::affine_transform<float> T;
    if(!arg_to_mni(resolution,vs,geo,T))
        return false;
    rotate(geo,vs,tipl::transformation_matrix<float>(T,geo,vs,voxel.dim,voxel.vs));
    voxel.report += " The diffusion MRI data were rotated to aligh AC-PC line at ";
    voxel.report += std::to_string(int(vs[0]));
    voxel.report += " mm isotropic resolution.";
    rotated_to_mni = true;
    return true;
}

void ImageModel::correct_motion(bool eddy)
{
    begin_prog("correcting motion...",true);
    tipl::affine_transform<float> arg;
    arg.rotation[0] = 0.01f;
    arg.rotation[1] = 0.01f;
    arg.rotation[2] = 0.01f;
    arg.translocation[0] = 0.01f;
    arg.translocation[0] = 0.01f;
    arg.translocation[0] = 0.01f;
    for(unsigned int i = 0;check_prog(i,src_bvalues.size());++i)
    {
        tipl::image<3,unsigned char> to;
        tipl::normalize(tipl::make_image(src_dwi_data[i],voxel.dim),to);
        tipl::filter::gaussian(to);
        tipl::filter::gaussian(to);
        bool terminated = false;
        if(src_bvalues[i] > 500.0f)
            tipl::reg::linear(dwi,voxel.vs,to,voxel.vs,
                                  arg,eddy ? tipl::reg::affine : tipl::reg::rigid_body,
                                  tipl::reg::correlation(),terminated,0.001,0,tipl::reg::narrow_bound);
        else
            tipl::reg::linear(dwi,voxel.vs,to,voxel.vs,
                              arg,eddy ? tipl::reg::affine : tipl::reg::rigid_body,
                              tipl::reg::mutual_information(),terminated,0.001,0,tipl::reg::narrow_bound);

        rotate_one_dwi(i,tipl::transformation_matrix<double>(arg,voxel.dim,voxel.vs,
                                                                     voxel.dim,voxel.vs));
        std::cout << "registeration at dwi (" << i+1 << "/" << src_bvalues.size() << ")=" << std::endl;
        std::cout << arg << std::flush;
    }
}
void ImageModel::crop(tipl::shape<3> range_min,tipl::shape<3> range_max)
{
    prog_init p("Removing background region");
    std::cout << "from:" << range_min << " to:" << range_max << std::endl;
    tipl::par_for2(src_dwi_data.size(),[&](unsigned int index,unsigned int id)
    {
        if(!id)
            check_prog(index,src_dwi_data.size());
        auto I = tipl::make_image(const_cast<unsigned short*>(src_dwi_data[index]),voxel.dim);
        tipl::image<3,unsigned short> I0 = I;
        tipl::crop(I0,range_min,range_max);
        std::fill(I.begin(),I.end(),0);
        std::copy(I0.begin(),I0.end(),I.begin());
    });
    tipl::crop(voxel.mask,range_min,range_max);
    tipl::crop(dwi,range_min,range_max);
    voxel.dim = voxel.mask.shape();
    calculate_dwi_sum(true);
}
void ImageModel::trim(void)
{
    tipl::shape<3> range_min,range_max;
    tipl::bounding_box(voxel.mask,range_min,range_max,0);
    crop(range_min,range_max);
}

float interpo_pos(float v1,float v2,float u1,float u2)
{
    float w = (u2-u1-v2+v1);
    return std::max<float>(0.0,std::min<float>(1.0,w == 0.0f? 0:(v1-u1)/w));
}

template<typename image_type>
void get_distortion_map(const image_type& v1,
                        const image_type& v2,
                        tipl::image<3>& dis_map)
{
    int h = v1.height(),w = v1.width();
    dis_map.resize(v1.shape());
    tipl::par_for(v1.depth()*h,[&](int z)
    {
        int base_pos = z*w;
        std::vector<float> line1(v1.begin()+base_pos,v1.begin()+base_pos+w),
                           line2(v2.begin()+base_pos,v2.begin()+base_pos+w);
        float sum1 = std::accumulate(line1.begin(),line1.end(),0.0f);
        float sum2 = std::accumulate(line2.begin(),line2.end(),0.0f);
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
                          const tipl::image<3>& dis_map,
                          image_type& v1_out,bool positive)
{
    int w = v1.width();
    v1_out.resize(v1.shape());
    tipl::par_for(v1.depth()*v1.height(),[&](int z)
    {
        std::vector<float> cdf_x1,cdf;
        cdf_x1.resize(size_t(w));
        cdf.resize(size_t(w));
        int base_pos = z*w;
        {
            tipl::pdf2cdf(v1.begin()+base_pos,v1.begin()+base_pos+w,&cdf_x1[0]);
            auto I1 = tipl::make_image(&cdf_x1[0],tipl::shape<1>(w));
            for(int x = 0,pos = base_pos;x < w;++x,++pos)
                cdf[size_t(x)] = tipl::estimate(I1,positive ? x+dis_map[pos] : x-dis_map[pos]);
            for(int x = 0,pos = base_pos;x < w;++x,++pos)
                v1_out[pos] = (x? std::max<float>(0.0f,cdf[size_t(x)] - cdf[size_t(x-1)]):0);
        }
    }
    );
}

void ImageModel::read_b0(tipl::image<3>& rev_b0) const
{
    rev_b0 = (src_bvalues[0] == 0.0f) ?  tipl::make_image(src_dwi_data[0],voxel.dim):
            tipl::make_image(src_dwi_data[size_t(std::min_element(src_bvalues.begin(),src_bvalues.end())-src_bvalues.begin())],voxel.dim);
}
bool ImageModel::read_rev_b0(const char* filename,tipl::image<3>& rev_b0) const
{
    tipl::image<3> v2;
    if(QString(filename).endsWith(".nii.gz") || QString(filename).endsWith(".nii"))
    {
        gz_nifti nii;
        if(!nii.load_from_file(filename))
        {
            error_msg = "Cannot load the image file";
            return false;
        }
        nii.toLPS(v2);
    }
    if(QString(filename).endsWith(".src.gz"))
    {
        ImageModel src2;
        if(!src2.load_from_file(filename))
        {
            error_msg = src2.error_msg;
            return false;
        }
        src2.read_b0(v2);
    }
    if(voxel.dim != v2.shape())
    {
        error_msg = "inconsistent image dimension between original DWI and reverse-phase DWI";
        return false;
    }
    rev_b0.swap(v2);
    return true;
}
bool phase_direction_at_AP_PA(const tipl::image<3>& v1,const tipl::image<3>& v2)
{
    tipl::image<2,float> px1,px2,py1,py2;
    tipl::project_x(v1,px1);
    tipl::project_x(v2,px2);
    tipl::project_y(v1,py1);
    tipl::project_y(v2,py2);
    float cx = float(tipl::correlation(px1.begin(),px1.end(),px2.begin()));
    float cy = float(tipl::correlation(py1.begin(),py1.end(),py2.begin()));
    std::cout << "projected x correction:" << cx << std::endl;
    std::cout << "projected y correction:" << cy << std::endl;
    return cx < cy;
}

bool ImageModel::distortion_correction(const char* filename)
{
    tipl::image<3> v1,v2;
    if(!read_rev_b0(filename,v2))
        return false;
    read_b0(v1);

    bool swap_xy = phase_direction_at_AP_PA(v1,v2);
    if(swap_xy)
    {
        tipl::swap_xy(v1);
        tipl::swap_xy(v2);
    }

    tipl::image<3> dis_map(v1.shape()),df,gx(v1.shape()),v1_gx(v1.shape()),v2_gx(v2.shape());


    tipl::filter::gaussian(v1);
    tipl::filter::gaussian(v2);

    tipl::gradient(v1,v1_gx,1,0);
    tipl::gradient(v2,v2_gx,1,0);

    get_distortion_map(v2,v1,dis_map);
    tipl::filter::gaussian(dis_map);
    tipl::filter::gaussian(dis_map);

    tipl::image<3> vv1,vv2;
    apply_distortion_map2(v1,dis_map,vv1,true);
    apply_distortion_map2(v2,dis_map,vv2,false);

    /*

    tipl::image<3> vv1,vv2;
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
    }*/

    std::vector<tipl::image<3,unsigned short> > dwi(src_dwi_data.size());
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
    return true;
}


bool create_acqparam_file(const char* acq_param,const tipl::image<3>& b0,const tipl::image<3>& rev_b0)
{
    bool is_appa = phase_direction_at_AP_PA(b0,rev_b0);
    std::cout << "phase encoding direction at " << (is_appa ? "anterior-posterior" : "left-right") << std::endl;
    unsigned int phase_dim = (is_appa ? 1 : 0);
    auto c1 = tipl::center_of_mass(b0);
    auto c2 = tipl::center_of_mass(rev_b0);
    bool phase_dir = c1[phase_dim] < c2[phase_dim];
    char pe[2][2][3] = { {"RL","LR"},{"AP","PA"}};
    std::cout << "source has phase encoding direction of " << pe[phase_dim][phase_dir ? 0:1] << std::endl;
    std::cout << "other_src has phase encoding direction of " << pe[phase_dim][phase_dir ? 1:0] << std::endl;
    std::cout << "create acqparams.txt" << std::endl;
    std::ofstream out(acq_param);
    if(!out)
    {
        std::cout << "ERROR: cannot write to acq param file " << acq_param << std::endl;
        return 1;
    }
    if(is_appa)
    {
        // anterior or posterior
        if(phase_dir)
            out << "0 -1 0 0.05\r\n0 1 0 0.05" << std::endl;
        else
            out << "0 1 0 0.05\r\n0 -1 0 0.05" << std::endl;
    }
    else
    {
        // left or right
        if(phase_dir)
            out << "-1 0 0 0.05\r\n1 0 0 0.05" << std::endl;
        else
            out << "1 0 0 0.05\r\n-1 0 0 0.05" << std::endl;
    }
    return true;
}

bool create_b0_appa_file(const char* b0_appa_file,const ImageModel& src,const tipl::image<3>& b0,const tipl::image<3>& rev_b0)
{
    std::cout << "create b0_appa.nii.gz" << std::endl;
    {
        tipl::matrix<4,4> trans;
        initial_LPS_nifti_srow(trans,b0.shape(),src.voxel.vs);

        tipl::shape<4> nifti_dim;
        std::copy(b0.shape().begin(),b0.shape().end(),nifti_dim.begin());
        nifti_dim[3] = 2;

        tipl::image<4,float> buffer(nifti_dim);
        std::copy(b0.begin(),b0.end(),buffer.begin());
        std::copy(rev_b0.begin(),rev_b0.end(),buffer.begin()+int64_t(b0.size()));
        if(!gz_nifti::save_to_file(b0_appa_file,buffer,src.voxel.vs,trans))
        {
            std::cout << "ERROR: Cannot wrtie a temporary b0_appa image volume to " << b0_appa_file << std::endl;
            return false;
        }
    }
    return true;
}


//QFileInfo(file_name.c_str()).absolutePath()

bool test_call_fsl(std::string command,std::string exec_name)
{
    QProcess program;
    std::cout << "test calling " << exec_name << "..." << std::endl;
    program.start(command.c_str());
    if(program.waitForStarted())
    {
        std::cout << exec_name << " successfully initiated." << std::endl;
        if(!program.waitForFinished(1000))
            program.kill();
        return true;
    }
    else
    {
        std::cout << "cannot initiate " << exec_name << std::endl;
        return false;
    }
}
#include <QCoreApplication>
std::string get_plugin_executive(std::string exec_name,std::string command)
{
    // if a command is specified, call and return
    if(!command.empty())
        return test_call_fsl(command,exec_name) ? command : std::string();
    // try direct calling
    if(test_call_fsl(exec_name,exec_name))
        return exec_name;
    // now search for fsl executives
    #ifdef _WIN32
    if(QFileInfo(QCoreApplication::applicationDirPath() + "/plugin/" + exec_name.c_str() + ".exe").exists())
    {
        std::cout << "found " << exec_name << ".exe" << std::endl;
        command = (QCoreApplication::applicationDirPath() +  + "/plugin/" + exec_name.c_str() + ".exe").toStdString();
    }
    else
    {
        std::cout << "Please download " << exec_name << ".exe (google \"" << exec_name << ".exe\" fsl) and copy it to " << QCoreApplication::applicationDirPath().toStdString() << "/plugin" << std::endl;
        return std::string();
    }
    #else
    std::string fsl_path;
    auto env = QProcess::systemEnvironment();
    for(int i =0;i < env.size();++i)
        if(env[i].startsWith("FSLDIR="))
        {
            fsl_path = env[i].split("=")[1].toStdString();
            std::cout << "FSL installation found at " << fsl_path << std::endl;
            if(fsl_path.back() == '/')
                fsl_path.pop_back();
            command = fsl_path + "/bin/" + exec_name;
            break;
        }
    if(fsl_path.empty())
        std::cout << "cannot find FSL installation" << std::endl;
    #endif
    return test_call_fsl(command,exec_name) ? command : std::string();
}

bool ImageModel::run_plugin(std::string program_name,std::vector<std::string> param,std::string working_dir,std::string exec)
{
    exec = get_plugin_executive(program_name,exec);
    if(exec.empty())
    {
        #ifdef _WIN32
        error_msg = (QString("Please download win_fsl and extract it to %1\\plugin").arg(QCoreApplication::applicationDirPath())).toStdString();
        #else
        error_msg = "cannot locate ";
        error_msg += program_name;
        #endif
        return false;
    }
    QProcess program;
    program.setEnvironment(program.environment() << "FSLOUTPUTTYPE=NIFTI_GZ");
    program.setWorkingDirectory(working_dir.c_str());
    std::cout << "running " << program_name << " at " << working_dir << " with" << std::endl;
    QStringList p;
    for(auto s:param)
    {
        std::cout << s << std::endl;
        p << s.c_str();
    }
    program.start(exec.c_str(),p);
    if(!program.waitForStarted())
    {
        error_msg = program_name + " failed to initiate.";
        return false;
    }
    if(program.waitForFinished(3000))
    {
        error_msg = QString::fromLocal8Bit(program.readAllStandardError()).toStdString()+QString::fromLocal8Bit(program.readAllStandardOutput()).toStdString();
        if(error_msg.empty())
            error_msg = program_name + " ended prematurely: unknown error";
        return false;
    }
    unsigned int proc_time = 0;
    while(!program.waitForFinished(1000) && check_prog((++proc_time)%360,360))
    {
        QString output = QString::fromLocal8Bit(program.readAllStandardOutput());
        if(output.isEmpty())
            continue;
        QStringList output_lines = output.remove('\r').split('\n');
        if(output_lines.back().isEmpty())
            output_lines.pop_back();
        for(int i = 0;i+1 < output_lines.size();++i)
            std::cout << output_lines[i].toStdString() << std::endl;
        set_title(output_lines.back().toStdString().c_str());
    }
    check_prog(1,1);
    if(prog_aborted())
    {
        program.kill();
        error_msg = "process aborted";
        return false;
    }
    std::cout << "completed." << std::endl;
    error_msg = QString::fromLocal8Bit(program.readAllStandardError()).toStdString();
    return error_msg.empty();
}

bool ImageModel::run_topup(std::string other_src,std::string exec)
{
    if(voxel.report.find("rotated") != std::string::npos)
    {
        error_msg = "TOPUP cannot be applied to rotated images";
        return false;
    }
    std::string topup_result = QFileInfo(file_name.c_str()).baseName().replace('.','_').toStdString();
    std::string acqparam_file = QFileInfo(file_name.c_str()).baseName().toStdString() + ".topup.acqparams.txt";
    std::string b0_appa_file = QFileInfo(file_name.c_str()).baseName().toStdString() + ".topup.b0_appa.nii.gz";

    tipl::image<3> b0,rev_b0;
    read_b0(b0);
    if(!read_rev_b0(other_src.c_str(),rev_b0))
        return false;

    // trim background
    {
        std::cout << "trimming background to speed up topup/eddy" << std::endl;
        auto temp_mask = voxel.mask;
        tipl::morphology::dilation(temp_mask);
        tipl::shape<3> range_min,range_max;
        tipl::bounding_box(temp_mask,range_min,range_max,0);
        // ensure even number in the dimension for topup
        for(int d = 0;d < 3;++d)
            if((range_max[d]-range_min[d]) % 2 != 0)
                range_max[d]--;
        crop(range_min,range_max);
        tipl::crop(b0,range_min,range_max);
        tipl::crop(rev_b0,range_min,range_max);
    }

    if(!create_acqparam_file(acqparam_file.c_str(),b0,rev_b0) ||
       !create_b0_appa_file(b0_appa_file.c_str(),*this,b0,rev_b0))
    {
        error_msg = "Cannot create temporary processing files at ";
        error_msg += QFileInfo(file_name.c_str()).absolutePath().toStdString();
        error_msg += ". Please check write permission";
        return false;
    }

    if(QFileInfo(QFileInfo(file_name.c_str()).absolutePath()+"/" + topup_result.c_str() + "_fieldcoef.nii.gz").exists())
    {
        std::cout << "found topup result.skipping" << std::endl;
        return true;
    }


    std::vector<std::string> param = {
        "--warpres=20,16,14,12,10,6,4,4,4",
        "--subsamp=2,2,2,2,2,1,1,1,1",  // This causes an error in odd number of slices
        "--fwhm=8,6,4,3,3,2,1,0,0",
        "--miter=5,5,5,5,5,10,10,20,20",
        "--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001",
        "--estmov=1,1,1,1,1,0,0,0,0",
        "--minmet=0,0,0,0,0,1,1,1,1",
        "--scale=1",
        QString("--imain=%1").arg(b0_appa_file.c_str()).toStdString().c_str(),
        QString("--datain=%1").arg(acqparam_file.c_str()).toStdString().c_str(),
        QString("--out=%1").arg(topup_result.c_str()).toStdString().c_str(),
        QString("--verbose=1").toStdString().c_str()};

    return run_plugin("topup",param,QFileInfo(file_name.c_str()).absolutePath().toStdString(),exec);
}

bool load_bvec(const char* file_name,std::vector<double>& b_table,bool flip_by = true);
bool ImageModel::load_topup_eddy_result(void)
{
    std::string preproc_file = file_name+".preproc.nii.gz";
    std::string bvec_file = file_name+".preproc.eddy_rotated_bvecs";

    gz_nifti in;
    if(!in.load_from_file(preproc_file))
    {
        error_msg = "failed to open result nifti files from ";
        error_msg = file_name;
        return false;
    }
    tipl::shape<3> dim;
    for(size_t index = 0;index < src_bvalues.size();++index)
    {
        if(!in.input_stream->good())
        {
            error_msg = "invalid results. please reload SRC file and rerun the TOPUP/EDDY pipeline";
            return false;
        }
        tipl::image<3> I;
        in >> I;
        tipl::lower_threshold(I,0);
        dim = I.shape();
        std::copy(I.begin(),I.end(),const_cast<unsigned short*>(src_dwi_data[index]));
    }

    std::vector<double> b_table;
    if(load_bvec(bvec_file.c_str(),b_table))
    {
        std::cout << "load rotated bvce from " << bvec_file << std::endl;
        for(size_t index = 0,i = 0;i < src_bvalues.size() && index+2 < b_table.size();++i,index += 3)
        {
            src_bvectors[i][0] = float(b_table[index]);
            src_bvectors[i][1] = float(b_table[index+1]);
            src_bvectors[i][2] = float(b_table[index+2]);
        }
    }
    voxel.dim = dim;
    voxel.report += " The suscetibility artifact was corrected using reversed phase-encoding b0 through FSL topup";
    voxel.report += b_table.empty() ? ".":"/eddy.";
    calculate_dwi_sum(true);
    return true;
}

bool ImageModel::run_applytopup(std::string exec)
{
    std::string topup_result = QFileInfo(file_name.c_str()).baseName().replace('.','_').toStdString();
    std::string acqparam_file = QFileInfo(file_name.c_str()).baseName().toStdString() + ".topup.acqparams.txt";
    std::string temp_nifti = file_name+".nii.gz";
    std::string preproc_file = file_name+".preproc";
    if(!save_to_nii_width_even_dimension(temp_nifti.c_str()))
        return false;

    std::vector<std::string> param = {
            QString("--imain=%1").arg(QFileInfo(temp_nifti.c_str()).fileName()).toStdString().c_str(),
            QString("--datain=%1").arg(acqparam_file.c_str()).toStdString().c_str(),
            QString("--topup=%1").arg(topup_result.c_str()).toStdString().c_str(),
            QString("--out=%1").arg(QFileInfo(preproc_file.c_str()).fileName()).toStdString().c_str(),
            "--inindex=1",
            "--method=jac",
            "--verbose=1"};

    if(!run_plugin("applytopup",param,QFileInfo(file_name.c_str()).absolutePath().toStdString(),exec) ||
       !load_topup_eddy_result())
        return false;

    QFile(temp_nifti.c_str()).remove();
    calculate_dwi_sum(true);
    return true;
}

bool ImageModel::run_eddy(std::string exec)
{
    std::string topup_result = QFileInfo(file_name.c_str()).baseName().replace('.','_').toStdString();
    std::string acqparam_file = QFileInfo(file_name.c_str()).baseName().toStdString() + ".topup.acqparams.txt";
    std::string temp_nifti = file_name+".nii.gz";
    std::string mask_nifti = file_name+".mask.nii.gz";
    std::string preproc_file = file_name+".preproc";
    std::string index_file = file_name+".index.txt";
    std::string bval_file = file_name+".bval";
    std::string bvec_file = file_name+".bvec";

    if(!save_to_nii_width_even_dimension(temp_nifti.c_str()))
        return false;
    if(!save_mask_nii(mask_nifti.c_str()))
    {
        error_msg = "cannot save mask file to ";
        error_msg += mask_nifti;
        return false;
    }
    if(!save_bval(bval_file.c_str()))
    {
        error_msg = "cannot save ";
        error_msg += bval_file;
        return false;
    }
    if(!save_bvec(bvec_file.c_str()))
    {
        error_msg = "cannot save ";
        error_msg += bvec_file;
        return false;
    }
    {
        std::ofstream out(index_file);
        for(size_t i = 0;i < src_bvalues.size();++i)
            out << " 1";
    }


    std::vector<std::string> param = {
            QString("--imain=%1").arg(QFileInfo(temp_nifti.c_str()).fileName()).toStdString().c_str(),
            QString("--mask=%1").arg(QFileInfo(mask_nifti.c_str()).fileName()).toStdString().c_str(),
            QString("--acqp=%1").arg(acqparam_file.c_str()).toStdString().c_str(),
            QString("--index=%1").arg(QFileInfo(index_file.c_str()).fileName()).toStdString().c_str(),
            QString("--bvecs=%1").arg(QFileInfo(bvec_file.c_str()).fileName()).toStdString().c_str(),
            QString("--bvals=%1").arg(QFileInfo(bval_file.c_str()).fileName()).toStdString().c_str(),
            QString("--topup=%1").arg(topup_result.c_str()).toStdString().c_str(),
            QString("--out=%1").arg(QFileInfo(preproc_file.c_str()).fileName()).toStdString().c_str(),
            "--verbose=1"
            };

    if(!run_plugin("eddy",param,QFileInfo(file_name.c_str()).absolutePath().toStdString(),exec))
    {
        std::cout << "eddy cannot process this data:" << std::endl;
        std::cout << error_msg << std::endl;
        return run_applytopup();
    }
    if(!load_topup_eddy_result())
        return false;
    QFile(temp_nifti.c_str()).remove();
    calculate_dwi_sum(true);
    return true;
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
            << " The maximum b-value was " << int(std::round(sorted_bvalues.back())) << " s/mm².";
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
        out << " s/mm².";

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
                << " The b-value was " << sorted_bvalues.back() << " s/mm².";
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
    prog_init p("saving ",std::filesystem::path(dwi_file_name).filename().string().c_str());
    for (unsigned int index = 0;check_prog(index,src_bvalues.size());++index)
    {
        std::ostringstream out;
        out << "image" << index;
        mat_writer.write(out.str().c_str(),src_dwi_data[index],
                         uint32_t(voxel.dim.plane_size()),uint32_t(voxel.dim.depth()));
    }
    mat_writer.write("mask",voxel.mask,uint32_t(voxel.dim.plane_size()));
    return true;
}

extern std::string src_error_msg;
bool find_bval_bvec(const char* file_name,QString& bval,QString& bvec);
bool load_4d_nii(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,bool need_bvalbvec);
void prepare_idx(const char* file_name,std::shared_ptr<gz_istream> in)
{
    if(!QString(file_name).endsWith(".gz"))
        return;
    std::string idx_name = file_name;
    idx_name += ".idx";
    {
        in->buffer_all = true;
        if(std::filesystem::exists(idx_name) &&
           QFileInfo(idx_name.c_str()).lastModified() > QFileInfo(file_name).lastModified())
        {
            std::cout << "using index file for accelerated loading:" << idx_name << std::endl;
            in->load_index(idx_name.c_str());
        }
        else
        {
            if(QFileInfo(file_name).size() > 134217728) // 128mb
            {
                std::cout << "prepare index file for future accelerated loading" << std::endl;
                in->sample_access_point = true;
            }
        }
    }
}
void save_idx(const char* file_name,std::shared_ptr<gz_istream> in)
{
    if(!QString(file_name).endsWith(".gz"))
        return;
    std::string idx_name = file_name;
    idx_name += ".idx";
    if(in->has_access_points() && in->sample_access_point)
    {
        std::cout << "saving index file for accelerated loading: " << idx_name << std::endl;
        in->save_index(idx_name.c_str());
    }
}

bool ImageModel::load_from_file(const char* dwi_file_name)
{
    file_name = dwi_file_name;
    if(!QFileInfo(dwi_file_name).exists())
    {
        error_msg = "File does not exist:";
        error_msg += dwi_file_name;
        return false;
    }
    if(QString(dwi_file_name).toLower().endsWith(".jpg") ||
       QString(dwi_file_name).toLower().endsWith(".tif"))
    {
        tipl::image<2,unsigned char> raw;
        {
            QImage fig;
            set_title("load picture");
            if(!fig.load(dwi_file_name))
            {
                error_msg = "Unsupported image format";
                return false;
            }
            set_title("converting to grascale");
            int pixel_bytes = fig.bytesPerLine()/fig.width();
            raw.resize(tipl::shape<2>(uint32_t(fig.width()),uint32_t(fig.height())));
            tipl::par_for(raw.height(),[&](int y){
                auto line = fig.scanLine(y);
                auto out = raw.begin() + int64_t(y)*raw.width();
                for(int x = 0;x < raw.width();++x,line += pixel_bytes)
                    out[x] = uint8_t(*line);
            });
        }
        set_title("generating mask");
        auto raw_ = tipl::make_image(&*raw.begin(),tipl::shape<3>(raw.width(),raw.height(),1));
        if(raw.width() > 2048)
        {
            tipl::downsample_with_padding2(raw_,dwi);
            while(dwi.width() > 2048)
                tipl::downsample_with_padding2(dwi);
        }
        else
            dwi = raw_;

        // increase contrast
        dwi -= 128;
        dwi *= 2;


        voxel.is_histology = true;
        voxel.vs = {0.05f,0.05f,0.05f};
        voxel.dim = dwi.shape();
        voxel.hist_image.swap(raw);
        voxel.report = "Histology image was loaded at a size of ";
        voxel.report += std::to_string(voxel.hist_image.width());
        voxel.report += " by ";
        voxel.report += std::to_string(voxel.hist_image.height());
        voxel.report += " pixels.";

        voxel.steps += "[Step T2][Reconstruction] open ";
        voxel.steps += std::filesystem::path(dwi_file_name).filename().string();
        voxel.steps += "\n";

        set_title("generating mask");
        tipl::segmentation::otsu(dwi,voxel.mask);
        tipl::negate(voxel.mask);
        for(int i = 0;i < int(dwi.width()/200);++i)
        {
            auto slice = voxel.mask.slice_at(0);
            tipl::morphology::dilation(slice);
        }
        for(int i = 0;i < int(dwi.width()/200);++i)
        {
            auto slice = voxel.mask.slice_at(0);
            tipl::morphology::erosion(slice);
        }
        tipl::morphology::defragment(voxel.mask);
        return true;
    }
    if(QString(dwi_file_name).toLower().endsWith(".nii.gz"))
    {
        QString bval,bvec;
        if(!find_bval_bvec(dwi_file_name,bval,bvec))
        {
            error_msg = "Cannot find bval bvec files for the NIFTI file";
            return false;
        }
        std::vector<std::shared_ptr<DwiHeader> > dwi_files;
        if(!load_4d_nii(dwi_file_name,dwi_files,true))
        {
            error_msg = src_error_msg;
            return false;
        }
        nifti_dwi.resize(dwi_files.size());
        src_dwi_data.resize(dwi_files.size());
        src_bvalues.resize(dwi_files.size());
        src_bvectors.resize(dwi_files.size());
        for(size_t index = 0;index < dwi_files.size();++index)
        {
            if(index == 0)
            {
                voxel.vs = dwi_files[0]->voxel_size;
                voxel.dim = dwi_files[0]->image.shape();
            }
            nifti_dwi[index].swap(dwi_files[index]->image);
            src_dwi_data[index] = &nifti_dwi[index][0];
            src_bvalues[index] = dwi_files[index]->bvalue;
            src_bvectors[index] = dwi_files[index]->bvec;
        }

        get_report(voxel.report);
        calculate_dwi_sum(true);
        voxel.steps += "[Step T2][Reconstruction] open ";
        voxel.steps += std::filesystem::path(dwi_file_name).filename().string();
        voxel.steps += "\n";
        return true;
    }

    if (!QString(dwi_file_name).toLower().endsWith(".src.gz") &&
        !QString(dwi_file_name).toLower().endsWith(".src"))
    {
        error_msg = "Unsupported file format";
        return false;
    }

    prepare_idx(dwi_file_name,mat_reader.in);
    if(!mat_reader.load_from_file(dwi_file_name) || prog_aborted())
    {
        if(!prog_aborted())
        {
            error_msg = QFileInfo(dwi_file_name).baseName().toStdString();
            error_msg = " is an invalid SRC file";
        }
        return false;
    }
    save_idx(dwi_file_name,mat_reader.in);


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
        const float* grad_dev_ptr = nullptr;
        std::vector<tipl::pointer_image<3,float> > grad_dev;
        size_t b0_pos = size_t(std::min_element(src_bvalues.begin(),src_bvalues.end())-src_bvalues.begin());
        if(src_bvalues[b0_pos] == 0.0f &&
           mat_reader.read("grad_dev",row,col,grad_dev_ptr) &&
           size_t(row)*size_t(col) == voxel.dim.size()*9)
        {
            for(unsigned int index = 0;index < 9;index++)
                grad_dev.push_back(tipl::make_image(const_cast<float*>(grad_dev_ptr+index*voxel.dim.size()),voxel.dim));
            if(std::fabs(grad_dev[0][0])+std::fabs(grad_dev[4][0])+std::fabs(grad_dev[8][0]) < 1.0f)
            {
                tipl::add_constant(grad_dev[0].begin(),grad_dev[0].end(),1.0);
                tipl::add_constant(grad_dev[4].begin(),grad_dev[4].end(),1.0);
                tipl::add_constant(grad_dev[8].begin(),grad_dev[8].end(),1.0);
            }
            // correct signals
            {
                set_title("apply gradient deviation correction");
                tipl::par_for(voxel.dim.size(),[&](size_t voxel_index)
                {
                    tipl::matrix<3,3,float> G;
                    for(unsigned int i = 0;i < 9;++i)
                        G[i] = grad_dev[i][voxel_index];

                    double b0_signal = double(src_dwi_data[b0_pos][voxel_index]);
                    if(b0_signal == 0.0)
                        return;
                    for(unsigned int index = 0;index < src_bvalues.size();++index)
                        if(src_bvalues[index] > 0.0f)
                        {
                            auto bvec = src_bvectors[index];
                            bvec.rotate(G);
                            double inv_l2 = 1.0/double(bvec.length2());
                            const_cast<unsigned short*>(src_dwi_data[index])[voxel_index] =
                                    uint16_t(std::pow(b0_signal,1.0-inv_l2)*
                                    std::pow(double(src_dwi_data[index][voxel_index]),inv_l2));
                        }
                });
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
    voxel.steps += std::filesystem::path(dwi_file_name).filename().string();
    voxel.steps += "\n";
    return true;
}

bool ImageModel::save_fib(const std::string& output_name)
{
    prog_init p("saving ",std::filesystem::path(output_name).filename().string().c_str());
    gz_mat_write mat_writer(output_name.c_str());
    if(!mat_writer)
    {
        error_msg = "Cannot save fib file";
        return false;
    }

    mat_writer.write("dimension",voxel.dim);
    mat_writer.write("voxel_size",voxel.vs);

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
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
bool ImageModel::save_to_nii(const char* nifti_file_name) const
{
    tipl::matrix<4,4> trans;
    initial_LPS_nifti_srow(trans,voxel.dim,voxel.vs);

    tipl::shape<4> nifti_dim;
    std::copy(voxel.dim.begin(),voxel.dim.end(),nifti_dim.begin());
    nifti_dim[3] = uint32_t(src_bvalues.size());

    tipl::image<4,unsigned short> buffer(nifti_dim);
    tipl::par_for(src_bvalues.size(),[&](size_t index)
    {
        std::copy(src_dwi_data[index],
                  src_dwi_data[index]+voxel.dim.size(),
                  buffer.begin() + long(index*voxel.dim.size()));
    });
    return gz_nifti::save_to_file(nifti_file_name,buffer,voxel.vs,trans);
}
bool ImageModel::save_to_nii_width_even_dimension(const char* nifti_file_name) const
{
    if(voxel.dim[0]%2 == 0 && voxel.dim[1]%2 == 0 && voxel.dim[2]%2 == 0)
    {
        if(!save_to_nii(nifti_file_name))
        {
            error_msg = "cannot save temporary nifti file";
            return false;
        }
    }
    else
    // padding
    {
        std::cout << "saving padded nifti file to " << nifti_file_name << std::endl;
        tipl::shape<3> new_dim(tipl::shape<3>(uint32_t(voxel.dim.width()+1)/2*2,uint32_t(voxel.dim.height()+1)/2*2,uint32_t(voxel.dim.depth()+1)/2*2));
        tipl::matrix<4,4> trans;
        initial_LPS_nifti_srow(trans,new_dim,voxel.vs);

        tipl::shape<4> nifti_dim;
        std::copy(new_dim.begin(),new_dim.end(),nifti_dim.begin());
        nifti_dim[3] = uint32_t(src_bvalues.size());

        // apply topup may output negative values
        tipl::image<4> buffer(nifti_dim);
        tipl::par_for(src_bvalues.size(),[&](size_t index)
        {
            auto I = buffer.slice_at(uint32_t(index));
            tipl::draw(tipl::make_image(src_dwi_data[index],voxel.dim),I,
                       tipl::vector<3>(0,0,0));
        });
        if(!gz_nifti::save_to_file(nifti_file_name,buffer,voxel.vs,trans))
        {
            error_msg = "cannot save temporary nifti file";
            return false;
        }
    }
    return true;
}
bool ImageModel::save_b0_to_nii(const char* nifti_file_name) const
{
    tipl::matrix<4,4> trans;
    initial_LPS_nifti_srow(trans,voxel.dim,voxel.vs);
    tipl::image<3> buffer(voxel.dim);
    std::copy(src_dwi_data[0],src_dwi_data[0]+buffer.size(),buffer.begin());
    return gz_nifti::save_to_file(nifti_file_name,buffer,voxel.vs,trans);
}
bool ImageModel::save_mask_nii(const char* nifti_file_name) const
{
    tipl::matrix<4,4> trans;
    initial_LPS_nifti_srow(trans,voxel.dim,voxel.vs);
    return gz_nifti::save_to_file(nifti_file_name,voxel.mask,voxel.vs,trans);
}

bool ImageModel::save_dwi_sum_to_nii(const char* nifti_file_name) const
{
    tipl::matrix<4,4> trans;
    initial_LPS_nifti_srow(trans,voxel.dim,voxel.vs);
    tipl::image<3,unsigned char> buffer(dwi);
    return gz_nifti::save_to_file(nifti_file_name,buffer,voxel.vs,trans);
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
    // correct b_table first
    if(voxel.check_btable)
        study_src->check_b_table();
    // apply preprocessing steps
    {
        std::istringstream commands(voxel.steps);
        std::string cmd;
        std::cout << "applying following commands to comapred SRC" << std::endl;
        while(std::getline(commands,cmd))
        {
            if(cmd.find("[Edit]") != std::string::npos &&
               cmd.find('=') == std::string::npos)
            {
                std::cout << cmd << std::endl;
                    study_src->command(cmd);
            }
        }
    }

    voxel.study_name = QFileInfo(file_name).baseName().toStdString();
    voxel.compare_voxel = &(study_src->voxel);
    {
        check_prog(0,0);
        // temporary store the mask
        auto mask = voxel.mask;
        // set all mask=1 for baseline and follow-up
        std::fill(voxel.mask.begin(),voxel.mask.end(),1);
        std::fill(study_src->voxel.mask.begin(),study_src->voxel.mask.end(),1);

        reconstruct<check_btable_process>("calculating fa map1");
        study_src->reconstruct<check_btable_process>("calculating fa map2");
        // restore mask

        voxel.mask.swap(mask);
        check_prog(0,0);
        begin_prog("registration between longitudinal scans");
        auto& Ib = voxel.fib_fa; // baseline
        auto& If = study_src->voxel.fib_fa; // follow-up
        auto& Ib_vs = voxel.vs;
        auto& If_vs = study_src->voxel.vs;
        bool terminated = false;
        check_prog(0,4);
        tipl::affine_transform<float> T;
        {
            tipl::vector<3> vs;
            tipl::shape<3> geo;
            study_src->arg_to_mni(2.0f,vs,geo,T);
        }
        begin_prog("registering longitudinal data",true);
        check_prog(0,4);
        tipl::reg::linear(Ib,Ib_vs,If,If_vs,
                T,tipl::reg::rigid_body,tipl::reg::correlation(),terminated,0.01,0,tipl::reg::large_bound);
        check_prog(1,4);
        tipl::reg::linear(Ib,Ib_vs,If,If_vs,
                T,tipl::reg::rigid_body,tipl::reg::correlation(),terminated,0.001,0,tipl::reg::large_bound);
        tipl::transformation_matrix<float> arg(T,Ib.shape(),Ib_vs,If.shape(),If_vs);
        // nonlinear part
        tipl::image<3,tipl::vector<3> > cdm_dis;

        //if(voxel.dt_deform)
        {
            set_title("nonlinear warping");
            check_prog(2,4);
            tipl::image<3> Iff(Ib.shape());
            tipl::resample(If,Iff,arg,tipl::cubic);
            tipl::match_signal(Ib,Iff);
            bool terminated = false;
            tipl::reg::cdm_param param;
            param.cdm_smoothness = 0.2f;
            param.iterations = 120;
            tipl::reg::cdm(Ib,Iff,cdm_dis,terminated,param);


            set_title("subvoxel nonlinear warping");
            check_prog(3,4);

            tipl::image<3> If2Ib(Ib.shape());
            tipl::resample_dis(If,If2Ib,arg,cdm_dis,tipl::cubic);
            tipl::image<3,tipl::vector<3> > cdm_dis2;

            param.multi_resolution = false;
            tipl::reg::cdm(Ib,If2Ib,cdm_dis2,terminated,param);
            tipl::accumulate_displacement(cdm_dis,cdm_dis2);

            /*
            if(1) // debug
            {
                tipl::image<3> If2Ib(voxel.dim);
                tipl::resample_dis(If,If2Ib,arg,cdm_dis,tipl::cubic);
                gz_nifti o1,o2,o3,o4;
                o1.set_voxel_size(voxel.vs);
                o1.load_from_image(Ib);
                o1.save_to_file("d:/Ib.nii.gz");

                o2.set_voxel_size(study_src->voxel.vs);
                o2.load_from_image(If);
                o2.save_to_file("d:/If.nii.gz");

                o3.set_voxel_size(voxel.vs);
                o3.load_from_image(Iff);
                o3.save_to_file("d:/Iff.nii.gz");

                o3.set_voxel_size(voxel.vs);
                o3.load_from_image(If2Ib);
                o3.save_to_file("d:/If2Ib.nii.gz");
            }
            */
        }
        study_src->rotate(Ib.shape(),voxel.vs,arg,cdm_dis);
        study_src->voxel.mask = voxel.mask;
        check_prog(4,4);
    }

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
    voxel.R2 = float(tipl::correlation(dwi.begin(),dwi.end(),study_src->dwi.begin()));
    return true;
}
