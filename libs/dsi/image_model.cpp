#include <QFileInfo>
#include <QDir>
#include <QInputDialog>
#include <QDateTime>
#include <QImage>
#include <QProcess>
#include <QApplication>
#include "image_model.hpp"
#include "odf_process.hpp"
#include "dti_process.hpp"
#include "fib_data.hpp"
#include "dwi_header.hpp"
#include "tracking/region/Regions.h"
#include <filesystem>
#include "reg.hpp"

bool load_4d_nii(const std::string& file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files,
                 bool search_bvalbvec,
                 bool must_have_bval_bvec,std::string& error_msg);

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




void src_data::draw_mask(tipl::color_image& buffer,int position)
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
void src_data::calculate_dwi_sum(bool update_mask)
{
    if(src_dwi_data.empty())
        return;
    {
        tipl::image<3> dwi_sum(voxel.dim);
        bool skip_b0 = tipl::max_value(src_bvalues) >= 100.0;
        tipl::adaptive_par_for(dwi_sum.size(),[&](size_t i)
        {
            for(size_t j = 0;j < src_dwi_data.size();++j)
            {
                if(skip_b0 && src_bvalues[j] < 100.0f)
                    continue;
                dwi_sum[i] += src_dwi_data[j][i];
            }
        });
        dwi = subject_image_pre(dwi_sum);
    }

    if(update_mask)
    {
        tipl::out() << "create mask from dwi sum";
        tipl::threshold(dwi,voxel.mask,25,1,0);
        if(dwi.depth() < 300)
        {
            tipl::morphology::defragment(voxel.mask);
            for(size_t i = 0;i < 6;++i)
                tipl::morphology::fit(voxel.mask,dwi);
            tipl::morphology::defragment(voxel.mask);
            tipl::morphology::smoothing(voxel.mask);
            tipl::morphology::defragment_slice(voxel.mask);
            for(size_t i = 0;i < 6;++i)
                tipl::morphology::fit(voxel.mask,dwi);
            tipl::morphology::defragment(voxel.mask);

        }
    }
    else
    {
        if(dwi.shape() == voxel.mask.shape())
            tipl::preserve(dwi.begin(),dwi.end(),voxel.mask.begin());
    }
}

bool src_data::warp_b0_to_image(dual_reg& r)
{
    tipl::progress prog("registering images");
    std::vector<tipl::image<3> > b0;
    if(!read_b0(b0))
        return false;
    r.I[0] = subject_image_pre(std::move(b0[0]));
    r.I[1] = subject_image_pre(tipl::image<3>(dwi));
    r.I[2] = voxel.mask;
    tipl::morphology::dilation(r.I[2]);
    r.modality_names = {"b0","dwi sum","mask"};
    r.IR = voxel.trans_to_mni;
    r.Ivs = voxel.vs;
    r.Is = r.I[0].shape();
    r.match_resolution(true);
    bool ended = false;
    std::thread thread([&](void)
    {
        r.linear_reg(tipl::prog_aborted);
        if(r.reg_type == tipl::reg::rigid_body)
        {
            tipl::image<3> mask(r.J[2]);
            tipl::filter::mean(mask);
            tipl::filter::mean(mask);
            tipl::filter::mean(mask);
            tipl::normalize(mask);
            r.It[0] *= mask;
        }
        if(r.r[0] < r.r[1]) // sum_dwi correlates better
        {
            tipl::out() << "use sum of dwi for nonlinear registration";
            r.J[0].swap(r.J[1]);
            r.I[0].swap(r.I[1]);
        }
        r.nonlinear_reg(tipl::prog_aborted);
        ended = true;
    });
    while(!ended)
        prog(0,1);
    thread.join();
    return !prog.aborted();
}
extern std::vector<std::string> t2w_template_list,iso_template_list;
bool src_data::mask_from_template(void)
{
    tipl::progress prog("generate mask from template");
    dual_reg r;
    if(!r.load_template(0,t2w_template_list[voxel.template_id]) ||
       !r.load_template(1,iso_template_list[voxel.template_id]))
    {
        error_msg = "no template t2w for generating mask";
        return false;
    }
    // remove skull from t2w
    tipl::preserve(r.It[0].begin(),r.It[0].end(),r.It[1].begin());
    if(!warp_b0_to_image(r))
        return false;
    // use iso to generate mask
    tipl::threshold(r.apply_warping<false,tipl::interpolation::linear>(r.It[1]),voxel.mask,50.0f);
    for(size_t i = 0;i < 4;++i)
    {
        tipl::morphology::smoothing(voxel.mask);
        tipl::morphology::fit(voxel.mask,dwi);
    }
    apply_mask = true;
    return true;
}
bool src_data::mask_from_unet(void)
{
    std::vector<tipl::image<3> > b0;
    if(!read_b0(b0))
        return false;
    std::string model_file_name = QCoreApplication::applicationDirPath().toStdString() + "/network/brain.t2w.seg5.net.gz";
    if(std::filesystem::exists(model_file_name))
    {
        tipl::progress p("generating a mask using unet",true);
        auto unet = tipl::ml3d::unet3d::load_model<tipl::io::gz_mat_read>(model_file_name.c_str());
        if(unet.get())
        {
            tipl::filter::gaussian(b0[0]);
            if(unet->forward(b0[0],voxel.vs,p))
            {
                tipl::threshold(unet->get_mask(),voxel.mask,0.5f,1,0);
                tipl::morphology::defragment(voxel.mask);
                return true;
            }
            else
                error_msg =  "failed to process the b0 image";
        }
        else
            error_msg =  "failed to load unet model";
    }
    else
        error_msg =  "no applicable unet model for generating a mask";
    return false;
}


bool src_data::correct_distortion_by_t2w(const std::string& t2w_filename)
{
    std::string msg = " Susceptibility distortion was corrected by nonlinearly warping the b0 image to the T2-weighted image.";
    if(tipl::contains(voxel.report,msg))
        return true;

    if(!apply_mask)
    {
        if(!mask_from_template())
            return false;
    }


    dual_reg r;
    if(!r.load_template(0,t2w_filename))
    {
        error_msg = r.error_msg;
        return false;
    }
    if(r.Itvs[2] > r.Itvs[0]*1.1f)
    {
        tipl::out() << "nonisotropic image found: regrid images applied";
        if(!tipl::command<void,tipl::io::gz_nifti>(r.It[0],r.Itvs,r.ItR,r.It_is_mni,
                "regrid","1",true,error_msg))
            return false;
        r.Its = r.It[0].shape();
    }

    tipl::progress p("distortion correction using t2w image",true);
    tipl::filter::gaussian(r.It[0]);
    r.reg_type = tipl::reg::rigid_body;
    if(!warp_b0_to_image(r))
        return false;
    voxel.R2 = r.r[0];
    {
        std::vector<tipl::image<3,unsigned short> > this_new_dwi(src_dwi_data.size());
        std::vector<const unsigned short*> new_src_dwi_data(src_dwi_data.size());
        tipl::progress prog("warping");
        size_t p = 0;
        tipl::adaptive_par_for(src_dwi_data.size(),[&](unsigned int index)
        {
            if(prog.aborted())
                return;
            prog(p++,src_dwi_data.size());
            auto new_I = r.apply_warping<true,tipl::interpolation::cubic>(tipl::image<3>(dwi_at(index)));
            tipl::lower_threshold(new_I,0.0f);
            this_new_dwi[index] = new_I;
            new_src_dwi_data[index] = this_new_dwi[index].data();
        });
        voxel.mask = r.apply_warping<true,tipl::interpolation::majority>(voxel.mask);
        if(prog.aborted())
            return false;
        this_new_dwi.swap(new_dwi);
        new_src_dwi_data.swap(src_dwi_data);
    }
    voxel.dim = r.Its;
    voxel.vs = r.Itvs;
    voxel.trans_to_mni = r.ItR;
    calculate_dwi_sum(false);
    voxel.recon_report << msg;
    return true;
}

void src_data::remove(unsigned int index)
{
    if(index >= src_dwi_data.size())
        return;
    src_dwi_data.erase(src_dwi_data.begin()+index);
    src_bvalues.erase(src_bvalues.begin()+index);
    src_bvectors.erase(src_bvectors.begin()+index);
    std::string remove_text(" Reconstruction was conducted on a subset of DWI.");
    if(voxel.report.find(remove_text) == std::string::npos)
        voxel.report += remove_text;
}

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

std::vector<size_t> src_data::get_sorted_dwi_index(void)
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
void src_data::flip_b_table(const unsigned char* order)
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



extern std::vector<std::string> fib_template_list;
std::shared_ptr<fib_data> src_data::get_template_fib(tipl::affine_transform<float>& arg)
{
    std::shared_ptr<fib_data> template_fib(new fib_data);
    if(!template_fib->load_template_fib(voxel.template_id,voxel.vs[0]))
        return template_fib;
    {
        tipl::progress prog("registering to template");
        auto iso = template_fib->get_iso();
        tipl::reg::linear<tipl::out>(
               tipl::reg::make_list(template_image_pre(tipl::image<3>(iso))),template_fib->vs,
               tipl::reg::make_list(subject_image_pre(tipl::image<3>(dwi))),voxel.vs,arg,tipl::reg::affine,tipl::prog_aborted);
        if(prog.aborted())
        {
            template_fib.reset();
            return template_fib;
        }
        float r = tipl::correlation(iso,tipl::resample<tipl::interpolation::linear>(dwi,iso.shape(),
                    tipl::transformation_matrix<float>(arg,template_fib->dim,template_fib->vs,voxel.dim,voxel.vs)));
        tipl::out() << "linear r: " << r << std::endl;
        if(r < 0.6f)
            template_fib.reset();
    }
    return template_fib;
}
bool src_data::check_b_table(bool use_template)
{
    if(!new_dwi.empty())
    {
        error_msg = "cannot check b-table after aligning ac-pc or rotating image volume. Please make sure the b-table directions are correct before rotating image volume.";
        return false;
    }

    tipl::progress prog("checking b_table");

    // reconstruct DTI using original data and b-table
    {
        tipl::image<3,unsigned char> mask(voxel.mask.shape());
        mask = 1;
        mask.swap(voxel.mask);

        auto other_output = voxel.other_output;
        voxel.other_output = std::string();

        if(!reconstruct2<ReadDWIData,
                Dwi2Tensor>("checking b-table"))
            throw std::runtime_error("aborted");

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

    tipl::affine_transform<float> arg;
    std::shared_ptr<fib_data> template_fib;
    if(use_template)
        template_fib = get_template_fib(arg);
    if(template_fib.get())
    {
        voxel.recon_report <<
        " The accuracy of b-table orientation was examined by comparing fiber orientations with those of a population-averaged template (Yeh et al., Neuroimage 178, 57-68, 2018).";
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
            tipl::matrix<3,3,float> jacobian;
            tipl::rotation_matrix(arg.rotation,jacobian.begin(),tipl::vdim<3>());
            jacobian.inv();
            auto T = tipl::transformation_matrix<float>(arg,template_fib->dim,template_fib->vs,voxel.dim,voxel.vs);
            for(tipl::pixel_index<3> index(template_geo);index < template_geo.size();++index)
            {
                if(template_fib->dir.fa[0][index.index()] < 0.2f)
                    continue;
                tipl::vector<3> pos(index);
                T(pos);
                pos.round();
                if(subject_geo.is_valid(pos))
                {
                    auto sub_dir = new_dir[0][tipl::pixel_index<3>(pos.begin(),subject_geo).index()];
                    sub_dir.rotate(jacobian);
                    sum_cos += std::abs(double(sub_dir*template_fib->dir.get_fib(index.index(),0)));
                    ++ncount;
                }
            }
            result[i] = float(sum_cos/double(ncount));
        }
        else
        // for animal studies, use fiber coherence index
            result[i] = evaluate_fib(subject_geo,otsu,fib_fa,[&](uint32_t pos,uint8_t fib){return new_dir[fib][pos];});
    }
    long best = long(std::max_element(result,result+24)-result);
    tipl::out sp;
    for(int i = 0;i < 24;++i)
    {
        if(i == best)
            sp << (txt[i]+1) << "=BEST";
        else
            sp << (txt[i]+1) << "=-" << int(100.0f*(result[best]-result[i])/(result[best]+1.0f)) << "%";
        if(i % 8 == 7)
            sp << std::endl;
        else
            sp << ",";
    }

    if(result[best] > result[0])
    {
        sp << "b-table corrected by " << txt[best] << " for " << file_name << std::endl;
        flip_b_table(order[best]);
        voxel.load_from_src(*this);
        error_msg = "The b-table was corrected by flipping ";
        error_msg += txt[best];
        voxel.recon_report << " " << error_msg << ".";
    }
    else
    {
        error_msg = "The b-table orientation is correct.";
        fib_fa[0].swap(voxel.fib_fa);
        fib_dir[0].swap(voxel.fib_dir);
    }
    return true;
}

bool src_data::load_intro(const std::string& file_name)
{
    std::ifstream file(file_name);
    if(!file)
    {
        error_msg = "cannot open ";
        error_msg += file_name;
        return false;
    }
    {
        voxel.intro = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        tipl::out() << "read intro: " << std::string(voxel.intro.begin(),
                                                     voxel.intro.begin()+std::min<size_t>(voxel.intro.size(),64)) << "...";
    }
    return true;
}
size_t sum_dif(const unsigned short* I1,const unsigned short* I2,size_t size)
{
    size_t dif = 0;
    for(size_t i = 0;i < size;++i)
        dif += std::abs(int(I1[i])-int(I2[i]));
    return dif;
}

std::vector<std::pair<size_t,size_t> > src_data::get_bad_slices(void)
{
    std::vector<std::pair<size_t,size_t> > result;
    std::mutex result_mutex;
    tipl::adaptive_par_for(src_dwi_data.size(),[&](size_t index)
    {        
        for(size_t z = 1,pos = voxel.dim.plane_size();z + 1 < size_t(voxel.dim.depth());++z,pos += voxel.dim.plane_size())
        {
            auto dif_lower = sum_dif(src_dwi_data[index] + pos-voxel.dim.plane_size(),
                                     src_dwi_data[index] + pos,voxel.dim.plane_size());
            auto dif_upper = sum_dif(src_dwi_data[index] + pos+voxel.dim.plane_size(),
                                     src_dwi_data[index] + pos,voxel.dim.plane_size());
            auto dif_upper_lower = sum_dif(src_dwi_data[index] + pos+voxel.dim.plane_size(),
                                      src_dwi_data[index] + pos-voxel.dim.plane_size(),voxel.dim.plane_size());
            if(dif_lower + dif_upper > dif_upper_lower*2)
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result.push_back(std::make_pair(index,z));
            }
        }
    });
    return result;
}

float masked_correlation(const unsigned short* I1_ptr,const unsigned short* I2_ptr,tipl::image<3,unsigned char>& mask)
{
    std::vector<float> I1,I2;
    I1.reserve(mask.size());
    I2.reserve(mask.size());
    for(size_t i = 0;i < mask.size();++i)
        if(mask[i])
        {
            I1.push_back(I1_ptr[i]);
            I2.push_back(I2_ptr[i]);
        }
    return float(tipl::correlation(I1.begin(),I1.end(),I2.begin()));
}
std::pair<float,float> src_data::quality_control_neighboring_dwi_corr(void)
{
    std::vector<std::pair<size_t,size_t> > corr_pairs;
    for(size_t i = 0;i < src_bvalues.size();++i)
    {
        if(src_bvalues[i] == 0.0f)
            continue;
        float min_dis = std::numeric_limits<float>::max();
        size_t min_j = 0;
        for(size_t j = 0;j < src_bvalues.size();++j)
        {
            if(i == j || src_bvalues[j] == 0.0f)
                continue;
            tipl::vector<3> v1(src_bvectors[i]),v2(src_bvectors[j]);
            v1 *= std::sqrt(src_bvalues[i]);
            v2 *= std::sqrt(src_bvalues[j]);
            if(v1 == v2)
                continue;
            float dis = std::min<float>(float((v1-v2).length()),
                                        float((v1+v2).length()));
            if(dis < min_dis)
            {
                min_dis = dis;
                min_j = j;
            }
        }
        if(i > min_j)
            corr_pairs.push_back(std::make_pair(i,min_j));
    }
    std::vector<float> masked_ndc(corr_pairs.size()),ndc(corr_pairs.size());
    for(size_t index = 0;index < corr_pairs.size();++index)
    {
        size_t i1 = corr_pairs[index].first;
        size_t i2 = corr_pairs[index].second;
        ndc[index] = float(tipl::correlation(src_dwi_data[i1],src_dwi_data[i1]+voxel.dim.size(),src_dwi_data[i2]));
        masked_ndc[index] = masked_correlation(src_dwi_data[i1],src_dwi_data[i2],voxel.mask);
    }
    return std::make_pair(tipl::mean(ndc),tipl::mean(masked_ndc));
}

float src_data::dwi_contrast(void)
{
    std::vector<size_t> dwi_self,dwi_neighbor,dwi_ortho;
    for(size_t i = 0;i < src_bvalues.size();++i)
    {
        if(src_bvalues[i] == 0.0f)
            continue;
        float min_dis1 = std::numeric_limits<float>::max();
        size_t min_j1 = 0;
        float min_dis2 = std::numeric_limits<float>::max();
        size_t min_j2 = 0;

        tipl::vector<3> v1(src_bvectors[i]);
        v1 *= std::sqrt(src_bvalues[i]);
        auto v1_length2 = v1.length2();
        auto v1_length = std::sqrt(v1_length2);

        for(size_t j = 0;j < src_bvalues.size();++j)
        {
            if(i == j || src_bvalues[j] == 0.0f)
                continue;
            tipl::vector<3> v2(src_bvectors[j]);
            v2 *= std::sqrt(src_bvalues[j]);
            // looking for the neighboring DWI
            if(v1 == v2)
                continue;
            float dis1 = std::min<float>(float((v1-v2).length()),
                                        float((v1+v2).length()));
            if(dis1 < min_dis1)
            {
                min_dis1 = dis1;
                min_j1 = j;
            }

            // looking for the contrast DWI
            auto p1 = v1;
            p1 *= v1*v2;
            p1 /= v1_length2;
            auto p2 = v2;
            p2 -= p1;
            p2 *= v1_length/p2.length();
            float dis2 = (v2-p2).length();
            if(dis2 < min_dis2)
            {
                min_dis2 = dis2;
                min_j2 = j;
            }
        }
        dwi_self.push_back(i);
        dwi_neighbor.push_back(min_j1);
        dwi_ortho.push_back(min_j2);
    }
    std::vector<float> ndc(dwi_self.size()),odc(dwi_self.size());
    tipl::adaptive_par_for(dwi_self.size(),[&](size_t index)
    {
        ndc[index] = masked_correlation(src_dwi_data[dwi_self[index]],src_dwi_data[dwi_neighbor[index]],voxel.mask);
        odc[index] = masked_correlation(src_dwi_data[dwi_self[index]],src_dwi_data[dwi_ortho[index]],voxel.mask);
    });
    return tipl::mean(ndc)/tipl::mean(odc);

}
bool is_human_size(tipl::shape<3> dim,tipl::vector<3> vs);
bool src_data::is_human_data(void) const
{
    return is_human_size(voxel.dim,voxel.vs);
}

int64_t src_data::bottom_top_difference(void)
{
    size_t size = dwi.plane_size()*std::min(3,dwi.height()/2);
    return std::accumulate(dwi.begin(),dwi.begin()+size,int64_t(0)) - std::accumulate(dwi.end()-size,dwi.end(),int64_t(0));
}
int64_t src_data::anterior_posterior_difference(void)
{
    tipl::shape<3> range_min,range_max;
    tipl::bounding_box(voxel.mask,range_min,range_max);
    auto I = voxel.mask;
    tipl::crop(I,range_min,range_max);
    size_t anterior_sum = 0;
    size_t posterior_sum = 0;
    for(tipl::pixel_index<3> pos(I.shape());pos < I.size();++pos)
    {
        if(!I[pos.index()])
            continue;
        if(pos.y() < (I.height() >> 1))
            ++anterior_sum;
        if(pos.y() > (I.height() >> 1))
            ++posterior_sum;
    }
    return posterior_sum - anterior_sum;
}
void src_data::correction_axis(void)
{
    int long_axis_dir = long_axis_direction();
    tipl::out() << "long axis direction: " << long_axis_dir;
    size_t op_count = 0;
    if(long_axis_dir == 0)
    {
        command("[Step T2][Edit][Image swap xy]");
        ++op_count;
    }
    else
        if(long_axis_dir == 2)
        {
            command("[Step T2][Edit][Image swap yz]");
            ++op_count;
        }

    int sym_axis_dir = symmetric_axis_direction();
    tipl::out() << "symmetric axis direction: " << int(sym_axis_dir);
    if(sym_axis_dir == 1)
    {
        command("[Step T2][Edit][Image swap xy]");
        ++op_count;
    }
    else
        if(sym_axis_dir == 2)
        {
            command("[Step T2][Edit][Image swap xz]");
            ++op_count;
        }

    if(!op_count)
        return;

    int64_t bottom_top_dif = bottom_top_difference();
    tipl::out() << "bottom and top slices difference: " << bottom_top_dif;
    if(bottom_top_dif < 0)
    {
        command("[Step T2][Edit][Image flip z]");
        ++op_count;
    }
    int64_t anterior_posterior_dif = anterior_posterior_difference();
    tipl::out() << "anterior and postieror mask difference: " << anterior_posterior_dif;
    if(anterior_posterior_dif < 0)
    {
        command("[Step T2][Edit][Image flip y]");
        ++op_count;
    }

    if(op_count & 1)
        command("[Step T2][Edit][Image flip x]");
}
bool src_data::run_steps(const std::string& reg_file_name,const std::string& ref_steps)
{
    std::istringstream in(ref_steps);
    std::string step;
    std::vector<std::string> cmds,params;
    while(std::getline(in,step))
    {
        if(step.empty())
            continue;
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
        if((tipl::ends_with(param,".sz") || tipl::ends_with(param,".gz")) &&
           !tipl::match_files(reg_file_name,param,file_name,param))
        {
            error_msg = step;
            error_msg += " cannot find a matched file for ";
            error_msg += file_name;
            return false;
        }
        cmds.push_back(cmd);
        params.push_back(param);
    }

    if(!cmds.empty())
    {
        tipl::progress prog("apply operations");
        for(size_t index = 0;prog(index,cmds.size());++index)
            if(!command(cmds[index],params[index]))
            {
                error_msg +=  "at ";
                error_msg += cmds[index];
                return false;
            }
    }
    return true;
}
bool src_data::command(std::string cmd,std::string param)
{
    if(cmd == "[Step T2][Reconstruction]")
        return true;
    tipl::progress prog_(cmd.c_str());
    if(!param.empty())
        tipl::out() << "param: " << param << std::endl;
    if(cmd == "[Step T2][File][Save Src File]" || cmd == "[Step T2][File][Save 4D NIFTI]")
    {
        if(param.empty())
        {
            error_msg = " please assign file name ";
            return false;
        }
        tipl::progress prog_("saving ",std::filesystem::path(param).filename().u8string().c_str());
        return save_to_file(param.c_str());
    }
    if(cmd == "[Step T2][File][Save B0]")
    {
        if(param.empty())
        {
            error_msg = " please assign file name ";
            return false;
        }
        tipl::progress prog_("saving ",std::filesystem::path(param).filename().u8string().c_str());
        return save_b0_to_nii(param.c_str());
    }
    if(cmd == "[Step T2][File][Save DWI Sum]")
    {
        if(param.empty())
        {
            error_msg = " please assign file name ";
            return false;
        }
        tipl::progress prog_("saving ",std::filesystem::path(param).filename().u8string().c_str());
        return save_dwi_sum_to_nii(param.c_str());
    }

    if(cmd == "[Step T2a][Open]")
    {
        if(!std::filesystem::exists(param))
        {
            error_msg = param;
            error_msg += " does not exist";
            return false;
        }
        ROIRegion region(voxel.dim,voxel.vs);
        region.load_region_from_file(param.c_str());
        region.save_region_to_buffer(voxel.mask);
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
            tipl::morphology::dilation(voxel.mask.slice_at(0));
        else
            tipl::morphology::dilation(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Unet]")
    {
        if(!mask_from_unet())
            return false;
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
    if(cmd == "[Step T2a][Slice Defragment]")
    {
        tipl::morphology::defragment_slice(voxel.mask);
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
    if(cmd == "[Step T2a][Fit]")
    {
        tipl::morphology::fit(voxel.mask,dwi);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Negate]")
    {
        tipl::morphology::negate(voxel.mask);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2a][Template]")
    {
        mask_from_template();
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
                    QInputDialog::getInt(nullptr,QApplication::applicationName(),"Please assign the threshold",
                                                         int(tipl::segmentation::otsu_threshold(dwi)),
                                                         0,255,10,&ok)
                    :QInputDialog::getInt(nullptr,QApplication::applicationName(),"Please assign the threshold");
            if (!ok)
                return true;
        }
        else
            threshold = std::stoi(param);
        tipl::threshold(dwi,voxel.mask,uint8_t(threshold));
        voxel.steps += cmd + "=" + std::to_string(threshold) + "\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Probablistic Masking]")
    {
        if(param.empty())
        {
            error_msg = " please assign file name ";
            return false;
        }
        tipl::io::gz_nifti in;
        if(!in.load_from_file(param))
        {
            error_msg = in.error_msg;
            return false;
        }
        tipl::image<3> prob;
        in >> prob;
        if(prob.shape() != dwi.shape())
        {
            error_msg = "mask has a different image dimension";
            return false;
        }
        tipl::adaptive_par_for(src_dwi_data.size(),[&](size_t index)
        {
            unsigned short* buf = const_cast<unsigned short*>(src_dwi_data[index]);
            for(size_t i = 0;i < prob.size();++i)
                buf[i] *= prob[i];
        });
        tipl::threshold(prob,voxel.mask,0.0f);
        calculate_dwi_sum(false);
        voxel.steps += cmd+"="+param+"\n";
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
        apply_mask = true;
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Overwrite Voxel Size]")
    {
        std::istringstream in(param);
        in >> voxel.vs[0] >> voxel.vs[1] >> voxel.vs[2];
        voxel.report = get_report();
        voxel.steps += cmd+"="+param+"\n";
        return true;
    }
    if(cmd == "[Step T2][Edit][Smooth Signals]")
    {
        smoothing();
        voxel.steps += cmd+"\n";
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
    if(cmd == "[Step T2][Edit][Align ACPC]")
    {
        if(!align_acpc(param.empty() ? voxel.vs[0] : std::stof(param)))
            return false;
        voxel.steps += param.empty() ? (cmd+"\n") : (cmd+"="+param+"\n");
        return true;
    }
    // correct for b-table orientation
    if(cmd == "[Step T2][B-table][Check B-table]")
    {
        if(!check_b_table(true))
            return false;
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][B-table][Check B-table2]")
    {
        if(!check_b_table(false))
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
    if(cmd == "[Step T2][B-table][swap bxby]")
    {
        for(size_t i = 0;i < src_bvectors.size();++i)
            std::swap(src_bvectors[i][0],src_bvectors[i][1]);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][B-table][swap bybz]")
    {
        for(size_t i = 0;i < src_bvectors.size();++i)
            std::swap(src_bvectors[i][1],src_bvectors[i][2]);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][B-table][swap bxbz]")
    {
        for(size_t i = 0;i < src_bvectors.size();++i)
            std::swap(src_bvectors[i][0],src_bvectors[i][2]);
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Corrections][TOPUP]")
    {
        if(!load_existing_corrections())
        {
            if(!get_rev_pe(param) || !run_topup() || !run_applytopup())
                return false;
        }
        voxel.steps += param.empty() ? (cmd+"\n") : (cmd+"="+param+"\n");
        return true;
    }
    if(cmd == "[Step T2][Corrections][TOPUP EDDY]")
    {
        if(!load_existing_corrections())
        {
            if(get_rev_pe(param))
            {
                if(!run_topup())
                    return false;
            }
            else
            {
                tipl::warning() << error_msg;
                tipl::warning() << "skip topup";
            }
            if(!run_eddy())
                return false;
        }
        voxel.steps += param.empty() ? (cmd+"\n") : (cmd+"="+param+"\n");
        return true;
    }
    if(cmd == "[Step T2][Corrections][EDDY]")
    {
        if(!load_existing_corrections())
        {
            if(!run_eddy())
                return false;
        }
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Corrections][Motion Correction]")
    {
        if(!correct_motion())
            return false;
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2][Corrections][Bias Field]")
    {
        if(!correct_bias_field())
            return false;
        voxel.steps += cmd+"\n";
        return true;
    }

    if(cmd == "[Step T2][Corrections][By T2w]")
    {
        if(!correct_distortion_by_t2w(param))
            return false;
        voxel.steps += cmd+"="+param+"\n";
        return true;
    }
    if(cmd == "[Step T2][Corrections][Volume Orientation Correction]")
    {
        correction_axis();
        voxel.steps += cmd+"\n";
        return true;
    }
    if(cmd == "[Step T2b(2)][Partial FOV]")
    {
        std::istringstream in(param);
        in >> voxel.partial_min >> voxel.partial_max;
        return true;
    }
    error_msg = "unknown command: ";
    error_msg += cmd;
    return false;
}

// 0: x  1: y  2: z
// 3: xy 4: yz 5: xz
void src_data::flip_dwi(unsigned char type)
{
    for(unsigned int index = 0;index < src_bvectors.size();++index)
    {
        if(type < 3)
            src_bvectors[index][type] = -src_bvectors[index][type];
        else
            std::swap(src_bvectors[index][type-3],src_bvectors[index][(type-2)%3]);
    }
    if(type >=3 )
        std::swap(voxel.vs[type-3],voxel.vs[(type-2)%3]);
    tipl::flip(dwi,type);
    tipl::flip(voxel.mask,type);
    tipl::progress prog("flip image");
    if(voxel.is_histology)
        tipl::flip(voxel.hist_image,type);
    else
    {
        size_t p = 0;
        tipl::adaptive_par_for(src_dwi_data.size(),[&](unsigned int index)
        {
            prog(p++,src_dwi_data.size());
            tipl::flip(dwi_at(index),type);
        });
    }
    voxel.dim = voxel.mask.shape();
}

tipl::matrix<3,3,float> get_inv_rotation(const Voxel& voxel,const tipl::transformation_matrix<double>& T)
{
    auto iT = T;
    iT.inverse();
    tipl::affine_transform<double> arg(iT.to_affine_transform(voxel.dim,voxel.vs,voxel.dim,voxel.vs));
    tipl::matrix<3,3,float> r;
    tipl::rotation_matrix(arg.rotation,r.begin(),tipl::vdim<3>());
    return r;
}
// used in eddy correction for each dwi
void src_data::rotate_one_dwi(unsigned int dwi_index,const tipl::transformation_matrix<double>& T)
{
    tipl::image<3> tmp(voxel.dim);
    tipl::resample<tipl::interpolation::cubic>(dwi_at(dwi_index),tmp,T);
    tipl::lower_threshold(tmp,0.0f);
    std::copy(tmp.begin(),tmp.end(),dwi_at(dwi_index).begin());
    // rotate b-table
    src_bvectors[dwi_index].rotate(get_inv_rotation(voxel,T));
    src_bvectors[dwi_index].normalize();
}

void src_data::rotate(const tipl::shape<3>& new_geo,
                        const tipl::vector<3>& new_vs,
                        const tipl::transformation_matrix<double>& T)
{
    std::vector<tipl::image<3,unsigned short> > rotated_dwi(src_dwi_data.size());
    std::vector<const unsigned short*> new_src_dwi_data(src_dwi_data.size());
    tipl::progress prog("rotating");
    size_t p = 0;
    tipl::adaptive_par_for(new_src_dwi_data.size(),[&](unsigned int index)
    {
        if(prog.aborted())
            return;
        prog(p++,src_dwi_data.size());
        rotated_dwi[index].resize(new_geo);
        tipl::resample<tipl::interpolation::cubic>(dwi_at(index),rotated_dwi[index],T);
        new_src_dwi_data[index] = rotated_dwi[index].data();
    });
    if(prog.aborted())
        return;
    rotated_dwi.swap(new_dwi);
    new_src_dwi_data.swap(src_dwi_data);

    tipl::matrix<3,3,float> r = get_inv_rotation(voxel,T);
    for (auto& vec : src_bvectors)
        {
            vec.rotate(r);
            vec.normalize();
        }
    voxel.dim = new_geo;
    voxel.vs = new_vs;
    auto trans = T.to_matrix();
    trans *= voxel.trans_to_mni;
    voxel.trans_to_mni = trans;

    // rotate the mask
    tipl::image<3,unsigned char> mask(voxel.dim);
    tipl::resample<tipl::interpolation::majority>(voxel.mask,mask,T);
    mask.swap(voxel.mask);

    calculate_dwi_sum(false);

}
void src_data::resample(float nv)
{
    if(voxel.vs[0] == nv &&
       voxel.vs[1] == nv &&
       voxel.vs[2] == nv)
        return;
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
    std::ostringstream out;
    out << " The images were resampled to " << std::fixed << std::setprecision(2) << nv << " mm isotropic resolution.";
    voxel.report += out.str();
}
void src_data::smoothing(void)
{
    size_t p = 0;
    tipl::progress prog("smoothing");
    tipl::adaptive_par_for(src_dwi_data.size(),[&](unsigned int index)
    {
        prog(p++,src_dwi_data.size());
        tipl::filter::gaussian(dwi_at(index));
    });
    calculate_dwi_sum(false);
}

bool src_data::add_other_image(const std::string& name,const std::string& filename)
{
    if(tipl::begins_with(filename,"http"))
    {
        voxel.other_image.push_back(tipl::image<3>());
        voxel.other_image_name.push_back(filename);
        voxel.other_image_trans.push_back(tipl::transformation_matrix<float>());
        voxel.other_image_voxel_size.push_back(tipl::vector<3>());
        return true;
    }
    tipl::progress prog("add other images");
    tipl::image<3> ref;
    tipl::vector<3> vs;
    tipl::transformation_matrix<float> trans;

    tipl::io::gz_nifti in;
    if(!in.load_from_file(filename.c_str()) || !in.toLPS(ref))
    {
        error_msg = "not a valid nifti file ";
        error_msg += filename;
        return false;
    }
    in.get_voxel_size(vs);

    tipl::out() << "add " << filename << " as " << name;

    bool has_registered = false;
    for(unsigned int index = 0;index < voxel.other_image.size();++index)
        if(ref.shape() == voxel.other_image[index].shape())
        {
            trans = voxel.other_image_trans[index];
            has_registered = true;
        }
    if(!has_registered && ref.shape() != voxel.dim)
    {
        tipl::out() << " and register image with DWI." << std::endl;
        trans = tipl::reg::linear<tipl::out>(
                        tipl::reg::make_list(subject_image_pre(tipl::image<3>(ref))),vs,
                        tipl::reg::make_list(subject_image_pre(tipl::image<3>(dwi))),voxel.vs,tipl::reg::rigid_body,tipl::prog_aborted);
    }
    else {
        if(has_registered)
            tipl::out() << " using previous registration." << std::endl;
        else
            tipl::out() << " treated as DWI space images." << std::endl;
    }
    if(name == "reg")
    {
        voxel.other_modality_subject.swap(ref);
        voxel.other_modality_trans = trans;
        voxel.other_modality_vs = vs;
    }
    else
    {
        voxel.other_image.push_back(std::move(ref));
        voxel.other_image_name.push_back(name);
        trans.inverse();
        voxel.other_image_trans.push_back(trans);
        voxel.other_image_voxel_size.push_back(vs);
    }
    return true;
}
extern std::vector<std::string> fa_template_list,iso_template_list;
void match_template_resolution(tipl::image<3>& VG,
                               tipl::vector<3>& VGvs,
                               tipl::image<3>& VF,
                               tipl::vector<3>& VFvs);
bool src_data::align_acpc(float reso)
{
    tipl::progress prog("align acpc",true);
    std::string msg = " The diffusion MRI data were rotated to align with the AC-PC line";
    if(voxel.report.find(msg) != std::string::npos)
    {
        tipl::out() << (error_msg = "image already aligned");
        return false;
    }
    std::ostringstream out;
    out << msg << " at an isotropic resolution of " << reso << " (mm).";
    tipl::out() << "align ap-pc" << std::endl;

    tipl::image<3> I,J;
    tipl::vector<3> Ivs,Jvs(reso,reso,reso);

    // prepare template images
    if(!tipl::io::gz_nifti::load_from_file(iso_template_list[voxel.template_id].c_str(),I,Ivs) && !
        tipl::io::gz_nifti::load_from_file(fa_template_list[voxel.template_id].c_str(),I,Ivs))
    {
        error_msg = "Failed to load/find MNI template.";
        return false;
    }
    if(reso < Ivs[0])
    {
        tipl::out() << (error_msg = "invalid resolution");
        return false;
    }

    // create an isotropic subject image for alignment
    tipl::scale(dwi,J,tipl::v(voxel.vs[0]/reso,voxel.vs[1]/reso,voxel.vs[2]/reso));



    match_template_resolution(I,Ivs,J,Jvs);

    prog(0,3);
    tipl::affine_transform<float> arg;
    {
        tipl::progress prog("linear registration");
        tipl::reg::linear<tipl::out>(
                    tipl::reg::make_list(template_image_pre(tipl::image<3>(I))),Ivs,
                    tipl::reg::make_list(subject_image_pre(tipl::image<3>(J))),Jvs,arg,tipl::reg::rigid_scaling,tipl::prog_aborted);
        if(prog.aborted())
            return false;
    }

    tipl::out() << arg << std::endl;
    prog(1,3);
    float r = tipl::correlation(I,tipl::resample<tipl::interpolation::cubic>(J,I.shape(),
                                  tipl::transformation_matrix<float>(arg,I.shape(),Ivs,J.shape(),Jvs)));
    tipl::out() << "R2 for ac-pc alignment: " << r*r << std::endl;
    prog(2,3);
    if(r*r < 0.3f)
    {
        error_msg = "Failed to align subject data to template.";
        return false;
    }
    arg.scaling[0] = 1.0f;
    arg.scaling[1] = 1.0f;
    arg.scaling[2] = 1.0f;

    tipl::shape<3> new_geo;
    new_geo[0] = I.width()*Ivs[0]/reso;
    new_geo[1] = I.height()*Ivs[1]/reso;
    new_geo[2] = I.depth()*Ivs[2]/reso;
    auto T = tipl::transformation_matrix<float>(arg,new_geo,tipl::v(reso,reso,reso),J.shape(),Jvs);

    // handle non isotropic resolution
    if(reso != voxel.vs[0] || reso != voxel.vs[1] || reso != voxel.vs[2])
    {
        tipl::transformation_matrix<double> T2;
        T2.sr[0] = double(reso/voxel.vs[0]);
        T2.sr[4] = double(reso/voxel.vs[1]);
        T2.sr[8] = double(reso/voxel.vs[2]);
        T.accumulate(T2);
    }
    rotate(new_geo,tipl::v(reso,reso,reso),T);
    voxel.report += out.str();
    return true;
}


template<typename T>
inline T B3_sym(T u)
{
    u = std::abs(u);
    if (u < T(1)) return (T(4) - T(6)*u*u + T(3)*u*u*u) / T(6);
    if (u < T(2)) {T v = T(2) - u;return (v*v*v) / T(6);}
    return T(0);
}
void correct_bias_field(tipl::image<3> I,
                        const tipl::image<3,unsigned char>& mask,
                        tipl::image<3>& log_bias_field,
                        const tipl::vector<3>& spacing)
{
    constexpr int spline_range = 3;
    constexpr int max_iters = 50;
    constexpr float tol = 1e-4f;

    std::vector<size_t> position;     // nearest control‐point grid index
    std::vector<double> logI;

    if(!log_bias_field.empty())
        for(size_t i = 0;i < mask.size();++i)
            if(mask[i])
                I[i] *= std::exp(-log_bias_field[i]);

    {
        auto otsu = tipl::segmentation::otsu_threshold_sharpening(I);
        double sum_signal = 0.0;
        size_t sum_signal_count = 0;
        for(size_t i = 0;i < mask.size();++i)
            if(mask[i])
            {
                position.push_back(i);
                if(I[i] <= otsu)
                    logI.push_back(std::numeric_limits<double>::max());
                else
                {
                    logI.push_back(std::log(I[i] + 1e-6f));
                    sum_signal += logI.back();
                    ++sum_signal_count;
                }
            }
        tipl::minus_constant(logI.begin(),logI.end(),sum_signal/sum_signal_count);
    }


    // 1) Compute control‐point grid size exactly as before
    tipl::shape<3> c_shape(int(std::ceil(1.0f/spacing[0])) + spline_range + spline_range,
                           int(std::ceil(1.0f/spacing[1])) + spline_range + spline_range,
                           int(std::ceil(1.0f/spacing[2])) + spline_range + spline_range);
    // 2) Precompute B3‐weights per sample
    std::vector<std::vector<std::pair<size_t,float>>> basis(position.size());
    tipl::vector<3> scale(1.0f/spacing[0]/float(I.width()-1),1.0f/spacing[1]/float(I.height()-1),1.0f/spacing[2]/float(I.depth()-1));
    tipl::par_for(position.size(), [&](size_t i)
    {
        tipl::pixel_index<3> pos(position[i],mask.shape());
        // normalized continuous coords
        float fx = pos[0]*scale[0] + spline_range,fy = pos[1]*scale[1] + spline_range,fz = pos[2]*scale[2] + spline_range;
        // nearest control‐point grid index
        int cx = int(std::floor(fx)),cy = int(std::floor(fy)),cz = int(std::floor(fz));
        auto& b = basis[i];
        double sumw = 0.0;
        for (int iz = cz - spline_range; iz < cz + spline_range; ++iz)
        {
            float wz = B3_sym(fz - iz);
            for (int iy = cy - spline_range; iy < cy + spline_range; ++iy)
            {
                float wyz = B3_sym(fy - iy)*wz;
                for (int ix = cx - spline_range; ix < cx + spline_range; ++ix)
                {
                    float wxyz = B3_sym(fx - ix) * wyz;
                    if(wxyz == 0.0f)
                        continue;
                    sumw += wxyz;
                    b.emplace_back(tipl::voxel2index(ix,iy,iz,c_shape), wxyz);
                }
            }
        }
        if (sumw > 0.0)
            for (auto& each : b)
                each.second /= sumw;
    });
    std::vector<double> ATA(c_shape.size()*c_shape.size());
    for (auto& each : basis)
        for (size_t i = 0,n = each.size(); i < n; ++i)
        {
            auto i_value = each[i].second;
            auto* row_ptr = &ATA[each[i].first * c_shape.size()];
            for (size_t j = i; j < n; ++j)
                row_ptr[each[j].first] += i_value * each[j].second;
        }
    for (int j = 0; j < c_shape.size(); ++j)
        ATA[j*c_shape.size() + j] += 1e-3f;// regularize
    std::vector<double> piv(c_shape.size());
    std::initializer_list<size_t> dim{c_shape.size(), c_shape.size()};
    if (!tipl::mat::ll_decomposition(ATA.begin(), piv.begin(), dim))
        return;

    tipl::image<3,double> cc_img(c_shape);   // control‐point coefficients
    std::vector<float> correction(position.size());
    double prev_rms = std::numeric_limits<double>::max();
    for (int iter = 0; iter < max_iters; ++iter)
    {
        // a) residuals
        std::vector<double> rhs(c_shape.size());
        for(size_t i = 0;i < logI.size();++i)
        {
            if(logI[i] == std::numeric_limits<double>::max())
                continue;
            auto resid = logI[i] - correction[i];
            for (auto& tpl : basis[i])
                rhs[tpl.first] += tpl.second * resid;
        }
        // c) solve via LL
        tipl::mat::ll_solve(ATA.begin(), piv.begin(), rhs.begin(), cc_img.begin(), dim);
        // d) update correction & RMS
        std::vector<double> each_sumsq(tipl::max_thread_count),each_count(tipl::max_thread_count);
        tipl::par_for<tipl::sequential_with_id>(correction.size(),[&](size_t i,size_t id)
        {
            double d = 0.0;
            for(const auto& each : basis[i])
                d += double(cc_img[each.first])*each.second;
            correction[i] += float(d);
            each_sumsq[id] += d*d;
            ++each_count[id];
        });
        double rms = std::sqrt(tipl::sum(each_sumsq.begin(),each_sumsq.end()) / double(tipl::sum(each_count.begin(),each_count.end())));
        if (std::abs(prev_rms - rms) < tol)
            break;
        prev_rms = rms;

    }
    log_bias_field.resize(mask.shape());
    for(size_t i = 0;i < position.size();++i)
        log_bias_field[position[i]] += correction[i];

}
bool src_data::has_bias_field_correction(void) const
{
    return tipl::contains(voxel.report,"bias field");
}
bool src_data::correct_bias_field(void)
{
    if(has_bias_field_correction())
    {
        tipl::warning() << "bias field correction has been previously applied";
        return true;
    }
    tipl::image<3>  bias_field;
    {
        tipl::progress prog("estimate bias field",true);
        for(size_t i = 0;prog(i,1);++i)
            ::correct_bias_field(dwi,voxel.mask,bias_field,tipl::vector<3>(1.0f,voxel.vs[0]/voxel.vs[1],voxel.vs[0]/voxel.vs[2]));
        if(prog.aborted())
            return false;
    }
    {
        tipl::progress prog("apply correction");
        for(auto& each : bias_field)
            each = std::exp(-each);
        tipl::par_for(src_dwi_data.size(),[&](unsigned int index)
        {
            dwi_at(index) *= bias_field;
        });
        calculate_dwi_sum(true);
        voxel.report += " The bias field was corrected using b0 image.";
    }
    return true;
}
extern bool has_cuda;
extern int gpu_count;
bool src_data::correct_motion(void)
{
    std::string msg = " Motion correction and eddy current correction was conducted with b-table rotated.";
    if(voxel.report.find(msg) != std::string::npos)
        return true;
    std::vector<tipl::affine_transform<float> > args(src_bvalues.size());
    {
        tipl::progress prog("apply motion correction...");
        unsigned int p = 0;
        auto I0 = tipl::image<3>(dwi_at(0));
        tipl::filter::gaussian(I0);
        tipl::filter::gaussian(I0);

        tipl::adaptive_par_for(src_bvalues.size(),[&](int i)
        {
            prog(++p,src_bvalues.size());
            if(prog.aborted() || !i)
                return;
            args[i] = args[i-1];
            auto Ii = tipl::image<3>(dwi_at(i));
            tipl::filter::gaussian(Ii);
            tipl::filter::gaussian(Ii);
            tipl::reg::linear_refine<tipl::out>(
                        tipl::reg::make_list(subject_image_pre(tipl::image<3>(I0))),voxel.vs,
                        tipl::reg::make_list(subject_image_pre(std::move(Ii))),voxel.vs,args[i],tipl::reg::rigid_body,tipl::prog_aborted);
            tipl::out() << "dwi (" << i+1 << "/" << src_bvalues.size() << ")" <<
                         " shift=" << tipl::vector<3>(args[i].translocation) <<
                         " rotation=" << tipl::vector<3>(args[i].rotation) << std::endl;
        });
        if(prog.aborted())
        {
            error_msg = "aborted";
            return false;
        }
    }


    // get ndc list
    std::vector<tipl::affine_transform<float> > new_args(args);

    {
        tipl::progress prog("estimate and registering...");
        unsigned int p = 0;
        tipl::adaptive_par_for(src_bvalues.size(),[&](int i)
        {
            prog(++p,src_bvalues.size());
            if(prog.aborted() || !i)
                return;
            // get the minimum q space distance
            float min_dis = std::numeric_limits<float>::max();
            std::vector<float> dis_list(src_bvalues.size());
            for(size_t j = 0;j < src_bvalues.size();++j)
            {
                if(j == i)
                    continue;
                tipl::vector<3> v1(src_bvectors[i]),v2(src_bvectors[j]);
                v1 *= std::sqrt(src_bvalues[i]);
                v2 *= std::sqrt(src_bvalues[j]);
                float dis = std::min<float>(float((v1-v2).length()),
                                            float((v1+v2).length()));
                dis_list[j] = dis;
                if(dis < min_dis)
                    min_dis = dis;
            }
            min_dis *= 1.5f;
            tipl::image<3> from(dwi.shape());
            for(size_t j = 0;j < dis_list.size();++j)
            {
                if(j == i)
                    continue;
                if(dis_list[j] <= min_dis)
                {
                    tipl::image<3> from_(dwi.shape());
                    tipl::resample<tipl::interpolation::cubic>(dwi_at(j),from_,
                        tipl::transformation_matrix<float>(args[j],voxel.dim,voxel.vs,voxel.dim,voxel.vs));
                    from += from_;
                }
            }
            tipl::filter::gaussian(from);
            tipl::filter::gaussian(from);
            auto Ii = tipl::image<3>(dwi_at(i));
            tipl::filter::gaussian(Ii);
            tipl::filter::gaussian(Ii);

            tipl::reg::linear_refine<tipl::out>(
                        tipl::reg::make_list(subject_image_pre(std::move(from))),voxel.vs,
                        tipl::reg::make_list(subject_image_pre(std::move(Ii))),voxel.vs,new_args[i],tipl::reg::rigid_body,tipl::prog_aborted,tipl::reg::corr);
            tipl::out() << "dwi (" << i+1 << "/" << src_bvalues.size() << ") = "
                      << " shift=" << tipl::vector<3>(new_args[i].translocation)
                      << " rotation=" << tipl::vector<3>(new_args[i].rotation) << std::endl;

        });

        if(prog.aborted())
        {
            error_msg = "aborted";
            return false;
        }
    }

    {
        tipl::progress prog("estimate and registering...");
        size_t total = 0;
        tipl::adaptive_par_for(src_bvalues.size(),[&](size_t i)
        {
            prog(total++,src_bvalues.size());
            rotate_one_dwi(i,tipl::transformation_matrix<float>(new_args[i],voxel.dim,voxel.vs,voxel.dim,voxel.vs));
        });
    }
    voxel.report += msg;
    return true;
}

void src_data::crop(tipl::shape<3> range_min,tipl::shape<3> range_max)
{
    tipl::progress prog("Removing background region");
    size_t p = 0;
    tipl::out() << "from: " << range_min << " to: " << range_max << std::endl;
    tipl::adaptive_par_for(src_dwi_data.size(),[&](unsigned int index)
    {
        prog(p++,src_dwi_data.size());
        tipl::image<3,unsigned short> I0;
        tipl::crop(dwi_at(index),I0,range_min,range_max);
        std::copy(I0.begin(),I0.end(),dwi_at(index).begin());
    });
    tipl::crop(voxel.mask,range_min,range_max);
    tipl::crop(dwi,range_min,range_max);
    voxel.dim = voxel.mask.shape();
}
void src_data::trim(void)
{
    tipl::shape<3> range_min,range_max;
    tipl::bounding_box(voxel.mask,range_min,range_max);
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
    tipl::adaptive_par_for(v1.depth()*h,[&](int z)
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
    tipl::adaptive_par_for(v1.depth()*v1.height(),[&](int z)
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

bool src_data::read_b0(std::vector<tipl::image<3> >& b0) const
{
    for(size_t index = 0;index < src_bvalues.size();++index)
        if(src_bvalues[index] == 0.0f)
            b0.push_back(std::move(tipl::image<3>(dwi_at(index))));

    if(b0.empty())
    {
        error_msg = "No b0 found in DWI data";
        return false;
    }
    return true;
}

tipl::vector<3> phase_direction_at_AP_PA(const tipl::image<3>& v1,const tipl::image<3>& v2)
{
    tipl::vector<3> c;
    tipl::image<2,float> px1,px2,py1,py2;
    tipl::project_x(v1,px1);
    tipl::project_x(v2,px2);
    tipl::project_y(v1,py1);
    tipl::project_y(v2,py2);
    c[0] = float(tipl::correlation(px1.begin(),px1.end(),px2.begin()));
    c[1] = float(tipl::correlation(py1.begin(),py1.end(),py2.begin()));
    tipl::out() << "projected correction: " << c << std::endl;
    return c;
}

/*
bool src_data::distortion_correction(const std::string& filename)
{
    tipl::image<3> v1,v2;
    if(!read_b0(v1) || !read_rev_b0(filename,v2))
        return false;

    auto c = phase_direction_at_AP_PA(v1,v2);
    bool swap_xy = c[0] < c[1];
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

    //
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
    // }

    std::vector<tipl::image<3,unsigned short> > dwi(src_dwi_data.size());
    for(size_t i = 0;i < src_dwi_data.size();++i)
    {
        v1 = dwi_at(i);
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
    voxel.report += " The phase distortion was correlated using data from an opposite phase encoding direction.";
    return true;
}
*/

#include <QCoreApplication>
#include <QRegularExpression>
extern bool has_cuda;
bool src_data::run_plugin(std::string exec_name,
                            std::string keyword,
                            size_t total_keyword_count,std::vector<std::string> param,std::string working_dir,std::string exec)
{
    if(exec.empty())
    {
        #ifdef _WIN32
        // search for plugin
        exec = (QCoreApplication::applicationDirPath() +  + "/plugin/" + exec_name.c_str() + ".exe").toStdString();
        if(!std::filesystem::exists(exec))
        {
            error_msg = QString("Cannot find %1").arg(exec.c_str()).toStdString();
            return false;
        }
        #else
        std::string fsl_path = "/usr/local/fsl/bin";
        auto index = QProcess::systemEnvironment().indexOf(QRegularExpression("^FSLDIR=.+"));
        if(index != -1)
            fsl_path = QProcess::systemEnvironment()[index].split("=")[1].toStdString() + "/bin/";
        if(std::filesystem::exists(fsl_path))
            tipl::out() << "FSL installation found at " << fsl_path << std::endl;
        else
        {
            tipl::out() << "FSL installation not found: cannot find environmental variable FSLDIR, try calling directly";
            fsl_path.clear();
        }
        exec = fsl_path + exec_name;
        if(exec_name == "eddy")
        {
            if(std::filesystem::exists(fsl_path + "eddy_cpu"))
                exec = fsl_path + "eddy_cpu";
            if(std::filesystem::exists(fsl_path + "eddy_openmp"))
                exec = fsl_path + "eddy_openmp";
            if(has_cuda && std::filesystem::exists(fsl_path + "eddy_cuda"))
                exec = fsl_path + "eddy_cuda";
        }
        #endif
    }

    QProcess program;
    program.setEnvironment(program.environment() << "FSLOUTPUTTYPE=NIFTI_GZ");
    program.setWorkingDirectory(working_dir.c_str());
    tipl::out() << "run " << exec << std::endl;
    tipl::out() << "path: " << working_dir << std::endl;
    QStringList p;
    for(auto s:param)
    {
        tipl::out() << s << std::endl;
        p << s.c_str();
    }
    program.start(exec.c_str(),p);
    if(!program.waitForStarted())
    {
        switch(program.error())
        {
        case QProcess::FailedToStart:
            error_msg = exec_name + " failed to start. Either " + exec_name + " is missing, or you have no permission to run it.";
        break;
        case QProcess::Crashed:
            error_msg = exec_name + " crashed some time after starting successfully.";
        break;
        case QProcess::WriteError:
            error_msg = "An error occurred when attempting to write to the process. For example, the process may not be running, or it may have closed its input channel.";
        break;
        case QProcess::ReadError:
            error_msg = "An error occurred when attempting to read from the process. For example, the process may not be running.";
        break;
        default:
            error_msg = program.readAllStandardError().toStdString() + program.readAllStandardOutput().toStdString();
        break;
        }
        tipl::error() << error_msg;
        return false;
    }
    unsigned int keyword_seen = 0;
    tipl::progress prog("calling external program");
    while(!program.waitForFinished(1000) && !prog.aborted())
    {
        prog(keyword_seen,total_keyword_count);
        QString output = QString::fromLocal8Bit(program.readAllStandardOutput());
        if(output.isEmpty())
            continue;
        if(output.contains(keyword.c_str()))
            ++keyword_seen;
        QStringList output_lines = output.remove('\r').split('\n');
        output_lines.removeAll("");
        for(int i = 0;i+1 < output_lines.size();++i)
            tipl::out() << output_lines[i].toStdString() << std::endl;
        tipl::out() << output_lines.back().toStdString();
        if(keyword_seen >= total_keyword_count)
            ++total_keyword_count;
    }
    if(prog.aborted())
    {
        program.kill();
        error_msg = "process aborted";
        return false;
    }

    if(program.exitCode() == QProcess::CrashExit)
    {
        error_msg = program.readAllStandardError().toStdString() + program.readAllStandardOutput().toStdString();
        if(error_msg.empty())
            error_msg = "process failed";
        tipl::error() << error_msg;
        return false;
    }
    tipl::out() << "process completed." << std::endl;
    error_msg.clear();
    return true;
}

void src_data::setup_topup_eddy_volume(void)
{
    topup_size = voxel.dim;
    // ensure even number in the dimension for topup
    for(int d = 0;d < 3;++d)
        if(topup_size[d] & 1)
            topup_size[d]++;
    if(rev_pe_src.get())
        rev_pe_src->topup_size = topup_size;
}

bool src_data::generate_topup_b0_acq_files(std::vector<tipl::image<3> >& b0,
                                           std::vector<tipl::image<3> >& rev_b0,
                                           std::string& b0_appa_file,
                                           std::string& report)
{
    if(b0.empty() || rev_b0.empty())
    {
        error_msg = "cannot find b0 for topup";
        return false;
    }

    // DSI Studio uses LPS orientation whereas and FSL uses LAS
    // The y direction is flipped
    auto c = phase_direction_at_AP_PA(b0[0],rev_b0[0]);
    if(c[0] == c[1])
    {
        error_msg = "Invalid phase encoding. Please select correct reversed phase encoding b0 file";
        return false;
    }
    bool is_appa = c[0] < c[1];
    unsigned int phase_dim = (is_appa ? 1 : 0);
    tipl::vector<3> c1,c2;
    {
        tipl::image<3,unsigned char> mb0,rev_mb0;
        tipl::threshold(b0[0],mb0,tipl::max_value(b0[0])*0.8f,1,0);
        tipl::threshold(rev_b0[0],rev_mb0,tipl::max_value(rev_b0[0])*0.8f,1,0);
        c1 = tipl::center_of_mass_weighted(mb0);
        c2 = tipl::center_of_mass_weighted(rev_mb0);
    }
    tipl::out() << "source com: " << c1 << std::endl;
    tipl::out() << "rev pe com: " << c2 << std::endl;
    bool phase_dir = c1[phase_dim] > c2[phase_dim];

    std::string acqstr,pe_id;
    {
        std::string acqstr1,acqstr2,pe_id1,pe_id2;
        if(is_appa)
        {
            acqstr1 = "0 -1 0 0.05";
            acqstr2 = "0 1 0 0.05";
            pe_id1 = "AP";
            pe_id2 = "PA";
            if(phase_dir)
                pe_id = "AP_PA";
            else
            {
                acqstr1.swap(acqstr2);
                pe_id1.swap(pe_id2);
                pe_id = "PA_AP";
            }
        }
        else
        {
            acqstr1 = "-1 0 0 0.05";
            acqstr2 = "1 0 0 0.05";
            pe_id1 = "LR";
            pe_id2 = "RL";
            if(phase_dir)
                pe_id = "LR_RL";
            else
            {
                acqstr1.swap(acqstr2);
                pe_id1.swap(pe_id2);
                pe_id = "RL_LR";
            }
        }

        report = " " + std::to_string(b0.size()) + " " + pe_id1 + " encoding and "
                + std::to_string(rev_b0.size()) + " " + pe_id2 + " encoding b0 images were used to estimate susceptibility using FSL topup.";

        acqstr += acqstr1 + "\n";
        acqstr += acqstr2 + "\n";
        acqstr.pop_back();
        for(size_t index = 1;index < b0.size();++index)
            b0[0] += b0[index];
        for(size_t index = 1;index < rev_b0.size();++index)
            rev_b0[0] += rev_b0[index];

        b0.resize(1);
        rev_b0.resize(1);

    }



    tipl::out() << "source and reverse phase encoding: " << pe_id << std::endl;

    {
        tipl::out() << "create acq params at " << acqparam_file() << std::endl;
        std::ofstream out(acqparam_file().c_str());
        if(!out)
        {
            tipl::out() << "cannot write to acq param file " << acqparam_file() << std::endl;
            return false;
        }
        tipl::out() << acqstr << std::flush;
        out << acqstr << std::flush;
        out.close();
    }


    {
        // allow for more space in the PE direction
        setup_topup_eddy_volume();
        tipl::reshape(b0[0],topup_size);
        tipl::reshape(rev_b0[0],topup_size);
    }

    {
        tipl::out() << "create topup needed b0 nii.gz file from " << pe_id << " b0" << std::endl;
        tipl::image<4,float> buffer(b0[0].shape().expand(2));
        std::copy(b0[0].begin(),b0[0].end(),buffer.begin());
        std::copy(rev_b0[0].begin(),rev_b0[0].end(),buffer.begin() + b0[0].size());
        if(!tipl::io::gz_nifti::save_to_file((b0_appa_file = file_name + ".topup." + pe_id + ".nii.gz"),
                                             buffer,voxel.vs,voxel.trans_to_mni))
        {
            tipl::error() << "cannot write a temporary b0_appa image volume to " << b0_appa_file;
            return false;
        }
    }
    return true;
}


bool load_bval(const std::string& file_name,std::vector<double>& bval);
bool load_bvec(const std::string& file_name,std::vector<double>& b_table,bool flip_by = true);
bool src_data::load_topup_eddy_result(void)
{
    if(!std::filesystem::exists(corrected_file()))
    {
        error_msg = "cannot find corrected output ";
        error_msg += corrected_file();
        return false;
    }
    if(!topup_eddy_report.empty())
        std::ofstream(corrected_file()+".report.txt") << topup_eddy_report;

    std::string bval_file = file_name+".bval";
    std::string bvec_file = file_name+".corrected.eddy_rotated_bvecs";

    if(std::filesystem::exists(bvec_file)) // has eddy
    {
        tipl::out() << "update b-table from eddy output" << std::endl;
        std::vector<double> bval,bvec;
        if(!load_bval(bval_file,bval) || !load_bvec(bvec_file,bvec))
        {
            error_msg = "cannot find bval and bvec. please run topup/eddy again";
            return false;
        }
        src_bvalues.resize(bval.size());
        src_bvectors.resize(bval.size());
        std::copy(bval.begin(),bval.end(),src_bvalues.begin());
        for(size_t index = 0,i = 0;i < src_bvalues.size() && index+2 < bvec.size();++i,index += 3)
        {
            src_bvectors[i][0] = float(bvec[index]);
            src_bvectors[i][1] = float(bvec[index+1]);
            src_bvectors[i][2] = float(bvec[index+2]);
        }
    }

    tipl::out() << "load topup/eddy results" << std::endl;
    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    if(!load_4d_nii(corrected_file(),dwi_files,false,false,error_msg))
        return false;
    nifti_dwi.resize(dwi_files.size());
    src_dwi_data.resize(dwi_files.size());
    src_bvalues.resize(dwi_files.size());
    src_bvectors.resize(dwi_files.size());
    for(size_t index = 0;index < dwi_files.size();++index)
    {
        tipl::reshape(dwi_files[index]->image,voxel.dim);
        nifti_dwi[index].swap(dwi_files[index]->image);
        src_dwi_data[index] = &nifti_dwi[index][0];
    }

    if(topup_eddy_report.empty())
    {
        std::ifstream file(corrected_file()+".report.txt");
        topup_eddy_report = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }
    voxel.report += topup_eddy_report;
    calculate_dwi_sum(true);
    apply_mask = true;
    return true;
}

bool src_data::run_applytopup(std::string exec)
{
    tipl::progress prog("run applytopup");
    if(!std::filesystem::exists(topup_result()))
    {
        error_msg = "applytopup cannot find ";
        error_msg += topup_result();
        return false;
    }

    if(!save_nii_for_applytopup_or_eddy(false))
        return false;
    std::vector<std::string> param;
    // to use blipup and blipdown, both must have the same DWI count
    if(rev_pe_src.get() && rev_pe_src->src_bvalues.size() == src_bvalues.size())
    {
        if(!rev_pe_src->save_nii_for_applytopup_or_eddy(false))
        {
            error_msg = rev_pe_src->error_msg;
            return false;
        }
        //two full acq of DWI
        param = {
                std::string("--imain=") + std::filesystem::path(temp_nifti()).filename().u8string() +","+
                                          std::filesystem::path(rev_pe_src->temp_nifti()).filename().u8string(),
                std::string("--datain=") + std::filesystem::path(acqparam_file()).filename().u8string(),
                std::string("--topup=") + std::filesystem::path(topup_output()).filename().u8string(),
                std::string("--out=") + std::filesystem::path(corrected_output()).filename().u8string(),
                "--inindex=1,2",
                "--method=jac",
                "--verbose=1"};        
        topup_eddy_report += " FSL's applytopup was used to correct for susceptibility distortion using two DWI dataset acquired at opposite phase encoding directions.";    }
    else
    {
        // one full acq of DWI
        param = {
                std::string("--imain=") + std::filesystem::path(temp_nifti()).filename().u8string(),
                std::string("--datain=") + std::filesystem::path(acqparam_file()).filename().u8string(),
                std::string("--topup=") + std::filesystem::path(topup_output()).filename().u8string(),
                std::string("--out=") + std::filesystem::path(corrected_output()).filename().u8string(),
                "--inindex=1",
                "--method=jac",
                "--verbose=1"};
        topup_eddy_report += " FSL's applytopup was used to correct for susceptibility distortion using DWI acquired from one phase encoding direction.";
    }
    if(!run_plugin("applytopup"," ",10,param,QFileInfo(file_name.c_str()).absolutePath().toStdString(),exec))
        return false;
    if(!load_topup_eddy_result())
    {
        error_msg += " please check if memory is enough to run applytopup";
        return false;
    }
    std::filesystem::remove(temp_nifti());
    if(rev_pe_src.get())
        std::filesystem::remove(rev_pe_src->file_name+".nii.gz");
    return true;
}

bool eddy_check_shell(const std::vector<float>& bvalues,std::string& cause)
{
    std::vector<float> shell_bvalue;
    std::vector<size_t> shells;
    shells.push_back(0);
    for (size_t i = 1; i < bvalues.size(); i++)
    {
        size_t j;
        for(j = 0; j < shells.size(); j++)
            if (std::abs(bvalues[shells[j]]-bvalues[i]) < 100)
                break;
        if(j == shells.size())
            shells.push_back(i);
    }

    shell_bvalue.resize(shells.size());
    std::vector<size_t> shell_count(shells.size(),1);
    for(size_t s = 0; s < shells.size(); s++)
    {
        shell_bvalue[s] = bvalues[shells[s]];
        for (size_t i = 0; i < bvalues.size(); i++)
        {
            if (std::abs(bvalues[shells[s]]-bvalues[i]) < 100 && i != shells[s])
            {
                shell_bvalue[s] += bvalues[i];
                shell_count[s]++;
            }
        }
        shell_bvalue[s] /= shell_count[s];
    }

    std::sort(shell_bvalue.begin(),shell_bvalue.end());
    for (unsigned int j = 0; j< shell_bvalue.size(); j++)
    {
        shell_count[j] = 0;
        for (unsigned int i=0; i< bvalues.size(); i++)
            if (std::abs(bvalues[i]-shell_bvalue[j]) <= 100)
                    shell_count[j]++;
    }
    {
        if(shell_bvalue.size() >= 7)
        {
            cause = "too many shells";
            return false;
        }
        auto scans_per_shell = uint32_t((double(bvalues.size() - shell_count[0]) / double(shell_bvalue.size() - 1)) + 0.5);
        if(tipl::max_value(shell_count.begin()+1,shell_count.end()) >= 2 * scans_per_shell)
        {
            cause = "not enough average sampling in the shell";
            return false;
        }
        if(3 * tipl::min_value(shell_count.begin()+1,shell_count.end()) < scans_per_shell)
        {
            cause = "low sampling shell";
            return false;
        }
    }
    return true;
}
bool src_data::run_eddy(std::string exec)
{
    if(voxel.report.find("rotated") != std::string::npos)
    {
        error_msg = "eddy cannot be applied to motion corrected or rotated images";
        return false;
    }
    bool has_topup = std::filesystem::exists(topup_result());
    if(!has_topup)
    {
        tipl::out() << "cannot find topup result: " << topup_result();
        tipl::out() << "run eddy without topup";
        setup_topup_eddy_volume();
        std::ofstream out(acqparam_file());
        out << "0 -1 0 0.05" << std::endl;
    }
    {
        std::string cause;
        if(!eddy_check_shell(src_bvalues,cause))
        {
            error_msg = "cannot run eddy due to "+cause;
            if(!has_topup)
                return false;
            tipl::warning() << error_msg;
            error_msg.clear();
            return run_applytopup();
        }
    }
    if(!save_nii_for_applytopup_or_eddy(true))
        return false;

    tipl::progress prog("run eddy");
    std::string mask_nifti = file_name+".mask.nii.gz";
    std::string index_file = file_name+".index.txt";
    std::string bval_file = file_name+".bval";
    std::string bvec_file = file_name+".bvec";
    {
        tipl::image<3,unsigned char> I(topup_size);
        tipl::reshape(voxel.mask,I);
        if(!tipl::io::gz_nifti::save_to_file(mask_nifti.c_str(),I,voxel.vs,voxel.trans_to_mni))
        {
            error_msg = "cannot save mask file to ";
            error_msg += mask_nifti;
            return false;
        }
    }

    {
        std::ofstream index_out(index_file),bval_out(bval_file),bvec_out(bvec_file);
        if(!index_out || !bval_out || !bvec_out)
        {
            error_msg = "cannot write temporary files to ";
            error_msg += QFileInfo(file_name.c_str()).absolutePath().toStdString();
            return false;
        }
        for(size_t i = 0;i < src_bvalues.size();++i)
        {
            index_out << " 1";
            bval_out << src_bvalues[i] << " ";
            bvec_out << src_bvectors[i][0] << " "
                     << -src_bvectors[i][1] << " "
                     << src_bvectors[i][2] << "\n";
        }
        if(rev_pe_src.get())
        {
            for(size_t i = 0;i < rev_pe_src->src_bvalues.size();++i)
            {
                index_out << " 2";
                bval_out << rev_pe_src->src_bvalues[i] << " ";
                bvec_out << rev_pe_src->src_bvectors[i][0] << " "
                         << -rev_pe_src->src_bvectors[i][1] << " "
                         << rev_pe_src->src_bvectors[i][2] << "\n";
            }
        }
    }

    std::vector<std::string> param = {
            std::string("--imain=") + std::filesystem::path(temp_nifti()).filename().u8string(),
            std::string("--mask=") + std::filesystem::path(mask_nifti).filename().u8string(),
            std::string("--acqp=") + std::filesystem::path(acqparam_file()).filename().u8string(),
            std::string("--index=") + std::filesystem::path(index_file).filename().u8string(),
            std::string("--bvecs=") + std::filesystem::path(bvec_file).filename().u8string(),
            std::string("--bvals=") + std::filesystem::path(bval_file).filename().u8string(),
            std::string("--out=") + std::filesystem::path(corrected_output()).filename().u8string(),
            "--verbose=1"
            };
    if(has_topup)
        param.push_back(std::string("--topup=") + std::filesystem::path(topup_output()).filename().u8string());

    topup_eddy_report += " The eddy current distortions of " + std::to_string(src_bvalues.size()) + " DWI";
    if(rev_pe_src.get())
        topup_eddy_report += " and " + std::to_string(rev_pe_src->src_bvalues.size()) + " opposite phase DWI";
    topup_eddy_report += " were corrected using FSL eddy.";
    if(!run_plugin(has_cuda ? "eddy_cuda" : "eddy","model",16,param,QFileInfo(file_name.c_str()).absolutePath().toStdString(),exec))
    {
        if(!has_topup)
            return false;
        return run_applytopup();
    }
    if(!load_topup_eddy_result())
    {
        if(has_cuda)
            error_msg += " please install CUDA toolkit to run eddy";
        else
            error_msg += " eddy terminated prematurely likely due to insufficient memory";
        return false;
    }
    std::filesystem::remove(temp_nifti());
    std::filesystem::remove(mask_nifti);
    return true;
}
std::string src_data::find_topup_reverse_pe(void)
{
    tipl::progress prog("searching for opposite direction scans..");
    // locate rsrc.gz file
    std::string rev_file_name(file_name);
    tipl::remove_suffix(rev_file_name,".sz");
    tipl::remove_suffix(rev_file_name,".src.gz");
    tipl::remove_suffix(rev_file_name,".nii.gz");
    if(std::filesystem::exists(rev_file_name+".rz"))
    {
        tipl::out() << "reversed pe SRC file found: " << rev_file_name+".rz" << std::endl;
        return rev_file_name+".rz";
    }
    if(std::filesystem::exists(rev_file_name+".rsrc.gz"))
    {
        tipl::out() << "reversed pe SRC file found: " << rev_file_name+".rsrc.gz" << std::endl;
        return rev_file_name+".rsrc.gz";
    }

    // locate reverse pe nifti files
    std::map<float,std::string,std::greater<float> > candidates;
    {
        std::vector<tipl::image<3> > b0;
        if(!read_b0(b0))
            return std::string();
        auto searching_path = std::filesystem::path(file_name).parent_path().string();
        if(searching_path.empty())
            searching_path = std::filesystem::current_path().string();
        tipl::out() << "searching for *.nii.gz at " << searching_path;
        auto files = tipl::search_files(searching_path,"*.nii.gz");
        tipl::out() << files.size() << " candidate files found";

        for(auto file : files)
        {
            // skip those nii files generated for topup or eddy
            if(tipl::contains(file,".src.gz.") ||
               tipl::contains(file,".sz.") ||
               tipl::contains(file,std::filesystem::path(file_name).filename().string()))
                continue;
            tipl::out() << "checking " << file;
            tipl::io::gz_nifti nii;
            if(!nii.load_from_file(file))
            {
                tipl::out() << "cannot open " << file << " " << nii.error_msg;
                continue;
            }
            if(nii.width()*nii.height()*nii.depth() != dwi.size())
            {
                tipl::out() << file << " has a different image size " <<
                               tipl::shape<3>(nii.width(),nii.height(),nii.depth())
                               << ", skipping" << std::endl;
                continue;
            }
            if(nii.dim(4) > 8)
            {
                tipl::out() << file << " is likely non-b0 4d nifti, skipping" << std::endl;
                continue;
            }
            tipl::out() << "candidate found: " << file << std::endl;
            tipl::image<3> b0_op,each;
            if(!(nii >> b0_op))
                continue;
            auto c = phase_direction_at_AP_PA(b0[0],b0_op);
            if(c[0] + c[1] < 1.8f) // 0.9 + 0.9
            {
                tipl::out() << "correlation with b0 is low. skipping...";
                continue;
            }
            if(c[0] == c[1])
            {
                tipl::out() << "not opposite direction image. skipping...";
                continue;
            }
            candidates[std::fabs(c[0]-c[1])] = file;
        }
    }
    if(candidates.empty())
        return std::string();
    tipl::out() << "reverse phase encoding image selected: " << candidates.begin()->second << std::endl;
    return candidates.begin()->second;
}
extern std::string topup_param_file;
bool src_data::get_rev_pe(std::string other_src)
{
    if(other_src.empty())
        other_src = find_topup_reverse_pe();
    if(other_src.empty())
    {
        error_msg = "cannot find rever pe data";
        return false;
    }
    if(!std::filesystem::exists(other_src))
    {
        error_msg = "find not exist: ";
        error_msg += other_src;
        return false;
    }
    std::shared_ptr<src_data> src2(new src_data);
    if(!src2->load_from_file(other_src))
    {
        error_msg = "cannot read ";
        error_msg += other_src;
        return false;
    }
    if(src2->voxel.dim != voxel.dim)
    {
        error_msg = "inconsistent image dimension in reverse pe data";
        return false;
    }
    rev_pe_src = src2;
    return true;
}
bool src_data::load_existing_corrections(void)
{
    if(std::filesystem::exists(file_name+".corrected.nii.gz"))
    {
        tipl::out() << "load previous results from " << file_name << ".corrected.nii.gz" <<std::endl;
        if(load_topup_eddy_result())
            return true;
        tipl::warning() << "failed to load previous results" << error_msg;
        tipl::out() << "run correction from scratch ";
    }
    return false;
}
bool src_data::run_topup(void)
{
    if(voxel.report.find("rotated") != std::string::npos)
    {
        error_msg = "topup cannot be applied to motion corrected or rotated images";
        return false;
    }  
    if(!rev_pe_src.get())
    {
        error_msg = "no reverse pe data for topup";
        return false;
    }
    // run topup
    topup_eddy_report.clear();

    tipl::progress prog("run topup");
    std::string b0_appa_file,topup_report;
    std::vector<tipl::image<3> > b0,rev_b0;
    if(!rev_pe_src->read_b0(rev_b0))
    {
        error_msg = rev_pe_src->error_msg;
        return false;
    }
    if(!read_b0(b0) || !generate_topup_b0_acq_files(b0,rev_b0,b0_appa_file,topup_report))
        return false;

    std::vector<std::string> param = {
        std::string("--imain=")+std::filesystem::path(b0_appa_file).filename().u8string(),
        std::string("--datain=")+std::filesystem::path(acqparam_file()).filename().u8string(),
        std::string("--out=")+std::filesystem::path(topup_output()).filename().u8string(),
        std::string("--iout=")+std::filesystem::path(file_name + ".topup.check_result").filename().u8string(),
        "--verbose=1"};

    if(!std::filesystem::exists(topup_param_file))
    {
        tipl::out() << "failed to find topup parameter file at " << topup_param_file;
        tipl::out() << "apply default parameters";
        for(auto each : {
                "--warpres=20,16,14,12,10,6,4,4,4",
                "--subsamp=2,2,2,2,2,1,1,1,1",  // This causes an error in odd number of slices
                "--fwhm=8,6,4,3,3,2,1,0,0",
                "--miter=5,5,5,5,5,10,10,20,20",
                "--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001",
                "--estmov=1,1,1,1,1,0,0,0,0",
                "--minmet=0,0,0,0,0,1,1,1,1",
                "--scale=1"})
            param.push_back(each);
        return false;
    }

    for(const auto& line: tipl::read_text_file(topup_param_file))
    {
        if(!line.empty() && line[0] == '-')
           param.push_back(line);
    }

    if(std::filesystem::exists(topup_result()))
    {
        tipl::out() << "found previous topup output:" << topup_result();
        tipl::out() << "skip topup";
    }
    else
    {
        if(!run_plugin("topup","level",9,param,
            QFileInfo(file_name.c_str()).absolutePath().toStdString(),std::string()))
            return false;
    }
    topup_eddy_report += topup_report;
    return true;
}


void calculate_shell(std::vector<float> sorted_bvalues,
                     std::vector<unsigned int>& shell);
std::string src_data::get_report(bool dwi_part_only)
{
    std::vector<float> sorted_bvalues(src_bvalues);
    std::sort(sorted_bvalues.begin(),sorted_bvalues.end());
    unsigned int num_dir = 0;
    for(size_t i = 0;i < src_bvalues.size();++i)
        if(src_bvalues[i] > 50)
            ++num_dir;
    std::ostringstream out;

    std::vector<unsigned int> shell;
    calculate_shell(src_bvalues,shell);
    if(shell.size() > 4 && shell[1] - shell[0] <= 6)
    {
        out << " A diffusion spectrum imaging scheme was used, and a total of " << num_dir
            << " diffusion sampling were acquired."
            << " The maximum b-value was " << int(std::round(sorted_bvalues.back())) << " s/mm².";
    }
    else
    if(shell.size() > 1)
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
        {
            if(num_dir < 100)
                out << " A DTI diffusion scheme was used, and a total of ";
            else
                out << " A HARDI scheme was used, and a total of ";
            out << num_dir
                << " diffusion sampling directions were acquired."
                << " The b-value was " << sorted_bvalues.back() << " s/mm².";
        }
    if(!dwi_part_only)
        out << " The in-plane resolution was " << std::fixed << std::setprecision(3) << voxel.vs[0] << " mm."
            << " The slice thickness was " << std::fixed << std::setprecision(2) << tipl::max_value(voxel.vs.begin(),voxel.vs.end()) << " mm.";
    return out.str();
}
extern int src_ver;
bool src_data::save_to_file(const std::string& filename)
{
    if(src_bvalues.empty())
    {
        error_msg = "no DWI data to save";
        return false;
    }
    tipl::progress prog("saving ",filename);
    if(tipl::ends_with(filename,"nii.gz"))
    {
        tipl::shape<4> nifti_dim;
        std::copy(voxel.dim.begin(),voxel.dim.end(),nifti_dim.begin());
        nifti_dim[3] = uint32_t(src_bvalues.size());

        tipl::image<4,unsigned short> buffer(nifti_dim);
        tipl::adaptive_par_for(src_bvalues.size(),[&](size_t index)
        {
            std::copy_n(src_dwi_data[index],voxel.dim.size(),
                      buffer.begin() + long(index*voxel.dim.size()));
        });
        if(!tipl::io::gz_nifti::save_to_file(filename,buffer,voxel.vs,voxel.trans_to_mni))
        {
            error_msg = "cannot save ";
            error_msg += filename;
            return false;
        }
        error_msg = "cannot save bval bvec";
        return save_bval((filename.substr(0,filename.size()-7)+".bval").c_str()) &&
               save_bvec((filename.substr(0,filename.size()-7)+".bvec").c_str());
    }
    if(tipl::ends_with(filename,"src.gz") ||
       tipl::ends_with(filename,".sz") ||
       tipl::ends_with(filename,".rz"))
    {
        std::string temp_file = filename + ".tmp";
        {

            tipl::io::gz_mat_write mat_writer(temp_file);
            if(!mat_writer)
            {
                error_msg = "cannot write to a temporary file " + temp_file;
                return false;
            }
            mat_writer.write("dimension",voxel.dim);
            mat_writer.write("voxel_size",voxel.vs);
            mat_writer.write("trans",voxel.trans_to_mni);
            mat_writer.write("version",src_ver);
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
            if(voxel.mask.empty())
                tipl::threshold(dwi,voxel.mask,0,1,0);

            if(tipl::ends_with(filename,".sz"))
            {
                mat_writer.apply_slope = true;
                mat_writer.apply_mask = apply_mask;
                mat_writer.mask_rows = voxel.dim.plane_size();
                mat_writer.mask_cols = voxel.dim.depth();
                mat_writer.si2vi = tipl::get_sparse_index(voxel.mask);
                if(!apply_mask && !mat_writer.si2vi.empty())
                {
                    bool no_signal_at_background = true;
                    for(auto each : mat_writer.si2vi)
                        if(dwi[each])
                        {
                            no_signal_at_background = false;
                            break;
                        }
                    if(no_signal_at_background)
                    {
                        tipl::out() << "no dwi signal in the background, enable apply mask";
                        mat_writer.apply_mask = true;
                    }
                }
            }
            if(mat_writer.apply_mask)
            {
                tipl::out() << "store masked DWI signals";
                mat_writer.write("mask",voxel.mask,voxel.dim.plane_size());
            }
            else
                tipl::out() << "store unmasked DWI signals";

            for (unsigned int index = 0;prog(index,src_bvalues.size());++index)
                mat_writer.write<tipl::io::masked_sloped>("image"+std::to_string(index),src_dwi_data[index],
                                                   voxel.dim.plane_size(),voxel.dim.depth());

            mat_writer.write("report",voxel.report + voxel.recon_report.str());
            mat_writer.write("steps",voxel.steps);
            if(voxel.intro.empty() && std::filesystem::exists(std::filesystem::path(filename).parent_path() / "README"))
                load_intro((std::filesystem::path(filename).parent_path() / "README").string());
            mat_writer.write("intro",voxel.intro);
        }
        if(prog.aborted())
        {
            std::filesystem::remove(temp_file);
            return true;
        }
        try{
            if(std::filesystem::exists(filename))
                std::filesystem::remove(filename);
            std::filesystem::rename(temp_file,filename);
            return true;
        }
        catch(std::runtime_error& e)
        {
            error_msg = e.what();
            std::filesystem::remove(temp_file);
            return false;
        }
    }
    error_msg = "unsupported file extension";
    return false;
}

void prepare_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in)
{
    if(file_name.back() != 'z' || !std::filesystem::exists(file_name))
        return;
    std::string idx_name = file_name;
    idx_name += ".idx";
    {
        in->buffer_all = true;
        if(std::filesystem::exists(idx_name) &&
           std::filesystem::last_write_time(idx_name) >
           std::filesystem::last_write_time(file_name))
        {
            tipl::out() << "using index file for accelerated loading: " << idx_name << std::endl;
            in->load_index(idx_name);
        }
        else
        {
            if(std::filesystem::file_size(file_name) > 134217728) // 128mb
            {
                tipl::out() << "prepare index file for future accelerated loading" << std::endl;
                in->sample_access_point = true;
            }
        }
    }
}
void save_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in)
{
    if(!tipl::ends_with(file_name,".gz"))
        return;
    std::string idx_name = file_name;
    idx_name += ".idx";
    if(in->has_access_points() && in->sample_access_point && !std::filesystem::exists(idx_name))
    {
        tipl::out() << "saving index file for accelerated loading: " << std::filesystem::path(idx_name).filename().u8string() << std::endl;
        in->save_index(idx_name);
    }
}
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs);
bool src_data::load_from_file(std::vector<std::shared_ptr<DwiHeader> >& dwi_files,bool sort_btable)
{
    if(dwi_files.empty())
    {
        error_msg = "no DWI data";
        return false;
    }
    file_name = dwi_files.front()->file_name;
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

    voxel.dim = dwi_files.front()->image.shape();
    voxel.vs = dwi_files.front()->voxel_size;
    voxel.trans_to_mni = dwi_files.front()->trans_to_mni;
    if(voxel.trans_to_mni == tipl::identity_matrix())
        initial_LPS_nifti_srow(voxel.trans_to_mni,voxel.dim,voxel.vs);

    nifti_dwi.resize(dwi_files.size());
    src_bvalues.resize(dwi_files.size());
    src_bvectors.resize(dwi_files.size());
    src_dwi_data.resize(dwi_files.size());
    for(size_t i = 0;i < dwi_files.size();++i)
    {
        src_bvalues[i] = dwi_files[i]->bvalue;
        src_bvectors[i] = dwi_files[i]->bvec;
        nifti_dwi[i].swap(dwi_files[i]->image);
        src_dwi_data[i] = nifti_dwi[i].data();
    }

    voxel.report = dwi_files.front()->report + get_report(!dwi_files.front()->report.empty());
    calculate_dwi_sum(true);
    dwi_files.clear();
    return true;
}
bool src_data::load_from_file(const std::vector<std::string>& nii_names,bool need_bval_bvec)
{
    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    for(auto& nii_name : nii_names)
    {
        tipl::out() << "opening " << nii_name;
        if(!load_4d_nii(nii_name,dwi_files,true,need_bval_bvec,error_msg))
            tipl::warning() << "skipping " << nii_name << ": " << error_msg;
    }
    return load_from_file(dwi_files,false);
}

size_t match_volume(tipl::const_pointer_image<3,unsigned char> mask,tipl::vector<3> vs);
QImage read_qimage(QString filename,std::string& error);
tipl::const_pointer_image<3,unsigned char> handle_mask(tipl::io::gz_mat_read& mat_reader);
extern int src_ver;
bool src_data::load_from_file(const std::string& dwi_file_name)
{
    tipl::progress prog("open ",dwi_file_name);
    if(voxel.steps.empty())
    {
        voxel.steps = "[Step T2][Reconstruction] open ";
        voxel.steps += std::filesystem::path(dwi_file_name).filename().u8string();
        voxel.steps += "\n";
    }

    if(!std::filesystem::exists(dwi_file_name))
    {
        error_msg = "file does not exist: ";
        error_msg += dwi_file_name;
        return false;
    }
    file_name = dwi_file_name;

    if(std::filesystem::path(dwi_file_name).extension() == ".jpg" ||
       std::filesystem::path(dwi_file_name).extension() == ".tif")
    {
        tipl::image<2,unsigned char> raw;
        {
            QImage fig = read_qimage(dwi_file_name.c_str(),error_msg);
            if(fig.isNull())
                return false;
            tipl::out() << "converting to grayscale";
            int pixel_bytes = fig.bytesPerLine()/fig.width();
            raw.resize(tipl::shape<2>(uint32_t(fig.width()),uint32_t(fig.height())));
            tipl::adaptive_par_for(raw.height(),[&](int y){
                auto line = fig.scanLine(y);
                auto out = raw.begin() + int64_t(y)*raw.width();
                for(int x = 0;x < raw.width();++x,line += pixel_bytes)
                    out[x] = uint8_t(*line);
            });
        }
        tipl::out() << "generating mask";
        auto raw_ = raw.alias(0,tipl::shape<3>(raw.width(),raw.height(),1));
        if(raw.width() > 2048)
        {
            tipl::downsample_with_padding(raw_,dwi);
            while(dwi.width() > 2048)
                tipl::downsample_with_padding(dwi);
        }
        else
            dwi = raw_;

        // increase contrast
        dwi -= 128;
        dwi *= 2;


        voxel.is_histology = true;
        voxel.dim = dwi.shape();
        voxel.vs = {0.05f,0.05f,0.05f};
        initial_LPS_nifti_srow(voxel.trans_to_mni,voxel.dim,voxel.vs);
        voxel.hist_image.swap(raw);
        voxel.report = "Histology image was loaded at a size of "
                     + std::to_string(voxel.hist_image.width()) + " by "
                     + std::to_string(voxel.hist_image.height()) + " pixels.";
        tipl::out() << "generating mask";
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
        tipl::out() << "image file loaded" << std::endl;
        return true;
    }
    if(tipl::ends_with(dwi_file_name,".nii.gz"))
    {
        std::vector<std::shared_ptr<DwiHeader> > dwi_files;
        if(!load_4d_nii(dwi_file_name,dwi_files,true,false,error_msg))
            return false;
        return load_from_file(dwi_files,false);
    }
    else
    {
        prepare_idx(dwi_file_name,mat_reader.in);
        if(!mat_reader.load_from_file(dwi_file_name,prog))
        {
            if(!prog.aborted())
                error_msg = "cannot open file: " + mat_reader.error_msg;
            return false;
        }
        save_idx(dwi_file_name,mat_reader.in);
        mat_reader.in->close();

        if (!mat_reader.read("dimension",voxel.dim) ||
            !mat_reader.read("voxel_size",voxel.vs))
        {
            error_msg = "incompatible SRC format";
            return false;
        }
        if(!mat_reader.read("trans",voxel.trans_to_mni))
            initial_LPS_nifti_srow(voxel.trans_to_mni,voxel.dim,voxel.vs);
        int this_src_ver(0);
        if(mat_reader.has("version") && (this_src_ver = mat_reader.read_as_value<int>("version")) > src_ver)
        {
            error_msg = "incompatible SRC format. please update DSI Studio to open this file.";
            return false;
        }
        if(!mat_reader.has("mask"))
            voxel.mask.clear();
        else
        {
            voxel.mask = handle_mask(mat_reader);
            apply_mask = true;
        }
        mat_reader.read("steps",voxel.steps);

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
        tipl::out() << "src ver: " << this_src_ver;
        tipl::out() << "dim: " << voxel.dim << " vs: " << voxel.vs;
        tipl::out() << "trans: " << voxel.trans_to_mni;
        tipl::out() << "dwi count: " << src_bvalues.size();

        mat_reader.read("report",voxel.report);
        if(!tipl::contains(voxel.report,"b-value"))
            voxel.report += get_report(!voxel.report.empty());

        if(!apply_mask && voxel.report.find(" eddy ") != std::string::npos)
            apply_mask = true;

        mat_reader.read("intro",voxel.intro);

        {
            tipl::progress p2("reading data");
            src_dwi_data.resize(src_bvalues.size());
            for (size_t index = 0;p2(index,src_dwi_data.size());++index)
                if(!mat_reader.read("image"+std::to_string(index),src_dwi_data[index]))
                {
                    error_msg = "cannot read image. incomplete file ?";
                    return false;
                }
            if(p2.aborted())
                return false;
        }

        {
            const float* grad_dev_ptr = nullptr;
            std::vector<tipl::pointer_image<3,float> > grad_dev;
            size_t b0_pos = size_t(std::min_element(src_bvalues.begin(),src_bvalues.end())-src_bvalues.begin());
            if(src_bvalues[b0_pos] == 0.0f &&
               mat_reader.read("grad_dev",row,col,grad_dev_ptr) &&
               size_t(row)*size_t(col) == voxel.dim.size()*9)
            {
                tipl::progress prog2("gradient deviation correction");

                for(unsigned int index = 0;index < 9;index++)
                    grad_dev.push_back(tipl::make_image(const_cast<float*>(grad_dev_ptr+index*voxel.dim.size()),voxel.dim));
                if(std::fabs(grad_dev[0][0])+std::fabs(grad_dev[4][0])+std::fabs(grad_dev[8][0]) < 1.0f)
                {
                    tipl::add_constant(grad_dev[0].begin(),grad_dev[0].end(),1.0);
                    tipl::add_constant(grad_dev[4].begin(),grad_dev[4].end(),1.0);
                    tipl::add_constant(grad_dev[8].begin(),grad_dev[8].end(),1.0);
                }
                // correct signals
                size_t progress = 0;
                tipl::adaptive_par_for(voxel.dim.size(),[&](size_t voxel_index)
                {
                    if(!prog2(++progress,voxel.dim.size()))
                        return;
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
                if(prog2.aborted())
                {
                    error_msg = "aborted";
                    return false;
                }
            }
        }
    }

    if(prog.aborted())
    {
        error_msg = "aborted";
        return false;
    }

    calculate_dwi_sum(voxel.mask.empty());
    // create mask if not loaded from SRC file
    if(is_human_size(voxel.dim,voxel.vs))
        voxel.template_id = 0;
    else
        voxel.template_id = match_volume(tipl::make_image(voxel.mask.data(),voxel.mask.shape()),voxel.vs);
    return true;
}
extern int fib_ver;
bool src_data::save_fib(void)
{
    check_output_file_name();

    tipl::out() << "saving " << output_file_name;
    std::string tmp_file = output_file_name + ".tmp";
    tipl::io::gz_mat_write mat_writer(tmp_file);
    if(!mat_writer)
    {
        error_msg = "cannot write to a temporary file " + tmp_file;
        return false;
    }
    if(tipl::ends_with(output_file_name,".fz"))
    {
        mat_writer.apply_slope = true;
        mat_writer.apply_mask = true;
        mat_writer.mask_rows = voxel.dim.plane_size();
        mat_writer.mask_cols = voxel.dim.depth();
        mat_writer.si2vi = tipl::get_sparse_index(voxel.mask);
    }
    mat_writer.write("dimension",voxel.dim);
    mat_writer.write("voxel_size",voxel.vs);
    mat_writer.write("trans",voxel.trans_to_mni);       
    mat_writer.write("version",fib_ver);
    mat_writer.write("mask",voxel.mask,voxel.dim.plane_size());
    if(voxel.qsdr)
    {
        mat_writer.write("template",std::filesystem::path(fa_template_list[voxel.template_id]).stem().stem().stem().string());
        mat_writer.write("R2",std::vector<float>({voxel.R2}));
    }

    if(!voxel.end(mat_writer))
    {
        mat_writer.close();
        error_msg = "aborted";
        return false;
    }
    if(!voxel.other_image.empty())
    {
        // for htmls
        {
            std::string other_image_text;
            for(unsigned int index = 0;index < voxel.other_image.size();++index)
                if(voxel.other_image[index].empty())
                {
                    other_image_text += voxel.other_image_name[index];
                    other_image_text += ",";
                    continue;
                }
            mat_writer.write("other_images",other_image_text);
        }
        for(unsigned int index = 0;index < voxel.other_image.size();++index)
            if(voxel.other_image[index].shape() == voxel.dim)
                mat_writer.write<tipl::io::masked_sloped>(voxel.other_image_name[index],
                                                          voxel.other_image[index],voxel.dim.plane_size());
    }
    mat_writer.write("report",voxel.report + voxel.recon_report.str());
    mat_writer.write("steps",voxel.steps + voxel.step_report.str() + "[Step T2b][Run reconstruction]\n");
    mat_writer.write("intro",voxel.intro);
    mat_writer.close();
    std::filesystem::rename(tmp_file,output_file_name);

    return true;
}

bool src_data::save_nii_for_applytopup_or_eddy(bool include_rev) const
{
    tipl::progress prog("prepare file for tupup/eddy");
    tipl::image<4> buffer(topup_size.expand(src_bvalues.size() +
                          uint32_t(rev_pe_src.get() && include_rev ? rev_pe_src->src_bvalues.size():0)));
    if(buffer.empty())
    {
        error_msg = "cannot create trimmed volume for applytopup or eddy";
        return false;
    }
    size_t p = 0;
    tipl::adaptive_par_for(src_bvalues.size(),[&](unsigned int index)
    {
        prog(p++,src_bvalues.size());
        auto out = buffer.slice_at(index);
        tipl::reshape(dwi_at(index),out);
    });
    if(prog.aborted())
        return false;
    p = 0;
    if(rev_pe_src.get() && include_rev)
        tipl::adaptive_par_for(rev_pe_src->src_bvalues.size(),[&](unsigned int index)
        {
            prog(p++,rev_pe_src->src_bvalues.size());
            auto out = buffer.slice_at(index+uint32_t(src_bvalues.size()));
            tipl::reshape(rev_pe_src->dwi_at(index),out);
        });
    tipl::out() << "save temporary nifti file at " << temp_nifti();
    if(!tipl::io::gz_nifti::save_to_file(temp_nifti().c_str(),buffer,voxel.vs,voxel.trans_to_mni,false,nullptr,std::move(prog)))
    {
        if(!tipl::prog_aborted)
            error_msg = "failed to write a temporary nifti file: " + temp_nifti() + ". Please check write permission.";
        return false;
    }
    return !prog.aborted();
}
bool src_data::save_b0_to_nii(const std::string& nifti_file_name) const
{
    tipl::image<3> buffer(voxel.dim);
    std::copy_n(src_dwi_data[0],buffer.size(),buffer.begin());
    return tipl::io::gz_nifti::save_to_file(nifti_file_name,buffer,voxel.vs,voxel.trans_to_mni);
}
bool src_data::save_mask_nii(const std::string& nifti_file_name) const
{
    return tipl::io::gz_nifti::save_to_file(nifti_file_name,voxel.mask,voxel.vs,voxel.trans_to_mni);
}

bool src_data::save_dwi_sum_to_nii(const std::string& nifti_file_name) const
{
    tipl::image<3,unsigned char> buffer(dwi);
    return tipl::io::gz_nifti::save_to_file(nifti_file_name,buffer,voxel.vs,voxel.trans_to_mni);
}

bool src_data::save_b_table(const std::string& file_name) const
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
bool src_data::save_bval(const std::string& file_name) const
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
bool src_data::save_bvec(const std::string& file_name) const
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
