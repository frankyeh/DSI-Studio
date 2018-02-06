#include "image_model.hpp"
#include "odf_process.hpp"
#include "dti_process.hpp"

void ImageModel::calculate_dwi_sum(void)
{
    dwi_sum.clear();
    dwi_sum.resize(voxel.dim);
    image::par_for(dwi_sum.size(),[&](unsigned int pos)
    {
        for (unsigned int index = 0;index < src_dwi_data.size();++index)
            dwi_sum[pos] += src_dwi_data[index][pos];
    });

    float max_value = *std::max_element(dwi_sum.begin(),dwi_sum.end());
    float min_value = max_value;
    for (unsigned int index = 0;index < dwi_sum.size();++index)
        if (dwi_sum[index] < min_value && dwi_sum[index] > 0)
            min_value = dwi_sum[index];


    image::minus_constant(dwi_sum,min_value);
    image::lower_threshold(dwi_sum,0.0f);
    float t = image::segmentation::otsu_threshold(dwi_sum);
    image::upper_threshold(dwi_sum,t*3.0f);
    image::normalize(dwi_sum,1.0);
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
std::pair<float,float> evaluate_fib(
        const image::geometry<3>& dim,
        const std::vector<std::vector<float> >& fib_fa,
        const std::vector<std::vector<float> >& fib_dir)
{
    unsigned char num_fib = fib_fa.size();
    char dx[13] = {1,0,0,1,1,0, 1, 1, 0, 1,-1, 1, 1};
    char dy[13] = {0,1,0,1,0,1,-1, 0, 1, 1, 1,-1, 1};
    char dz[13] = {0,0,1,0,1,1, 0,-1,-1, 1, 1, 1,-1};
    std::vector<image::vector<3> > dis(13);
    for(unsigned int i = 0;i < 13;++i)
    {
        dis[i] = image::vector<3>(dx[i],dy[i],dz[i]);
        dis[i].normalize();
    }
    float otsu = *std::max_element(fib_fa[0].begin(),fib_fa[0].end())*0.1;
    std::vector<std::vector<unsigned char> > connected(fib_fa.size());
    for(unsigned int index = 0;index < connected.size();++index)
        connected[index].resize(dim.size());
    float connection_count = 0;
    for(image::pixel_index<3> index(dim);index < dim.size();++index)
    {
        if(fib_fa[0][index.index()] <= otsu)
            continue;
        unsigned int index3 = index.index()+index.index()+index.index();
        for(unsigned char fib1 = 0;fib1 < num_fib;++fib1)
        {
            if(fib_fa[fib1][index.index()] <= otsu)
                break;
            for(unsigned int j = 0;j < 2;++j)
            for(unsigned int i = 0;i < 13;++i)
            {
                image::vector<3,int> pos;
                pos = j ? image::vector<3,int>(index[0] + dx[i],index[1] + dy[i],index[2] + dz[i])
                          :image::vector<3,int>(index[0] - dx[i],index[1] - dy[i],index[2] - dz[i]);
                if(!dim.is_valid(pos))
                    continue;
                image::pixel_index<3> other_index(pos[0],pos[1],pos[2],dim);
                unsigned int other_index3 = other_index.index()+other_index.index()+other_index.index();
                if(std::abs(image::vector<3>(&fib_dir[fib1][index3])*dis[i]) <= 0.8665)
                    continue;
                for(unsigned char fib2 = 0;fib2 < num_fib;++fib2)
                    if(fib_fa[fib2][other_index.index()] > otsu &&
                            std::abs(image::vector<3>(&fib_dir[fib2][other_index3])*dis[i]) > 0.8665)
                    {
                        connected[fib1][index.index()] = 1;
                        connected[fib2][other_index.index()] = 1;
                        connection_count += fib_fa[fib2][other_index.index()];
                    }
            }
        }
    }
    float no_connection_count = 0;
    for(image::pixel_index<3> index(dim);index < dim.size();++index)
    {
        for(unsigned int i = 0;i < num_fib;++i)
            if(fib_fa[i][index.index()] > otsu && !connected[i][index.index()])
            {
                no_connection_count += fib_fa[i][index.index()];
            }

    }

    return std::make_pair(connection_count,no_connection_count);
}
void flip_fib_dir(std::vector<float>& fib_dir,const unsigned char* order)
{
    for(unsigned int j = 0;j+2 < fib_dir.size();j += 3)
    {
        float x = fib_dir[j+order[0]];
        float y = fib_dir[j+order[1]];
        float z = fib_dir[j+order[2]];
        fib_dir[j] = x;
        fib_dir[j+1] = y;
        fib_dir[j+2] = z;
        if(order[3])
            fib_dir[j] = -fib_dir[j];
        if(order[4])
            fib_dir[j+1] = -fib_dir[j+1];
        if(order[5])
            fib_dir[j+2] = -fib_dir[j+2];
    }
}
std::string ImageModel::check_b_table(void)
{
    if(baseline.get())
        baseline->check_b_table();
    set_title("checking b-table");
    bool output_dif = voxel.output_diffusivity;
    bool output_tensor = voxel.output_tensor;
    voxel.output_diffusivity = false;
    voxel.output_tensor = false;
    reconstruct<check_btable_process>();
    voxel.output_diffusivity = output_dif;
    voxel.output_tensor = output_tensor;
    std::vector<std::vector<float> > fib_fa(1);
    std::vector<std::vector<float> > fib_dir(1);
    fib_fa[0].swap(voxel.fib_fa);
    fib_dir[0].swap(voxel.fib_dir);

    const unsigned char order[18][6] = {
                            {0,1,2,1,0,0},
                            {0,1,2,0,1,0},
                            {0,1,2,0,0,1},
                            {0,2,1,1,0,0},
                            {0,2,1,0,1,0},
                            {0,2,1,0,0,1},
                            {1,0,2,1,0,0},
                            {1,0,2,0,1,0},
                            {1,0,2,0,0,1},
                            {1,2,0,1,0,0},
                            {1,2,0,0,1,0},
                            {1,2,0,0,0,1},
                            {2,1,0,1,0,0},
                            {2,1,0,0,1,0},
                            {2,1,0,0,0,1},
                            {2,0,1,1,0,0},
                            {2,0,1,0,1,0},
                            {2,0,1,0,0,1}};
    const char txt[18][7] = {".012fx",".012fy",".012fz",
                             ".021fx",".021fy",".021fz",
                             ".102fx",".102fy",".102fz",
                             ".120fx",".120fy",".120fz",
                             ".210fx",".210fy",".210fz",
                             ".201fx",".201fy",".201fz"};

    float result[18] = {0};
    float cur_score = evaluate_fib(voxel.dim,fib_fa,fib_dir).first;
    for(int i = 0;i < 18;++i)
    {
        std::vector<std::vector<float> > new_dir(fib_dir);
        flip_fib_dir(new_dir[0],order[i]);
        result[i] = evaluate_fib(voxel.dim,fib_fa,new_dir).first;
    }
    int best = std::max_element(result,result+18)-result;

    if(result[best] > cur_score)
    {
        flip_b_table(order[best]);
        voxel.load_from_src(*this);
        return txt[best];
    }
    return std::string();
}

float ImageModel::quality_control_neighboring_dwi_corr(void)
{
    // correlation of neighboring DWI < 1750
    std::vector<std::pair<int,int> > corr_pairs;
    for(int i = 0;i < src_bvalues.size();++i)
    {
        if(src_bvalues[i] > 1750.0f || src_bvalues[i] == 0.0f)
            continue;
        float max_cos = 0.0;
        int max_j = 0;
        for(int j = 0;j < src_bvalues.size();++j)
            if(std::abs(src_bvalues[j]-src_bvalues[i]) < 200.0f && i != j)
            {
                float cos = std::abs(src_bvectors[i]*src_bvectors[j]);
                if(cos > max_cos)
                {
                    max_cos = cos;
                    max_j = j;
                }
            }
        if(max_j > i)
            corr_pairs.push_back(std::make_pair(i,max_j));
    }
    float self_cor = 0.0f;
    unsigned int count = 0;
    image::par_for(corr_pairs.size(),[&](int index)
    {
        int i1 = corr_pairs[index].first;
        int i2 = corr_pairs[index].second;
        std::vector<float> I1,I2;
        I1.reserve(voxel.dim.size());
        I2.reserve(voxel.dim.size());
        for(int i = 0;i < voxel.dim.size();++i)
            if(voxel.mask[i])
            {
                I1.push_back(src_dwi_data[i1][i]);
                I2.push_back(src_dwi_data[i2][i]);
            }
        self_cor += image::correlation(I1.begin(),I1.end(),I2.begin());
        ++count;
    });
    self_cor/= (float)count;
    return self_cor;
}
bool ImageModel::is_human_data(void) const
{
    return voxel.dim[0]*voxel.vs[0] > 100 && voxel.dim[1]*voxel.vs[1] > 120 && voxel.dim[2]*voxel.vs[2] > 40;
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
void ImageModel::rotate_b_table(unsigned char dim)
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
void ImageModel::flip(unsigned char type)
{
    if(type < 3)
        flip_b_table(type);
    else
        rotate_b_table(type-3);
    image::flip(dwi_sum,type);
    image::flip(voxel.mask,type);
    for(unsigned int i = 0;i < voxel.grad_dev.size();++i)
    {
        auto I = image::make_image((float*)&*(voxel.grad_dev[i].begin()),voxel.dim);
        image::flip(I,type);
    }
    for (unsigned int index = 0;check_prog(index,src_dwi_data.size());++index)
    {
        auto I = image::make_image((unsigned short*)src_dwi_data[index],voxel.dim);
        image::flip(I,type);
    }
    voxel.dim = dwi_sum.geometry();
}
// used in eddy correction for each dwi
void ImageModel::rotate_dwi(unsigned int dwi_index,const image::transformation_matrix<double>& affine)
{
    image::basic_image<float,3> tmp(voxel.dim);
    auto I = image::make_image((unsigned short*)src_dwi_data[dwi_index],voxel.dim);
    image::resample(I,tmp,affine,image::cubic);
    image::lower_threshold(tmp,0);
    std::copy(tmp.begin(),tmp.end(),I.begin());
    // rotate b-table
    image::matrix<3,3,float> iT = image::inverse(affine.get());
    image::vector<3> v;
    image::vector_rotation(src_bvectors[dwi_index].begin(),v.begin(),iT,image::vdim<3>());
    v.normalize();
    src_bvectors[dwi_index] = v;
}

void ImageModel::rotate(const image::basic_image<float,3>& ref,
                        const image::transformation_matrix<double>& affine,
                        bool super_resolution)
{
    image::geometry<3> new_geo = ref.geometry();
    std::vector<image::basic_image<unsigned short,3> > dwi(src_dwi_data.size());
    image::par_for2(src_dwi_data.size(),[&](unsigned int index,unsigned int id)
    {
        if(!id)
            check_prog(index,src_dwi_data.size());
        dwi[index].resize(new_geo);
        auto I = image::make_image((unsigned short*)src_dwi_data[index],voxel.dim);
        if(super_resolution)
            image::resample_with_ref(I,ref,dwi[index],affine);
        else
            image::resample(I,dwi[index],affine,image::cubic);
        src_dwi_data[index] = &(dwi[index][0]);
    });
    check_prog(0,0);
    dwi.swap(new_dwi);

    // rotate b-table
    image::matrix<3,3,float> iT = image::inverse(affine.get());
    for (unsigned int index = 0;index < src_bvalues.size();++index)
    {
        image::vector<3> tmp;
        image::vector_rotation(src_bvectors[index].begin(),tmp.begin(),iT,image::vdim<3>());
        tmp.normalize();
        src_bvectors[index] = tmp;
    }

    if(!voxel.grad_dev.empty())
    {
        // <R*Gra_dev*b_table,ODF>
        // = <(R*Gra_dev*inv(R))*R*b_table,ODF>
        float det = std::abs(iT.det());
        begin_prog("rotating grad_dev");
        for(unsigned int index = 0;check_prog(index,voxel.dim.size());++index)
        {
            image::matrix<3,3,float> grad_dev,G_invR;
            for(unsigned int i = 0; i < 9; ++i)
                grad_dev[i] = voxel.grad_dev[i][index];
            G_invR = grad_dev*affine.get();
            grad_dev = iT*G_invR;
            for(unsigned int i = 0; i < 9; ++i)
                voxel.grad_dev[i][index] = grad_dev[i]/det;
        }
        std::vector<image::basic_image<float,3> > new_gra_dev(voxel.grad_dev.size());
        begin_prog("rotating grad_dev volume");
        for (unsigned int index = 0;check_prog(index,new_gra_dev.size());++index)
        {
            new_gra_dev[index].resize(new_geo);
            image::resample(voxel.grad_dev[index],new_gra_dev[index],affine,image::cubic);
            voxel.grad_dev[index] = image::make_image((float*)&(new_gra_dev[index][0]),voxel.dim);
        }
        new_gra_dev.swap(voxel.new_grad_dev);
    }
    voxel.dim = new_geo;
    calculate_dwi_sum();
    voxel.calculate_mask(dwi_sum);

}
void ImageModel::trim(void)
{
    image::geometry<3> range_min,range_max;
    image::bounding_box(voxel.mask,range_min,range_max,0);
    for (unsigned int index = 0;check_prog(index,src_dwi_data.size());++index)
    {
        auto I = image::make_image((unsigned short*)src_dwi_data[index],voxel.dim);
        image::basic_image<unsigned short,3> I0 = I;
        image::crop(I0,range_min,range_max);
        std::fill(I.begin(),I.end(),0);
        std::copy(I0.begin(),I0.end(),I.begin());
    }
    image::crop(voxel.mask,range_min,range_max);
    voxel.dim = voxel.mask.geometry();
    calculate_dwi_sum();
    voxel.calculate_mask(dwi_sum);
}
void ImageModel::distortion_correction(const ImageModel& rhs)
{
    image::basic_image<float,3> v1(dwi_sum),v2(rhs.dwi_sum),d;
    v1 /= image::mean(v1);
    v2 /= image::mean(v2);
    image::filter::gaussian(v1);
    image::filter::gaussian(v2);
    bool swap_xy = false;
    bool swap_ap = false;

    {
        image::vector<3,float> m1 = image::center_of_mass(v1); // should be d+
        image::vector<3,float> m2 = image::center_of_mass(v2); // should be d-
        m1 -= m2;
        std::cout << m1 << std::endl;
        if(std::abs(m1[1]) > std::abs(m1[0]))
        {
            image::swap_xy(v1);
            image::swap_xy(v2);
            std::swap(m1[1],m1[0]);
            swap_xy = true;
        }
        if(m1[0] > 0)
        {
            v1.swap(v2);
            swap_ap = true;
        }
    }


    distortion_estimate(v1,v2,d);
    check_prog(0,0);
    if(prog_aborted())
        return;
    begin_prog("applying warp");
    std::vector<image::basic_image<unsigned short,3> > dwi(src_dwi_data.size());
    distortion_map m;
    m = d;
    for(int i = 0;check_prog(i,src_dwi_data.size());++i)
    {
        //dwi[i] = image::make_image((unsigned short*)dwi_data[i],voxel.dim);
        if(prog_aborted())
            return;
        image::basic_image<float,3> dwi1 = image::make_image((unsigned short*)src_dwi_data[i],voxel.dim);
        image::basic_image<float,3> dwi2 = image::make_image((unsigned short*)rhs.src_dwi_data[i],voxel.dim);
        if(swap_xy)
        {
            image::swap_xy(dwi1);
            image::swap_xy(dwi2);
        }
        if(swap_ap)
            dwi1.swap(dwi2);
        image::basic_image<float,3> v;
        if(i == 1)
        {
            image::filter::gaussian(dwi1);
            image::filter::gaussian(dwi2);
            m.calculate_original(dwi1,dwi2,v);
        }
        else
            v = dwi1;
        if(swap_xy)
            image::swap_xy(v);
        image::lower_threshold(v,0);
        dwi[i] = v;
    }
    if(prog_aborted())
        return;
    d *= 50.0f;
    image::swap_xy(d);
    dwi[0] = d;
    new_dwi.swap(dwi);
    for(int i = 0;i < new_dwi.size();++i)
        src_dwi_data[i] = &(new_dwi[i][0]);
    calculate_dwi_sum();
    voxel.calculate_mask(dwi_sum);
}


void calculate_shell(const std::vector<float>& sorted_bvalues,
                     std::vector<unsigned int>& shell)
{
    shell.clear();
    std::vector<float> dif_dis;
    for(int i = 1;i < sorted_bvalues.size();++i)
        if(sorted_bvalues[i-1] != 0.0)
            dif_dis.push_back(sorted_bvalues[i] - sorted_bvalues[i-1]);
    if(dif_dis.empty())
        return;
    std::sort(dif_dis.begin(),dif_dis.end());

    float gap = *std::max_element(dif_dis.begin(),dif_dis.end())*0.1;
    if(gap < 100)
        gap = 100;

    if(sorted_bvalues.front() != 0.0f)
        shell.push_back(0);
    else
    {
        for(int i = 1;i < sorted_bvalues.size();++i)
            if(sorted_bvalues[i] != 0)
            {
                shell.push_back(i);
                break;
            }
    }
    for(unsigned int index = shell.back()+1;index < sorted_bvalues.size();++index)
        if(std::abs(sorted_bvalues[index]-sorted_bvalues[shell.back()]) > gap)
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
    return is_dsi() && (shell[1] - shell[0] <= 3);
}

bool ImageModel::is_dsi(void)
{
    if(shell.empty())
        calculate_shell();
    return shell.size() > 4 && (shell[1] - shell[0] <= 6);
}
bool ImageModel::need_scheme_balance(void)
{
    if(shell.empty())
        calculate_shell();
    if(is_dsi() || shell.size() > 6)
        return false;
    for(int i = 0;i < shell.size()-1;++i)
        if(shell[i]-shell[i] < 128)
            return true;
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
    std::ostringstream out;
    if(is_dsi())
    {
        out << " A diffusion spectrum imaging scheme was used, and a total of " <<
               src_bvalues.size()-(src_bvalues.front() == 0 ? 1:0)
            << " diffusion sampling were acquired."
            << " The maximum b-value was " << (int)std::round(src_bvalues.back()) << " s/mm2.";
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
            out << (int)std::round(src_bvalues[
                index == shell.size()-1 ? (src_bvalues.size()+shell.back())/2 : (shell[index+1] + shell[index])/2]);
        }
        out << " s/mm2.";

        out << " The number of diffusion sampling directions were ";
        for(unsigned int index = 0;index < shell.size()-1;++index)
            out << shell[index+1] - shell[index] << (shell.size() == 2 ? " ":", ");
        out << "and " << src_bvalues.size()-shell.back() << ", respectively.";
    }
    else
        if(shell.size() == 1)
        {
            int dir_num = int(src_bvalues.size()-(src_bvalues.front() == 0 ? 1:0));
            if(dir_num < 100)
                out << " A DTI diffusion scheme was used, and a total of ";
            else
                out << " A HARDI scheme was used, and a total of ";
            out << dir_num
                << " diffusion sampling directions were acquired."
                << " The b-value was " << src_bvalues.back() << " s/mm2.";
        }

    out << " The in-plane resolution was " << voxel.vs[0] << " mm."
        << " The slice thickness was " << voxel.vs[2] << " mm.";
    report = out.str();
}

bool ImageModel::load_from_file(const char* dwi_file_name)
{
    file_name = dwi_file_name;
    if (!mat_reader.load_from_file(dwi_file_name))
    {
        error_msg = "Cannot open file";
        return false;
    }
    unsigned int row,col;

    const unsigned short* dim_ptr = 0;
    if (!mat_reader.read("dimension",row,col,dim_ptr))
    {
        error_msg = "Cannot find dimension matrix";
        return false;
    }
    const float* voxel_size = 0;
    if (!mat_reader.read("voxel_size",row,col,voxel_size))
    {
        //error_msg = "Cannot find voxel size matrix";
        //return false;
        std::fill(voxel.vs.begin(),voxel.vs.end(),2.0);
    }
    else
        std::copy(voxel_size,voxel_size+3,voxel.vs.begin());

    if (dim_ptr[0]*dim_ptr[1]*dim_ptr[2] <= 0)
    {
        error_msg = "Invalid dimension setting";
        return false;
    }
    voxel.dim[0] = dim_ptr[0];
    voxel.dim[1] = dim_ptr[1];
    voxel.dim[2] = dim_ptr[2];

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

    const char* report_buf = 0;
    if(mat_reader.read("report",row,col,report_buf))
        voxel.report = std::string(report_buf,report_buf+row*col);
    else
        get_report(voxel.report);

    src_dwi_data.resize(src_bvalues.size());
    for (unsigned int index = 0;index < src_bvalues.size();++index)
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
        const float* grad_dev = 0;
        if(mat_reader.read("grad_dev",row,col,grad_dev) && row*col == voxel.dim.size()*9)
        {
            for(unsigned int index = 0;index < 9;index++)
                voxel.grad_dev.push_back(image::make_image((float*)grad_dev+index*voxel.dim.size(),voxel.dim));
            if(std::fabs(voxel.grad_dev[0][0])+std::fabs(voxel.grad_dev[4][0])+std::fabs(voxel.grad_dev[8][0]) < 1.0)
            {
                image::add_constant(voxel.grad_dev[0].begin(),voxel.grad_dev[0].end(),1.0);
                image::add_constant(voxel.grad_dev[4].begin(),voxel.grad_dev[4].end(),1.0);
                image::add_constant(voxel.grad_dev[8].begin(),voxel.grad_dev[8].end(),1.0);
            }
        }

    }

    // create mask;
    calculate_dwi_sum();

    const unsigned char* mask_ptr = 0;
    if(mat_reader.read("mask",row,col,mask_ptr))
    {
        voxel.mask.resize(voxel.dim);
        if(row*col == voxel.dim.size())
            std::copy(mask_ptr,mask_ptr+row*col,voxel.mask.begin());
    }
    else
        voxel.calculate_mask(dwi_sum);
    return true;
}
bool ImageModel::load_baseline(const char* dwi_file_name)
{
    std::shared_ptr<ImageModel> bl(new ImageModel);
    if(!bl->load_from_file(dwi_file_name))
    {
        error_msg = bl->error_msg;
        return false;
    }
    baseline = bl;
    baseline->voxel.load_from_src(*baseline.get());
    voxel.baseline = &(baseline->voxel);
    return true;
}

void ImageModel::save_to_file(gz_mat_write& mat_writer)
{

    set_title("saving");

    // dimension
    {
        short dim[3];
        dim[0] = voxel.dim[0];
        dim[1] = voxel.dim[1];
        dim[2] = voxel.dim[2];
        mat_writer.write("dimension",dim,1,3);
    }

    // voxel size
    mat_writer.write("voxel_size",&*voxel.vs.begin(),1,3);

    std::vector<float> float_data;
    std::vector<short> short_data;
    voxel.ti.save_to_buffer(float_data,short_data);
    mat_writer.write("odf_vertices",&*float_data.begin(),3,voxel.ti.vertices_count);
    mat_writer.write("odf_faces",&*short_data.begin(),3,voxel.ti.faces.size());

}
void ImageModel::save_fib(const std::string& ext)
{
    std::string output_name = file_name;
    output_name += ext;
    begin_prog("saving data");
    gz_mat_write mat_writer(output_name.c_str());
    save_to_file(mat_writer);
    voxel.end(mat_writer);
    std::string final_report = voxel.report.c_str();
    final_report += voxel.recon_report.str();
    mat_writer.write("report",final_report.c_str(),1,final_report.length());
}
bool ImageModel::save_to_nii(const char* nifti_file_name) const
{
    gz_nifti header;
    float vs[4];
    std::copy(voxel.vs.begin(),voxel.vs.end(),vs);
    vs[3] = 1.0;
    header.set_voxel_size(vs);
    image::geometry<4> nifti_dim;
    std::copy(voxel.dim.begin(),voxel.dim.end(),nifti_dim.begin());
    nifti_dim[3] = src_bvalues.size();
    image::basic_image<unsigned short,4> buffer(nifti_dim);
    for(unsigned int index = 0;index < src_bvalues.size();++index)
    {
        std::copy(src_dwi_data[index],
                  src_dwi_data[index]+voxel.dim.size(),
                  buffer.begin() + (size_t)index*voxel.dim.size());
    }
    image::flip_xy(buffer);
    header << buffer;
    return header.save_to_file(nifti_file_name);
}
bool ImageModel::save_b0_to_nii(const char* nifti_file_name) const
{
    gz_nifti header;
    header.set_voxel_size(voxel.vs);
    image::basic_image<unsigned short,3> buffer(src_dwi_data[0],voxel.dim);
    image::flip_xy(buffer);
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
            << src_bvectors[index][1] << " "
            << src_bvectors[index][2] << std::endl;
    }
    return out.good();
}
