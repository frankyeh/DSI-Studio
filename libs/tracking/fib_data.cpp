#include <QCoreApplication>
#include "fib_data.hpp"
#include "fa_template.hpp"
#include "atlas.hpp"


extern std::vector<atlas> atlas_list;


bool odf_data::read(gz_mat_read& mat_reader)
{
    unsigned int row,col;
    {
        if(mat_reader.read("odfs",row,col,odfs))
            odfs_size = row*col;
        else
        {
            for(unsigned int index = 0;1;++index)
            {
                const float* odf = 0;
                std::ostringstream out;
                out << "odf" << index;
                std::string name = out.str();
                if(!mat_reader.read(name.c_str(),row,col,odf))
                    break;
                if(odf_blocks.size() <= index)
                {
                    odf_blocks.resize(index+1);
                    odf_block_size.resize(index+1);
                }
                odf_blocks[index] = odf;
                odf_block_size[index] = row*col;
            }
        }
    }
    if(!has_odfs())
        return false;

    // dimension
    image::geometry<3> dim;
    {
        const unsigned short* dim_buf = 0;
        if (!mat_reader.read("dimension",row,col,dim_buf))
            return false;
        std::copy(dim_buf,dim_buf+3,dim.begin());
    }
    // odf_vertices
    {
        const float* odf_buffer;
        if (!mat_reader.read("odf_vertices",row,col,odf_buffer))
            return false;
        half_odf_size = col / 2;
    }
    const float* fa0 = 0;
    if (!mat_reader.read("fa0",row,col,fa0))
        return false;

    if (odfs)
    {
        voxel_index_map.resize(dim);
        for (unsigned int index = 0,j = 0;index < voxel_index_map.size();++index)
        {
            if (fa0[index] == 0.0)
            {
                unsigned int from = j*(half_odf_size);
                unsigned int to = from + half_odf_size;
                if (to > odfs_size)
                    break;
                bool odf_is_zero = true;
                for (;from < to;++from)
                    if (odfs[from] != 0.0)
                    {
                        odf_is_zero = false;
                        break;
                    }
                if (!odf_is_zero)
                    continue;
            }
            ++j;
            voxel_index_map[index] = j;
        }
    }

    if (!odf_blocks.empty())
    {
        odf_block_map1.resize(dim);
        odf_block_map2.resize(dim);

        int voxel_index = 0;
        for(unsigned int i = 0;i < odf_block_size.size();++i)
            for(unsigned int j = 0;j < odf_block_size[i];j += half_odf_size)
            {
                unsigned int k_end = j + half_odf_size;
                bool is_odf_zero = true;
                for(unsigned int k = j;k < k_end;++k)
                    if(odf_blocks[i][k] != 0.0)
                    {
                        is_odf_zero = false;
                        break;
                    }
                if(!is_odf_zero)
                    for(;voxel_index < odf_block_map1.size();++voxel_index)
                        if(fa0[voxel_index] != 0.0)
                            break;
                if(voxel_index >= odf_block_map1.size())
                    break;
                odf_block_map1[voxel_index] = i;
                odf_block_map2[voxel_index] = j;
                ++voxel_index;
            }
    }
    return true;
}


const float* odf_data::get_odf_data(unsigned int index) const
{
    if (odfs != 0)
    {
        if (index >= voxel_index_map.size() || voxel_index_map[index] == 0)
            return 0;
        return odfs+(voxel_index_map[index]-1)*half_odf_size;
    }

    if (!odf_blocks.empty())
    {
        if (index >= odf_block_map2.size())
            return 0;
        return odf_blocks[odf_block_map1[index]] + odf_block_map2[index];
    }
    return 0;
}



void fiber_directions::check_index(unsigned int index)
{
    if (fa.size() <= index)
    {
        ++index;
        fa.resize(index);
        findex.resize(index);
        num_fiber = index;
    }
}

bool fiber_directions::add_data(gz_mat_read& mat_reader)
{
    unsigned int row,col;

    // odf_vertices
    {
        const float* odf_buffer;
        if (mat_reader.read("odf_vertices",row,col,odf_buffer))
        {
                odf_table.resize(col);
            for (unsigned int index = 0;index < odf_table.size();++index,odf_buffer += 3)
            {
                odf_table[index][0] = odf_buffer[0];
                odf_table[index][1] = odf_buffer[1];
                odf_table[index][2] = odf_buffer[2];
            }
            half_odf_size = col / 2;
        }
        else
        {
            odf_table.resize(2);
            half_odf_size = 1;
            odf_faces.clear();
        }
    }
    // odf_faces
    {
        const unsigned short* odf_buffer;
        if(mat_reader.read("odf_faces",row,col,odf_buffer))
        {
            odf_faces.resize(col);
            for (unsigned int index = 0;index < odf_faces.size();++index,odf_buffer += 3)
            {
                odf_faces[index][0] = odf_buffer[0];
                odf_faces[index][1] = odf_buffer[1];
                odf_faces[index][2] = odf_buffer[2];
            }
        }
    }

    for (unsigned int index = 0;check_prog(index,mat_reader.size());++index)
    {
        std::string matrix_name = mat_reader.name(index);
        if (matrix_name == "image")
        {
            check_index(0);
            mat_reader.read(index,row,col,fa[0]);
            findex_buf.resize(1);
            findex_buf[0].resize(row*col);
            findex[0] = &*(findex_buf[0].begin());
            odf_table.resize(2);
            half_odf_size = 1;
            odf_faces.clear();
            continue;
        }

        // prefix started here
        std::string prefix_name(matrix_name.begin(),matrix_name.end()-1);
        int store_index  = matrix_name[matrix_name.length()-1]-'0';
        if (store_index < 0 || store_index > 9)
            continue;

        if (prefix_name == "index")
        {
            check_index(store_index);
            mat_reader.read(index,row,col,findex[store_index]);
            continue;
        }
        if (prefix_name == "fa")
        {
            check_index(store_index);
            mat_reader.read(index,row,col,fa[store_index]);
            continue;
        }
        if (prefix_name == "dir")
        {
            const float* dir_ptr;
            mat_reader.read(index,row,col,dir_ptr);
            check_index(store_index);
            dir.resize(findex.size());
            dir[store_index] = dir_ptr;
            continue;
        }

        int prefix_name_index = 0;
        for(;prefix_name_index < index_name.size();++prefix_name_index)
            if(index_name[prefix_name_index] == prefix_name)
                break;
        if(prefix_name_index == index_name.size())
        {
            index_name.push_back(prefix_name);
            index_data.push_back(std::vector<const float*>());
        }

        if(index_data[prefix_name_index].size() <= store_index)
            index_data[prefix_name_index].resize(store_index+1);
        mat_reader.read(index,row,col,index_data[prefix_name_index][store_index]);

    }



    // adding the primary fiber index
    index_name.insert(index_name.begin(),fa.size() == 1 ? "fa":"qa");
    index_data.insert(index_data.begin(),fa);

    // check index_data integrity
    for(int index = 1;index < index_data.size();++index)
    {
        for(int j = 0;j < index_data[index].size();++j)
            if(!index_data[index][j] || index_data[index].size() != num_fiber)
            {
                index_data.erase(index_data.begin()+index);
                index_name.erase(index_name.begin()+index);
                --index;
                break;
            }
    }


    if(num_fiber == 0)
        error_msg = "No image data found";
    return num_fiber;
}

bool fiber_directions::set_tracking_index(int new_index)
{
    if(new_index >= index_data.size() || new_index < 0)
        return false;
    fa = index_data[new_index];
    return true;
}

bool fiber_directions::set_tracking_index(const std::string& name)
{
    return set_tracking_index(std::find(index_name.begin(),index_name.end(),name)-index_name.begin());
}
float fiber_directions::get_fa(unsigned int index,unsigned char order) const
{
    if(order >= fa.size())
        return 0.0;
    return fa[order][index];
}
const float* fiber_directions::get_dir(unsigned int index,unsigned int order) const
{
    if(!dir.empty())
        return dir[order] + index + (index << 1);
    if(order >= findex.size())
        return &*(odf_table[0].begin());
    return &*(odf_table[findex[order][index]].begin());
}


bool tracking_data::get_nearest_dir_fib(unsigned int space_index,
                     const image::vector<3,float>& ref_dir, // reference direction, should be unit vector
                     unsigned char& fib_order_,
                     unsigned char& reverse_,
                     float threshold,
                     float cull_cos_angle) const
{
    if(space_index >= dim.size() || fa[0][space_index] <= threshold)
        return false;
    float max_value = cull_cos_angle;
    unsigned char fib_order;
    unsigned char reverse;
    for (unsigned char index = 0;index < fib_num;++index)
    {
        if (fa[index][space_index] <= threshold)
            continue;
        float value = cos_angle(ref_dir,space_index,index);
        if (-value > max_value)
        {
            max_value = -value;
            fib_order = index;
            reverse = 1;
        }
        else
            if (value > max_value)
            {
                max_value = value;
                fib_order = index;
                reverse = 0;
            }
    }
    if (max_value == cull_cos_angle)
        return false;
    fib_order_ = fib_order;
    reverse_ = reverse;
    return true;
}
void tracking_data::read(const fib_data& fib)
{
    dim = fib.dim;
    vs = fib.vs;
    odf_table = fib.dir.odf_table;
    fib_num = fib.dir.num_fiber;
    fa = fib.dir.fa;
    findex = fib.dir.findex;
    dir = fib.dir.dir;
    other_index = fib.dir.index_data;
}
bool tracking_data::get_dir(unsigned int space_index,
                     const image::vector<3,float>& dir, // reference direction, should be unit vector
                     image::vector<3,float>& main_dir,
                            float threshold,
                            float cull_cos_angle) const
{
    unsigned char fib_order;
    unsigned char reverse;
    if (!get_nearest_dir_fib(space_index,dir,fib_order,reverse,threshold,cull_cos_angle))
        return false;
    main_dir = get_dir(space_index,fib_order);
    if(reverse)
    {
        main_dir[0] = -main_dir[0];
        main_dir[1] = -main_dir[1];
        main_dir[2] = -main_dir[2];
    }
    return true;
}

const float* tracking_data::get_dir(unsigned int space_index,unsigned char fib_order) const
{
    if(!dir.empty())
        return dir[fib_order] + space_index + (space_index << 1);
    return &*(odf_table[findex[fib_order][space_index]].begin());
}

float tracking_data::cos_angle(const image::vector<3>& cur_dir,unsigned int space_index,unsigned char fib_order) const
{
    if(!dir.empty())
    {
        const float* dir_at = dir[fib_order] + space_index + (space_index << 1);
        return cur_dir[0]*dir_at[0] + cur_dir[1]*dir_at[1] + cur_dir[2]*dir_at[2];
    }
    return cur_dir*odf_table[findex[fib_order][space_index]];
}

float tracking_data::get_track_specific_index(unsigned int space_index,unsigned int index_num,
                         const image::vector<3,float>& dir) const
{
    if(space_index >= dim.size() || fa[0][space_index] == 0.0)
        return 0.0;
    unsigned char fib_order = 0;
    float max_value = std::abs(cos_angle(dir,space_index,0));
    for (unsigned char index = 1;index < fib_num;++index)
    {
        if (fa[index][space_index] == 0.0)
            continue;
        float value = cos_angle(dir,space_index,index);
        if (-value > max_value)
        {
            max_value = -value;
            fib_order = index;
        }
        else
            if (value > max_value)
            {
                max_value = value;
                fib_order = index;
            }
    }
    return other_index[index_num][fib_order][space_index];
}

bool tracking_data::is_white_matter(const image::vector<3,float>& pos,float t) const
{
    return image::estimate(image::make_image(fa[0],dim),pos) > t;
}


bool fib_data::load_from_file(const char* file_name)
{
    if (!mat_reader.load_from_file(file_name) || prog_aborted())
    {
        error_msg = prog_aborted() ? "loading process aborted" : "cannot open file";
        return false;
    }
    return load_from_mat();
}
bool fib_data::load_from_mat(void)
{
    {
        unsigned int row,col;
        const char* report_buf = 0;
        if(mat_reader.read("report",row,col,report_buf))
            report = std::string(report_buf,report_buf+row*col);

        const unsigned short* dim_buf = 0;
        if (!mat_reader.read("dimension",row,col,dim_buf))
        {
            error_msg = "cannot find dimension matrix";
            return false;
        }
        std::copy(dim_buf,dim_buf+3,dim.begin());
    }
    if(!dir.add_data(mat_reader))
    {
        error_msg = dir.error_msg;
        return false;
    }

    if(dir.fa.empty())
    {
        error_msg = "Empty FA matrix";
        return false;
    }
    odf.read(mat_reader);

    view_item.push_back(item());
    view_item.back().name =  dir.fa.size() == 1 ? "fa":"qa";
    view_item.back().image_data = image::make_image(dir.fa[0],dim);
    view_item.back().set_scale(dir.fa[0],dir.fa[0]+dim.size());
    for(unsigned int index = 1;index < dir.index_name.size();++index)
    {
        view_item.push_back(item());
        view_item.back().name =  dir.index_name[index];
        view_item.back().image_data = image::make_image(dir.index_data[index][0],dim);
        view_item.back().set_scale(dir.index_data[index][0],dir.index_data[index][0]+dim.size());
    }
    view_item.push_back(item());
    view_item.back().name = "color";

    unsigned int row,col;
    for (unsigned int index = 0;check_prog(index,mat_reader.size());++index)
    {
        std::string matrix_name = mat_reader.name(index);
        if (matrix_name == "voxel_size")
        {
            const float* size_buf = 0;
            mat_reader.read(index,row,col,size_buf);
            if (!size_buf || row*col != 3)
                return false;
            vs = size_buf;
            if(vs[0]*vs[1]*vs[2] == 0.0)
                vs[0] = vs[1] = vs[2] = 2.0;
            continue;
        }
        if (matrix_name == "trans")
        {
            const float* trans = 0;
            mat_reader.read(index,row,col,trans);
            trans_to_mni.resize(16);
            std::copy(trans,trans+16,trans_to_mni.begin());
            is_qsdr = true;
            continue;
        }
        if (matrix_name == "image")
            continue;
        std::string prefix_name(matrix_name.begin(),matrix_name.end()-1);
        if (prefix_name == "index" || prefix_name == "fa" || prefix_name == "dir")
            continue;
        const float* buf = 0;
        mat_reader.read(index,row,col,buf);
        if (row*col != dim.size() || !buf)
            continue;
        if(matrix_name.length() >= 2 && matrix_name[matrix_name.length()-2] == '_' &&
           (matrix_name[matrix_name.length()-1] == 'x' ||
            matrix_name[matrix_name.length()-1] == 'y' ||
            matrix_name[matrix_name.length()-1] == 'z' ||
            matrix_name[matrix_name.length()-1] == 'd'))
            continue;
        if(matrix_name[matrix_name.length()-1] >= '0' && matrix_name[matrix_name.length()-1] <= '9')
            continue;
        view_item.push_back(item());
        view_item.back().name = matrix_name;
        view_item.back().image_data = image::make_image(buf,dim);
        view_item.back().set_scale(buf,buf+dim.size());
    }
    if (!dim[2])
    {
        error_msg = "invalid dimension";
        return false;
    }

    if(is_qsdr && !view_item.empty())
    {
        unsigned int row,col;
        const float* mx = 0;
        const float* my = 0;
        const float* mz = 0;
        const short* native_geo = 0;
        for(unsigned int i = 0; i < view_item.size();++i)
        {
            std::string name;
            if(i)
                name = view_item[i].name;
            if(mat_reader.read((name+"_x").c_str(),row,col,mx) &&
               mat_reader.read((name+"_y").c_str(),row,col,my) &&
               mat_reader.read((name+"_z").c_str(),row,col,mz) &&
                 mat_reader.read((name+"_d").c_str(),row,col,native_geo))
            {
                view_item[i].mx = image::make_image(mx,dim);
                view_item[i].my = image::make_image(my,dim);
                view_item[i].mz = image::make_image(mz,dim);
                view_item[i].native_geo = image::geometry<3>(native_geo[0],native_geo[1],native_geo[2]);
            }
        }
    }

    is_human_data = dim[0]*vs[0] > 100 && dim[1]*vs[1] > 120 && dim[2]*vs[2] > 40;
    db.read_db(this);
    return true;
}

size_t fib_data::get_name_index(const std::string& index_name) const
{
    for(unsigned int index_num = 0;index_num < view_item.size();++index_num)
        if(view_item[index_num].name == index_name)
            return index_num;
    return view_item.size();
}
void fib_data::get_index_list(std::vector<std::string>& index_list) const
{
    for (int index = 0; index < view_item.size(); ++index)
        if(view_item[index].name != "color")
            index_list.push_back(view_item[index].name);
}
image::const_pointer_image<float,3> fib_data::get_view_volume(const std::string& view_name) const
{
    unsigned int view_index = get_name_index(view_name);
    if(view_index == view_item.size() || view_item[view_index].name == "color")
        return image::const_pointer_image<float,3>();
    return view_item[view_index].image_data;
}
std::pair<float,float> fib_data::get_value_range(const std::string& view_name) const
{
    unsigned int view_index = get_name_index(view_name);
    if(view_index == view_item.size())
        return std::make_pair((float)0.0,(float)0.0);
    if(view_item[view_index].name == "color")
        return std::make_pair(view_item[0].min_value,view_item[0].max_value);
    return std::make_pair(view_item[view_index].min_value,view_item[view_index].max_value);
}

void fib_data::get_slice(const std::string& view_name,
               unsigned char d_index,unsigned int pos,
               image::color_image& show_image,const image::value_to_color<float>& v2c)
{
    unsigned int view_index = get_name_index(view_name);
    if(view_index == view_item.size())
        return;

    if(view_item[view_index].name == "color")
    {
        {
            image::basic_image<float,2> buf;
            image::reslicing(view_item[0].image_data, buf, d_index, pos);
            v2c.convert(buf,show_image);
        }

        if(view_item[view_index].color_map_buf.empty())
        {
            view_item[view_index].color_map_buf.resize(dim);
            for (unsigned int index = 0;index < dim.size();++index)
                view_item[view_index].color_map_buf[index] = index;
        }
        image::basic_image<unsigned int,2> buf;
        image::reslicing(view_item[view_index].color_map_buf, buf, d_index, pos);
        for (unsigned int index = 0;index < buf.size();++index)
        {
            const float* d = dir.get_dir(buf[index],0);
            show_image[index].r = std::abs((float)show_image[index].r*d[0]);
            show_image[index].g = std::abs((float)show_image[index].g*d[1]);
            show_image[index].b = std::abs((float)show_image[index].b*d[2]);
        }
    }
    else
    {
        image::basic_image<float,2> buf;
        image::reslicing(view_item[view_index].image_data, buf, d_index, pos);
        v2c.convert(buf,show_image);
    }

}

void fib_data::get_voxel_info2(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const
{
    unsigned int index = (z*dim[1]+y)*dim[0] + x;
    if (index >= dim.size())
        return;
    for(unsigned int i = 0;i < dir.num_fiber;++i)
        if(dir.get_fa(index,i) == 0.0f)
        {
            buf.push_back(0.0f);
            buf.push_back(0.0f);
            buf.push_back(0.0f);
        }
        else
        {
            const float* d = dir.get_dir(index,i);
            buf.push_back(d[0]);
            buf.push_back(d[1]);
            buf.push_back(d[2]);
        }
}
void fib_data::get_voxel_information(int x,int y,int z,std::vector<float>& buf) const
{
    if(!dim.is_valid(x,y,z))
        return;
    int index = (z*dim[1]+y)*dim[0] + x;
    if (index >= dim.size())
        return;
    for(unsigned int i = 0;i < view_item.size();++i)
        if(view_item[i].name != "color")
            buf.push_back(view_item[i].image_data.empty() ? 0.0 : view_item[i].image_data[index]);
}
void fib_data::get_index_titles(std::vector<std::string>& titles)
{
    std::vector<std::string> index_list;
    get_index_list(index_list);
    for(unsigned int index = 0;index < index_list.size();++index)
    {
        titles.push_back(index_list[index]+" mean");
        titles.push_back(index_list[index]+" sd");
    }
}
extern fa_template fa_template_imp;
bool fib_data::can_map_to_mni(void)
{
    if(!is_human_data)
        return false;
    if(is_qsdr)
        return true;
    if(!has_reg())
    {
        begin_prog("running normalization");
        run_normalization(1,true);
        while(check_prog(reg.get_prog(),18) && !prog_aborted())
            ;
        check_prog(16,16);
    }
    return true;
}

void fib_data::run_normalization(int factor,bool background)
{
    auto lambda = [this,factor]()
    {
        image::basic_image<float,3> from(dir.fa[0],dim),to(fa_template_imp.I);
        image::filter::gaussian(from);
        from -= image::segmentation::otsu_threshold(from);
        image::lower_threshold(from,0.0);
        image::normalize(from,1.0);
        image::normalize(to,1.0);
        reg.run_reg(from,vs,fa_template_imp.I,fa_template_imp.vs,
                    factor,image::reg::corr,image::reg::affine,thread.terminated,std::thread::hardware_concurrency());
    };

    if(background)
    {
        thread.run(lambda);
    }
    else
        lambda();
}

void fib_data::subject2mni(image::vector<3>& pos)
{
    if(!is_human_data)
        return;
    if(is_qsdr)
    {
        pos.to(trans_to_mni);
        return;
    }
    reg(pos);
    fa_template_imp.to_mni(pos);
}

void fib_data::get_atlas_roi(int atlas_index,int roi_index,std::vector<image::vector<3,short> >& points)
{
    points.clear();
    for (image::pixel_index<3>index(dim); index < dim.size(); ++index)
    {
        image::vector<3> mni(index.begin());
        subject2mni(mni);
        if (!atlas_list[atlas_index].is_labeled_as(mni, roi_index))
            continue;
        points.push_back(image::vector<3,short>(index.begin()));
    }
}

void fib_data::get_mni_mapping(image::basic_image<image::vector<3,float>,3 >& mni_position)
{
    mni_position.resize(dim);
    for (image::pixel_index<3>index(dim); index < dim.size();++index)
        if(dir.get_fa(index.index(),0) > 0)
        {
            image::vector<3,float> mni(index.begin());
            subject2mni(mni);
            mni_position[index.index()] = mni;
        }
}
void fib_data::get_profile(const std::vector<float>& tract_data,
                 std::vector<float>& profile_)
{
    if(tract_data.size() < 6)
        return;
    image::geometry<3> dim(64,80,3);
    profile_.resize(dim.size());
    auto profile = image::make_image(&profile_[0],dim);
    std::fill(profile.begin(),profile.end(),0);
    for(int j = 0;j < tract_data.size();j += 3)
    {
        image::vector<3> v(&(tract_data[j]));
        subject2mni(v);
        // x = -60 ~ 60    total  120
        // y = -90 ~ 60    total  150
        // z = -50 ~ 70    total  120
        int x = std::floor(v[0]+60);
        int y = std::floor(v[1]+90);
        int z = std::floor(v[2]+50);
        x >>= 1; // 2 mm
        y >>= 1; // 2 mm
        z >>= 1; // 2 mm
        float w = std::abs(j-int(tract_data.size() >> 1));
        if(x > 0 && x < profile.width())
        {
            if(y > 0 && y < profile.height())
                profile.at(x,y,0) += w;
            if(z > 0 && z < profile.height())
                profile.at(x,z,1) += w;
        }
        if(z > 0 && z < profile.width() && y > 0 && y < profile.height())
            profile.at(z,y,2) += w;
    }
    auto s1 = profile.slice_at(0);
    auto s2 = profile.slice_at(1);
    auto s3 = profile.slice_at(2);
    image::filter::gaussian(s1);
    image::filter::gaussian(s1);
    image::filter::gaussian(s2);
    image::filter::gaussian(s2);
    image::filter::gaussian(s3);
    image::filter::gaussian(s3);
    float m = *std::max_element(profile.begin(),profile.end());
    if(m != 0.0)
        image::multiply_constant(profile,1.8/m);
    image::minus_constant(profile,0.9);

}

bool track_recognition::can_recognize(void)
{
    if(!track_list.empty())
        return true;
    std::string file_name(QCoreApplication::applicationDirPath().toLocal8Bit().begin());
    file_name += "/network.bin";
    std::string track_label(QCoreApplication::applicationDirPath().toLocal8Bit().begin());
    track_label += "/network_label.txt";
    std::ifstream in(track_label.c_str());
    if(in && cnn.load_from_file(file_name.c_str()))
    {
        std::string line;
        while(std::getline(in,line))
        {
            track_list.push_back(line);
            track_name.push_back(line);
            std::string& name = track_name.back();
            if(name.back() == 'L' && name[name.length()-2] == '_')
                name = std::string("left ") + name.substr(0,name.length()-2);
            if(name.back() == 'R' && name[name.length()-2] == '_')
                name = std::string("right ") + name.substr(0,name.length()-2);
            std::transform(name.begin(),name.end(),name.begin(),::tolower);
            std::replace(name.begin(),name.end(),'_',' ');
        }
        if(track_list.size() != cnn.get_output_size())
            return false;
        return true;
    }
    return false;
}


void track_recognition::clear(void)
{
    cnn_name.clear();
    cnn_data.clear();
    thread.clear();
}



void track_recognition::add_sample(fib_data* handle,unsigned char index,const std::vector<float>& tracks)
{
    int insert_place = cnn_data.data.empty() ? 0:dist(cnn_data.data.size());
    cnn_data.data.insert(cnn_data.data.begin()+insert_place,std::vector<float>());
    handle->get_profile(tracks,cnn_data.data[insert_place]);
    cnn_data.data_label.insert(cnn_data.data_label.begin()+insert_place,index);
}
