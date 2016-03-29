#include "fib_data.hpp"
#include "fa_template.hpp"
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


bool tracking::get_nearest_dir_fib(unsigned int space_index,
                     const image::vector<3,float>& ref_dir, // reference direction, should be unit vector
                     unsigned char& fib_order_,
                     unsigned char& reverse_) const
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
void tracking::read(const fib_data& fib)
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
bool tracking::get_dir(unsigned int space_index,
                     const image::vector<3,float>& dir, // reference direction, should be unit vector
                     image::vector<3,float>& main_dir) const
{
    unsigned char fib_order;
    unsigned char reverse;
    if (!get_nearest_dir_fib(space_index,dir,fib_order,reverse))
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

const float* tracking::get_dir(unsigned int space_index,unsigned char fib_order) const
{
    if(!dir.empty())
        return dir[fib_order] + space_index + (space_index << 1);
    return &*(odf_table[findex[fib_order][space_index]].begin());
}

float tracking::cos_angle(const image::vector<3>& cur_dir,unsigned int space_index,unsigned char fib_order) const
{
    if(!dir.empty())
    {
        const float* dir_at = dir[fib_order] + space_index + (space_index << 1);
        return cur_dir[0]*dir_at[0] + cur_dir[1]*dir_at[1] + cur_dir[2]*dir_at[2];
    }
    return cur_dir*odf_table[findex[fib_order][space_index]];
}

float tracking::get_track_specific_index(unsigned int space_index,unsigned int index_num,
                         const image::vector<3,float>& dir) const
{
    unsigned char fib_order;
    unsigned char reverse;
    if (!get_nearest_dir_fib(space_index,dir,fib_order,reverse))
        return 0.0;
    return other_index[index_num][fib_order][space_index];
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
    view_item.back().image_data = image::make_image(dim,dir.fa[0]);
    view_item.back().set_scale(dir.fa[0],dir.fa[0]+dim.size());
    for(unsigned int index = 1;index < dir.index_name.size();++index)
    {
        view_item.push_back(item());
        view_item.back().name =  dir.index_name[index];
        view_item.back().image_data = image::make_image(dim,dir.index_data[index][0]);
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
        view_item.back().image_data = image::make_image(dim,buf);
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
                view_item[i].mx = image::make_image(dim,mx);
                view_item[i].my = image::make_image(dim,my);
                view_item[i].mz = image::make_image(dim,mz);
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
            image::vector<3,float> dir(dir.get_dir(buf[index],0));
            show_image[index].r = std::abs((float)show_image[index].r*(float)dir[0]);
            show_image[index].g = std::abs((float)show_image[index].g*(float)dir[1]);
            show_image[index].b = std::abs((float)show_image[index].b*(float)dir[2]);
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
    {
        image::vector<3,float> dir(dir.get_dir(index,i));
        buf.push_back(dir[0]);
        buf.push_back(dir[1]);
        buf.push_back(dir[2]);
    }
}
void fib_data::get_voxel_information(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const
{
    unsigned int index = (z*dim[1]+y)*dim[0] + x;
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

void track_recognition::clear(void)
{
    cnn_test_label.clear();
    cnn_name.clear();
    cnn_test_data.clear();
    cnn_data.clear();
    cnn_label.clear();
    thread.clear();
}

void track_recognition::get_profile(fib_data* handle,
                 const std::vector<float>& tract_data,
                 image::basic_image<float,3>& profile)
{
    profile.resize(image::geometry<3>(64,80,3));
    std::fill(profile.begin(),profile.end(),0);
    for(unsigned int j = 0;j < tract_data.size();j += 3)
    {
        image::vector<3> v(&(tract_data[j]));
        handle->subject2mni(v);
        // x = -60 ~ 60    total  120
        // y = -90 ~ 60    total  150
        // z = -50 ~ 70    total  120
        int x = std::floor(v[0]+60);
        int y = std::floor(v[1]+90);
        int z = std::floor(v[2]+50);
        x >>= 1; // 2 mm
        y >>= 1; // 2 mm
        z >>= 1; // 2 mm
        if(x > 0 && x < profile.width())
        {
            if(y > 0 && y < profile.height())
                profile.at(x,y,0) = 3;
            if(z > 0 && z < profile.height())
                profile.at(x,z,1) = 3;
        }
        if(z > 0 && z < profile.width() && y > 0 && y < profile.height())
            profile.at(z,y,2) = 3;
    }
    image::filter::gaussian(profile.slice_at(0));
    image::filter::gaussian(profile.slice_at(1));
    image::filter::gaussian(profile.slice_at(2));
    image::minus_constant(profile,float(1));
}

void track_recognition::add_sample(fib_data* handle,int index,const std::vector<float>& tracks,int cv_fold)
{
    image::basic_image<float,3> profile;
    get_profile(handle,tracks,profile);
    if(cnn_test_data.size()*cv_fold < cnn_data.size()) // 20-fold cv
    {
        cnn_test_data.push_back(std::vector<float>(profile.begin(),profile.end()));
        cnn_test_label.push_back(index);
    }
    else
    {
        int insert_place = cnn_data.empty() ? 0:dist(cnn_data.size());
        cnn_data.insert(cnn_data.begin()+insert_place,std::vector<float>(profile.begin(),profile.end()));
        cnn_label.insert(cnn_label.begin()+insert_place,index);
    }
}

class timer
{
 public:
    timer():  t1(std::chrono::high_resolution_clock::now()){};
    double elapsed(){return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count();}
    void restart(){t1 = std::chrono::high_resolution_clock::now();}
    void start(){t1 = std::chrono::high_resolution_clock::now();}
    void stop(){t2 = std::chrono::high_resolution_clock::now();}
    double total(){stop();return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();}
    ~timer(){}
 private:
    std::chrono::high_resolution_clock::time_point t1, t2;
} t;

bool track_recognition::train(void)
{
    try{
        cnn.reset();
        cnn << "64,80,3|conv,tanh,3|62,78,6|avg_pooling,tanh,2|31,39,6|full,tanh|1,1,60|full,tanh"
            << image::geometry<3>(1,1,cnn_name.size());
    }
    catch(std::runtime_error error)
    {
        err_msg = error.what();
        return false;
    }

    thread.run([this]
    {
        auto on_enumerate_epoch = [this](){

            std::cout << "time:" << t.elapsed() << std::endl;
            int num_success(0);
            int num_total(0);
            std::vector<int> result;
            cnn.test(cnn_test_data,result);
            for (int i = 0; i < cnn_test_data.size(); i++)
            {
                if (result[i] == cnn_test_label[i])
                    num_success++;
                num_total++;
            }
            std::cout << "accuracy:" << num_success * 100.0 / num_total << "% (" << num_success << "/" << num_total << ")" << std::endl;
            t.restart();
            };
        t.start();
        cnn.train(cnn_data,cnn_label, 20,thread.terminated, on_enumerate_epoch,0.002);
    });
    return true;
}
