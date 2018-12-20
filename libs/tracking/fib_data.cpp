#include <QCoreApplication>
#include <QFileInfo>
#include "fib_data.hpp"
#include "fa_template.hpp"
#include "tessellated_icosahedron.hpp"

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
    tipl::geometry<3> dim;
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

    for(int index = 1;index < index_data.size();++index)
    {
        // check index_data integrity
        for(int j = 0;j < index_data[index].size();++j)
            if(!index_data[index][j] || index_data[index].size() != num_fiber)
            {
                index_data.erase(index_data.begin()+index);
                index_name.erase(index_name.begin()+index);
                --index;
                break;
            }

        // identify dt indices
        if(index_name[index].find("inc_") != std::string::npos ||
           index_name[index].find("dec_") != std::string::npos)
        {
            dt_index_name.push_back(index_name[index]);
            dt_index_data.push_back(index_data[index]);
            index_data.erase(index_data.begin()+index);
            index_name.erase(index_name.begin()+index);
            --index;
            continue;
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
    cur_index = new_index;
    return true;
}
bool fiber_directions::set_tracking_index(const std::string& name)
{
    return set_tracking_index(std::find(index_name.begin(),index_name.end(),name)-index_name.begin());
}
bool fiber_directions::set_dt_index(int new_index)
{
    if(new_index >= dt_index_data.size() || new_index < 0)
    {
        dt_fa.clear();
        return false;
    }
    dt_fa = dt_index_data[new_index];
    dt_cur_index = new_index;
    return true;
}
bool fiber_directions::set_dt_index(const std::string& name)
{
    return set_dt_index(std::find(dt_index_name.begin(),dt_index_name.end(),name)-dt_index_name.begin());
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
                     const tipl::vector<3,float>& ref_dir, // reference direction, should be unit vector
                     unsigned char& fib_order_,
                     unsigned char& reverse_,
                     float threshold,
                     float cull_cos_angle,
                     float dt_threshold) const
{
    if(space_index >= dim.size())
        return false;
    float max_value = cull_cos_angle;
    unsigned char fib_order;
    unsigned char reverse;
    for (unsigned char index = 0;index < fib_num;++index)
    {
        if (fa[index][space_index] <= threshold)
            continue;
        if (!dt_fa.empty() && dt_fa[index][space_index] <= dt_threshold) // for differential tractography
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
    dt_fa = fib.dir.dt_fa;
    findex = fib.dir.findex;
    dir = fib.dir.dir;
    other_index = fib.dir.index_data;
    threshold_name = fib.dir.index_name[fib.dir.cur_index];
    if(!dt_fa.empty())
        dt_threshold_name = fib.dir.dt_index_name[fib.dir.dt_cur_index];
}
bool tracking_data::get_dir(unsigned int space_index,
                     const tipl::vector<3,float>& dir, // reference direction, should be unit vector
                     tipl::vector<3,float>& main_dir,
                            float threshold,
                            float cull_cos_angle,
                            float dt_threshold) const
{
    unsigned char fib_order;
    unsigned char reverse;
    if (!get_nearest_dir_fib(space_index,dir,fib_order,reverse,threshold,cull_cos_angle,dt_threshold))
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

float tracking_data::cos_angle(const tipl::vector<3>& cur_dir,unsigned int space_index,unsigned char fib_order) const
{
    if(!dir.empty())
    {
        const float* dir_at = dir[fib_order] + space_index + (space_index << 1);
        return cur_dir[0]*dir_at[0] + cur_dir[1]*dir_at[1] + cur_dir[2]*dir_at[2];
    }
    return cur_dir*odf_table[findex[fib_order][space_index]];
}

float tracking_data::get_track_specific_index(unsigned int space_index,unsigned int index_num,
                         const tipl::vector<3,float>& dir) const
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

bool tracking_data::is_white_matter(const tipl::vector<3,float>& pos,float t) const
{
    return tipl::estimate(tipl::make_image(fa[0],dim),pos) > t && pos[2] > 0.5;
}


bool fib_data::load_from_file(const char* file_name)
{
    tipl::image<float,3> I;
    tipl::vector<3,float> vs_;
    if(QFileInfo(file_name).completeSuffix() == "nii" ||
       QFileInfo(file_name).completeSuffix() == "nii.gz")
    {
        gz_nifti header;
        if(!header.load_from_file(file_name))
        {
            error_msg = "Invalid NIFTI format";
            return false;
        }
        if(header.dim(4) == 3)
        {
            tipl::image<float,3> x,y,z;
            header.get_voxel_size(vs_);
            header.toLPS(x,false);
            header.toLPS(y,false);
            header.toLPS(z,false);

            dim = x.geometry();
            vs = vs_;

            dir.check_index(0);
            dir.num_fiber = 1;
            dir.findex_buf.resize(1);
            dir.findex_buf[0].resize(x.size());
            dir.findex[0] = &*(dir.findex_buf[0].begin());
            dir.fa_buf.resize(1);
            dir.fa_buf[0].resize(x.size());
            dir.fa[0] = &*(dir.fa_buf[0].begin());

            dir.index_name.push_back("fiber");
            dir.index_data.push_back(dir.fa);
            tessellated_icosahedron ti;
            ti.init(8);
            dir.odf_faces = ti.faces;
            dir.odf_table = ti.vertices;
            dir.half_odf_size = ti.half_vertices_count;
            tipl::par_for(x.size(),[&](int i)
            {
                tipl::vector<3> v(-x[i],y[i],z[i]);
                float length = v.length();
                if(length == 0.0f)
                    return;
                v /= length;
                dir.fa_buf[0][i] = length;
                dir.findex_buf[0][i] = ti.discretize(v);
            });

            view_item.push_back(item());
            view_item.back().name =  "fiber";
            view_item.back().image_data = tipl::make_image(dir.fa[0],dim);
            view_item.back().set_scale(dir.fa[0],dir.fa[0]+dim.size());
            return true;
        }
        else
        {
            header.toLPS(I);
            header.get_voxel_size(vs_);
        }
    }
    else
    if(QFileInfo(file_name).fileName() == "2dseq")
    {
        tipl::io::bruker_2dseq bruker_header;
        if(!bruker_header.load_from_file(file_name))
        {
            error_msg = "Invalid 2dseq format";
            return false;
        }
        bruker_header.get_image().swap(I);
        bruker_header.get_voxel_size(vs_);
    }
    if(!I.empty())
    {
        mat_reader.add("dimension",I.geometry().begin(),3,1);
        mat_reader.add("voxel_size",&*vs_.begin(),3,1);
        mat_reader.add("image",&*I.begin(),I.size(),1);
        load_from_mat();
        dir.index_name[0] = "image";
        view_item[0].name = "image";
        return true;
    }
    if (!mat_reader.load_from_file(file_name) || prog_aborted())
    {
        error_msg = prog_aborted() ? "Loading process aborted" : "Invalid file format";
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
    view_item.back().image_data = tipl::make_image(dir.fa[0],dim);
    view_item.back().set_scale(dir.fa[0],dir.fa[0]+dim.size());
    for(unsigned int index = 1;index < dir.index_name.size();++index)
    {
        view_item.push_back(item());
        view_item.back().name =  dir.index_name[index];
        view_item.back().image_data = tipl::make_image(dir.index_data[index][0],dim);
        view_item.back().set_scale(dir.index_data[index][0],dir.index_data[index][0]+dim.size());
    }
    view_item.push_back(item());
    view_item.back() = view_item[0];
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
        view_item.back().image_data = tipl::make_image(buf,dim);
        view_item.back().set_scale(buf,buf+dim.size());
    }
    if (!dim[2])
    {
        error_msg = "invalid dimension";
        return false;
    }

    {
        const float* mx = 0;
        const float* my = 0;
        const float* mz = 0;
        if(!is_qsdr &&
           mat_reader.read("mni_x",row,col,mx) &&
           mat_reader.read("mni_y",row,col,my) &&
           mat_reader.read("mni_z",row,col,mz))
        {
            mni_position.resize(dim);
            for(int i = 0;i < dim.size();++i)
            {
                mni_position[i][0] = mx[i];
                mni_position[i][1] = my[i];
                mni_position[i][2] = mz[i];
            }
        }
        if(is_qsdr &&
           mat_reader.read("native_x",row,col,mx) &&
           mat_reader.read("native_y",row,col,my) &&
           mat_reader.read("native_z",row,col,mz))
        {
            native_position.resize(dim);
            for(int i = 0;i < dim.size();++i)
            {
                native_position[i][0] = mx[i];
                native_position[i][1] = my[i];
                native_position[i][2] = mz[i];
            }
        }
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
                view_item[i].mx = tipl::make_image(mx,dim);
                view_item[i].my = tipl::make_image(my,dim);
                view_item[i].mz = tipl::make_image(mz,dim);
                view_item[i].native_geo = tipl::geometry<3>(native_geo[0],native_geo[1],native_geo[2]);
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

std::pair<float,float> fib_data::get_value_range(const std::string& view_name) const
{
    unsigned int view_index = get_name_index(view_name);
    if(view_index == view_item.size())
        return std::make_pair((float)0.0,(float)0.0);
    if(view_item[view_index].name == "color")
        return std::make_pair(view_item[0].min_value,view_item[0].max_value);
    return std::make_pair(view_item[view_index].min_value,view_item[view_index].max_value);
}

void fib_data::get_slice(unsigned int view_index,
               unsigned char d_index,unsigned int pos,
               tipl::color_image& show_image,const tipl::value_to_color<float>& v2c)
{
    if(view_item[view_index].name == "color")
    {
        {
            tipl::image<float,2> buf;
            tipl::volume2slice(view_item[0].image_data, buf, d_index, pos);
            v2c.convert(buf,show_image);
        }

        if(view_item[view_index].color_map_buf.empty())
        {
            view_item[view_index].color_map_buf.resize(dim);
            std::iota(view_item[view_index].color_map_buf.begin(),
                      view_item[view_index].color_map_buf.end(),0);
        }
        tipl::image<unsigned int,2> buf;
        tipl::volume2slice(view_item[view_index].color_map_buf, buf, d_index, pos);
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
        tipl::image<float,2> buf;
        tipl::volume2slice(view_item[view_index].image_data, buf, d_index, pos);
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
    {
        if(view_item[i].name == "color")
            continue;
        if(view_item[i].image_data.geometry() != dim)
        {
            tipl::vector<3> pos(x,y,z);
            pos.to(view_item[i].iT);
            buf.push_back(tipl::estimate(view_item[i].image_data,pos));
        }
        else
            buf.push_back(view_item[i].image_data.empty() ? 0.0 : view_item[i].image_data[index]);
    }
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

void fib_data::run_normalization(bool background)
{
    prog = 0;
    auto lambda = [this]()
    {
        if(fa_template_imp.I.empty() && !fa_template_imp.load_from_file())
        {
            std::cout << fa_template_imp.error_msg << std::endl;
            prog = 5;
            return;
        }
        auto& It = fa_template_imp.I;
        tipl::transformation_matrix<float> T;
        tipl::image<float,3> Is(dir.fa[0],dim);
        tipl::filter::gaussian(Is);
        prog = 1;
        tipl::reg::two_way_linear_mr(It,fa_template_imp.vs,Is,vs,T,tipl::reg::affine,
                                     tipl::reg::mutual_information(),thread.terminated);
        prog = 2;
        if(thread.terminated)
            return;
        tipl::image<float,3> Iss(It.geometry());
        tipl::resample_mt(Is,Iss,T,tipl::linear);
        prog = 3;
        tipl::image<tipl::vector<3>,3> dis;
        tipl::reg::cdm(It,Iss,dis,thread.terminated,2.0f,0.95f);
        if(thread.terminated)
            return;
        prog = 4;
        tipl::image<tipl::vector<3,float>,3 > mni(Is.geometry());
        if(thread.terminated)
            return;
        T.inverse();
        mni.for_each_mt([&](tipl::vector<3,float>& v,const tipl::pixel_index<3>& pos)
        {
            tipl::vector<3> p(pos),d;
            T(p);
            v = p;
            tipl::estimate(dis,v,d,tipl::linear);
            v += d;
            fa_template_imp.to_mni(v);
        });
        if(thread.terminated)
            return;
        mni_position.swap(mni);
        prog = 5;
    };

    if(background)
    {
        thread.run(lambda);
    }
    else
    {
        std::cout << "Subject normalization to MNI space." << std::endl;
        lambda();
    }
}

void fib_data::subject2mni(tipl::vector<3>& pos)
{
    if(!is_human_data)
        return;
    if(is_qsdr)
    {
        pos.to(trans_to_mni);
        return;
    }
    tipl::vector<3> p;
    tipl::estimate(mni_position,pos,p);
    pos = p;
}
void fib_data::subject2mni(tipl::pixel_index<3>& index,tipl::vector<3>& pos)
{
    if(!is_human_data)
        return;
    if(is_qsdr)
    {
        pos = index;
        pos.to(trans_to_mni);
        return;
    }
    if(!mni_position.empty())
        mni_position[index.index()];
    return;
}

void fib_data::get_atlas_roi(atlas& at,int roi_index,std::vector<tipl::vector<3,short> >& points,float& r)
{
    if(get_mni_mapping().empty())
        return;
    // this will load the files from storage to prevent GUI multishread crash
    at.is_labeled_as(tipl::vector<3>(0,0,0), roi_index);
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<std::vector<tipl::vector<3,short> > > buf(thread_count);
    r = 1.0;
    mni_position.for_each_mt2([&](const tipl::vector<3>& mni,const tipl::pixel_index<3>& index,int id)
    {
        tipl::vector<3> rmni(mni);
        rmni.round();
        if (at.is_labeled_as(rmni, roi_index))
            buf[id].push_back(tipl::vector<3,short>(index.begin()));
    });
    points.clear();
    for(int i = 0;i < buf.size();++i)
        points.insert(points.end(),buf[i].begin(),buf[i].end());


}
const tipl::image<tipl::vector<3,float>,3 >& fib_data::get_mni_mapping(void)
{
    if(!mni_position.empty())
        return mni_position;
    if(is_qsdr)
    {
        mni_position.resize(dim);
        mni_position.for_each_mt([&](tipl::vector<3>& mni,const tipl::pixel_index<3>& index)
        {
            mni = index.begin();
            mni.to(trans_to_mni);
        });
        return mni_position;
    }
    run_normalization(false);
    return mni_position;
}
void smoothed_tracks(const std::vector<float>& track,std::vector<float>& smoothed);
void resample_tracks(const std::vector<float>& track,std::vector<float>& new_track,float interval);
bool fib_data::get_profile(const std::vector<float>& tract,
                 std::vector<float>& profile_)
{
    if(tract.size() < 6)
        return false;
    std::vector<float> tract_data;

    {
        std::vector<float> smoothed_track_in_mni;
        std::vector<float> tract_in_mni;
        for(int j = 0;j < tract.size();j += 3)
        {
            tipl::vector<3> v(&(tract[j]));
            subject2mni(v);
            tract_in_mni.push_back(v[0]);
            tract_in_mni.push_back(v[1]);
            tract_in_mni.push_back(v[2]);
        }
        smoothed_tracks(tract_in_mni,smoothed_track_in_mni);
        resample_tracks(smoothed_track_in_mni,tract_data,0.5);
    }

    tipl::geometry<3> dim(60,75,3);
    profile_.resize(dim.size());
    auto profile = tipl::make_image(&profile_[0],dim);
    std::fill(profile.begin(),profile.end(),0);
    float length_2 = tract_data.size() >> 1;
    bool has_point = false;
    for(int j = 0;j < tract_data.size();j += 3)
    {
        tipl::vector<3> v(&(tract_data[j]));
        // x = -60 ~ 60    total  120
        // y = -90 ~ 60    total  150
        // z = -50 ~ 70    total  120
        int x = std::floor((v[0]+60)*0.5+0.5);
        int y = std::floor((v[1]+90)*0.5+0.5);
        int z = std::floor((v[2]+50)*0.5+0.5);
        if(x < 0 || y < 0 || z < 0 ||
           x >= 60 || y >= 75 || z >= 60)
            continue;
        float w = std::fabs(j-length_2)*0.5f/(float)length_2+0.5;
        profile.at(x,y,0) = w;
        profile.at(x,z,1) = w;
        profile.at(z,y,2) = w;
        has_point = true;
    }
    return has_point;
}

bool track_recognition::can_recognize(void)
{
    if(!track_list.empty())
        return true;
    std::string file_name(QCoreApplication::applicationDirPath().toLocal8Bit().begin());
    file_name += "/network.net.gz";
    std::string track_label(QCoreApplication::applicationDirPath().toLocal8Bit().begin());
    track_label += "/network_label.txt";
    std::ifstream in(track_label.c_str());
    if(in && cnn.load_from_file<gz_istream>(file_name.c_str()))
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
        {
            std::cout << "Invalid network_label file" << std::endl;
            return false;
        }
        return true;
    }
    std::cout << "Cannot open " << file_name << std::endl;
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
    std::vector<float> profile;
    if(!handle->get_profile(tracks,profile))
        return;
    cnn_data.data.push_back(std::vector<float>());
    cnn_data.data.back().swap(profile);
    cnn_data.data_label.push_back(index);
}
