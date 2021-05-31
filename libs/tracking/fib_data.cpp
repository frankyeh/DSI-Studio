#include <QCoreApplication>
#include <QFileInfo>
#include "fib_data.hpp"
#include "tessellated_icosahedron.hpp"
#include "tract_model.hpp"
extern std::vector<std::string> fa_template_list;
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
                const float* odf = nullptr;
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
    if (!mat_reader.read("dimension",dim))
        return false;
    // odf_vertices
    {
        const float* odf_buffer;
        if (!mat_reader.read("odf_vertices",row,col,odf_buffer))
            return false;
        half_odf_size = col / 2;
    }
    const float* fa0 = nullptr;
    if (!mat_reader.read("fa0",row,col,fa0))
        return false;

    if (odfs)
    {
        voxel_index_map.resize(dim);
        for (unsigned int index = 0,j = 0;index < voxel_index_map.size();++index)
        {
            if (fa0[index] == 0.0f)
            {
                unsigned int from = j*(half_odf_size);
                unsigned int to = from + half_odf_size;
                if (to > odfs_size)
                    break;
                bool odf_is_zero = true;
                for (;from < to;++from)
                    if (odfs[from] != 0.0f)
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
                    if(odf_blocks[i][k] != 0.0f)
                    {
                        is_odf_zero = false;
                        break;
                    }
                if(!is_odf_zero)
                    for(;voxel_index < odf_block_map1.size();++voxel_index)
                        if(fa0[voxel_index] != 0.0f)
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
    if (odfs != nullptr)
    {
        if (index >= voxel_index_map.size() || voxel_index_map[index] == 0)
            return nullptr;
        return odfs+(voxel_index_map[index]-1)*half_odf_size;
    }

    if (!odf_blocks.empty())
    {
        if (index >= odf_block_map2.size())
            return nullptr;
        return odf_blocks[odf_block_map1[index]] + odf_block_map2[index];
    }
    return nullptr;
}

extern bool has_gui;
tipl::const_pointer_image<float,3> item::get_image(void)
{
    if(!image_ready)
    {
        // delay read routine
        unsigned int row,col;
        const float* buf = nullptr;
        has_gui = false;
        if (!mat_reader->read(image_index,row,col,buf))
        {
            dummy.resize(image_data.geometry());
            image_data = tipl::make_image(&*dummy.begin(),dummy.geometry());
        }
        else
        {
            mat_reader->in->flush();
            image_data = tipl::make_image(buf,image_data.geometry());
        }
        has_gui = true;
        image_ready = true;
        set_scale(image_data.begin(),image_data.end());
    }
    return image_data;
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

    for (unsigned int index = 0;index < mat_reader.size();++index)
    {
        std::string matrix_name = mat_reader.name(index);
        if (matrix_name == "image")
        {
            check_index(0);
            mat_reader.read(index,row,col,fa[0]);
            findex_buf.resize(1);
            findex_buf[0].resize(size_t(row)*size_t(col));
            findex[0] = &*(findex_buf[0].begin());
            odf_table.resize(2);
            half_odf_size = 1;
            odf_faces.clear();
            continue;
        }

        // read fiber wise index (e.g. my_fa0,my_fa1,my_fa2)
        std::string prefix_name(matrix_name.begin(),matrix_name.end()-1); // the "my_fa" part
        auto last_ch = matrix_name[matrix_name.length()-1]; // the index value part
        if (last_ch < '0' || last_ch > '9')
            continue;
        uint32_t store_index = uint32_t(last_ch-'0');
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

        auto prefix_name_index = size_t(std::find(index_name.begin(),index_name.end(),prefix_name)-index_name.begin());
        if(prefix_name_index == index_name.size())
        {
            index_name.push_back(prefix_name);
            index_data.push_back(std::vector<const float*>());
        }

        if(index_data[prefix_name_index].size() <= size_t(store_index))
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
        if(index_name[index].find("inc_") == 0 ||
           index_name[index].find("dec_") == 0)
        {
            dt_index_name.push_back(index_name[index]);
            dt_index_data.push_back(index_data[index]);
            //index_data.erase(index_data.begin()+index);
            //index_name.erase(index_name.begin()+index);
            //--index;
            //continue;
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

float fiber_directions::get_fa(size_t index,unsigned char order) const
{
    if(order >= fa.size())
        return 0.0;
    return fa[order][index];
}
float fiber_directions::get_dt_fa(size_t index,unsigned char order) const
{
    if(order >= dt_fa.size())
        return 0.0;
    return dt_fa[order][index];
}


const float* fiber_directions::get_dir(size_t index,unsigned int order) const
{
    if(!dir.empty())
        return dir[order] + index + (index << 1);
    if(order >= findex.size())
        return &*(odf_table[0].begin());
    return &*(odf_table[findex[order][index]].begin());
}

float fiber_directions::cos_angle(const tipl::vector<3>& cur_dir,unsigned int space_index,unsigned char fib_order) const
{
    if(!dir.empty())
    {
        const float* dir_at = dir[fib_order] + space_index + (space_index << 1);
        return cur_dir[0]*dir_at[0] + cur_dir[1]*dir_at[1] + cur_dir[2]*dir_at[2];
    }
    return cur_dir*odf_table[findex[fib_order][space_index]];
}


float fiber_directions::get_track_specific_index(unsigned int space_index,const std::vector<const float*>& index,
                         const tipl::vector<3,float>& dir) const
{
    if(fa[0][space_index] == 0.0)
        return 0.0;
    unsigned char fib_order = 0;
    float max_value = std::abs(cos_angle(dir,space_index,0));
    for (unsigned char index = 1;index < fa.size();++index)
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
    return index[fib_order][space_index];
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
    unsigned char fib_order = 0;
    unsigned char reverse = 0;
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
    if (max_value <= cull_cos_angle)
        return false;
    fib_order_ = fib_order;
    reverse_ = reverse;
    return true;
}
void tracking_data::read(std::shared_ptr<fib_data> fib)
{
    dim = fib->dim;
    vs = fib->vs;
    odf_table = fib->dir.odf_table;
    fib_num = uint8_t(fib->dir.num_fiber);
    fa = fib->dir.fa;
    dt_fa = fib->dir.dt_fa;
    findex = fib->dir.findex;
    dir = fib->dir.dir;
    other_index = fib->dir.index_data;
    if(!fib->dir.index_name.empty())
        threshold_name = fib->dir.get_threshold_name();
    if(!dt_fa.empty())
        dt_threshold_name = fib->dir.get_dt_threshold_name();
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
    return &odf_table[findex[fib_order][space_index]][0];
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



bool tracking_data::is_white_matter(const tipl::vector<3,float>& pos,float t) const
{
    return tipl::estimate(tipl::make_image(fa[0],dim),pos) > t && pos[2] > 0.5;
}

size_t match_template(float volume);
void initial_LPS_nifti_srow(tipl::matrix<4,4,float>& T,const tipl::geometry<3>& geo,const tipl::vector<3>& vs)
{
    std::fill(T.begin(),T.end(),0.0f);
    T[0] = -vs[0];
    T[5] = -vs[1];
    T[10] = vs[2];
    T[3] = vs[0]*(geo[0]-1);
    T[7] = vs[1]*(geo[1]-1);
    T[15] = 1.0f;
}

fib_data::fib_data(tipl::geometry<3> dim_,tipl::vector<3> vs_):dim(dim_),vs(vs_)
{
    initial_LPS_nifti_srow(trans_to_mni,dim,vs);
}

fib_data::fib_data(tipl::geometry<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4,float>& trans_to_mni_):
    dim(dim_),vs(vs_),trans_to_mni(trans_to_mni_)
{}

bool load_fib_from_tracks(const char* file_name,tipl::image<float,3>& I,tipl::vector<3>& vs);
void prepare_idx(const char* file_name,std::shared_ptr<gz_istream> in);
void save_idx(const char* file_name,std::shared_ptr<gz_istream> in);
bool fib_data::load_from_file(const char* file_name)
{
    tipl::image<float,3> I;
    gz_nifti header;
    fib_file_name = file_name;
    if((QFileInfo(file_name).fileName().endsWith(".nii") ||
        QFileInfo(file_name).fileName().endsWith(".nii.gz")) &&
        header.load_from_file(file_name))
    {
        if(header.dim(4) == 3)
        {
            tipl::image<float,3> x,y,z;
            header.get_voxel_size(vs);
            header.toLPS(x,false);
            header.toLPS(y,false);
            header.toLPS(z,false);
            dim = x.geometry();
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

            view_item.push_back(item("fiber",dir.fa[0],dim));
            return true;
        }
        else
        if(header.dim(4) && header.dim(4) % 3 == 0)
        {
            uint32_t fib_num = header.dim(4)/3;
            for(uint32_t i = 0;i < fib_num;++i)
            {
                tipl::image<float,3> x,y,z;
                header.get_voxel_size(vs);
                header.toLPS(x,false);
                header.toLPS(y,false);
                header.toLPS(z,false);
                if(i == 0)
                {
                    dim = x.geometry();
                    dir.check_index(fib_num-1);
                    dir.num_fiber = fib_num;
                    dir.findex_buf.resize(fib_num);
                    dir.fa_buf.resize(fib_num);
                }

                dir.findex_buf[i].resize(x.size());
                dir.findex[i] = &*(dir.findex_buf[i].begin());
                dir.fa_buf[i].resize(x.size());
                dir.fa[i] = &*(dir.fa_buf[i].begin());

                tessellated_icosahedron ti;
                ti.init(8);
                dir.odf_faces = ti.faces;
                dir.odf_table = ti.vertices;
                dir.half_odf_size = ti.half_vertices_count;
                tipl::par_for(x.size(),[&](uint32_t j)
                {
                    tipl::vector<3> v(x[j],y[j],-z[j]);
                    float length = float(v.length());
                    if(length == 0.0f || std::isnan(length))
                        return;
                    v /= length;
                    dir.fa_buf[i][j] = length;
                    dir.findex_buf[i][j] = short(ti.discretize(v));
                });

            }
            dir.index_name.push_back("fiber");
            dir.index_data.push_back(dir.fa);
            view_item.push_back(item("fiber",dir.fa[0],dim));
            return true;
        }
        else
        {
            header.toLPS(I);
            header.get_voxel_size(vs);
            header.get_image_transformation(trans_to_mni);
            is_mni_image = true;
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
        bruker_header.get_voxel_size(vs);

        std::ostringstream out;
        out << "Image resolution is (" << vs[0] << "," << vs[1] << "," << vs[2] << ")." << std::endl;
        report = out.str();

    }
    else
    if(QString(file_name).endsWith("trk.gz") ||
       QString(file_name).endsWith("trk") ||
       QString(file_name).endsWith("tck") ||
       QString(file_name).endsWith("tt.gz"))
    {
        if(!load_fib_from_tracks(file_name,I,vs))
        {
            error_msg = "Invalid track format";
            return false;
        }
    }
    if(!I.empty())
    {
        mat_reader.add("dimension",I.geometry());
        mat_reader.add("voxel_size",vs);
        mat_reader.add("image",I);
        load_from_mat();
        dir.index_name[0] = "image";
        view_item[0].name = "image";
        trackable = false;
        return true;
    }
    if(!QFileInfo(file_name).exists())
    {
        error_msg = "File not exist";
        return false;
    }


    //  prepare idx file
    prepare_idx(file_name,mat_reader.in);
    if(mat_reader.in->has_access_points())
    {
        mat_reader.delay_read = true;
        mat_reader.in->buffer_all = false;
    }
    if (!mat_reader.load_from_file(file_name) || prog_aborted())
    {
        error_msg = prog_aborted() ? "Loading process aborted" : "Invalid file format";
        return false;
    }
    save_idx(file_name,mat_reader.in);

    if(!load_from_mat())
        return false;
    return true;
}
bool fib_data::save_mapping(const std::string& index_name,const std::string& file_name,const tipl::value_to_color<float>& v2c)
{
    if(index_name == "fiber")
    {
        gz_nifti file;
        file.set_voxel_size(vs);
        tipl::image<float,4> buf(tipl::geometry<4>(
                                 dim.width(),
                                 dim.height(),
                                 dim.depth(),3*int(dir.num_fiber)));

        for(unsigned int j = 0,index = 0;j < dir.num_fiber;++j)
        for(int k = 0;k < 3;++k)
        for(unsigned int i = 0;i < dim.size();++i,++index)
            buf[index] = dir.get_dir(i,j)[k];

        tipl::flip_xy(buf);
        file << buf;
        return file.save_to_file(file_name.c_str());
    }
    if(index_name == "odfs" && odf.has_odfs())
    {
        gz_nifti file;
        file.set_voxel_size(vs);
        tipl::image<float,4> buf(tipl::geometry<4>(
                                 dim.width(),
                                 dim.height(),
                                 dim.depth(),
                                 dir.half_odf_size));
        for(unsigned int pos = 0;pos < dim.size();++pos)
        {
            auto* ptr = odf.get_odf_data(pos);
            if(ptr!= nullptr)
                std::copy(ptr,ptr+dir.half_odf_size,buf.begin()+pos*dir.half_odf_size);

        }
        tipl::flip_xy(buf);
        file << buf;
        return file.save_to_file(file_name.c_str());
    }
    size_t index = get_name_index(index_name);
    if(index >= view_item.size())
        return false;

    if(index_name == "color")
    {
        tipl::image<tipl::rgb,3> buf(dim);
        for(int z = 0;z < buf.depth();++z)
        {
            tipl::color_image I;
            get_slice(uint32_t(index),uint8_t(2),uint32_t(z),I,v2c);
            std::copy(I.begin(),I.end(),buf.begin()+size_t(z)*buf.plane_size());
        }
        return gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni);
    }


    if(QFileInfo(QString(file_name.c_str())).completeSuffix().toLower() == "mat")
    {
        tipl::io::mat_write file(file_name.c_str());
        file << view_item[index].get_image();
        return true;
    }
    else
    {
        tipl::image<float,3> buf(view_item[index].get_image());
        if(view_item[index].get_image().geometry() != dim)
        {
            tipl::image<float,3> new_buf(dim);
            tipl::resample(buf,new_buf,view_item[index].iT,tipl::cubic);
            new_buf.swap(buf);
        }
        return gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni);
    }
}
bool is_human_size(tipl::geometry<3> dim,tipl::vector<3> vs)
{
    return dim[0]*vs[0] > 130 && dim[1]*vs[1] > 180;
}
bool fib_data::load_from_mat(void)
{
    mat_reader.read("report",report);
    mat_reader.read("steps",steps);
    if (!mat_reader.read("dimension",dim))
    {
        error_msg = "cannot find dimension matrix";
        return false;
    }
    if(!dim.size())
    {
        error_msg = "invalid dimension";
        return false;
    }
    if (!mat_reader.read("voxel_size",vs))
    {
        error_msg = "cannot find voxel_size matrix";
        return false;
    }
    if (mat_reader.read("trans",trans_to_mni))
        is_qsdr = true;

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

    view_item.push_back(item(dir.fa.size() == 1 ? "fa":"qa",dir.fa[0],dim));
    for(unsigned int index = 1;index < dir.index_name.size();++index)
        view_item.push_back(item(dir.index_name[index],dir.index_data[index][0],dim));
    view_item.push_back(item("color",dir.fa[0],dim));

    // read other DWI space volume
    for (unsigned int index = 0;index < mat_reader.size();++index)
    {
        std::string matrix_name = mat_reader.name(index);
        if (matrix_name == "image")
            continue;
        std::string prefix_name(matrix_name.begin(),matrix_name.end()-1);
        char post_fix = matrix_name[matrix_name.length()-1];
        if(post_fix >= '0' && post_fix <= '9')
        {
            if (prefix_name == "index" || prefix_name == "fa" || prefix_name == "dir" ||
                std::find_if(view_item.begin(),
                             view_item.end(),
                             [&prefix_name](const item& view)
                             {return view.name == prefix_name;}) != view_item.end())
                continue;
        }
        if (size_t(mat_reader[index].get_rows())*size_t(mat_reader[index].get_cols()) != dim.size())
            continue;
        view_item.push_back(item(matrix_name,dim,&mat_reader,index));
    }

    is_human_data = is_human_size(dim,vs); // 1 percentile head size in mm
    db.read_db(this);

    if(is_qsdr)
    {
        if(mat_reader.has("native_mapping"))
        {
            mat_reader.read("native_dimension",native_geo);
            mat_reader.read("native_voxel_size",native_vs);
        }
        for(unsigned int i = 0; i < view_item.size();++i)
        {
            mat_reader.read((view_item[i].name+"_dimension").c_str(),view_item[i].native_geo);
            mat_reader.read((view_item[i].name+"_trans").c_str(),view_item[i].native_trans);
        }

        std::string template_name = mat_reader.read<std::string>("template_name");
        // matching templates
        for(size_t index = 0;index < fa_template_list.size();++index)
        {
            if(QFileInfo(fa_template_list[index].c_str()).baseName().toStdString() == template_name)
            {
                set_template_id(index);
                return true;
            }
            gz_nifti read;
            if(!read.load_from_file(fa_template_list[index]))
                continue;
            tipl::vector<3> Itvs;
            tipl::image<float,3> dummy;
            read.toLPS(dummy,true,false);
            read.get_voxel_size(Itvs);
            if(std::abs(dim[0]-read.nif_header2.dim[1]*Itvs[0]/vs[0]) < 2.0f)
            {
                set_template_id(index);
                return true;
            }
        }
        // older version of QSDR
        set_template_id(0);
        return true;
    }

    if(!is_mni_image)
        initial_LPS_nifti_srow(trans_to_mni,dim,vs);

    // template matching
    // check if there is any mapping files exist
    for(size_t index = 0;index < fa_template_list.size();++index)
    {
        QString name = QFileInfo(fa_template_list[index].c_str()).baseName().toLower();
        if(QFileInfo(fib_file_name.c_str()).fileName().contains(name) ||
           QFileInfo(QString(fib_file_name.c_str())+"."+name+".mapping.gz").exists() ||
           QFileInfo(QString(fib_file_name.c_str())+"."+name+".inv.mapping.gz").exists())
        {
            set_template_id(index);
            return true;
        }
    }

    if(!is_human_data)
    {
        size_t count = 0;
        for(size_t i = 0;i < dim.size();++i)
            if(dir.fa[0][i] > 0.0f)
                ++count;
        set_template_id(match_template(count*2.0f*vs[0]*vs[1]*vs[2]));
        return true;
    }
    set_template_id(0);
    return true;
}

const tipl::image<tipl::vector<3,float>,3 >& fib_data::get_native_position(void) const
{
    if(native_position.empty() && mat_reader.has("native_mapping"))
    {
        unsigned int row,col;
        const float* mapping = nullptr;
        if(mat_reader.read("native_mapping",row,col,mapping))
        {
            native_position.resize(dim);
            std::copy(mapping,mapping+col*row,&native_position[0][0]);
        }
    }
    return native_position;
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
            tipl::volume2slice(view_item[0].get_image(), buf, d_index, pos);
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
        tipl::volume2slice(view_item[view_index].get_image(), buf, d_index, pos);
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
        if(view_item[i].name == "color" || !view_item[i].image_ready)
            continue;
        if(view_item[i].get_image().geometry() != dim)
        {
            tipl::vector<3> pos(x,y,z);
            pos.to(view_item[i].iT);
            buf.push_back(tipl::estimate(view_item[i].get_image(),pos));
        }
        else
            buf.push_back(view_item[i].get_image().size() ? view_item[i].get_image()[index] : 0.0);
    }
}
extern std::vector<std::string> fa_template_list,iso_template_list,atlas_file_list,track_atlas_file_list;
void fib_data::set_template_id(size_t new_id)
{
    if(new_id != template_id)
    {
        template_id = new_id;
        template_I.clear();
        mni_position.clear();
        atlas_list.clear();
        tractography_atlas_file_name.clear();
        tractography_name_list.clear();
        track_atlas.reset();
        // populate atlas list
        {
            std::string atlas_file = fa_template_list[template_id]+".atlas.txt";
            std::ifstream in(atlas_file);
            std::string line;
            while(in >> line)
            {
                for(size_t j = 0;j < atlas_file_list.size();++j)
                    if(QFileInfo(atlas_file_list[j].c_str()).baseName().toStdString() == line)
                    {
                        atlas_list.push_back(std::make_shared<atlas>());
                        atlas_list.back()->name = line;
                        atlas_list.back()->filename = atlas_file_list[j];
                        break;
                    }
            }
        }
        for(size_t j = 0;j < track_atlas_file_list.size();++j)
        {
            if(QFileInfo(track_atlas_file_list[j].c_str()).baseName() ==
               QFileInfo(fa_template_list[template_id].c_str()).baseName())
            {
                tractography_atlas_file_name = track_atlas_file_list[j];
                std::string tractography_name_list_file_name = tractography_atlas_file_name + ".txt";
                std::ifstream in(tractography_name_list_file_name);
                if(in)
                    std::copy(std::istream_iterator<std::string>(in),
                    std::istream_iterator<std::string>(),std::back_inserter(tractography_name_list));
                break;
            }
        }
    }
}
bool fib_data::load_template(void)
{
    if(!template_I.empty())
        return true;
    gz_nifti read;
    tipl::image<float,3> I;
    tipl::vector<3> I_vs;
    if(!read.load_from_file(fa_template_list[template_id].c_str()))
    {
        error_msg = "cannot load ";
        error_msg += fa_template_list[template_id];
        return false;
    }
    tipl::matrix<4,4,float> tran;
    read.toLPS(I);
    read.get_voxel_size(I_vs);
    read.get_image_transformation(template_trans_to_mni);
    float ratio = float(I.width()*I_vs[0])/float(dim[0]*vs[0]);
    if(ratio < 0.25f || ratio > 4.0f)
    {
        error_msg = "image resolution mismatch: ratio=";
        error_msg += std::to_string(ratio);
        return false;
    }
    template_shift[0] = template_trans_to_mni[3];
    template_shift[1] = template_trans_to_mni[7];
    template_shift[2] = template_trans_to_mni[11];
    template_I.swap(I);
    template_vs = I_vs;

    // load iso template if exists
    {
        gz_nifti read2;
        if(!iso_template_list[template_id].empty() &&
           read2.load_from_file(iso_template_list[template_id].c_str()))
            read2.toLPS(template_I2);
    }
    template_I *= 1.0f/float(tipl::mean(template_I));
    if(!template_I2.empty())
        template_I2 *= 1.0f/float(tipl::mean(template_I2));


    if(is_mni_image)
    {
        need_normalization = false;
        return true;
    }
    if(!is_qsdr)
    {
        need_normalization = true;
        return true;
    }

    std::string template_name = mat_reader.read<std::string>("template_name");
    if(template_name.empty())
        need_normalization = std::abs(float(dim[0])-template_I.width()*template_vs[0]/vs[0]) > 2;
    else
        need_normalization = QFileInfo(fa_template_list[template_id].c_str()).baseName().toStdString() != template_name;
    return true;
}

bool fib_data::load_track_atlas()
{
    if(tractography_atlas_file_name.empty() || tractography_name_list.empty())
    {
        error_msg = "no tractography atlas associated with the current template";
        return false;
    }
    if(!track_atlas.get())
    {
        track_atlas = std::make_shared<TractModel>(dim,vs,trans_to_mni);
        if(!track_atlas->load_from_file(tractography_atlas_file_name.c_str()))
        {
            error_msg = "failed to load tractography atlas";
            return false;
        }
        if(!load_template())
        {
            error_msg = "failed to load template";
            return false;
        }
        if(track_atlas->geo != template_I.geometry())
        {
            error_msg = "dimension mismatch between tractography atlas and template";
            return false;
        }

        {
            prog_init p("warping atlas tracks to subject space");
            run_normalization(true,true);
            if(prog_aborted())
                return false;
        }

        // warp tractography atlas to subject space
        auto& tract_data = track_atlas->get_tracts();
        tipl::par_for(tract_data.size(),[&](size_t i)
        {
            for(size_t j = 0;j < tract_data[i].size();j += 3)
            {
                tipl::vector<3> p(&tract_data[i][j]);
                template_to_mni(p);
                mni2subject(p);
                tract_data[i][j] = p[0];
                tract_data[i][j+1] = p[1];
                tract_data[i][j+2] = p[2];
            }
        });
    }
    return true;
}

void fib_data::template_to_mni(tipl::vector<3>& p)
{
    p[0] = -p[0];
    if(template_vs[0] != 1.0f)
        p[0] *= template_vs[0];
    p[1] = -p[1];
    if(template_vs[1] != 1.0f)
        p[1] *= template_vs[1];
    if(template_vs[2] != 1.0f)
        p[2] *= template_vs[2];
    p += template_shift;
}

void fib_data::template_from_mni(tipl::vector<3>& p)
{
    p -= template_shift;
    if(template_vs[2] != 1.0f)
        p[2] /= template_vs[2];
    if(template_vs[1] != 1.0f)
        p[1] /= template_vs[1];
    p[1] = -p[1];
    if(template_vs[0] != 1.0f)
        p[0] /= template_vs[0];
    p[0] = -p[0];


}
//---------------------------------------------------------------------------
unsigned int fib_data::find_nearest(const float* trk,unsigned int length,bool contain,float false_distance)
{
    auto norm1 = [](const float* v1,const float* v2){return std::fabs(v1[0]-v2[0])+std::fabs(v1[1]-v2[1])+std::fabs(v1[2]-v2[2]);};
    struct min_min{
        inline float operator()(float min_dis,const float* v1,const float* v2)
        {
            float d1 = std::fabs(v1[0]-v2[0]);
            if(d1 > min_dis)
                return min_dis;
            d1 += std::fabs(v1[1]-v2[1]);
            if(d1 > min_dis)
                return min_dis;
            d1 += std::fabs(v1[2]-v2[2]);
            if(d1 > min_dis)
                return min_dis;
            return d1;
        }
    }min_min_fun;
    if(length <= 6)
        return 9999;
    float best_distance = contain ? 50.0f : false_distance;
    const auto& tract_data = track_atlas->get_tracts();
    const auto& tract_cluster = track_atlas->get_cluster_info();
    size_t best_index = tract_data.size();
    if(contain)
    {
        for(size_t i = 0;i < tract_data.size();++i)
        {
            bool skip = false;
            float max_dis = 0.0f;
            for(size_t n = 0;n < length;n += 6)
            {
                float min_dis = norm1(&tract_data[i][0],trk+n);
                for(size_t m = 0;m < tract_data[i].size() && min_dis > max_dis;m += 3)
                    min_dis = min_min_fun(min_dis,&tract_data[i][m],trk+n);
                max_dis = std::max<float>(min_dis,max_dis);
                if(max_dis > best_distance)
                {
                    skip = true;
                    break;
                }
            }
            if(!skip)
            {
                best_distance = max_dis;
                best_index = i;
            }
        }
    }
    else
    {
        for(size_t i = 0;i < tract_data.size();++i)
        {
            if(min_min_fun(best_distance,&tract_data[i][0],trk) >= best_distance ||
                min_min_fun(best_distance,&tract_data[i][tract_data[i].size()-3],trk+length-3) >= best_distance ||
                min_min_fun(best_distance,&tract_data[i][tract_data[i].size()/3/2*3],trk+(length/3/2*3)) >= best_distance)
                continue;

            bool skip = false;
            float max_dis = 0.0f;
            for(size_t m = 0;m < tract_data[i].size();m += 3)
            {
                const float* tim = &tract_data[i][m];
                const float* trk_length = trk+length;
                float min_dis = norm1(tim,trk);
                for(const float* trk_n = trk;trk_n < trk_length && min_dis > max_dis;trk_n += 3)
                    min_dis = min_min_fun(min_dis,tim,trk_n);
                max_dis = std::max<float>(min_dis,max_dis);
                if(max_dis > best_distance)
                {
                    skip = true;
                    break;
                }
            }
            if(!skip)
            for(size_t n = 0;n < length;n += 3)
            {
                const float* ti0 = &tract_data[i][0];
                const float* ti_end = ti0+tract_data[i].size();
                const float* trk_n = trk+n;
                float min_dis = norm1(ti0,trk_n);
                for(const float* tim = ti0;tim < ti_end && min_dis > max_dis;tim += 3)
                    min_dis = min_min_fun(min_dis,tim,trk_n);
                max_dis = std::max<float>(min_dis,max_dis);
                if(max_dis > best_distance)
                {
                    skip = true;
                    break;
                }
            }
            if(!skip)
            {
                best_distance = max_dis;
                best_index = i;
            }
        }
    }
    if(best_index == tract_data.size())
        return 9999;
    return tract_cluster[best_index];
}
//---------------------------------------------------------------------------

bool fib_data::recognize(std::shared_ptr<TractModel>& trk,std::vector<unsigned int>& result,float tolerance)
{
    if(!load_track_atlas())
        return false;
    result.resize(trk->get_tracts().size());
    tipl::par_for(trk->get_tracts().size(),[&](size_t i)
    {
        if(trk->get_tracts()[i].empty())
            return;
        result[i] = find_nearest(&(trk->get_tracts()[i][0]),uint32_t(trk->get_tracts()[i].size()),false,tolerance);
    });
    return true;
}

bool fib_data::recognize(std::shared_ptr<TractModel>& trk,std::map<float,std::string,std::greater<float> >& result,bool contain)
{
    if(!load_track_atlas())
        return false;
    std::vector<float> count(tractography_name_list.size());
    tipl::par_for(trk->get_tracts().size(),[&](size_t i)
    {
        if(trk->get_tracts()[i].empty())
            return;
        unsigned int index = find_nearest(&(trk->get_tracts()[i][0]),uint32_t(trk->get_tracts()[i].size()),contain,50.0f);
        if(index < count.size())
            ++count[index];
    });
    float sum = std::accumulate(count.begin(),count.end(),0.0f);
    if(sum != 0.0f)
        tipl::multiply_constant(count,1.0f/sum);
    result.clear();
    for(size_t i = 0;i < count.size();++i)
        result[count[i]] = tractography_name_list[i];
    return true;
}
void fib_data::recognize_report(std::shared_ptr<TractModel>& trk,std::string& report)
{
    std::map<float,std::string,std::greater<float> > result;
    if(!recognize(trk,result,true)) // true: connectometry may only show part of pathways. enable containing condition
        return;
    size_t n = 0;
    std::ostringstream out;
    for(auto& r : result)
    {
        if(r.first < 0.01f) // only report greater than 1%
            break;
        if(n)
            out << (n == result.size()-1 ? (result.size() == 2 ? " and ":", and ") : ", ");
        out <<  r.second;
        ++n;
    }
    report += out.str();
}


void animal_reg(const tipl::image<float,3>& from,tipl::vector<3> from_vs,
          const tipl::image<float,3>& to,tipl::vector<3> to_vs,
          tipl::transformation_matrix<double>& T,bool& terminated)
{
    float PI = 3.14159265358979323846f;
    float directions[5][3]={
        {0.0f,0.0f,0.0f},
        {PI*0.5f,0.0f,0.0f},
        {PI*-0.5f,0.0f,0.0f},
        {PI*0.5f,0.0f,PI},
        {PI*-0.5f,0.0f,PI}
    };
    float cost = std::numeric_limits<float>::max();
    tipl::par_for(5,[&](int i)
    {
         tipl::affine_transform<double> arg;
         std::copy(directions[i],directions[i]+3,arg.rotation);
         float cur_cost = tipl::reg::linear_mr(from,from_vs,to,to_vs,arg,
            tipl::reg::affine,tipl::reg::mutual_information(),terminated,0.01,tipl::reg::large_bound);
         if(cur_cost < cost)
         {
             cost = cur_cost;
             T = tipl::transformation_matrix<double>(arg,from.geometry(),from_vs,to.geometry(),to_vs);
         }
    });
}

void fib_data::run_normalization(bool background,bool inv)
{
    if(!need_normalization ||
       (!inv && !mni_position.empty()) ||
       (inv && !inv_mni_position.empty()))
        return;
    std::string output_file_name(fib_file_name);
    output_file_name += ".";
    output_file_name += QFileInfo(fa_template_list[template_id].c_str()).baseName().toLower().toStdString();
    output_file_name += inv ? ".inv.mapping.gz" : ".mapping.gz";
    gz_mat_read in;
    if(in.load_from_file(output_file_name.c_str()))
    {
        // check if the current fib files has the same recon steps as the one generating the maps
        std::string check_steps;
        in.read("steps",check_steps);
        if(check_steps.empty() || check_steps == steps)
        {
            tipl::image<tipl::vector<3,float>,3 > mni(inv ? template_I.geometry() : dim);
            const float* ptr = nullptr;
            unsigned int row,col;
            in.read("mapping",row,col,ptr);
            if(row == 3 && col == mni.size() && ptr)
            {
                std::copy(ptr,ptr+col*row,&mni[0][0]);
                if(inv)
                    inv_mni_position.swap(mni);
                else
                    mni_position.swap(mni);
                prog = 5;
                return;
            }
        }
    }
    if(background)
        begin_prog("running normalization");
    prog = 0;
    bool terminated = false;
    auto lambda = [this,output_file_name,inv,&terminated]()
    {
        tipl::transformation_matrix<double> T;

        auto It = template_I;
        auto It2 = template_I2;
        tipl::image<float,3> Is(dir.fa[0],dim);
        tipl::image<float,3> Is2;

        for(unsigned int i = 0;i < view_item.size();++i)
            if(view_item[i].name == std::string("iso"))
                Is2 = view_item[i].get_image();
        if(Is2.empty()) // DTI reconstruction
        for(unsigned int i = 0;i < view_item.size();++i)
            if(view_item[i].name == std::string("md"))
                Is2 = view_item[i].get_image();

        unsigned int downsampling = 0;

        while(Is.size() > It.size())
        {
            tipl::downsampling(Is);
            ++downsampling;
        }

        tipl::filter::gaussian(Is);
        prog = 1;
        if(!has_manual_atlas)
        {
            auto tvs = vs;
            tvs *= std::sqrt((It.plane_size()*template_vs[0]*template_vs[1])/
                    (Is.plane_size()*vs[0]*vs[1]));
            if(template_vs[0] < 1.0f) // animal
            {
                if(Is2.empty() || It2.empty())
                    animal_reg(It,template_vs,Is,tvs,T,terminated);
                else
                    animal_reg(It2,template_vs,Is2,tvs,T,terminated);
            }
            else
                tipl::reg::two_way_linear_mr(It,template_vs,Is,tvs,T,tipl::reg::affine,
                                             tipl::reg::mutual_information(),terminated);

            for(unsigned int i = 0;i < downsampling;++i)
                tipl::multiply_constant(T.data,T.data+12,2.0f);
        }
        else
            T = manual_template_T;

        if(terminated)
            return;
        prog = 2;
        tipl::image<float,3> Iss(It.geometry());
        tipl::resample_mt(Is,Iss,T,tipl::linear);
        tipl::image<float,3> Iss2(It.geometry());
        if(!Is2.empty())
            tipl::resample_mt(Is2,Iss2,T,tipl::linear);
        prog = 3;
        tipl::image<tipl::vector<3>,3> dis,inv_dis;
        tipl::reg::cdm_pre(It,It2,Iss,Iss2);
        if(Iss2.geometry() == Iss.geometry())
        {
            set_title("dual normalization");
            if(inv)
                tipl::reg::cdm2(It,It2,Iss,Iss2,dis,terminated);
            else
                tipl::reg::cdm2(Iss,Iss2,It,It2,inv_dis,terminated);
        }
        else
        {
            if(inv)
                tipl::reg::cdm(It,Iss,dis,terminated);
            else
                tipl::reg::cdm(Iss,It,inv_dis,terminated);
        }

        if(terminated)
            return;
        prog = 4;
        tipl::image<tipl::vector<3,float>,3 > mni(inv ? template_I.geometry() : dim);

        if(terminated)
            return;
        if(inv)
        {
            mni.for_each_mt([&](tipl::vector<3,float>& v,const tipl::pixel_index<3>& pos)
            {
                tipl::vector<3> p(pos);
                v = p;
                v += dis[pos.index()];
                T(v);
            });
            inv_mni_position.swap(mni);
        }
        else {
            T.inverse();
            mni.for_each_mt([&](tipl::vector<3,float>& v,const tipl::pixel_index<3>& pos)
            {
                tipl::vector<3> p(pos),d;
                T(p);
                v = p;
                tipl::estimate(inv_dis,v,d,tipl::linear);
                v += d;
                template_to_mni(v);
            });
            mni_position.swap(mni);
        }

        gz_mat_write out(output_file_name.c_str());
        if(out)
        {
            if(inv)
                out.write("mapping",&inv_mni_position[0][0],3,inv_mni_position.size());
            else
                out.write("mapping",&mni_position[0][0],3,mni_position.size());
            out.write("steps",steps);
        }
        prog = 5;
    };

    if(background)
    {
        std::thread t(lambda);
        while(check_prog(prog,5))
        {
            if(prog_aborted())
            {
                terminated = true;
                t.join();
                return;
            }
            if(inv)
            {
                if(!inv_mni_position.empty())
                    break;
            }
            else
            {
                if(!mni_position.empty())
                    break;
            }
        }
        check_prog(0,0);
        t.join();
    }
    else
    {
        std::cout << "Subject normalization to MNI space." << std::endl;
        lambda();
    }
}

bool fib_data::can_map_to_mni(void)
{
    if(!load_template())
        return false;
    run_normalization(true,false);
    if(prog_aborted())
    {
        error_msg = "action aborted by user";
        return false;
    }
    return true;
}

void sub2mni(tipl::vector<3>& pos,const tipl::matrix<4,4,float>& trans)
{
    if(trans[0] != 1.0f)
        pos[0] *= trans[0];
    if(trans[5] != 1.0f)
        pos[1] *= trans[5];
    if(trans[10] != 1.0f)
        pos[2] *= trans[10];
    pos[0] += trans[3];
    pos[1] += trans[7];
    pos[2] += trans[11];
}
void mni2sub(tipl::vector<3>& pos,const tipl::matrix<4,4,float>& trans)
{
    pos[0] -= trans[3];
    pos[1] -= trans[7];
    pos[2] -= trans[11];
    if(trans[0] != 1.0f)
        pos[0] /= trans[0];
    if(trans[5] != 1.0f)
        pos[1] /= trans[5];
    if(trans[10] != 1.0f)
        pos[2] /= trans[10];
}


void fib_data::mni2subject(tipl::vector<3>& pos)
{
    if(!need_normalization)
    {
        mni2sub(pos,trans_to_mni);
        return;
    }
    if(!inv_mni_position.empty())
    {
        template_from_mni(pos);
        tipl::vector<3> p;
        if(pos[2] < 0.0f)
            pos[2] = 0.0f;
        tipl::estimate(inv_mni_position,pos,p);
        pos = p;
    }
}

void fib_data::subject2mni(tipl::vector<3>& pos)
{
    if(!need_normalization)
    {
        sub2mni(pos,trans_to_mni);
        return;
    }
    if(!mni_position.empty())
    {
        tipl::vector<3> p;
        if(pos[2] < 0.0f)
            pos[2] = 0.0f;
        tipl::estimate(mni_position,pos,p);
        pos = p;
    }
}
void fib_data::subject2mni(tipl::pixel_index<3>& index,tipl::vector<3>& pos)
{
    if((is_qsdr || is_mni_image) && mni_position.empty())
    {
        pos = index;
        mni2sub(pos,trans_to_mni);
        return;
    }
    if(mni_position.empty())
        mni_position[index.index()];
    return;
}

void fib_data::get_atlas_roi(std::shared_ptr<atlas> at,unsigned int roi_index,std::vector<tipl::vector<3,short> >& points,float& r)
{
    if(get_mni_mapping().empty() || !at->load_from_file())
        return;
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<std::vector<tipl::vector<3,short> > > buf(thread_count);
    r = 1.0;
    mni_position.for_each_mt2([&](const tipl::vector<3>& mni,const tipl::pixel_index<3>& index,int id)
    {
        if (at->is_labeled_as(mni, roi_index))
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
    if(!need_normalization)
    {
        mni_position.resize(dim);
        mni_position.for_each_mt([&](tipl::vector<3>& mni,const tipl::pixel_index<3>& index)
        {
            mni = index.begin();
            sub2mni(mni,trans_to_mni);
        });
        return mni_position;
    }
    if(load_template())
        run_normalization(false,false);
    return mni_position;
}
