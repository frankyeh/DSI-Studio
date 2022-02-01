#include <QCoreApplication>
#include <QFileInfo>
#include <QDateTime>
#include "fib_data.hpp"
#include "tessellated_icosahedron.hpp"
#include "tract_model.hpp"
#include "roi.hpp"
#include "mac_filesystem.hpp"

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
    tipl::shape<3> dim;
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

        unsigned int voxel_index = 0;
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

tipl::const_pointer_image<3,float> item::get_image(void)
{
    if(!image_ready)
    {
        static std::mutex mat_load;
        std::lock_guard<std::mutex> lock(mat_load);
        if(image_ready)
            return image_data;
        // delay read routine
        unsigned int row,col;
        const float* buf = nullptr;
        bool has_gui_ = has_gui;
        has_gui = false;
        if (!mat_reader->read(image_index,row,col,buf))
        {
            std::cout << "ERROR: reading " << name << std::endl;
            dummy.resize(image_data.shape());
            image_data = tipl::make_image(&*dummy.begin(),dummy.shape());
        }
        else
        {
            std::cout << name << " loaded" << std::endl;
            mat_reader->in->flush();
            image_data = tipl::make_image(buf,image_data.shape());
        }
        has_gui = has_gui_;
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

    if(num_fiber == 0)
    {
        error_msg = "Invalid FIB format";
        return 0;
    }


    // adding the primary fiber index
    index_name.insert(index_name.begin(),fa.size() == 1 ? "fa":"qa");
    index_data.insert(index_data.begin(),fa);

    for(size_t index = 1;index < index_data.size();++index)
    {
        // check index_data integrity
        for(size_t j = 0;j < index_data[index].size();++j)
            if(!index_data[index][j] || index_data[index].size() != num_fiber)
            {
                index_data.erase(index_data.begin()+int64_t(index));
                index_name.erase(index_name.begin()+int64_t(index));
                --index;
                break;
            }

        // identify dt indices
        if(index_name[index].find("inc_") == 0 ||
           index_name[index].find("dec_") == 0)
        {
            dt_index_name.push_back(index_name[index]);
            dt_index_data.push_back(index_data[index]);
        }
    }
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
void fiber_directions::add_dt_index(const std::string& name,tipl::image<3>&& I)
{
    new_dT.push_back(std::make_shared<tipl::image<3> >());
    new_dT.back()->swap(I);
    std::vector<const float*> new_ptr(num_fiber);
    std::fill(new_ptr.begin(),new_ptr.end(),&*new_dT.back()->begin());
    dt_index_data.push_back(new_ptr);
    dt_index_name.push_back(name);
}

const float* fiber_directions::get_fib(size_t space_index,unsigned int order) const
{
    if(!dir.empty())
        return dir[order] + space_index + space_index + space_index;
    if(order >= findex.size())
        return &*(odf_table[0].begin());
    return &*(odf_table[findex[order][space_index]].begin());
}

float fiber_directions::cos_angle(const tipl::vector<3>& cur_dir,size_t space_index,unsigned char fib_order) const
{
    if(!dir.empty())
    {
        const float* dir_at = dir[fib_order] + space_index + space_index + space_index;
        return cur_dir[0]*dir_at[0] + cur_dir[1]*dir_at[1] + cur_dir[2]*dir_at[2];
    }
    return cur_dir*odf_table[findex[fib_order][space_index]];
}


float fiber_directions::get_track_specific_metrics(size_t space_index,
                         const std::vector<const float*>& metrics,
                         const tipl::vector<3,float>& dir) const
{
    if(fa[0][space_index] == 0.0f)
        return 0.0;
    unsigned char fib_order = 0;
    float max_value = std::abs(cos_angle(dir,space_index,0));
    for (unsigned char index = 1;index < uint8_t(fa.size());++index)
    {
        if (fa[index][space_index] == 0.0f)
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
    return metrics[fib_order][space_index];
}


bool tracking_data::get_dir(size_t space_index,
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
    if(fib->has_high_reso)
    {
        has_high_reso = true;
        high_reso_ratio = fib->vs[0]/fib->high_reso->vs[0];
        high_reso.reset(new tracking_data);
        high_reso->read(fib->high_reso);
    }
}
bool tracking_data::get_dir(size_t space_index,
                            const tipl::vector<3,float>& dir, // reference direction, should be unit vector
                            tipl::vector<3,float>& main_dir,
                            float threshold,
                            float cull_cos_angle,
                            float dt_threshold) const
{
    unsigned char fib_order;
    unsigned char reverse;
    if (!get_dir(space_index,dir,fib_order,reverse,threshold,cull_cos_angle,dt_threshold))
        return false;
    main_dir = get_fib(space_index,fib_order);
    if(reverse)
    {
        main_dir[0] = -main_dir[0];
        main_dir[1] = -main_dir[1];
        main_dir[2] = -main_dir[2];
    }
    return true;
}

const float* tracking_data::get_fib(size_t space_index,unsigned char fib_order) const
{
    if(!dir.empty())
        return dir[fib_order] + space_index + (space_index << 1);
    return &odf_table[findex[fib_order][space_index]][0];
}

float tracking_data::cos_angle(const tipl::vector<3>& cur_dir,size_t space_index,unsigned char fib_order) const
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
void initial_LPS_nifti_srow(tipl::matrix<4,4>& T,const tipl::shape<3>& geo,const tipl::vector<3>& vs)
{
    std::fill(T.begin(),T.end(),0.0f);
    T[0] = -vs[0];
    T[5] = -vs[1];
    T[10] = vs[2];
    T[3] = vs[0]*(geo[0]-1);
    T[7] = vs[1]*(geo[1]-1);
    T[15] = 1.0f;
}

fib_data::fib_data(tipl::shape<3> dim_,tipl::vector<3> vs_):dim(dim_),vs(vs_)
{
    initial_LPS_nifti_srow(trans_to_mni,dim,vs);
}

fib_data::fib_data(tipl::shape<3> dim_,tipl::vector<3> vs_,const tipl::matrix<4,4>& trans_to_mni_):
    dim(dim_),vs(vs_),trans_to_mni(trans_to_mni_)
{}

bool load_fib_from_tracks(const char* file_name,
                          tipl::image<3>& I,
                          tipl::vector<3>& vs,
                          tipl::matrix<4,4>& trans_to_mni);
void prepare_idx(const char* file_name,std::shared_ptr<gz_istream> in);
void save_idx(const char* file_name,std::shared_ptr<gz_istream> in);
bool read_fib_mat_with_idx(const char* file_name,gz_mat_read& mat_reader)
{
    prepare_idx(file_name,mat_reader.in);
    if(mat_reader.in->has_access_points())
    {
        mat_reader.delay_read = true;
        mat_reader.in->buffer_all = false;
    }
    if (!mat_reader.load_from_file(file_name) || progress::aborted())
        return false;
    save_idx(file_name,mat_reader.in);
    return true;
}
bool fib_data::load_from_file(const char* file_name)
{
    tipl::image<3> I;
    gz_nifti header;
    fib_file_name = file_name;
    if((QFileInfo(file_name).fileName().endsWith(".nii") ||
        QFileInfo(file_name).fileName().endsWith(".nii.gz")) &&
        header.load_from_file(file_name))
    {
        if(header.dim(4) == 3)
        {
            tipl::image<3> x,y,z;
            header.get_voxel_size(vs);
            header.toLPS(x,false);
            header.toLPS(y,false);
            header.toLPS(z,false);
            dim = x.shape();
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
            match_template();
            return true;
        }
        else
        if(header.dim(4) && header.dim(4) % 3 == 0)
        {
            uint32_t fib_num = header.dim(4)/3;
            for(uint32_t i = 0;i < fib_num;++i)
            {
                tipl::image<3> x,y,z;
                header.get_voxel_size(vs);
                header.toLPS(x,false);
                header.toLPS(y,false);
                header.toLPS(z,false);
                if(i == 0)
                {
                    dim = x.shape();
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
            match_template();
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
        if(!load_fib_from_tracks(file_name,I,vs,trans_to_mni))
        {
            error_msg = "Invalid track format";
            return false;
        }
    }
    if(!I.empty())
    {
        mat_reader.add("dimension",I.shape());
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


    if(!read_fib_mat_with_idx(file_name,mat_reader))
    {
        error_msg = progress::aborted() ? "Loading process aborted" : "Invalid file format";
        return false;
    }


    // check if initiate surrogate analysis for large data
    if(has_gui &&
       !mat_reader.has("odfs") && !mat_reader.has("odf0") && // not ODF FIB files
       !mat_reader.has("subject_names") &&                        // not connectometry DB
       !mat_reader.has("dirs") &&                        // 4D dirs matrix requires more implementation in surrogate mat_reader
        mat_reader.read("dimension",dim) &&
        mat_reader.read("voxel_size",vs) &&
        (dim[0] > 256 || dim[1] > 256 || dim[2] > 256))
    {
        high_reso.reset(new fib_data);
        std::cout << "initiate surrogate analysis" << std::endl;
        std::string surrogate_file_name = file_name;
        surrogate_file_name.resize(surrogate_file_name.size()-7);
        surrogate_file_name += ".fibs.gz";
        // if no surrogate FIB or surrogate FIB is older, then generate one
        if(!std::filesystem::exists(surrogate_file_name) || QFileInfo(surrogate_file_name.c_str()).lastModified() < QFileInfo(file_name).lastModified())
        {
            progress prog_("create surrogate FIB file");
            size_t largest_dim = tipl::max_value(dim);
            size_t downsampling = 0;
            tipl::vector<3> low_reso_vs(vs);
            tipl::shape<3> low_reso_dim(dim);
            while(largest_dim > 256)
            {
                ++downsampling;
                largest_dim >>= 1;
                low_reso_vs *= 2.0f;
                low_reso_dim[0] = (low_reso_dim[0]+1) >> 1;
                low_reso_dim[1] = (low_reso_dim[1]+1) >> 1;
                low_reso_dim[2] = (low_reso_dim[2]+1) >> 1;
            }
            std::cout << "preparing surrogate FIB file" << std::endl;
            gz_mat_write out(surrogate_file_name.c_str());
            out.write("dimension",low_reso_dim);
            out.write("voxel_size",low_reso_vs);
            // output odf vertices and faces
            {
                tessellated_icosahedron ti;
                ti.init(8);
                std::vector<float> float_data;
                std::vector<short> short_data;
                ti.save_to_buffer(float_data,short_data);
                out.write("odf_vertices",float_data,3);
                out.write("odf_faces",short_data,3);
            }
            // QSDR
            if(mat_reader.read("trans",trans_to_mni))
            {
                // downsample mapping matrix
                if(!get_native_position().empty())
                {
                    tipl::image<3,tipl::vector<3,float> > new_mapping;
                    for(size_t j = 0;j < downsampling;++j)
                        if(j == 0)
                            tipl::downsample_with_padding(native_position,new_mapping);
                        else
                            tipl::downsample_with_padding(new_mapping);
                    out.write("mapping",&new_mapping[0][0],3,new_mapping.size());
                }
                // convert trans_to_mni
                for(size_t j = 0;j < downsampling;++j)
                {
                    trans_to_mni[0] *= 2.0f;
                    trans_to_mni[5] *= 2.0f;
                    trans_to_mni[10] *= 2.0f;
                }
                out.write("trans",trans_to_mni.begin(),4,4);

                tipl::shape<3> native_shape;
                tipl::vector<3> native_vs;
                if(mat_reader.read("native_dimension",native_shape) &&
                   mat_reader.read("native_voxel_size",native_vs))
                {
                    out.write("native_dimension",native_shape);
                    out.write("native_voxel_size",native_vs);
                }
            }

            for(size_t index = 0;progress::at(index,mat_reader.size());++index)
            {
                tipl::io::mat_matrix& matrix = mat_reader[index];
                progress::show(std::string("loading ") + matrix.get_name());
                if(matrix.is_type<char>()) // report, steps, ...etc
                {
                    std::string content;
                    mat_reader.read(matrix.get_name().c_str(),content);
                    out.write(matrix.get_name().c_str(),content);
                    std::cout << "write " << matrix.get_name() << ":" << content << std::endl;
                }

                if(matrix.get_name() == "dir0")
                {
                    if(matrix.has_delay_read() && !matrix.read(*(mat_reader.in.get())))
                    {
                        error_msg = "failed to create surrogate FIB file";
                        return false;
                    }
                    progress::show("write dir0 in downsampled volume");
                    auto ptr = reinterpret_cast<const float*>(matrix.get_data(tipl::io::mat_type_info<float>::type));
                    tipl::image<3,tipl::vector<3> > new_image,J(dim);
                    for(size_t j = 0;j < J.size();++j)
                    {
                        J[j] = tipl::vector<3>(*ptr,*(ptr+1),*(ptr+2));
                        ptr += 3;
                    }
                    for(size_t j = 0;j < downsampling;++j)
                        if(j == 0)
                            tipl::downsample_no_average(J,new_image);
                        else
                            tipl::downsample_no_average(new_image,new_image);
                    out.write(matrix.get_name().c_str(),&new_image[0][0],uint32_t(3*dim.plane_size()),uint32_t(dim.depth()));
                }

                if(size_t(matrix.get_cols())*size_t(matrix.get_rows()) == dim.size()) // image volumes, including fa, and fiber index
                {
                    progress::show(std::string("write ") + matrix.get_name() + " in downsampled volume");
                    if(matrix.has_delay_read() && !matrix.read(*(mat_reader.in.get())))
                    {
                        error_msg = "failed to create surrogate FIB file";
                        return false;
                    }
                    if(matrix.is_type<float>()) // qa, fa...etc.
                    {
                        auto J = tipl::make_image(reinterpret_cast<const float*>(matrix.get_data(tipl::io::mat_type_info<float>::type)),dim);
                        tipl::image<3> new_image;
                        for(size_t j = 0;j < downsampling;++j)
                            if(j == 0)
                                tipl::downsample_with_padding(J,new_image);
                            else
                                tipl::downsample_with_padding(new_image);
                        out.write(matrix.get_name().c_str(),new_image);
                    }
                    if(matrix.is_type<short>()) // index0,index1
                    {
                        auto J = tipl::make_image(reinterpret_cast<const unsigned short*>(matrix.get_data(tipl::io::mat_type_info<unsigned short>::type)),dim);
                        tipl::image<3,unsigned short> new_index;
                        for(size_t j = 0;j < downsampling;++j)
                            if(j == 0)
                                tipl::downsample_no_average(J,new_index);
                            else
                                tipl::downsample_no_average(new_index,new_index);
                        out.write(matrix.get_name().c_str(),new_index);
                    }
                }
            }

            if(progress::aborted())
            {
                error_msg = "failed to load surrogate FIB file";
                return false;
            }
        }
        progress::show("reading surrogate FIB file");
        high_reso->mat_reader.swap(mat_reader);
        high_reso->fib_file_name = fib_file_name;
        if(!read_fib_mat_with_idx(surrogate_file_name.c_str(),mat_reader))
        {
            error_msg = "failed to load surrogate FIB file";
            return false;
        }
        fib_file_name = surrogate_file_name;
        has_high_reso = true;
    }

    if(!load_from_mat())
        return false;

    return true;
}
bool fib_data::save_mapping(const std::string& index_name,const std::string& file_name)
{
    if(index_name == "fiber" || index_name == "dirs") // command line exp use "dirs"
    {
        tipl::image<4,float> buf(tipl::shape<4>(
                                 dim.width(),
                                 dim.height(),
                                 dim.depth(),3*uint32_t(dir.num_fiber)));

        for(unsigned int j = 0,index = 0;j < dir.num_fiber;++j)
        for(int k = 0;k < 3;++k)
        for(unsigned int i = 0;i < dim.size();++i,++index)
            buf[index] = dir.get_fib(i,j)[k];

        return gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni);
    }
    if(index_name.length() == 4 && index_name.substr(0,3) == "dir" && index_name[3]-'0' >= 0 && index_name[3]-'0' < int(dir.num_fiber))
    {
        unsigned char dir_index = uint8_t(index_name[3]-'0');
        tipl::image<4,float> buf(tipl::shape<4>(dim[0],dim[1],dim[2],3));
        for(unsigned int j = 0,ptr = 0;j < 3;++j)
        for(size_t index = 0;index < dim.size();++index,++ptr)
            if(dir.fa[dir_index][index] > 0.0f)
                buf[ptr] = dir.get_fib(index,dir_index)[j];
        return gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni);
    }
    if(index_name == "odfs" && odf.has_odfs())
    {
        tipl::image<4,float> buf(tipl::shape<4>(
                                 dim.width(),
                                 dim.height(),
                                 dim.depth(),
                                 dir.half_odf_size));
        for(unsigned int pos = 0;pos < dim.size();++pos)
        {
            auto* ptr = odf.get_odf_data(pos);
            if(ptr!= nullptr)
                std::copy(ptr,ptr+dir.half_odf_size,buf.begin()+int64_t(pos)*dir.half_odf_size);

        }
        return gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni);
    }
    size_t index = get_name_index(index_name);
    if(index >= view_item.size())
        return false;

    if(index_name == "color")
    {
        tipl::image<3,tipl::rgb> buf(dim);
        for(int z = 0;z < buf.depth();++z)
        {
            tipl::color_image I;
            get_slice(uint32_t(index),uint8_t(2),uint32_t(z),I);
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
        tipl::image<3> buf(view_item[index].get_image());
        if(view_item[index].get_image().shape() != dim)
        {
            tipl::image<3> new_buf(dim);
            tipl::resample_mt<tipl::interpolation::cubic>(buf,new_buf,tipl::transformation_matrix<float>(view_item[index].iT));
            new_buf.swap(buf);
        }
        return gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni);
    }
}
bool is_human_size(tipl::shape<3> dim,tipl::vector<3> vs)
{
    return dim[2] > 5 && dim[0]*vs[0] > 130 && dim[1]*vs[1] > 130;
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
    if(has_high_reso)
    {
        std::cout << "reading original mat file" << std::endl;
        if(!high_reso->load_from_mat())
        {
            error_msg = high_reso->error_msg;
            return false;
        }
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
    is_histology = (dim[2] == 2 && dim[0] > 400 && dim[1] > 400);
    db.read_db(this);

    if(is_qsdr)
    {
        mat_reader.read("native_dimension",native_geo);
        mat_reader.read("native_voxel_size",native_vs);
        for(unsigned int i = 0; i < view_item.size();++i)
        {
            view_item[i].native_geo = native_geo;
            view_item[i].native_trans.sr[0] = view_item[i].native_trans.sr[4] = view_item[i].native_trans.sr[8] = 1.0;
            mat_reader.read((view_item[i].name+"_dimension").c_str(),view_item[i].native_geo);
            mat_reader.read((view_item[i].name+"_trans").c_str(),view_item[i].native_trans);
        }

        // matching templates
        for(size_t index = 0;index < fa_template_list.size();++index)
        {
            if(QString(fib_file_name.c_str()).contains(QFileInfo(fa_template_list[index].c_str()).baseName(),Qt::CaseInsensitive))
            {
                set_template_id(index);
                return true;
            }
            gz_nifti read;
            if(!read.load_from_file(fa_template_list[index]))
                continue;
            tipl::vector<3> Itvs;
            tipl::image<3> dummy;
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

    match_template();
    return true;
}
void fib_data::match_template(void)
{
    if(is_human_size(dim,vs))
        set_template_id(0);
    else
        set_template_id(::match_template(std::count_if(dir.fa[0],dir.fa[0]+dim.size(),[](float v){return v > 0.0f;})*2.0f*vs[0]*vs[1]*vs[2]));
}

const tipl::image<3,tipl::vector<3,float> >& fib_data::get_native_position(void) const
{
    if(native_position.empty() && mat_reader.has("mapping"))
    {
        unsigned int row,col;
        const float* mapping = nullptr;
        if(mat_reader.read("mapping",row,col,mapping))
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
    for (size_t index = 0; index < view_item.size(); ++index)
        if(view_item[index].name != "color")
            index_list.push_back(view_item[index].name);
}

bool fib_data::add_dT_index(const std::string& index_name)
{
    std::string metrics;
    for(size_t i = 0;i < view_item.size();++i)
    {
        if(i)
            metrics += ",";
        metrics += view_item[i].name;
        // find if the prefix of index_name matches the name of the index
        if(index_name.length() <= view_item[i].name.length() ||
           index_name.find(view_item[i].name) != 0 ||
           index_name[view_item[i].name.length()] != '-')
            continue;
        std::string post_fix = index_name.substr(view_item[i].name.length()+1);
        for(size_t j = 0;j < view_item.size();++j)
            if(post_fix == view_item[j].name)
            {
                tipl::image<3> Ibuf,Jbuf;
                auto J = view_item[j].get_image();
                auto I = view_item[i].get_image();
                if(J.shape() != dim)
                {
                    std::cout << view_item[j].name << " has a dimension of " << J.shape() <<
                                 " warpping to the DWI space..." << std::endl;
                    Jbuf.resize(dim);
                    tipl::resample_mt(J,Jbuf,tipl::transformation_matrix<float>(view_item[j].iT));
                    J = tipl::make_image(&*Jbuf.begin(),Jbuf.shape());
                }
                if(I.shape() != dim)
                {
                    std::cout << view_item[i].name << " has a dimension of " << I.shape() <<
                                 " warpping to the DWI space..." << std::endl;
                    Ibuf.resize(dim);
                    tipl::resample_mt(I,Ibuf,tipl::transformation_matrix<float>(view_item[i].iT));
                    I = tipl::make_image(&*Ibuf.begin(),Ibuf.shape());
                }

                tipl::image<3> new_metrics(dim);
                std::cout << "new metric: (" << view_item[i].name << " - " << view_item[j].name << ")/" << view_item[i].name << " x 100%" << std::endl;
                /*
                if(!raw)
                {
                    std::vector<float> x,y;
                    for(size_t k = 0;k < I.size();++k)
                        if(dir.fa[0][k] > 0.0f && I[k] > 0.0f && J[k] > 0.0f)
                        {
                            x.push_back(I[k]); // x = I
                            y.push_back(J[k]); // y = J
                        }
                    float a,b,r2; // I ~ a*J+b
                    tipl::linear_regression(x.begin(),x.end(),y.begin(),a,b,r2);
                    std::cout << "matching " << view_item[j].name << " ~ a*" << view_item[i].name << " + b" << std::endl;
                    std::cout << "a:" << a << " b:" << b << " r2:" << r2 << std::endl;
                    for(size_t k = 0;k < I.size();++k)
                        if(dir.fa[0][k] > 0.0f && I[k] > 0.0f && J[k] > 0.0f)
                            new_metrics[k] = (a*I[k]+b)/J[k]-1.0f;
                }
                // else
                */
                {
                    for(size_t k = 0;k < I.size();++k)
                        if(dir.fa[0][k] > 0.0f && I[k] > 0.0f && J[k] > 0.0f)
                            new_metrics[k] = 1.0f-J[k]/I[k];
                }
                dir.add_dt_index(index_name,std::move(new_metrics));
                return true;
            }
        error_msg = "cannot find ";
        error_msg += post_fix;
        error_msg += " among ";
        error_msg += metrics;
        return false;
    }
    error_msg = "cannot find matched metrics among ";
    error_msg += metrics;
    return false;
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
               tipl::color_image& show_image)
{
    if(view_item[view_index].name == "color")
    {
        {
            tipl::image<2,float> buf;
            tipl::volume2slice(view_item[0].get_image(), buf, d_index, pos);
            view_item[view_index].v2c.convert(buf,show_image);
        }

        if(view_item[view_index].color_map_buf.empty())
        {
            view_item[view_index].color_map_buf.resize(dim);
            std::iota(view_item[view_index].color_map_buf.begin(),
                      view_item[view_index].color_map_buf.end(),0);
        }
        tipl::image<2,unsigned int> buf;
        tipl::volume2slice(view_item[view_index].color_map_buf, buf, d_index, pos);
        for (unsigned int index = 0;index < buf.size();++index)
        {
            const float* d = dir.get_fib(buf[index],0);
            show_image[index].r = std::abs((float)show_image[index].r*d[0]);
            show_image[index].g = std::abs((float)show_image[index].g*d[1]);
            show_image[index].b = std::abs((float)show_image[index].b*d[2]);
        }
    }
    else
    {
        tipl::image<2,float> buf;
        tipl::volume2slice(view_item[view_index].get_image(), buf, d_index, pos);
        view_item[view_index].v2c.convert(buf,show_image);
    }

}

void fib_data::get_voxel_info2(int x,int y,int z,std::vector<float>& buf) const
{
    if(!dim.is_valid(x,y,z))
        return;
    size_t space_index = tipl::pixel_index<3>(x,y,z,dim).index();
    if (space_index >= dim.size())
        return;
    for(unsigned int i = 0;i < dir.num_fiber;++i)
        if(dir.fa[i][space_index] == 0.0f)
        {
            buf.push_back(0.0f);
            buf.push_back(0.0f);
            buf.push_back(0.0f);
        }
        else
        {
            const float* d = dir.get_fib(space_index,i);
            buf.push_back(d[0]);
            buf.push_back(d[1]);
            buf.push_back(d[2]);
        }
}
void fib_data::get_voxel_information(int x,int y,int z,std::vector<float>& buf) const
{
    if(!dim.is_valid(x,y,z))
        return;
    size_t space_index = tipl::pixel_index<3>(x,y,z,dim).index();
    if (space_index >= dim.size())
        return;
    for(unsigned int i = 0;i < view_item.size();++i)
    {
        if(view_item[i].name == "color" || !view_item[i].image_ready)
            continue;
        if(view_item[i].get_image().shape() != dim)
        {
            tipl::vector<3> pos(x,y,z);
            pos.to(view_item[i].iT);
            buf.push_back(tipl::estimate(view_item[i].get_image(),pos));
        }
        else
            buf.push_back(view_item[i].get_image().size() ? view_item[i].get_image()[space_index] : 0.0);
    }
}

void fib_data::get_iso_fa(tipl::image<3>& iso_fa_) const
{
    size_t index = get_name_index("iso");
    if(view_item.size() == index)
        index = get_name_index("md");
    if(view_item.size() == index)
        index = 0;
    tipl::image<3> iso_fa(view_item[index].get_image());
    tipl::add(iso_fa,view_item[0].get_image());
    iso_fa.swap(iso_fa_);
}

extern std::vector<std::string> fa_template_list,iso_template_list,track_atlas_file_list;
extern std::vector<std::vector<std::string> > template_atlas_list;


void apply_trans(tipl::vector<3>& pos,const tipl::matrix<4,4>& trans)
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
/*
void mni2temp(tipl::vector<3>& pos,const tipl::matrix<4,4>& trans)
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
*/
void fib_data::set_template_id(size_t new_id)
{
    if(new_id != template_id)
    {
        template_id = new_id;
        template_I.clear();
        s2t.clear();
        atlas_list.clear();
        track_atlas.reset();
        // populate atlas list
        for(size_t i = 0;i < template_atlas_list[template_id].size();++i)
        {
            atlas_list.push_back(std::make_shared<atlas>());
            atlas_list.back()->name = QFileInfo(template_atlas_list[template_id][i].c_str()).baseName().toStdString();
            atlas_list.back()->filename = template_atlas_list[template_id][i];
            atlas_list.back()->template_to_mni = trans_to_mni;
        }
        // populate tract names
        tractography_atlas_file_name = track_atlas_file_list[template_id];
        tractography_name_list.clear();
        if(std::filesystem::exists(tractography_atlas_file_name))
        {
            std::ifstream in(track_atlas_file_list[template_id]+".txt");
            if(in)
                std::copy(std::istream_iterator<std::string>(in),
                          std::istream_iterator<std::string>(),std::back_inserter(tractography_name_list));
        }
        // populate other modality name
        t1w_template_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".T1W.nii.gz").toStdString();
        wm_template_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".WM.nii.gz").toStdString();
        mask_template_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".mask.nii.gz").toStdString();
    }
}

bool fib_data::load_template(void)
{
    if(!template_I.empty())
        return true;
    gz_nifti read;
    tipl::image<3> I;
    tipl::vector<3> I_vs;
    if(!read.load_from_file(fa_template_list[template_id].c_str()))
    {
        error_msg = "cannot load ";
        error_msg += fa_template_list[template_id];
        return false;
    }
    read.toLPS(I);
    read.get_voxel_size(I_vs);
    read.get_image_transformation(template_to_mni);
    float ratio = float(I.width()*I_vs[0])/float(dim[0]*vs[0]);
    if(ratio < 0.25f || ratio > 8.0f)
    {
        error_msg = "image resolution mismatch: ratio=";
        error_msg += std::to_string(ratio);
        return false;
    }

    is_template_space = is_mni_image || (is_qsdr && std::abs(float(dim[0])-I.width()*I_vs[0]/vs[0]) < 2);

    if(is_template_space)
    {
        // set template space to current space
        template_I.resize(dim);
        template_vs = vs;
        template_to_mni = trans_to_mni;
        return true;
    }

    template_I.swap(I);
    template_vs = I_vs;

    unsigned int downsampling = 0;
    while((!is_human_data && template_I.width()/3 > int(dim[0])) ||
          (is_human_data && template_vs[0]*2.0f <= int(vs[0])))
    {
        std::cout << "downsampling template by 2x to match subject resolution" << std::endl;
        template_vs *= 2.0f;
        template_to_mni[0] *= 2.0f;
        template_to_mni[5] *= 2.0f;
        template_to_mni[10] *= 2.0f;
        tipl::downsampling(template_I);
        ++downsampling;
    }

    for(size_t i = 0;i < atlas_list.size();++i)
        atlas_list[i]->template_to_mni = template_to_mni;

    // load iso template if exists
    {
        gz_nifti read2;
        if(!iso_template_list[template_id].empty() &&
           read2.load_from_file(iso_template_list[template_id].c_str()))
        {
            read2.toLPS(template_I2);
            for(unsigned int i = 0;i < downsampling;++i)
                tipl::downsampling(template_I2);
        }

    }
    template_I *= 1.0f/float(tipl::mean(template_I));
    if(!template_I2.empty())
        template_I2 *= 1.0f/float(tipl::mean(template_I2));

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

        // find left right pairs
        std::vector<unsigned int> pair(tractography_name_list.size(),uint32_t(tractography_name_list.size()));
        for(unsigned int i = 0;i < tractography_name_list.size();++i)
            for(unsigned int j = i + 1;j < tractography_name_list.size();++j)
                if(tractography_name_list[i].size() == tractography_name_list[j].size() &&
                   tractography_name_list[i].back() == 'L' && tractography_name_list[j].back() == 'R' &&
                   tractography_name_list[i].substr(0,tractography_name_list[i].length()-1) ==
                   tractography_name_list[j].substr(0,tractography_name_list[j].length()-1))
                {
                    pair[i] = j;
                    pair[j] = i;
                }

        // copy tract from one side to another
        const auto& tracts = track_atlas->get_tracts();
        auto& cluster = track_atlas->get_cluster_info();

        std::vector<std::vector<float> > new_tracts;
        std::vector<unsigned int> new_cluster;
        for(size_t i = 0;i < tracts.size();++i)
            if(pair[cluster[i]] < tractography_name_list.size())
            {
                new_tracts.push_back(tracts[i]);
                auto& tract = new_tracts.back();
                // mirror in the x
                for(size_t pos = 0;pos < tract.size();pos += 3)
                    tract[pos] = track_atlas->geo.width()-tract[pos];
                new_cluster.push_back(pair[cluster[i]]);
            }

        // add adds
        track_atlas->add_tracts(new_tracts);
        cluster.insert(cluster.end(),new_cluster.begin(),new_cluster.end());

        if(!load_template())
            return false;

        {
            progress prog_("warping atlas tracks to subject space");
            run_normalization(true,true);
            if(progress::aborted())
                return false;
        }

        // warp tractography atlas to subject space
        auto& tract_data = track_atlas->get_tracts();
        auto T = tipl::from_space(track_atlas->trans_to_mni).to(template_to_mni);
        tipl::par_for(tract_data.size(),[&](size_t i)
        {
            for(size_t j = 0;j < tract_data[i].size();j += 3)
            {
                tipl::vector<3> p(&tract_data[i][j]);
                apply_trans(p,T); // from tract atlas space to current template space
                temp2sub(p);
                tract_data[i][j] = p[0];
                tract_data[i][j+1] = p[1];
                tract_data[i][j+2] = p[2];
            }
        });
        return true;
    }
    return true;
}

//---------------------------------------------------------------------------
unsigned int fib_data::find_nearest(const float* trk,unsigned int length,bool contain,float false_distance)
{
    return ::find_nearest(trk,length,track_atlas->get_tracts(),track_atlas->get_cluster_info(),contain,false_distance);
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

float two_way_linear_cuda(tipl::const_pointer_image<3,float> I,
                         const tipl::vector<3>& Ivs,
                         tipl::const_pointer_image<3,float> J,
                         const tipl::vector<3>& Jvs,
                         tipl::transformation_matrix<float>& T,
                         tipl::reg::reg_type reg_type,
                         bool& terminated,
                         tipl::affine_transform<float>* arg_min);

void animal_reg(const tipl::image<3>& from,tipl::vector<3> from_vs,
          const tipl::image<3>& to,tipl::vector<3> to_vs,
          tipl::transformation_matrix<float>& T,bool& terminated)
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
    progress::show("linear registration for animal data");
    tipl::par_for(5,[&](int i)
    {
         tipl::affine_transform<double> arg;
         std::copy(directions[i],directions[i]+3,arg.rotation);
         float cur_cost = tipl::reg::linear_mr<tipl::reg::mutual_information>(from,from_vs,to,to_vs,arg,
                                                    tipl::reg::affine,terminated,0.001,tipl::reg::large_bound);
         if(cur_cost < cost)
         {
             cost = cur_cost;
             T = tipl::transformation_matrix<float>(arg,from.shape(),from_vs,to.shape(),to_vs);
         }
    });
}

void fib_data::run_normalization(bool background,bool inv)
{
    if(is_template_space||
       (!inv && !s2t.empty()) ||
       (inv && !t2s.empty()))
        return;
    std::string output_file_name(fib_file_name);
    output_file_name += ".";
    output_file_name += QFileInfo(fa_template_list[template_id].c_str()).baseName().toLower().toStdString();
    output_file_name += ".map.gz";

    gz_mat_read in;

    // check 1. mapping files was created later than the FIB file
    //       2. the recon steps are the same
    //       3. has inv_mapping matrix (new format after Sep 2021

    if(QFileInfo(output_file_name.c_str()).lastModified() > QFileInfo(fib_file_name.c_str()).lastModified() &&
       in.load_from_file(output_file_name.c_str()) && in.has("from2to") && in.has("to2from")
       && in.read<std::string>("steps") == steps)
    {
        const float* t2s_ptr = nullptr;
        unsigned int t2s_row,t2s_col,s2t_row,s2t_col;
        const float* s2t_ptr = nullptr;
        if(in.read("to2from",t2s_row,t2s_col,t2s_ptr) &&
           in.read("from2to",s2t_row,s2t_col,s2t_ptr))
        {
            if(size_t(t2s_col)*size_t(t2s_row) == template_I.size()*3 &&
               size_t(s2t_col)*size_t(s2t_row) == dim.size()*3)
            {
                std::cout << "loading mapping fields from " << output_file_name << std::endl;
                t2s.clear();
                t2s.resize(template_I.shape());
                s2t.clear();
                s2t.resize(dim);
                std::copy(t2s_ptr,t2s_ptr+t2s_col*t2s_row,&t2s[0][0]);
                std::copy(s2t_ptr,s2t_ptr+s2t_col*s2t_row,&s2t[0][0]);
                prog = 5;
                return;
            }
        }
    }
    progress prog_("running normalization");
    prog = 0;
    bool terminated = false;
    auto lambda = [this,output_file_name,&terminated]()
    {
        tipl::transformation_matrix<float> T;

        auto It = template_I;
        auto It2 = template_I2;
        tipl::image<3> Is(dir.fa[0],dim);
        tipl::image<3> Is2;

        {
            size_t iso_index = get_name_index("iso");
            if(view_item.size() == iso_index)
                iso_index = get_name_index("md");
            if(view_item.size() != iso_index)
                Is2 = view_item[iso_index].get_image();
        }
        bool no_iso = Is2.empty() || It2.empty();

        prog = 1;
        if(!has_manual_atlas)
        {
            if(is_human_data)
            {
                if constexpr (tipl::use_cuda)
                    two_way_linear_cuda(It,template_vs,Is,vs,T,tipl::reg::affine,terminated,nullptr);
                else
                    tipl::reg::two_way_linear_mr<tipl::reg::mutual_information>(It,template_vs,Is,vs,T,tipl::reg::affine,terminated);
            }
            else
            {
                auto tvs = vs;
                tvs *= std::sqrt((It.plane_size()*template_vs[0]*template_vs[1])/
                                                        (Is.plane_size()*vs[0]*vs[1]));
                if(no_iso)
                    animal_reg(It,template_vs,Is,tvs,T,terminated);
                else
                    animal_reg(It2,template_vs,Is2,tvs,T,terminated);
            }
        }
        else
            T = manual_template_T;

        std::cout << "T:" << std::endl;
        std::cout << T;

        if(terminated)
            return;
        prog = 2;
        tipl::image<3> Iss(It.shape());
        tipl::resample_mt(Is,Iss,T);
        tipl::image<3> Iss2(It2.shape());
        if(!no_iso)
            tipl::resample_mt(Is2,Iss2,T);
        prog = 3;
        tipl::image<3,tipl::vector<3> > dis,inv_dis;
        tipl::reg::cdm_pre(It,It2,Iss,Iss2);
        if(no_iso)
        {
            progress::show("single modality normalization");
            tipl::reg::cdm(It,Iss,dis,terminated);
        }
        else
        {
            progress::show("dual modality normalization");
            tipl::reg::cdm2(It,It2,Iss,Iss2,dis,terminated);
        }

        tipl::invert_displacement(dis,inv_dis);

        if(terminated)
            return;
        gz_mat_write out(output_file_name.c_str());
        if(out)
        {
            out.write("to_dim",dis.shape());
            out.write("to_vs",template_vs);
            out.write("from_dim",dim);
            out.write("from_vs",vs);
            out.write("steps",steps);
        }

        {
            prog = 4;
            progress::show("calculating template to subject warp field");
            tipl::image<3,tipl::vector<3,float> > pos;
            tipl::displacement_to_mapping(dis,pos,T);
            if(out)
                out.write("to2from",&pos[0][0],3,pos.size());
            t2s.swap(pos);
        }
        if(terminated)
            return;
        {
            prog = 5;
            progress::show("calculating subject to template warp field");
            tipl::image<3,tipl::vector<3,float> > pos(dim);
            tipl::inv_displacement_to_mapping(inv_dis,pos,T);
            if(out)
                out.write("from2to",&pos[0][0],3,pos.size());
            s2t.swap(pos);
        }
        tipl::compose_mapping(tipl::make_image(dir.fa[0],dim),t2s,Iss);
        std::cout << "nonlinear:" << tipl::correlation(Iss.begin(),Iss.end(),It.begin()) << std::endl;
        prog = 6;
    };

    if(background)
    {
        std::thread t(lambda);
        while(progress::at(prog,6))
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if(progress::aborted())
            terminated = true;
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
    if(progress::aborted())
    {
        error_msg = "action aborted by user";
        return false;
    }
    return true;
}

void fib_data::temp2sub(tipl::vector<3>& pos)
{
    if(is_template_space)
        return;
    if(!t2s.empty())
    {
        tipl::vector<3> p;
        if(pos[2] < 0.0f)
            pos[2] = 0.0f;
        tipl::estimate(t2s,pos,p);
        pos = p;
    }
}

void fib_data::sub2temp(tipl::vector<3>& pos)
{
    if(is_template_space)
        return;
    if(!s2t.empty())
    {
        tipl::vector<3> p;
        if(pos[2] < 0.0f)
            pos[2] = 0.0f;
        tipl::estimate(s2t,pos,p);
        pos = p;
    }
}
void fib_data::sub2mni(tipl::vector<3>& pos)
{
    sub2temp(pos);
    apply_trans(pos,template_to_mni);
}

std::shared_ptr<atlas> fib_data::get_atlas(const std::string atlas_name)
{
    std::string name_list;
    for(auto at : atlas_list)
        if(at->name == atlas_name)
            return at;
        else {
            if(!name_list.empty())
                name_list += ",";
            name_list += at->name;
        }
    error_msg = atlas_name;
    error_msg += " is not one of the following built-in atlases:";
    error_msg += name_list;
    return std::shared_ptr<atlas>();
}

bool fib_data::get_atlas_roi(const std::string& atlas_name,const std::string& region_name,std::vector<tipl::vector<3,short> >& points)
{
    if(region_name.empty())
    {
        error_msg = "please assign region name of an atlas.";
        return false;
    }
    auto at = get_atlas(atlas_name);
    if(!at.get())
        return false;
    auto roi_index = uint32_t(std::find(at->get_list().begin(),at->get_list().end(),region_name)-at->get_list().begin());
    if(roi_index == at->get_list().size())
    {
        bool ok = false;
        roi_index = uint32_t(QString(region_name.c_str()).toInt(&ok));
        if(!ok)
        {
            error_msg = region_name;
            error_msg += " is not one of the regions in ";
            error_msg += atlas_name;
            return false;
        }
    }
    return get_atlas_roi(at,roi_index,points);
}
bool fib_data::get_atlas_roi(std::shared_ptr<atlas> at,unsigned int roi_index,std::vector<tipl::vector<3,short> >& points)
{
    if(get_sub2temp_mapping().empty() || !at->load_from_file())
    {
        error_msg = "no mni mapping";
        return false;
    }
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<std::vector<tipl::vector<3,short> > > buf(thread_count);

    // trigger atlas loading to avoid crash in multi thread
    if(!at->load_from_file())
    {
        error_msg = "cannot read atlas file ";
        error_msg += at->filename;
        return false;
    }
    tipl::par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
        [&](const tipl::pixel_index<3>& index,size_t id)
    {
        if (at->is_labeled_as(s2t[index.index()], roi_index))
            buf[id].push_back(tipl::vector<3,short>(index.begin()));
    });
    points.clear();
    for(size_t i = 0;i < buf.size();++i)
        points.insert(points.end(),buf[i].begin(),buf[i].end());
    return true;
}

bool fib_data::get_atlas_all_roi(std::shared_ptr<atlas> at,std::vector<std::vector<tipl::vector<3,short> > >& points)
{
    if(get_sub2temp_mapping().empty() || !at->load_from_file())
        return false;

    // trigger atlas loading to avoid crash in multi thread
    if(!at->load_from_file())
    {
        error_msg = "cannot read atlas file ";
        error_msg += at->filename;
        return false;
    }

    points.clear();
    points.resize(at->get_list().size());
    std::vector<std::mutex> push_back_mutex(points.size());
    if(at->is_multiple_roi)
    {
        tipl::par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
                    [&](const tipl::pixel_index<3>& index)
        {
            std::vector<uint16_t> region_indicies;
            at->region_indices_at(s2t[index.index()],region_indicies);
            if(region_indicies.empty())
                return;
            tipl::vector<3,short> point(index.begin());
            for(unsigned int i = 0;i < region_indicies.size();++i)
            {
                auto region_index = region_indicies[i];
                std::lock_guard<std::mutex> lock(push_back_mutex[region_index]);
                points[region_index].push_back(point);
            }
        });
    }
    else
    {
        tipl::par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
            [&](const tipl::pixel_index<3>& index)
        {
            int region_index = at->region_index_at(s2t[index.index()]);
            if(region_index < 0 || region_index >= int(points.size()))
                return;
            std::lock_guard<std::mutex> lock(push_back_mutex[uint32_t(region_index)]);
            points[uint32_t(region_index)].push_back(tipl::vector<3,short>(index.begin()));
        });
    }
    return true;
}

const tipl::image<3,tipl::vector<3,float> >& fib_data::get_sub2temp_mapping(void)
{
    if(!s2t.empty())
        return s2t;
    if(is_template_space)
    {
        s2t.resize(dim);
        tipl::par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
                      [&](const tipl::pixel_index<3>& index)
        {
            s2t[index.index()] = index.begin();
        });
        return s2t;
    }
    if(load_template())
        run_normalization(false,false);
    return s2t;
}
