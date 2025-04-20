#include <filesystem>
#include <unordered_set>
#include <QCoreApplication>
#include <QFileInfo>
#include <QDateTime>
#include "reg.hpp"
#include "fib_data.hpp"
#include "tessellated_icosahedron.hpp"
#include "tract_model.hpp"
#include "roi.hpp"
#include "cmd/img.hpp"

extern std::vector<std::string> fa_template_list;
bool odf_data::read(fib_data& fib)
{
    if(!odf_map.empty())
        return true;
    if(!fib.has_odfs())
        return false;
    tipl::progress prog("reading odf data");
    unsigned int row,col;
    odf_map.clear();
    odf_map.resize(fib.dim);
    const float* odf_buf = nullptr;
    for(size_t i = 0,si = 0;prog(0,1) && fib.mat_reader.read("odf"+std::to_string(i),row,col,odf_buf);++i)
        for(size_t j = 0;j < col;++j,++si,odf_buf += row)
            odf_map[fib.mat_reader.si2vi[si]] = odf_buf;
    return !prog.aborted();
}
void slice_model::get_minmax(void)
{
    if(max_value != 0.0f)
        return;
    if(!get_image().data() || name == "dti_fa" || name == "qa" || name == "fa") // e.g., fa, qa
    {
        contrast_min = min_value = 0.0f;
        contrast_max = max_value = 1.0f;
        return;
    }
    float slope;
    if(handle && handle->mat_reader.index_of(name) < handle->mat_reader.size() &&
       handle->mat_reader[handle->mat_reader.index_of(name)].get_sub_data(name+".inter",min_value) &&
       handle->mat_reader[handle->mat_reader.index_of(name)].get_sub_data(name+".slope",slope))
    {
        max_value = min_value + 255.99f*slope;
        tipl::out() << "min: " << min_value << " max:" << max_value;
    }
    else
        tipl::minmax_value(image_data.begin(),image_data.end(),min_value,max_value);
    if(std::isnan(min_value) || std::isinf(min_value) ||
       std::isnan(max_value) || std::isinf(max_value))
    {
        min_value = 0.0f;
        max_value = 1.0f;
    }
    contrast_min = 0;
    contrast_max = max_value;
    if(image_data.size() < 256*256*256 && contrast_min != contrast_max)
        contrast_max = min_value+(tipl::segmentation::otsu_median(image_data)-min_value)*2.0f;
    v2c.set_range(contrast_min,contrast_max);
    v2c.two_color(min_color,max_color);
}
void slice_model::get_slice(
               unsigned char d_index,unsigned int pos,
               tipl::color_image& show_image)
{
    v2c.convert(tipl::volume2slice(get_image(),d_index,pos),show_image);
}

tipl::const_pointer_image<3,float> slice_model::get_image(void)
{
    if(!image_data.data() && handle)
    {
        auto prior_show_prog = tipl::show_prog;
        tipl::show_prog = false;
        image_data = tipl::make_image(handle->mat_reader.read_as_type<float>(name),handle->dim);
        max_value = 0.0f;
        tipl::out() << name << " loaded" << std::endl;
        tipl::show_prog = prior_show_prog;
    }
    return image_data;
}

void slice_model::get_image_in_dwi(tipl::image<3>& I)
{
    if(iT != tipl::identity_matrix())
        tipl::resample(get_image(),I,iT);
    else
        I = get_image();
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

extern float odf8_vec[642][3];
extern unsigned short odf8_face[1280][3];


bool fiber_directions::add_data(fib_data& fib)
{
    tipl::progress prog("loading image volumes");
    auto& mat_reader = fib.mat_reader;
    dim = fib.dim;

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
            odf_table.resize(642);
            for (unsigned int index = 0;index < odf_table.size();++index)
            {
                odf_table[index][0] = odf8_vec[index][0];
                odf_table[index][1] = odf8_vec[index][1];
                odf_table[index][2] = odf8_vec[index][2];
            }
            half_odf_size = 321;
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
        else
        {
            odf_faces.resize(1280);
            for (unsigned int index = 0;index < odf_faces.size();++index)
            {
                odf_faces[index][0] = odf8_face[index][0];
                odf_faces[index][1] = odf8_face[index][1];
                odf_faces[index][2] = odf8_face[index][2];
            }
        }
    }

    for (unsigned int index = 0;prog(index,mat_reader.size());++index)
    {
        auto& mat_data = mat_reader[index];
        std::string matrix_name = mat_data.name;
        size_t total_size = mat_reader.cols(index)*mat_reader.rows(index);
        if(total_size != dim.size() && total_size != dim.size()*3)
            continue;
        if(tipl::begins_with(matrix_name,"subjects")) // database
            continue;
        if (matrix_name == "image")
        {
            check_index(0);
            mat_reader.read(index,fa[0]);
            findex_buf.resize(1);
            findex_buf[0].resize(total_size);
            findex[0] = &*(findex_buf[0].begin());
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
            if(!mat_reader.read(index,findex[store_index]))
                goto mat_reader_error;
            continue;
        }
        if (prefix_name == "fa")
        {
            check_index(store_index);
            if(!mat_reader.read(index,fa[store_index]))
                goto mat_reader_error;
            continue;
        }
        if (prefix_name == "dir")
        {
            const float* dir_ptr;
            if(!mat_reader.read(index,dir_ptr))
                goto mat_reader_error;
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
        if(!mat_reader.read(index,index_data[prefix_name_index][store_index]))
            goto mat_reader_error;
    }
    if(prog.aborted())
        return false;
    if(num_fiber == 0)
    {
        error_msg = "Invalid FIB format";
        return false;
    }

    // adding the primary fiber index
    index_name.insert(index_name.begin(),fa.size() == 1 ? "fa":"qa");
    index_data.insert(index_data.begin(),fa);
    fa_otsu = tipl::segmentation::otsu_threshold(tipl::make_image(fa[0],dim));

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
    }
    return num_fiber;

    mat_reader_error:
    error_msg = mat_reader.error_msg;
    return false;
}

bool fiber_directions::set_tracking_index(int new_index)
{
    if(new_index >= index_data.size() || new_index < 0)
        return false;
    fa = index_data[new_index];
    fa_otsu = tipl::segmentation::otsu_threshold(tipl::make_image(fa[0],dim));
    cur_index = new_index;
    return true;
}
bool fiber_directions::set_tracking_index(const std::string& name)
{
    return set_tracking_index(std::find(index_name.begin(),index_name.end(),name)-index_name.begin());
}


tipl::vector<3> fiber_directions::get_fib(size_t space_index,unsigned int order) const
{
    if(!dir.empty())
        return tipl::vector<3>(dir[order] + space_index + space_index + space_index);
    if(order >= findex.size())
        return odf_table[0];
    return odf_table[findex[order][space_index]];
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

void tracking_data::read(std::shared_ptr<fib_data> fib)
{
    dim = fib->dim;
    vs = fib->vs;
    odf_table = fib->dir.odf_table;
    fib_num = uint8_t(fib->dir.num_fiber);
    fa = fib->dir.fa;
    fa_otsu = fib->dir.fa_otsu;
    dt_fa = fib->dir.dt_fa;
    findex = fib->dir.findex;
    dir = fib->dir.dir;
    if(!fib->dir.index_name.empty())
        threshold_name = fib->dir.get_threshold_name();
    if(!dt_fa.empty())
        dt_metrics= fib->dir.dt_metrics;
}

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
void prepare_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
void save_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
bool fib_data::load_from_file(const std::string& file_name)
{
    tipl::progress prog("opening ",file_name);
    tipl::image<3> I;
    tipl::io::gz_nifti header;
    fib_file_name = file_name;
    if((tipl::ends_with(file_name,".nii") ||
        tipl::ends_with(file_name,".nii.gz")) &&
        header.load_from_file(file_name))
    {
        if(header.dim(4) == 3)
        {
            tipl::image<3> x,y,z;
            header >> x;
            header >> y;
            header >> z;
            header.get_voxel_size(vs);
            header.get_image_transformation(trans_to_mni);
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
            dir.odf_faces = ti.faces;
            dir.odf_table = ti.vertices;
            dir.half_odf_size = ti.half_vertices_count;
            tipl::adaptive_par_for(x.size(),[&](int i)
            {
                tipl::vector<3> v(-x[i],y[i],z[i]);
                float length = v.length();
                if(length == 0.0f)
                    return;
                v /= length;
                dir.fa_buf[0][i] = length;
                dir.findex_buf[0][i] = ti.discretize(v);
            });

            slices.push_back(std::make_shared<slice_model>("fiber",dir.fa[0],dim));
            match_template();
            tipl::out() << "NIFTI file loaded" << std::endl;
            return true;
        }
        else
        if(header.dim(4) && header.dim(4) % 3 == 0)
        {
            uint32_t fib_num = header.dim(4)/3;
            for(uint32_t i = 0;i < fib_num;++i)
            {
                tipl::image<3> x,y,z;
                header >> x;
                header >> y;
                header >> z;
                header.get_voxel_size(vs);
                header.get_image_transformation(trans_to_mni);
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
                dir.odf_faces = ti.faces;
                dir.odf_table = ti.vertices;
                dir.half_odf_size = ti.half_vertices_count;
                tipl::adaptive_par_for(x.size(),[&](uint32_t j)
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
            slices.push_back(std::make_shared<slice_model>("fiber",dir.fa[0],dim));
            match_template();
            tipl::out() << "NIFTI file loaded" << std::endl;
            return true;
        }
        else
        {
            header.toLPS(I);
            header.get_voxel_size(vs);
            header.get_image_transformation(trans_to_mni);
            tipl::out() << ((is_mni = header.is_mni()) ? "image treated as MNI-space image." : "image treated used as subject-space image" )<< std::endl;
        }
    }
    else
    if(std::filesystem::path(file_name).filename().string() == "2dseq")
    {
        tipl::io::bruker_2dseq bruker_header;
        if(!bruker_header.load_from_file(file_name))
        {
            error_msg = "Invalid 2dseq format";
            return false;
        }
        bruker_header.get_image().swap(I);
        bruker_header.get_voxel_size(vs);
        initial_LPS_nifti_srow(trans_to_mni,I.shape(),vs);
        std::ostringstream out;
        out << "Image resolution is (" << vs[0] << "," << vs[1] << "," << vs[2] << ")." << std::endl;
        report = out.str();
    }
    else
    if(tipl::ends_with(file_name,"trk.gz") ||
       tipl::ends_with(file_name,"trk") ||
       tipl::ends_with(file_name,"tck") ||
       tipl::ends_with(file_name,"tt.gz"))
    {
        if(!load_fib_from_tracks(file_name.c_str(),I,vs,trans_to_mni))
        {
            error_msg = "Invalid track format";
            return false;
        }
    }
    if(!I.empty())
    {
        mat_reader.push_back(std::make_shared<tipl::io::mat_matrix>("dimension",I.shape().data(),1,3));
        mat_reader.push_back(std::make_shared<tipl::io::mat_matrix>("voxel_size",vs.data(),1,3));
        mat_reader.push_back(std::make_shared<tipl::io::mat_matrix>("image",I.data(),uint32_t(I.plane_size()),I.depth()));
        load_from_mat();
        dir.index_name[0] = "image";
        slices[0]->name = "image";
        slices[0]->max_value = 0.0;// this allows calculating the min and max contrast
        trackable = false;
        tipl::out() << "image file loaded: " << I.shape() << std::endl;
        return true;
    }
    if(!std::filesystem::exists(file_name))
    {
        error_msg = "file does not exist: ";
        error_msg += file_name;
        return false;
    }

    prepare_idx(file_name,mat_reader.in);
    if(mat_reader.in->has_access_points())
    {
        mat_reader.delay_read = true;
        mat_reader.in->buffer_all = false;
    }
    if (!mat_reader.load_from_file(file_name,prog))
    {
        error_msg = mat_reader.error_msg;
        return false;
    }
    save_idx(file_name,mat_reader.in);


    if(!load_from_mat())
        return false;
    return true;
}


bool fib_data::save_slice(const std::string& index_name,const std::string& file_name)
{
    tipl::progress prog("saving ",file_name.c_str());
    auto save = [this,file_name](const auto& buf)->bool
    {
        if(!tipl::io::gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni,is_mni))
        {
            error_msg = "cannot save file ";
            error_msg += file_name;
            return false;
        }
        return true;
    };

    if(index_name == "fiber" || index_name == "dirs") // command line exp use "dirs"
    {
        tipl::image<4,float> buf(dim.expand(3*uint32_t(dir.num_fiber)));
        for(unsigned int j = 0,index = 0;j < dir.num_fiber;++j)
        for(int k = 0;k < 3;++k)
        for(size_t i = 0;i < dim.size();++i,++index)
            buf[index] = dir.get_fib(i,j)[k];
        return save(buf);
    }
    if(index_name.length() == 4 && index_name.substr(0,3) == "dir" && index_name[3]-'0' >= 0 && index_name[3]-'0' < int(dir.num_fiber))
    {
        tipl::image<4,float> buf(dim.expand(3));
        unsigned char dir_index = uint8_t(index_name[3]-'0');
        for(unsigned int j = 0,ptr = 0;j < 3;++j)
        for(size_t index = 0;index < dim.size();++index,++ptr)
            if(dir.fa[dir_index][index] > 0.0f)
                buf[ptr] = dir.get_fib(index,dir_index)[j];
        return save(buf);
    }
    if(index_name == "odfs")
    {
        odf_data odf;
        if(!odf.read(*this))
        {
            error_msg = odf.error_msg;
            return false;
        }
        tipl::image<4,float> buf(dim.expand(dir.half_odf_size));
        for(size_t pos = 0;pos < dim.size();++pos)
        {
            auto* ptr = odf.get_odf_data(pos);
            if(ptr!= nullptr)
                std::copy(ptr,ptr+dir.half_odf_size,buf.begin()+int64_t(pos)*dir.half_odf_size);
        }
        return save(buf);
    }
    size_t index = get_name_index(index_name);
    if(index >= slices.size())
    {
        error_msg = "cannot find metrics ";
        error_msg += index_name;
        return false;
    }

    if(index_name == "color")
    {
        tipl::image<3,tipl::rgb> buf(dim);
        for(int z = 0;z < buf.depth();++z)
        {
            tipl::color_image I;
            slices[index]->get_slice(uint8_t(2),uint32_t(z),I);
            std::copy(I.begin(),I.end(),buf.begin()+size_t(z)*buf.plane_size());
        }
        return save(buf);
    }


    if(tipl::ends_with(file_name,".mat"))
    {
        tipl::io::mat_write file(file_name.c_str());
        if(!file)
        {
            error_msg = "cannot save file ";
            error_msg += file_name;
            return false;
        }
        file << slices[index]->get_image();
        return true;
    }
    else
    {
        tipl::image<3> buf(slices[index]->get_image());
        if(slices[index]->get_image().shape() != dim)
        {
            tipl::image<3> new_buf(dim);
            tipl::resample<tipl::interpolation::cubic>(buf,new_buf,slices[index]->iT);
            new_buf.swap(buf);
        }
        return save(buf);
    }    
}
bool is_human_size(tipl::shape<3> dim,tipl::vector<3> vs)
{
    return dim[2] > 5 && dim[0]*vs[0] > 100 && dim[1]*vs[1] > 130;
}
extern int fib_ver;
bool check_fib_dim_vs(tipl::io::gz_mat_read& mat_reader,
                      tipl::shape<3>& dim,tipl::vector<3>& vs,tipl::matrix<4,4>& trans,bool& is_mni)
{
    int this_fib_ver(0);
    if(mat_reader.has("version") && (this_fib_ver = mat_reader.read_as_value<int>("version")) > fib_ver)
    {
        mat_reader.error_msg = "Incompatible FIB format. please update DSI Studio to open this new FIB file.";
        return false;
    }
    if (!mat_reader.read("dimension",dim))
    {
        mat_reader.error_msg = "cannot find dimension matrix";
        return false;
    }
    if(!dim.size())
    {
        mat_reader.error_msg = "invalid dimension";
        return false;
    }
    if (!mat_reader.read("voxel_size",vs))
    {
        mat_reader.error_msg = "cannot find voxel size matrix";
        return false;
    }
    // older version of gqi.fz does not have trans matrix
    if(!mat_reader.read("trans",trans))
        initial_LPS_nifti_srow(trans,dim,vs);
    if(!is_mni)
    {
        // now decide whether the fib is qsdr
        // in fib version >= 20250408, qsdr fib == has R2 (all .fz file will have "trans" matrix now)
        if(mat_reader.has("R2"))
            is_mni = true;
        // in fib version <= 202408, qsdr fib == has "trans" matrix (template fib.gz files don't have R2 matrix)
        if(this_fib_ver <= 202408 && mat_reader.has("trans"))
            is_mni = true;
    }
    tipl::out() << "fib_ver: " << this_fib_ver;
    tipl::out() << "dim: " << dim << " vs: " << vs;
    tipl::out() << "trans: " << trans;
    tipl::out() << "is qsdr: " << (is_mni ? "yes" : "no");
    return true;
}
tipl::const_pointer_image<3,unsigned char> handle_mask(tipl::io::gz_mat_read& mat_reader)
{
    const unsigned char* mask_ptr = nullptr;
    tipl::shape<3> dim;
    if(mat_reader.read("dimension",dim))
    {
        if(!mat_reader.read("mask",mask_ptr))
        {

            auto mask_mat = std::make_shared<tipl::io::mat_matrix>("mask");
            mask_mat->resize(tipl::shape<2>(dim.plane_size(),dim.depth()));
            auto& mask_buffer = mask_mat->data_buf;

            const float* fa0_ptr = nullptr;
            if(mat_reader.read("fa0",fa0_ptr) ||    // create mask from fib's fa map
               mat_reader.read("image0",fa0_ptr) || // create mask from src's b0
               mat_reader.read("image",fa0_ptr))    // create mask from t1w/t2w images
            {
                for(size_t i = 0;i < dim.size();++i)
                    if(fa0_ptr[i] > 0.0f)
                        mask_buffer[i] = 1;
            }
            mask_ptr = mask_buffer.data();
            mat_reader.push_back(mask_mat);
            tipl::out() << "mask created from the images";
        }
    }
    if(mask_ptr)
    {
        mat_reader.si2vi = tipl::get_sparse_index(tipl::make_image(mask_ptr,dim));
        mat_reader.mask_cols = dim.plane_size();
        mat_reader.mask_rows = dim.depth();
        tipl::out() << "mask voxels: " << mat_reader.si2vi.size();
    }
    return tipl::make_image(mask_ptr,mask_ptr ? dim : tipl::shape<3>(0,0,0));
}
bool fib_data::load_from_mat(void)
{
    if(!check_fib_dim_vs(mat_reader,dim,vs,trans_to_mni,is_mni))
    {
        error_msg = mat_reader.error_msg;
        return false;
    }
    mask = handle_mask(mat_reader);
    if(mask.empty())
    {
        error_msg = "invalid fib file: cannot create mask";
        return false;
    }
    mat_reader.read("report",report);
    mat_reader.read("steps",steps);
    mat_reader.read("intro",intro);
    mat_reader.read("other_images",other_images);


    if(!dir.add_data(*this))
    {
        error_msg = dir.error_msg;
        return false;
    }
    slices.push_back(std::make_shared<slice_model>(dir.fa.size() == 1 ? "fa":"qa",dir.fa[0],dim));
    for(unsigned int index = 1;index < dir.index_name.size();++index)
        slices.push_back(std::make_shared<slice_model>(dir.index_name[index],dir.index_data[index][0],dim));

    for (unsigned int index = 0;index < mat_reader.size();++index)
    {
        std::string matrix_name = mat_reader[index].name;
        if (matrix_name == "image" ||
            matrix_name == "mask" ||
            matrix_name.find("subjects") == 0)
            continue;
        std::string prefix_name(matrix_name.begin(),matrix_name.end()-1);
        char post_fix = matrix_name[matrix_name.length()-1];
        if(post_fix >= '0' && post_fix <= '9')
        {
            if (prefix_name == "index" || prefix_name == "fa" || prefix_name == "dir" ||
                std::find_if(slices.begin(),
                             slices.end(),
                             [&prefix_name](const auto& view)
                             {return view->name == prefix_name;}) != slices.end())
                continue;
        }
        if (mat_reader.rows(index)*mat_reader.cols(index) != dim.size())
            continue;
        slices.push_back(std::make_shared<slice_model>(matrix_name,this));
    }

    is_human_data = is_human_size(dim,vs); // 1 percentile head size in mm
    is_histology = (dim[2] == 2 && dim[0] > 400 && dim[1] > 400);





    if(!db.read_db(this))
    {
        error_msg = db.error_msg;
        return false;
    }

    if(is_mni)
    {
        // matching templates
        matched_template_id = 0;

        std::string template_name;
        if(mat_reader.read("template",template_name))
        {
            tipl::out() << "template: " << template_name;
            for(size_t index = 0;index < fa_template_list.size();++index)
            {
                auto name = std::filesystem::path(fa_template_list[index]).stem().stem().stem().string();
                if(template_name == name)
                {
                    matched_template_id = index;
                    set_template_id(matched_template_id);
                    return true;
                }
            }
        }

        for(int index = fa_template_list.size()-1;index >= 0;--index)
            if(tipl::contains(std::filesystem::path(fib_file_name).filename().string(),
                              std::filesystem::path(fa_template_list[index]).filename().string()))
            {
                matched_template_id = index;
                tipl::out() << "matched template (by file name): " <<
                                   std::filesystem::path(fa_template_list[index]).filename().string() << std::endl;
                set_template_id(matched_template_id);
                return true;
            }

        for(size_t index = 0;index < fa_template_list.size();++index)
        {
            tipl::io::gz_nifti read;
            if(!read.load_from_file(fa_template_list[index]))
                continue;
            tipl::vector<3> Itvs;
            tipl::shape<3> Itdim;
            read.get_image_dimension(Itdim);
            read.get_voxel_size(Itvs);
            if(std::abs(dim[0]-Itdim[0]*Itvs[0]/vs[0]) < 4.0f)
            {
                matched_template_id = index;
                tipl::out() << "matched template (by image size): " <<
                                   QFileInfo(fa_template_list[index].c_str()).baseName().toStdString() << std::endl;
                set_template_id(matched_template_id);
                return true;
            }
        }
        tipl::out() << "no matched template, use default: " <<
            std::filesystem::path(fa_template_list[matched_template_id]).stem().stem().stem() << std::endl;
        set_template_id(matched_template_id);
        return true;
    }

    match_template();
    return true;
}

bool read_fib_data(tipl::io::gz_mat_read& mat_reader)
{
    // get all data in delayed read condition
    tipl::progress prog("reading data");
    for(size_t i = 0;prog(i,mat_reader.size());++i)
    {
        auto& mat = mat_reader[i];
        if(mat.has_delay_read() && !mat.read(*(mat_reader.in.get())))
            return false;
    }
    if(prog.aborted())
        return false;
    return true;
}

bool save_fz(tipl::io::gz_mat_read& mat_reader,
              tipl::io::gz_mat_write& matfile,
              const std::vector<std::string>& skip_list,
              const std::vector<std::string>& skip_head_list)
{
    tipl::progress prog("saving");
    if(!mat_reader.si2vi.empty())
    {
        tipl::out() << "enabled masked format";
        matfile.apply_slope = true;
        matfile.apply_mask = true;
        matfile.mask_rows = mat_reader.mask_rows;
        matfile.mask_cols = mat_reader.mask_cols;
        matfile.si2vi = mat_reader.si2vi;
    }
    else
        tipl::out() << "no mask information. saving unmasked format";

    for(unsigned int index = 0;prog(index,mat_reader.size());++index)
    {
        if(!matfile)
        {
            mat_reader.error_msg = "cannot write to file. please check write permission.";
            return false;
        }
        const auto& name = mat_reader[index].name;
        bool skip = false;
        for(const auto& each : skip_list)
            if(name == each)
            {
                skip = true;
                break;
            }
        for(const auto& each : skip_head_list)
            if(name.find(each) == 0)
            {
                skip = true;
                break;
            }
        if(skip)
        {
            tipl::out() << "skip " << name;
            continue;
        }
        mat_reader.flush(index);

        // apply mask
        if(mat_reader[index].size() == matfile.mask_cols*matfile.mask_rows && mat_reader[index].name != "mask")
        {
            tipl::out() << "convert " << name << " data into masked format";
            if(mat_reader[index].is_type<float>())
                matfile.write<tipl::io::masked_sloped>(name,mat_reader.read_as_type<float>(index),
                                                       matfile.mask_rows,matfile.mask_cols);
            if(mat_reader[index].is_type<short>()) // index no slope
                matfile.write<tipl::io::masked>(name,mat_reader.read_as_type<short>(index),
                                                matfile.mask_rows,matfile.mask_cols);
            if(mat_reader[index].is_type<char>())
                matfile.write<tipl::io::masked>(name,mat_reader.read_as_type<char>(index),
                                                matfile.mask_rows,matfile.mask_cols);
            for(auto each : mat_reader[index].sub_data)
                matfile.write(*each.get());
            continue;
        }

        tipl::out() << "store " << name << " as is";
        matfile.write(mat_reader[index]);
    }
    return !prog.aborted();
}
bool modify_fib(tipl::io::gz_mat_read& mat_reader,
                const std::string& cmd,
                const std::string& param)
{
    tipl::out() << "fib command: " << cmd << " " << param;
    if(cmd == "remove")
    {
        if(!mat_reader.remove(param[0] >= '0' && param[0] <= '9' ? std::stoi(param) : int(mat_reader.index_of(param))))
        {
            mat_reader.error_msg = "invalid index";
            return false;
        }
        return true;
    }
    if(cmd == "rename")
    {
        auto data = tipl::split(param,' ');
        if(data.size() != 2)
        {
            mat_reader.error_msg = "invalid renaming command";
            return false;
        }

        auto row = std::stoi(data[0]);
        if(row >= mat_reader.size())
        {
            mat_reader.error_msg = "invalid row to rename";
            return false;
        }
        mat_reader[row].set_name(data[1]);
        return true;
    }

    if(!read_fib_data(mat_reader))
        return false;

    tipl::shape<3> dim;
    tipl::vector<3> vs;
    tipl::matrix<4,4,float> trans((tipl::identity_matrix()));
    bool is_mni;
    if(!check_fib_dim_vs(mat_reader,dim,vs,trans,is_mni))
        return false;

    handle_mask(mat_reader);

    if(cmd == "save" || cmd == "save_mini")
    {
        tipl::io::gz_mat_write matfile(param);
        if(!matfile)
        {
            mat_reader.error_msg = "cannot save file to ";
            mat_reader.error_msg += param;
            return false;
        }
        if(tipl::ends_with(param,".fib.gz"))
        {
            mat_reader.error_msg = "cannot save file to fib.gz format";
            return false;
        }
        if(cmd == "save_mini")
            return save_fz(mat_reader,matfile,{"odf_faces","odf_vertices","z0","mapping","dti_fa","md","ad","rd","fa3","fa4","rdi","index3","index4"},{"nrdi","subject"});
        return save_fz(mat_reader,matfile,{"odf_faces","odf_vertices","z0","mapping"},{"subject"});
    }


    tipl::progress prog(cmd.c_str());
    size_t p = 0;
    bool failed = false;
    bool first_mat = true;
    tipl::adaptive_par_for(mat_reader.size(),[&](unsigned int i)
    {
        if(!prog(p++,mat_reader.size()) || failed)
            return;
        auto& mat = mat_reader[i];
        size_t mat_size = mat_reader.cols(i)*mat_reader.rows(i);
        auto new_vs = vs;
        auto new_trans = trans;
        if(mat_size == 3*dim.size())
        {
            for(size_t d = 0;d < 3;++d)
            {
                tipl::image<3> new_image(dim);
                auto ptr = mat.get_data<float>()+d;
                for(size_t j = 0;j < dim.size();++j,ptr += 3)
                    new_image[j] = *ptr;
                if(!tipl::command<tipl::out,tipl::io::gz_nifti>(new_image,new_vs,new_trans,is_mni,cmd,param,true,mat_reader.error_msg))
                {
                    mat_reader.error_msg = "cannot perform ";
                    mat_reader.error_msg += cmd;
                    failed = true;
                    return;
                }
                if(d == 0)
                    mat.resize(tipl::vector<2,unsigned int>(3*new_image.width()*new_image.height(),new_image.depth()));
                for(size_t d = 0;d < 3;++d)
                {
                    auto ptr = mat.get_data<float>()+d;
                    for(size_t j = 0;j < dim.size();++j,ptr += 3)
                        *ptr = new_image[j];
                }
            }
        }
        if(mat_size == dim.size()) // image volumes, including fa, and fiber index
        {
            if(mat.is_type<short>() && (cmd == "normalize" || cmd.find("filter") != std::string::npos || cmd.find("_value") != std::string::npos))
                return;
            variant_image new_image;
            new_image.vs = vs;
            new_image.T = trans;
            new_image.shape = dim;
            if(!new_image.read_mat_image(i,mat_reader))
                return;
            if(tipl::begins_with(mat.name,"index"))
                new_image.interpolation = false;
            tipl::out() << mat_reader[i].name;
            if(!new_image.command(cmd,param))
            {
                mat_reader.error_msg = "cannot perform ";
                mat_reader.error_msg += cmd;
                failed = true;
                return;
            }

            new_image.write_mat_image(i,mat_reader);

            if(first_mat)
            {
                std::copy(new_image.shape.begin(),new_image.shape.end(),const_cast<unsigned int*>(mat_reader.read_as_type<unsigned int>("dimension")));
                std::copy(new_image.vs.begin(),new_image.vs.end(),const_cast<float*>(mat_reader.read_as_type<float>("voxel_size")));
                if(mat_reader.has("trans"))
                    std::copy(new_image.T.begin(),new_image.T.end(),const_cast<float*>(mat_reader.read_as_type<float>("trans")));
                first_mat = false;
            }
        }

    });
    if(failed)
        return false;
    return !prog.aborted();
}
bool fib_data::load_at_resolution(const std::string& file_name,float reso)
{
    tipl::progress prog("opening ",file_name);
    tipl::out() << "resample to resolution:" << reso;
    fib_file_name = file_name;
    if (!mat_reader.load_from_file(file_name,prog) ||
        !modify_fib(mat_reader,"regrid",std::to_string(reso)))
    {
        error_msg = mat_reader.error_msg;
        return false;
    }
    if(!load_from_mat())
        return false;
    return true;
}
bool fib_data::save_to_file(const std::string& file_name)
{
    if(!modify_fib(mat_reader,"save",file_name))
    {
        error_msg = mat_reader.error_msg;
        return false;
    }
    fib_file_name = file_name;
    return true;
}
void fib_data::remove_slice(size_t index)
{
    mat_reader.remove(slices[index]->name);
    slices.erase(slices.begin()+index);
}
size_t match_volume(tipl::const_pointer_image<3,unsigned char> mask,tipl::vector<3> vs);
void fib_data::match_template(void)
{
    if(is_human_size(dim,vs))
        set_template_id(0);
    else
    {
        tipl::out() << "image volume smaller than human young adult. try matching a template...";
        set_template_id(match_volume(handle_mask(mat_reader),vs));
    }
    tipl::out() << "matched template: " << std::filesystem::path(fa_template_list[template_id]).stem().stem().stem();
}

size_t fib_data::get_name_index(const std::string& index_name) const
{
    for(unsigned int index_num = 0;index_num < slices.size();++index_num)
        if(slices[index_num]->name == index_name)
            return index_num;
    for(unsigned int index_num = 0;index_num < slices.size();++index_num)
        if(slices[index_num]->name.find(index_name) != std::string::npos)
            return index_num;
    return slices.size();
}
std::vector<std::string> fib_data::get_index_list(void) const
{
    std::vector<std::string> index_list;
    for (const auto& each : slices)
        if(!each->optional())
            index_list.push_back(each->name);
    return index_list;
}

bool fib_data::set_dt_index(const std::pair<std::string,std::string>& name_pair,size_t type)
{

    auto find_index = [&](const std::string& metric,size_t& index,std::string& m_name)->bool
    {
        error_msg.clear();
        index = slices.size();
        m_name = metric;
        if(metric == "zero")
            return true;
        index = get_name_index(metric);
        if(index == slices.size())
        {
            error_msg = "cannot find the metric: ";
            error_msg += metric;
            return false;
        }
        if(slices[index]->registering)
        {
            error_msg = "registration undergoing. please wait until registration complete.";
            return false;
        }
        if(slices[index]->name != metric)
            tipl::warning() << "specified " << metric << " but not found. The analysis will use " << (m_name = slices[index]->name);
        return true;
    };

    std::string m1_name,m2_name;
    std::pair<size_t,size_t> pair;
    if(!find_index(name_pair.first,pair.first,m1_name) ||
       !find_index(name_pair.second,pair.second,m2_name))
        return false;

    tipl::image<3> m1(dim),m2(dim);
    if(pair.first < slices.size())
        slices[pair.first]->get_image_in_dwi(m1);
    if(pair.second < slices.size())
        slices[pair.second]->get_image_in_dwi(m2);

    dir.dt_fa_data = std::move(tipl::image<3>(dim));
    auto& dif = dir.dt_fa_data;
    switch(type)
    {
        case 0: // (m1-m2)÷m1
            dir.dt_metrics = "(" + m1_name + "-" + m2_name + ")/" + m1_name;
            for(size_t k = 0;k < m1.size();++k)
                if(dir.fa[0][k] > 0.0f && m1[k] > 0.0f && m2[k] > 0.0f)
                    dif[k] = 1.0f-m2[k]/m1[k];

        break;
        case 1: // (m1-m2)÷m2
            dir.dt_metrics = "(" + m1_name + "-" + m2_name + ")/" + m2_name;
            for(size_t k = 0;k < m1.size();++k)
                if(dir.fa[0][k] > 0.0f && m1[k] > 0.0f && m2[k] > 0.0f)
                    dif[k] = m1[k]/m2[k]-1.0f;
        break;
        case 2: // m1-m2
            dir.dt_metrics = m1_name + "-" + m2_name;
            for(size_t k = 0;k < m1.size();++k)
                if(dir.fa[0][k] > 0.0f && m1[k] > 0.0f && m2[k] > 0.0f)
                    dif[k] = m1[k]-m2[k];
        break;
        case 3: // (m2-m1)÷m1
            dir.dt_metrics = "(" + m2_name + "-" + m1_name + ")/" + m1_name;
            for(size_t k = 0;k < m1.size();++k)
                if(dir.fa[0][k] > 0.0f && m1[k] > 0.0f && m2[k] > 0.0f)
                    dif[k] = m2[k]/m1[k]-1.0f;

        break;
        case 4: // (m2-m1)÷m2
            dir.dt_metrics = "(" + m2_name + "-" + m1_name + ")/" + m2_name;
            for(size_t k = 0;k < m1.size();++k)
                if(dir.fa[0][k] > 0.0f && m1[k] > 0.0f && m2[k] > 0.0f)
                    dif[k] = 1.0f-m1[k]/m2[k];
        break;
        case 5: // m2-m1
            dir.dt_metrics = m2_name + "-" + m1_name;
            for(size_t k = 0;k < m1.size();++k)
                if(dir.fa[0][k] > 0.0f && m1[k] > 0.0f && m2[k] > 0.0f)
                    dif[k] = m2[k]-m1[k];
        break;
        case 6: // m1/max(m1)
            dir.dt_metrics = m1_name + "/max(" + m1_name + ")";
            {
                float max_v = tipl::max_value(m1);
                if(max_v > 0.0f)
                    for(size_t k = 0;k < m1.size();++k)
                        dif[k] = m1[k]/max_v;
            }
        break;
        case 7: // m2/max(m2)
            dir.dt_metrics = m2_name + "/max(" + m2_name + ")";
            {
                float max_v = tipl::max_value(m2);
                if(max_v > 0.0f)
                    for(size_t k = 0;k < m2.size();++k)
                        dif[k] = m2[k]/max_v;
            }
        break;
    }
    tipl::out() << "dt metrics:" << dir.dt_metrics;
    dir.dt_fa = std::vector<const float*>(size_t(dir.num_fiber),(const float*)&dif[0]);
    return true;
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
            auto d = dir.get_fib(space_index,i);
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
    for(const auto& each : slices)
        if(each->image_ready())
        {
            auto I = each->get_image();
            if(I.empty())
                buf.push_back(0.0f);
            else
            {
                if(I.shape() != dim)
                {
                    tipl::vector<3> pos(x,y,z);
                    pos.to(each->iT);
                    buf.push_back(tipl::estimate(I,pos));
                }
                else
                    buf.push_back(I[space_index]);
            }
        }
}




extern std::vector<std::string> fa_template_list,iso_template_list;
extern std::vector<std::vector<std::string> > atlas_file_name_list;


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

void apply_inverse_trans(tipl::vector<3>& pos,const tipl::matrix<4,4>& trans)
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
bool fib_data::add_atlas(const std::string& file_name)
{
    if(!std::filesystem::exists(file_name) || !tipl::ends_with(file_name,"nii.gz"))
        return false;
    atlas_list.push_back(std::make_shared<atlas>());
    atlas_list.back()->name = QFileInfo(file_name.c_str()).baseName().toStdString();
    atlas_list.back()->filename = file_name;
    atlas_list.back()->template_to_mni = template_I.empty() ? trans_to_mni : template_to_mni;
    return true;
}
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
        for(size_t i = 0;i < atlas_file_name_list[template_id].size();++i)
            add_atlas(atlas_file_name_list[template_id][i]);
        // populate other modality name
        t1w_template_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".T1W.nii.gz").toStdString();
        t2w_template_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".T2W.nii.gz").toStdString();
        wm_template_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".WM.nii.gz").toStdString();
        mask_template_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".mask.nii.gz").toStdString();

        // handle tractography atlas
        tractography_atlas_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".tt.gz").toStdString();
        tractography_name_list.clear();
        track_atlas.reset();
        std::ifstream in(tractography_atlas_file_name+".txt");
        if(std::filesystem::exists(tractography_atlas_file_name) && in)
        {
            std::copy(std::istream_iterator<std::string>(in),std::istream_iterator<std::string>(),std::back_inserter(tractography_name_list));
            auto tractography_atlas_roi_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".roi.nii.gz").toStdString();
            if(std::filesystem::exists(tractography_atlas_roi_file_name))
            {
                tractography_atlas_roi = std::make_shared<atlas>();
                tractography_atlas_roi->name = "tractography atlas ROI";
                tractography_atlas_roi->filename = tractography_atlas_roi_file_name;
                tractography_atlas_roi->template_to_mni = trans_to_mni;
            }
            auto tractography_atlas_roa_file_name = QString(fa_template_list[template_id].c_str()).replace(".QA.nii.gz",".roa.nii.gz").toStdString();
            if(std::filesystem::exists(tractography_atlas_roa_file_name))
            {
                tractography_atlas_roa = std::make_shared<atlas>();
                tractography_atlas_roa->name = "tractography atlas ROA";
                tractography_atlas_roa->filename = tractography_atlas_roa_file_name;
                tractography_atlas_roa->template_to_mni = trans_to_mni;
            }
        }

        alternative_mapping_index = 0;
        alternative_mapping = { "" };
        auto files = tipl::search_files(std::filesystem::path(fa_template_list[template_id]).parent_path().string(),"*.mz");
        alternative_mapping.insert(alternative_mapping.end(),files.begin(),files.end());

    }
}
std::vector<std::string> fib_data::get_tractography_all_levels(void)
{
    std::vector<std::string> list;
    std::string last_insert;
    for(size_t index = 0;index < tractography_name_list.size();++index)
    {
        auto sep_count = std::count(tractography_name_list[index].begin(),tractography_name_list[index].end(),'_');
        if(sep_count >= 2) // sub-bundles
        {
            // get their parent bundle
            auto insert = tractography_name_list[index].substr(0,tractography_name_list[index].find_last_of('_'));
            if(insert != last_insert)
                list.push_back((last_insert = insert).c_str());
        }
        list.push_back(tractography_name_list[index].c_str());
    }
    return list;
}
std::vector<std::string> fib_data::get_tractography_level0(void)
{
    std::vector<std::string> list;
    for (const auto &each : tractography_name_list)
    {
        auto level0 = each.substr(0, each.find("_"));
        if(list.empty() || level0 != list.back())
            list.push_back(level0);
    }
    return list;
}
std::vector<std::string> fib_data::get_tractography_level1(const std::string& group)
{
    std::vector<std::string> list;
    std::string last;
    for (const auto &each : tractography_name_list)
    {
        auto sep = tipl::split(each,'_');
        if(sep.size() >= 2 && group == sep[0] && last != sep[1])
        {
            list.push_back(sep[1]);
            last = sep[1];
        }
    }
    return list;
}
std::vector<std::string> fib_data::get_tractography_level2(const std::string& group1,const std::string& group2)
{
    std::vector<std::string> list;
    std::string last;
    for (const auto &each : tractography_name_list)
    {
        auto sep = tipl::split(each,'_');
        if(sep.size() >= 3 && group1 == sep[0] && group2 == sep[1] && last != sep[2])
        {
            list.push_back(sep[2]);
            last = sep[2];
        }
    }
    return list;
}

bool fib_data::load_template(void)
{
    if(!template_I.empty())
        return true;
    if(is_mni && template_id == matched_template_id)
    {
        template_I.resize(dim); // this will skip the load_template function
        template_vs = vs;
        template_to_mni = trans_to_mni;
        return true;
    }
    tipl::io::gz_nifti read;
    if(!read.load_from_file(fa_template_list[template_id].c_str()))
    {
        error_msg = "cannot load " + fa_template_list[template_id];
        return false;
    }
    read.toLPS(template_I);
    read.get_voxel_size(template_vs);
    read.get_image_transformation(template_to_mni);
    float ratio = float(template_I.width()*template_vs[0])/float(dim[0]*vs[0]);
    if(ratio < 0.25f || ratio > 8.0f)
    {
        error_msg = "image resolution mismatch: ratio=" + std::to_string(ratio);
        return false;
    }
    unsigned int downsampling = 0;
    while((!is_human_data && template_I.width()/3 > int(dim[0])) ||
          (is_human_data && template_vs[0]*2.0f <= int(vs[0])))
    {
        tipl::out() << "downsampling template by 2x to match subject resolution" << std::endl;
        template_vs *= 2.0f;
        template_to_mni[0] *= 2.0f;
        template_to_mni[5] *= 2.0f;
        template_to_mni[10] *= 2.0f;
        tipl::downsampling(template_I);
        ++downsampling;
    }

    for(size_t i = 0;i < atlas_list.size();++i)
        atlas_list[i]->template_to_mni = template_to_mni;
    if(tractography_atlas_roi.get())
        tractography_atlas_roi->template_to_mni = template_to_mni;

    // load iso template if exists
    {
        tipl::io::gz_nifti read2;
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
void fib_data::temp2sub(std::vector<std::vector<float> >&tracts) const
{
    tipl::adaptive_par_for(tracts.size(),[&](size_t i)
    {
        if(tracts.size() < 6)
            return;
        auto beg = tracts[i].begin();
        auto end = tracts[i].end();
        for(;beg != end;beg += 3)
        {
            tipl::vector<3> p(beg);
            temp2sub(p);
            beg[0] = p[0];
            beg[1] = p[1];
            beg[2] = p[2];
        }
    });
}
bool fib_data::load_track_atlas(bool symmetric)
{
    if(!std::filesystem::exists(tractography_atlas_file_name))
    {
        error_msg = "no tractography atlas in ";
        error_msg += QFileInfo(fa_template_list[template_id].c_str()).baseName().toStdString();
        error_msg += " template";
        return false;
    }
    if(tractography_name_list.empty())
    {
        error_msg = "no label text file for the tractography atlas";
        return false;
    }
    if(!track_atlas.get())
    {
        tipl::progress prog(symmetric ?  "loading symmetric tractography atlas" : "loading asymmetric tractography atlas");
        if(!map_to_mni(tipl::show_prog))
            return false;
        // load the tract to the template space
        track_atlas = std::make_shared<TractModel>(template_I.shape(),template_vs,template_to_mni);
        track_atlas->is_mni = true;
        if(!track_atlas->load_tracts_from_file(tractography_atlas_file_name.c_str(),this,true))
        {
            error_msg = "failed to load tractography atlas: ";
            error_msg += tractography_atlas_file_name;
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
        auto& cluster = track_atlas->tract_cluster;

        std::vector<std::vector<float> > new_tracts;
        std::vector<unsigned int> new_cluster;

        if(symmetric)
        {
            for(size_t i = 0;i < tracts.size();++i)
                if(pair[cluster[i]] < tractography_name_list.size())
                {
                    new_tracts.push_back(tracts[i]);
                    auto& tract = new_tracts.back();
                    // mirror in the x
                    for(size_t pos = 0;pos < tract.size();pos += 3)
                        tract[pos] = (template_I.shape()[0]-1)-tract[pos];
                    new_cluster.push_back(pair[cluster[i]]);
                }
        }
        else
        {
            new_tracts = tracts;
            new_cluster = cluster;
        }

        // add adds
        track_atlas->add_tracts(new_tracts);
        cluster.insert(cluster.end(),new_cluster.begin(),new_cluster.end());


        // get distance scaling
        auto& s2t = get_sub2temp_mapping();
        if(s2t.empty())
            return false;
        tract_atlas_jacobian = float((s2t[0]-s2t[1]).length());
        // warp tractography atlas to subject space
        temp2sub(track_atlas->get_tracts());

        auto& tract_data = track_atlas->get_tracts();
        // get min max length
        std::vector<float> min_length(tractography_name_list.size()),max_length(tractography_name_list.size());
        tipl::adaptive_par_for(tract_data.size(),[&](size_t i)
        {
            if(tract_data.size() <= 6)
                return;
            auto c = cluster[i];
            if(c < tractography_name_list.size())
            {
                double length = track_atlas->get_tract_length_in_mm(i);
                if(min_length[c] == 0)
                    min_length[c] = float(length);
                min_length[c] = std::min(min_length[c],float(length));
                max_length[c] = std::max(max_length[c],float(length));
            }
        });
        tract_atlas_min_length.swap(min_length);
        tract_atlas_max_length.swap(max_length);
    }
    return true;
}
//---------------------------------------------------------------------------
std::vector<size_t> fib_data::get_track_ids(const std::string& tract_name)
{
    std::vector<size_t> track_ids;
    for(size_t i = 0;i < tractography_name_list.size();++i)
        if(tipl::contains_case_insensitive(tractography_name_list[i],tract_name))
        track_ids.push_back(i);
    return track_ids;
}
//---------------------------------------------------------------------------
std::pair<float,float> fib_data::get_track_minmax_length(const std::string& tract_name)
{
    std::pair<float,float> minmax(0.0f,0.0f);
    auto track_ids = get_track_ids(tract_name);
    if(track_ids.empty())
        return std::make_pair(0.0f,0.0f);
    for(size_t i = 0;i < track_ids.size();++i)
    {
        if(i == 0)
        {
            minmax.first = tract_atlas_min_length[track_ids[0]];
            minmax.second= tract_atlas_max_length[track_ids[0]];
        }
        else
        {
            minmax.first  = std::min<float>(minmax.first,tract_atlas_min_length[track_ids[i]]);
            minmax.second = std::max<float>(minmax.second,tract_atlas_max_length[track_ids[i]]);
        }
    }
    return minmax;
}

//---------------------------------------------------------------------------
template<typename T,typename U>
unsigned int find_nearest_contain(const float* trk,unsigned int length,
                          const T& tract_data,// = track_atlas->get_tracts();
                          const U& tract_cluster)
{
    struct norm1_imp{
        inline float operator()(const float* v1,const float* v2)
        {
            return std::fabs(v1[0]-v2[0])+std::fabs(v1[1]-v2[1])+std::fabs(v1[2]-v2[2]);
        }
    } norm1;

    struct min_min_imp{
        inline float operator()(float min_dis,const float* v1,const float* v2)
        {
            float d1 = std::fabs(v1[0]-v2[0]);
            if(d1 > min_dis)                    return min_dis;
            d1 += std::fabs(v1[1]-v2[1]);
            if(d1 > min_dis)                    return min_dis;
            d1 += std::fabs(v1[2]-v2[2]);
            if(d1 > min_dis)                    return min_dis;
            return d1;
        }
    }min_min;
    size_t best_index = tract_data.size();
    float best_distance = std::numeric_limits<float>::max();
    for(size_t i = 0;i < tract_data.size();++i)
    {
        bool skip = false;
        float max_dis = 0;
        for(size_t n = 0;n < length;n += 6)
        {
            float min_dis = norm1(&tract_data[i][0],trk+n);
            for(size_t m = 0;m < tract_data[i].size() && min_dis > max_dis;m += 3)
                min_dis = min_min(min_dis,&tract_data[i][m],trk+n);
            if(min_dis > max_dis)
                max_dis = min_dis;
            if(max_dis > best_distance)
            {
                skip = true;
                break;
            }
        }
        if(!skip && max_dis < best_distance)
        {
            best_distance = max_dis;
            best_index = i;
        }
    }
    return tract_cluster[best_index];
}

//---------------------------------------------------------------------------

bool fib_data::recognize(std::shared_ptr<TractModel>& trk,
                         std::vector<unsigned int>& labels,
                         std::vector<unsigned int>& label_count)
{
    if(!load_track_atlas(false/*asymmetric*/))
        return false;
    labels.resize(trk->get_tracts().size());

    tipl::progress prog("recognizing tracks");
    size_t total = 0;
    tipl::par_for(trk->get_tracts().size(),[&](size_t i)
    {
        if(trk->get_tracts()[i].empty() || prog.aborted())
            return;
        prog(total++,trk->get_tracts().size());
        labels[i] = find_nearest_contain(&(trk->get_tracts()[i][0]),uint32_t(trk->get_tracts()[i].size()),track_atlas->get_tracts(),track_atlas->tract_cluster);
    },std::thread::hardware_concurrency());
    if(prog.aborted())
        return false;
    std::vector<unsigned int> count(tractography_name_list.size());
    for(auto l : labels)
    {
        if(l < count.size())
            ++count[l];
    }
    label_count.swap(count);
    return true;
}

bool fib_data::recognize(std::shared_ptr<TractModel>& trk,
               std::vector<unsigned int>& labels,
               std::vector<std::string> & label_names)
{
    if(!load_track_atlas(false/*asymmetric*/))
        return false;
    std::vector<unsigned int> c,count;
    recognize(trk,c,count);
    std::multimap<unsigned int,unsigned int,std::greater<unsigned int> > tract_list;
    for(unsigned int i = 0;i < count.size();++i)
        if(count[i])
            tract_list.insert(std::make_pair(count[i],i));

    unsigned int index = 0;
    labels.resize(c.size());
    for(auto p : tract_list)
    {
        for(size_t j = 0;j < c.size();++j)
            if(c[j] == p.second)
                labels[j] = index;
        label_names.push_back(tractography_name_list[p.second]);
        ++index;
    }
    return true;
}

std::multimap<float,std::string,std::greater<float> > fib_data::recognize_and_sort(std::shared_ptr<TractModel> trk)
{
    std::multimap<float,std::string,std::greater<float> > result;
    std::vector<unsigned int> labels,count;
    if(!load_track_atlas(false/*asymmetric*/) || !recognize(trk,labels,count))
        return result;
    auto sum = std::accumulate(count.begin(),count.end(),0);
    for(size_t i = 0;i < count.size();++i)
        if(count[i])
            result.insert(std::make_pair(float(count[i])/float(sum),tractography_name_list[i]));
    return result;
}
void fib_data::recognize_report(std::shared_ptr<TractModel>& trk,std::string& report)
{
    auto result = recognize_and_sort(trk);
    if(result.empty())
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


bool fib_data::map_to_mni(bool background)
{
    if(!load_template())
        return false;
    if(is_mni && template_id == matched_template_id)
        return true;
    if(!s2t.empty() && !t2s.empty())
        return true;
    std::string output_file_name(fib_file_name);
    output_file_name += ".";
    output_file_name += QFileInfo(fa_template_list[template_id].c_str()).baseName().toLower().toStdString();
    if(alternative_mapping_index)
        output_file_name += std::to_string(alternative_mapping_index);
    output_file_name += ".mz";
    if(std::filesystem::exists(output_file_name))
    {
        tipl::out() << "use existing mapping";
        tipl::progress p("open ",output_file_name);
        if(load_mapping(output_file_name))
            return true;
        if(!error_msg.empty())
        {
            tipl::out() << "cannot use existing mapping file: " << error_msg;
            error_msg.clear();
        }
    }

    tipl::progress p("normalization");
    prog = 0;
    auto lambda = [this,output_file_name]()
    {
        prog = 1;
        dual_reg reg;

        reg.modality_names = {"qa","iso"};

        size_t iso_index = get_name_index("iso");
        if(slices.size() == iso_index)
            iso_index = get_name_index("rd");

        reg.I[0] = subject_image_pre(tipl::image<3>(dir.fa[0],dim));
        if(iso_index < slices.size())
            reg.I[1] = subject_image_pre(tipl::image<3>(slices[iso_index]->get_image()));

        reg.Is = dim;
        reg.Ivs = vs;
        reg.IR = trans_to_mni;

        reg.It[0] = template_image_pre(template_I);
        reg.It[1] = template_image_pre(template_I2);

        reg.Its = template_I.shape();
        reg.Itvs = template_vs;
        reg.ItR = template_to_mni;

        // not FIB file, use t1w/t1w or others as template
        if(dir.index_name[0] == "image")
        {
            reg.modality_names = {"t1w","t2w","qa","iso"};
            tipl::out() << "reloading all t1w/t2w/qa/iso";
            if(!reg.load_template(0,t1w_template_file_name) ||
               !reg.load_template(1,t2w_template_file_name) ||
               !reg.load_template(2,fa_template_list[template_id]) ||
               !reg.load_template(3,iso_template_list[template_id]))
            {
                error_msg = "cannot perform normalization";
                tipl::prog_aborted = true;
                return;
            }
            reg.match_resolution(true);
            tipl::out() << "try using t1w image for registration..." << std::endl;
            reg.cost_type = tipl::reg::mutual_info;
            reg.linear_reg(tipl::prog_aborted);
            if(tipl::prog_aborted)
                return;
            auto best_index = std::max_element(reg.r.begin(),reg.r.end())-reg.r.begin();
            float best_r = reg.r[best_index];
            auto It = reg.It;
            auto arg = reg.arg;

            tipl::out() << "try using skull-stripped t1w for registration..." << std::endl;
            tipl::preserve(It[0],It[3]);
            tipl::preserve(It[1],It[3]);
            reg.It.swap(It);
            reg.linear_reg(tipl::prog_aborted);
            if(tipl::prog_aborted)
                return;
            if(best_r > tipl::max_value(reg.r))
            {
                tipl::out() << "using with-skull registration";
                reg.arg = arg; //restore linear transformation
                reg.It.swap(It); // restore It
            }
            else
            {
                tipl::out() << "using without-skull registration";
                best_index = std::max_element(reg.r.begin(),reg.r.end())-reg.r.begin();
            }
            reg.It[0] = reg.It[best_index];
            reg.It[1].clear();
            reg.cost_type = tipl::reg::corr;
            tipl::out() << "using " << reg.modality_names[best_index] << " for registration";
            reg.modality_names = {reg.modality_names[best_index]};
        }


        prog = 2;
        if(has_manual_atlas)
            reg.arg = manual_template_T;
        else
        {
            if(alternative_mapping_index && alternative_mapping_index < alternative_mapping.size() &&
              !reg.load_alternative_warping(alternative_mapping[alternative_mapping_index]))
            {
                error_msg = reg.error_msg;
                tipl::prog_aborted = true;
                return;
            }
            reg.match_resolution(false);
            reg.linear_reg(tipl::prog_aborted);
        }

        if(tipl::prog_aborted)
            return;
        prog = 3;

        reg.nonlinear_reg(tipl::prog_aborted);
        if(reg.r[0] < 0.3f)
        {
            error_msg = "cannot perform normalization";
            tipl::prog_aborted = true;
        }
        if(tipl::prog_aborted)
            return;

        reg.to_It_space(template_I.shape(),template_to_mni);
        s2t.swap(reg.from2to);
        t2s.swap(reg.to2from);
        prog = 4;
        if(!reg.save_warping(output_file_name.c_str()))
            tipl::error() << reg.error_msg;
    };

    if(background)
    {
        reg_threads.push_back(std::make_shared<std::thread>(lambda));
        while(p(prog,4))
            std::this_thread::yield();
        if(p.aborted())
        {
            error_msg = "aborted.";
            prog = 0;
        }
        return !p.aborted();
    }
    lambda();
    return !p.aborted();
}
bool fib_data::load_mapping(const std::string& file_name)
{
    if(tipl::ends_with(file_name,".mz"))
    {
        if(std::filesystem::last_write_time(file_name) < std::filesystem::last_write_time(fib_file_name))
        {
            error_msg = "The mapping file was created before the fib file.";
            return false;
        }
        dual_reg map;
        if(!map.load_warping(file_name))
        {
            error_msg = map.error_msg;
            return false;
        }
        if(map.from2to.shape() != dim)
        {
            error_msg = "image size mismatch in the mapping file";
            return false;
        }
        if(map.to2from.shape() != template_I.shape() || map.ItR != template_to_mni)
        {
            tipl::out() << "transform mappings";
            tipl::out() << "dim: " << map.to2from.shape() << "->" << template_I.shape();
            tipl::out() << "trans: " << map.ItR;
            tipl::out() << "new trans:" << template_to_mni;
            map.to_It_space(template_I.shape(),template_to_mni);
        }

        s2t.swap(map.from2to);
        t2s.swap(map.to2from);

        prog = 6;
        return true;
    }

    tipl::io::gz_nifti nii;
    tipl::out() << "opening " << file_name;
    if(!nii.load_from_file(file_name))
    {
        error_msg = nii.error_msg;
        return false;
    }
    tipl::image<3> shiftx,shifty,shiftz;
    tipl::matrix<4,4,float> trans;
    nii >> shiftx;
    nii >> shifty;
    nii >> shiftz;
    nii.get_image_transformation(trans);
    tipl::out() << "dimension: " << shiftx.shape();
    tipl::out() << "trans_to_mni: " << trans;

    if(shiftx.shape() != dim || shifty.shape() != dim || shiftz.shape() != dim)
    {
        error_msg = "image size does not match";
        return false;
    }
    auto T = template_to_mni;
    T.inv();
    s2t.resize(dim);
    t2s.resize(template_I.shape());
    tipl::out() << s2t[0];
    tipl::adaptive_par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
                  [&](const tipl::pixel_index<3>& index)
    {
        s2t[index.index()] = index;
        apply_trans(s2t[index.index()],trans);
        s2t[index.index()][0] += shiftx[index.index()];
        s2t[index.index()][1] += shifty[index.index()];
        s2t[index.index()][2] += shiftz[index.index()];
        apply_trans(s2t[index.index()],T);

    });
    tipl::out() << s2t[0];
    return true;
}

void fib_data::temp2sub(tipl::vector<3>& pos) const
{
    if(is_mni && template_id == matched_template_id)
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
    if(is_mni && template_id == matched_template_id)
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
void fib_data::mni2sub(tipl::vector<3>& pos)
{
    apply_inverse_trans(pos,template_to_mni);
    temp2sub(pos);
}
std::shared_ptr<atlas> fib_data::get_atlas(const std::string atlas_name)
{
    for(auto at : atlas_list)
        if(at->name == atlas_name)
            return at;
    if(!add_atlas(atlas_name))
    {
        error_msg = atlas_name + " is not one of the built-in atlases or an readable nifti file";
        return std::shared_ptr<atlas>();
    }
    return atlas_list.back();
}

bool fib_data::get_atlas_roi(const std::string& atlas_name,const std::string& region_name,std::vector<tipl::vector<3,short> >& points)
{
    auto at = get_atlas(atlas_name);
    if(!at.get())
    {
        error_msg = "cannot find atlas: ";
        error_msg += atlas_name;
        return false;
    }
    return get_atlas_roi(at,region_name,points);
}
bool fib_data::get_atlas_roi(std::shared_ptr<atlas> at,const std::string& region_name,std::vector<tipl::vector<3,short> >& points)
{
    if(!at.get())
        return false;
    auto roi_index = uint32_t(std::find(at->get_list().begin(),at->get_list().end(),region_name)-at->get_list().begin());
    if(roi_index == at->get_list().size())
    {
        bool ok = false;
        roi_index = uint32_t(QString(region_name.c_str()).toInt(&ok));
        if(!ok)
        {
            error_msg = "cannot find region: ";
            error_msg += region_name;
            return false;
        }
    }
    return get_atlas_roi(at,roi_index,points);
}
bool fib_data::get_atlas_roi(std::shared_ptr<atlas> at,unsigned int roi_index,
                             const tipl::shape<3>& new_geo,const tipl::matrix<4,4>& to_diffusion_space,
                             std::vector<tipl::vector<3,short> >& points)
{
    if(get_sub2temp_mapping().empty() || !at->load_from_file())
    {
        error_msg = "no mni mapping";
        return false;
    }
    tipl::out() << "loading " << at->get_list()[roi_index] << " from " << at->name << std::endl;

    std::vector<std::vector<tipl::vector<3,short> > > buf(tipl::max_thread_count);

    // trigger atlas loading to avoid crash in multi thread
    if(!at->load_from_file())
    {
        error_msg = "cannot read atlas file ";
        error_msg += at->filename;
        return false;
    }
    if(new_geo == dim && to_diffusion_space == tipl::identity_matrix())
    {
        tipl::adaptive_par_for<tipl::sequential_with_id>(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
            [&](const tipl::pixel_index<3>& index,size_t id)
        {
            if (at->is_labeled_as(s2t[index.index()], roi_index))
                buf[id].push_back(tipl::vector<3,short>(index.begin()));
        });
    }
    else
    {
        tipl::adaptive_par_for<tipl::sequential_with_id>(tipl::begin_index(new_geo),tipl::end_index(new_geo),
            [&](const tipl::pixel_index<3>& index,size_t id)
        {
            tipl::vector<3> p(index),p2;
            p.to(to_diffusion_space);
            if(!tipl::estimate(s2t,p,p2))
                return;
            if (at->is_labeled_as(p2, roi_index))
                buf[id].push_back(tipl::vector<3,short>(index.begin()));
        });
    }
    tipl::aggregate_results(std::move(buf),points);
    return true;
}

bool fib_data::get_atlas_all_roi(std::shared_ptr<atlas> at,
                                 const tipl::shape<3>& new_geo,const tipl::matrix<4,4>& to_diffusion_space,
                                 std::vector<std::vector<tipl::vector<3,short> > >& points,
                                 std::vector<std::string>& names)
{
    if(get_sub2temp_mapping().empty())
    {
        error_msg = "cannot warp subject image to the template space";
        return false;
    }
    // trigger atlas loading to avoid crash in multi thread
    if(!at.get())
    {
        error_msg = "cannot load atlas";
        return false;
    }
    if(!at->load_from_file())
    {
        error_msg = "cannot read atlas file ";
        error_msg += at->filename;
        return false;
    }
    std::vector<std::vector<std::vector<tipl::vector<3,short> > > > region_voxels(std::thread::hardware_concurrency());
    for(auto& region : region_voxels)
    {
        region.clear();
        region.resize(at->get_list().size());
    }

    bool need_trans = (new_geo != dim || to_diffusion_space != tipl::identity_matrix());
    auto shape = need_trans ? new_geo : dim;
    tipl::par_for<tipl::sequential_with_id>(tipl::begin_index(shape),tipl::end_index(shape),
                [&](const tipl::pixel_index<3>& index,size_t id)
    {
        tipl::vector<3> p2;
        if(need_trans)
        {
            tipl::vector<3> p(index);
            p.to(to_diffusion_space);
            if(!tipl::estimate(s2t,p,p2))
                return;
        }
        else
            p2 = s2t[index.index()];

        if(at->is_multiple_roi)
        {
            std::vector<uint16_t> region_indicies;
            at->region_indices_at(p2,region_indicies);
            if(region_indicies.empty())
                return;
            tipl::vector<3,short> point(index.begin());
            for(unsigned int i = 0;i < region_indicies.size();++i)
            {
                auto region_index = region_indicies[i];
                region_voxels[id][region_index].push_back(point);
            }
        }
        else
        {
            int region_index = at->region_index_at(p2);
            if(region_index < 0 || region_index >= int(region_voxels[id].size()))
                return;
            region_voxels[id][uint32_t(region_index)].push_back(tipl::vector<3,short>(index.begin()));
        }
    },std::thread::hardware_concurrency());

    for(size_t i = 0;i < at->get_list().size();++i)
    {
        names.push_back(at->get_list()[i]);
        std::vector<tipl::vector<3,short> > region_points(std::move(region_voxels[0][i]));
        // aggregating results from all threads
        for(size_t j = 1;j < region_voxels.size();++j)
            region_points.insert(region_points.end(),
                    std::make_move_iterator(region_voxels[j][i].begin()),std::make_move_iterator(region_voxels[j][i].end()));
        std::sort(region_points.begin(),region_points.end());
        points.push_back(std::move(region_points));
    }
    return true;
}

const tipl::image<3,tipl::vector<3,float> >& fib_data::get_sub2temp_mapping(void)
{
    if(s2t.empty() &&
       map_to_mni(tipl::show_prog) &&
       is_mni && template_id == matched_template_id)
    {
        s2t.resize(dim);
        tipl::adaptive_par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
                      [&](const tipl::pixel_index<3>& index)
        {
            s2t[index.index()] = index.begin();
        });
    }
    return s2t;
}

