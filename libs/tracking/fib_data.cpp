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

extern std::vector<std::string> fa_template_list;
bool odf_data::read(tipl::io::gz_mat_read& mat_reader)
{
    if(!odf_map.empty())
        return true;
    tipl::progress prog("reading odf data");
    unsigned int row,col;
    const float* fa0 = nullptr;
    tipl::shape<3> dim;
    if (!mat_reader.read("fa0",row,col,fa0) || !mat_reader.read("dimension",dim))
    {
        error_msg = "invalid FIB file format";
        return false;
    }
    std::vector<const float*> odf_buf;
    std::vector<size_t> odf_buf_count;
    size_t odf_count = 0;
    {
        while(mat_reader.get_col_row((std::string("odf")+std::to_string(odf_buf_count.size())).c_str(),row,col))
        {
            odf_buf_count.push_back(col);
            odf_count += col;
        }

        odf_buf.resize(odf_buf_count.size());
        for(size_t i = 0;prog(i,odf_buf_count.size());++i)
        {
            if(!mat_reader.read((std::string("odf")+std::to_string(i)).c_str(),row,col,odf_buf[i]))
            {
                error_msg = "failed to read ODF data";
                return false;
            }
        }
        if(prog.aborted())
            return false;
    }
    if (odf_buf.empty())
    {
        error_msg = "no ODF data found";
        return false;
    }

    tipl::out() << "odf count: " << odf_count << std::endl;

    size_t mask_count = 0;
    {
        for(size_t i = 0;i < dim.size();++i)
            if(fa0[i] != 0.0f)
                ++mask_count;
        tipl::out() << "mask count: " << mask_count << std::endl;
        if(odf_count < mask_count)
        {
            error_msg = "incomplete ODF data";
            return false;
        }
    }

    size_t voxel_index = 0;
    odf_count = 0; // count ODF again and now ignoring 0 odf to see if it matches.
    odf_map.resize(dim);
    for(size_t i = 0;prog(i,odf_buf.size());++i)
    {
        // row: half_odf_size
        auto odf_ptr = odf_buf[i];
        for(size_t j = 0;j < odf_buf_count[i];++j,odf_ptr += row)
        {
            bool is_odf_zero = true;
            for(size_t k = 0;k < row;++k)
                if(odf_ptr[k] != 0.0f)
                {
                    is_odf_zero = false;
                    break;
                }
            if(is_odf_zero)
                continue;
            ++odf_count;
            for(;voxel_index < odf_map.size();++voxel_index)
                if(fa0[voxel_index] != 0.0f)
                    break;
            if(voxel_index >= odf_map.size())
                break;
            odf_map[voxel_index] = odf_ptr;
            ++voxel_index;
        }
    }
    if(prog.aborted())
        return false;
    tipl::out() << "odf count (excluding 0): " << odf_count << std::endl;
    if(odf_count != mask_count)
    {
        error_msg = "ODF count does not match the mask";
        return false;
    }
    return true;
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
        auto prior_show_prog = tipl::show_prog;
        tipl::show_prog = false;
        if (!mat_reader->read(image_index,row,col,buf))
        {
            tipl::out() << "ERROR: reading " << name << std::endl;
            dummy.resize(image_data.shape());
            image_data = dummy.alias();
        }
        else
        {
            tipl::out() << name << " loaded" << std::endl;
            mat_reader->in->flush();
            image_data = tipl::make_image(buf,image_data.shape());
        }
        tipl::show_prog = prior_show_prog;
        image_ready = true;
        if(max_value == 0.0f)
            get_minmax();
    }
    return image_data;
}

void item::get_image_in_dwi(tipl::image<3>& I)
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

bool fiber_directions::add_data(tipl::io::gz_mat_read& mat_reader)
{
    tipl::progress prog("loading image volumes");
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
    mat_reader.read("dimension",dim);
    for (unsigned int index = 0;prog(index,mat_reader.size());++index)
    {
        std::string matrix_name = mat_reader.name(index);
        size_t total_size = mat_reader[index].get_cols()*mat_reader[index].get_rows();
        if(total_size != dim.size() && total_size != dim.size()*3)
            continue;
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
        if(matrix_name.find("subjects") == 0) // database
            continue;
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
            fa_otsu = tipl::segmentation::otsu_threshold(tipl::make_image(fa[0],dim));
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
    if(prog.aborted())
        return 0;
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
    }
    return num_fiber;
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
    {
        dt_fa_data = fib->dir.dt_fa_data;
        dt_threshold_name = fib->dir.dt_threshold_name;
    }
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
void prepare_idx(const char* file_name,std::shared_ptr<tipl::io::gz_istream> in);
void save_idx(const char* file_name,std::shared_ptr<tipl::io::gz_istream> in);
bool fib_data::load_from_file(const char* file_name)
{
    tipl::progress prog("open FIB file ",std::filesystem::path(file_name).filename().string().c_str());
    tipl::image<3> I;
    tipl::io::gz_nifti header;
    fib_file_name = file_name;
    if((QFileInfo(file_name).fileName().endsWith(".nii") ||
        QFileInfo(file_name).fileName().endsWith(".nii.gz")) &&
        header.load_from_file(file_name))
    {
        if(header.dim(4) == 3)
        {
            tipl::image<3> x,y,z;
            header.get_voxel_size(vs);
            header >> x;
            header >> y;
            header >> z;
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
                header.get_voxel_size(vs);
                header >> x;
                header >> y;
                header >> z;
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
            tipl::out() << "NIFTI file loaded" << std::endl;
            return true;
        }
        else
        {
            header.toLPS(I);
            header.get_voxel_size(vs);
            header.get_image_transformation(trans_to_mni);
            is_mni = QFileInfo(file_name).fileName().toLower().contains("mni");
            if(is_mni)
                tipl::out() << "The file name contains 'mni'. The image is used as MNI-space image." << std::endl;
            else
                tipl::out() << "The image is used as subject-space image" << std::endl;
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
        view_item[0].max_value = 0.0;// this allows calculating the min and max contrast
        trackable = false;
        tipl::out() << "image file loaded: " << I.shape() << std::endl;
        return true;
    }
    if(!QFileInfo(file_name).exists())
    {
        error_msg = "file does not exist";
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
    tipl::out() << "FIB file loaded" << std::endl;
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
        for(size_t i = 0;i < dim.size();++i,++index)
            buf[index] = dir.get_fib(i,j)[k];

        return tipl::io::gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni,is_mni);
    }
    if(index_name.length() == 4 && index_name.substr(0,3) == "dir" && index_name[3]-'0' >= 0 && index_name[3]-'0' < int(dir.num_fiber))
    {
        unsigned char dir_index = uint8_t(index_name[3]-'0');
        tipl::image<4,float> buf(tipl::shape<4>(dim[0],dim[1],dim[2],3));
        for(unsigned int j = 0,ptr = 0;j < 3;++j)
        for(size_t index = 0;index < dim.size();++index,++ptr)
            if(dir.fa[dir_index][index] > 0.0f)
                buf[ptr] = dir.get_fib(index,dir_index)[j];
        return tipl::io::gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni,is_mni);
    }
    if(index_name == "odfs")
    {
        tipl::image<4,float> buf(tipl::shape<4>(
                                 dim.width(),
                                 dim.height(),
                                 dim.depth(),
                                 dir.half_odf_size));
        odf_data odf;
        if(!odf.read(mat_reader))
        {
            error_msg = odf.error_msg;
            return false;
        }
        for(size_t pos = 0;pos < dim.size();++pos)
        {
            auto* ptr = odf.get_odf_data(pos);
            if(ptr!= nullptr)
                std::copy(ptr,ptr+dir.half_odf_size,buf.begin()+int64_t(pos)*dir.half_odf_size);

        }
        return tipl::io::gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni,is_mni);
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
        return tipl::io::gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni,is_mni);
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
            tipl::resample<tipl::interpolation::cubic>(buf,new_buf,view_item[index].iT);
            new_buf.swap(buf);
        }
        return tipl::io::gz_nifti::save_to_file(file_name.c_str(),buf,vs,trans_to_mni,is_mni);
    }
}
bool is_human_size(tipl::shape<3> dim,tipl::vector<3> vs)
{
    return dim[2] > 5 && dim[0]*vs[0] > 100 && dim[1]*vs[1] > 130;
}
bool fib_data::load_from_mat(void)
{
    tipl::out() << "loading fiber and image data" << std::endl;
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
        is_mni = true;
    if(!dir.add_data(mat_reader))
    {
        error_msg = dir.error_msg;
        return false;
    }
    tipl::out() << "initiating data" << std::endl;
    /*
    {
        tipl::out() << "reading original mat file" << std::endl;
        if(!high_reso->load_from_mat())
        {
            error_msg = high_reso->error_msg;
            return false;
        }
    }
    */

    view_item.push_back(item(dir.fa.size() == 1 ? "fa":"qa",dir.fa[0],dim));
    for(unsigned int index = 1;index < dir.index_name.size();++index)
        view_item.push_back(item(dir.index_name[index],dir.index_data[index][0],dim));
    view_item.push_back(item("color",dir.fa[0],dim));

    for (unsigned int index = 0;index < mat_reader.size();++index)
    {
        std::string matrix_name = mat_reader.name(index);
        if (matrix_name == "image" || matrix_name.find("subjects") == 0)
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

    if(!db.read_db(this))
    {
        error_msg = db.error_msg;
        return false;
    }

    if(is_mni)
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
        matched_template_id = 0;
        for(size_t index = 0;index < fa_template_list.size();++index)
            if(QString(fib_file_name.c_str()).contains(QFileInfo(fa_template_list[index].c_str()).baseName(),Qt::CaseInsensitive))
            {
                matched_template_id = index;
                tipl::out() << "matched template (by file name): " <<
                                   QFileInfo(fa_template_list[index].c_str()).baseName().toStdString() << std::endl;
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

        tipl::out() << "No matched template, use default: " << QFileInfo(fa_template_list[matched_template_id].c_str()).baseName().toStdString() << std::endl;
        set_template_id(matched_template_id);
        return true;
    }

    {
        if(trans_to_mni == tipl::identity_matrix())
            initial_LPS_nifti_srow(trans_to_mni,dim,vs);
    }

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
bool img_command_float32_std(tipl::image<3>& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             const std::string& cmd,const std::string& param1,std::string& error_msg);
bool modify_fib(tipl::io::gz_mat_read& mat_reader,
                const std::string& cmd,
                const std::string& param)
{
    if(cmd == "save")
    {
        tipl::io::gz_mat_write matfile(param.c_str());
        if(!matfile)
        {
            mat_reader.error_msg = "cannot save file to ";
            mat_reader.error_msg += param;
            return false;
        }
        tipl::progress prog("saving");
        for(unsigned int index = 0;prog(index,mat_reader.size());++index)
            if(!matfile.write(mat_reader[index]))
            {
                mat_reader.error_msg = "failed to write buffer to file ";
                mat_reader.error_msg += param;
                return false;
            }
        return true;
    }
    if(cmd == "remove")
    {
        auto row = std::stoi(param);
        if(row >= mat_reader.size())
        {
            mat_reader.error_msg = "invalid row to remove";
            return false;
        }
        mat_reader.remove(row);
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
    bool is_mni = false;
    if(!mat_reader.read("dimension",dim) || !mat_reader.read("voxel_size",vs))
    {
        mat_reader.error_msg = "not a valid fib file";
        return false;
    }
    if(mat_reader.has("trans"))
    {
        mat_reader.read("trans",trans);
        is_mni = true;
    }

    tipl::progress prog(cmd.c_str());
    size_t p = 0;
    bool failed = false;
    tipl::par_for(mat_reader.size(),[&](unsigned int i)
    {
        if(!prog(p++,mat_reader.size()) || failed)
            return;
        auto& mat = mat_reader[i];
        auto new_vs = vs;
        auto new_trans = trans;
        if(size_t(mat.get_cols())*size_t(mat.get_rows()) == 3*dim.size())
        {
            for(size_t d = 0;d < 3;++d)
            {
                tipl::image<3> new_image(dim);
                auto ptr = mat.get_data<float>()+d;
                for(size_t j = 0;j < dim.size();++j,ptr += 3)
                    new_image[j] = *ptr;
                if(!img_command_float32_std(new_image,new_vs,new_trans,is_mni,cmd,param,mat_reader.error_msg))
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
        if(size_t(mat.get_cols())*size_t(mat.get_rows()) == dim.size()) // image volumes, including fa, and fiber index
        {
            if(mat.is_type<short>() && (cmd == "normalize" || cmd.find("filter") != std::string::npos || cmd.find("_value") != std::string::npos))
                return;
            tipl::image<3> new_image;
            if(mat.is_type<short>()) // index0,index1
                new_image = tipl::make_image(mat.get_data<short>(),dim);
            else
                new_image = tipl::make_image(mat.get_data<float>(),dim);

            if(!img_command_float32_std(new_image,new_vs,new_trans,is_mni,cmd,param,mat_reader.error_msg))
            {
                mat_reader.error_msg = "cannot perform ";
                mat_reader.error_msg += cmd;
                failed = true;
                return;
            }
            mat.resize(tipl::vector<2,unsigned int>(new_image.width()*new_image.height(),new_image.depth()));
            if(mat.is_type<short>()) // index0,index1
                std::copy(new_image.begin(),new_image.end(),mat.get_data<short>());
            else
                std::copy(new_image.begin(),new_image.end(),mat.get_data<float>());

            if(mat.get_name() == "fa0")
            {
                std::copy(new_image.shape().begin(),new_image.shape().end(),mat_reader.get_data<unsigned int>("dimension"));
                std::copy(new_vs.begin(),new_vs.end(),mat_reader.get_data<float>("voxel_size"));
                if(is_mni)
                    std::copy(new_trans.begin(),new_trans.end(),mat_reader.get_data<float>("trans"));
            }
        }

    });
    if(failed)
        return false;
    return !prog.aborted();
}

bool fib_data::resample_to(float resolution)
{
    if(!modify_fib(mat_reader,"regrid",std::to_string(resolution)))
    {
        error_msg = mat_reader.error_msg;
        return false;
    }
    mat_reader.read("dimension",dim);
    mat_reader.read("voxel_size",vs);
    mat_reader.read("trans",trans_to_mni);
    for(auto& item : view_item)
        item.set_image(tipl::make_image(item.get_image().begin(),dim));
    return true;
}
size_t match_volume(float volume);
void fib_data::match_template(void)
{
    if(is_human_size(dim,vs))
    {
        tipl::out() << "default template set to young adult";
        set_template_id(0);
    }
    else
    {
        tipl::out() << "image volume smaller than human young adult. try matching a template...";
        set_template_id(match_volume(std::count_if(dir.fa[0],dir.fa[0]+dim.size(),[](float v){return v > 0.0f;})*2.0f*vs[0]*vs[1]*vs[2]));
        tipl::out() << "default template: " << template_id;
    }
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
    for(unsigned int index_num = 0;index_num < view_item.size();++index_num)
        if(view_item[index_num].name.find(index_name) != std::string::npos)
            return index_num;
    return view_item.size();
}
void fib_data::get_index_list(std::vector<std::string>& index_list) const
{
    for (size_t index = 0; index < view_item.size(); ++index)
        if(view_item[index].name != "color")
            index_list.push_back(view_item[index].name);
}

bool fib_data::set_dt_index(const std::pair<int,int>& pair,size_t type)
{
    tipl::image<3> I(dim),J(dim);
    std::string name;
    int i = pair.first;
    int j = pair.second;
    if(i >= 0 && i < view_item.size())
    {
        if(view_item[i].registering)
        {
            error_msg = "Registration undergoing. Please wait until registration complete.";
            return false;
        }
        name += view_item[i].name;
        view_item[i].get_image_in_dwi(I);
    }

    if(j >= 0 && j < view_item.size())
    {
        if(view_item[j].registering)
        {
            error_msg = "Registration undergoing. Please wait until registration complete.";
            return false;
        }
        name += "-";
        name += view_item[j].name;
        view_item[j].get_image_in_dwi(J);
    }
    if(name.empty())
    {
        dir.dt_fa.clear();
        dir.dt_threshold_name.clear();
        return true;
    }

    std::shared_ptr<tipl::image<3> > new_metrics(new tipl::image<3>(dim));
    auto& K = (*new_metrics);
    switch(type)
    {
        case 0: // (m1-m2)÷m1
            for(size_t k = 0;k < I.size();++k)
                if(dir.fa[0][k] > 0.0f && I[k] > 0.0f && J[k] > 0.0f)
                    K[k] = 1.0f-J[k]/I[k];
        break;
        case 1: // (m1-m2)÷m2
            for(size_t k = 0;k < I.size();++k)
                if(dir.fa[0][k] > 0.0f && I[k] > 0.0f && J[k] > 0.0f)
                    K[k] = I[k]/J[k]-1.0f;
        break;
        case 2: // m1-m2
            for(size_t k = 0;k < I.size();++k)
                if(dir.fa[0][k] > 0.0f && I[k] > 0.0f && J[k] > 0.0f)
                    K[k] = I[k]-J[k];
        case 3: // m1/max(m1)
            {
                float max_v = tipl::max_value(I);
                if(max_v > 0.0f)
                    for(size_t k = 0;k < I.size();++k)
                        K[k] = I[k]/max_v;
            }
        break;
    }

    dir.dt_fa_data = new_metrics;
    dir.dt_fa = std::vector<const float*>(size_t(dir.num_fiber),(const float*)&K[0]);
    dir.dt_threshold_name = name;
    return true;
}


void fib_data::get_slice(unsigned int view_index,
               unsigned char d_index,unsigned int pos,
               tipl::color_image& show_image)
{
    if(view_item[view_index].name == "color")
    {
        view_item[view_index].v2c.convert(tipl::volume2slice(view_item[0].get_image(),d_index,pos),show_image);

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
            auto d = dir.get_fib(buf[index],0);
            show_image[index].r = std::abs(float(show_image[index].r)*d[0]);
            show_image[index].g = std::abs(float(show_image[index].g)*d[1]);
            show_image[index].b = std::abs(float(show_image[index].b)*d[2]);
        }
    }
    else
    {
        view_item[view_index].v2c.convert(tipl::volume2slice(view_item[view_index].get_image(),d_index,pos),show_image);
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
        {
            atlas_list.push_back(std::make_shared<atlas>());
            atlas_list.back()->name = QFileInfo(atlas_file_name_list[template_id][i].c_str()).baseName().toStdString();
            atlas_list.back()->filename = atlas_file_name_list[template_id][i];
            atlas_list.back()->template_to_mni = trans_to_mni;
        }
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
        error_msg = "cannot load ";
        error_msg += fa_template_list[template_id];
        return false;
    }
    read.toLPS(template_I);
    read.get_voxel_size(template_vs);
    read.get_image_transformation(template_to_mni);
    float ratio = float(template_I.width()*template_vs[0])/float(dim[0]*vs[0]);
    if(ratio < 0.25f || ratio > 8.0f)
    {
        error_msg = "image resolution mismatch: ratio=";
        error_msg += std::to_string(ratio);
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
    tipl::par_for(tracts.size(),[&](size_t i)
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
bool fib_data::load_track_atlas()
{
    tipl::progress prog("loading tractography atlas");
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
        if(!map_to_mni())
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
        tipl::par_for(tract_data.size(),[&](size_t i)
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
    if(!load_track_atlas())
        return false;
    labels.resize(trk->get_tracts().size());
    tipl::par_for(trk->get_tracts().size(),[&](size_t i)
    {
        if(trk->get_tracts()[i].empty())
            return;
        labels[i] = find_nearest_contain(&(trk->get_tracts()[i][0]),uint32_t(trk->get_tracts()[i].size()),track_atlas->get_tracts(),track_atlas->tract_cluster);
    });

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
    if(!load_track_atlas())
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

bool fib_data::recognize_and_sort(std::shared_ptr<TractModel>& trk,std::multimap<float,std::string,std::greater<float> >& result)
{
    if(!load_track_atlas())
        return false;
    std::vector<unsigned int> labels,count;
    if(!recognize(trk,labels,count))
        return false;
    auto sum = std::accumulate(count.begin(),count.end(),0);
    result.clear();
    for(size_t i = 0;i < count.size();++i)
        if(count[i])
            result.insert(std::make_pair(float(count[i])/float(sum),tractography_name_list[i]));
    return true;
}
void fib_data::recognize_report(std::shared_ptr<TractModel>& trk,std::string& report)
{
    std::multimap<float,std::string,std::greater<float> > result;
    if(!recognize_and_sort(trk,result)) // true: connectometry may only show part of pathways. enable containing condition
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
    output_file_name += ".map.gz";

    if(std::filesystem::exists(output_file_name))
    {
        tipl::progress prog("checking existing mapping file");
        if(load_mapping(output_file_name.c_str(),false/* not external*/))
            return true;
        tipl::out() << "new mapping file needed: " << error_msg;
        error_msg.clear();
    }

    tipl::progress prog_("running normalization");
    prog = 0;
    bool terminated = false;
    auto lambda = [this,output_file_name,&terminated]()
    {
        dual_reg reg;
        reg.It = template_I;
        reg.It2 = template_I2;
        reg.Itvs = template_vs;
        reg.ItR = template_to_mni;

        reg.I = tipl::image<3>(dir.fa[0],dim);
        reg.Ivs = vs;
        reg.IR = trans_to_mni;

        // not FIB file, use t1w as template
        if(dir.index_name[0] == "image")
        {
            if(!tipl::io::gz_nifti::load_to_space(t1w_template_file_name.c_str(),reg.It,template_to_mni))
            {
                prog = 6;
                error_msg = "cannot perform normalization";
                terminated = true;
                return;
            }
            tipl::out() << "using structure image for normalization" << std::endl;
            reg.It2.clear();
        }

        {
            size_t iso_index = get_name_index("iso");
            if(view_item.size() != iso_index)
                reg.I2 = view_item[iso_index].get_image();
        }

        prog = 1;
        if(has_manual_atlas)
            reg.arg = manual_template_T;
        else
            reg.linear_reg(tipl::reg::affine,0/*mutual info*/,terminated);

        if(terminated)
        {
            prog = 6;
            return;
        }
        prog = 3;
        if(dir.index_name[0] == "image")
        {
            tipl::out() << "matching t1w t2w contrast" << std::endl;
            if(tipl::io::gz_nifti::load_to_space(t2w_template_file_name.c_str(),reg.It2,template_to_mni))
                reg.matching_contrast();
            else
                reg.It2.clear();
        }
        prog = 4;
        if(reg.nonlinear_reg(terminated,true) < 0.3f)
        {
            error_msg = "cannot perform normalization";
            terminated = true;
        }
        if(terminated)
        {
            prog = 6;
            return;
        }
        prog = 5;
        if(!reg.save_warping(output_file_name.c_str()))
            tipl::out() << reg.error_msg;
        s2t.swap(reg.from2to);
        t2s.swap(reg.to2from);
        prog = 6;
    };

    if(background)
    {
        std::thread t(lambda);
        while(prog_(prog,6))
            std::this_thread::yield();
        if(prog_.aborted())
        {
            error_msg = "aborted.";
            terminated = true;
        }
        t.join();
        return !terminated;
    }

    tipl::out() << "Subject normalization to MNI space." << std::endl;
    lambda();
    return !terminated;
}
bool fib_data::load_mapping(const char* file_name,bool external)
{
    if(tipl::ends_with(file_name,".map.gz"))
    {
        tipl::io::gz_mat_read in;
        if(!in.load_from_file(file_name))
        {
            error_msg = in.error_msg;
            return false;
        }
        if(!in.has("from2to") || !in.has("to2from"))
        {
            error_msg = "invalid mapping file format";
            return false;
        }
        if(!external) // additional check for internal mapping
        {
            // check 1. mapping files was created later than the FIB file
            if(QFileInfo(file_name).lastModified() < QFileInfo(fib_file_name.c_str()).lastModified())
                return false;
            //       2. the recon steps are the same
            if(in.read<std::string>("steps") != steps)
                return false;
            //       3. check method version (new after Aug 2023)
            constexpr int method_ver = 202308; // 999999 is for external loading mapping
            if(!in.has("method_ver") || std::stoi(in.read<std::string>("method_ver")) < method_ver)
                return false;
        }

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
                    tipl::out() << "loading mapping fields from " << file_name << std::endl;
                    t2s.clear();
                    t2s.resize(template_I.shape());
                    s2t.clear();
                    s2t.resize(dim);
                    std::copy(t2s_ptr,t2s_ptr+t2s_col*t2s_row,&t2s[0][0]);
                    std::copy(s2t_ptr,s2t_ptr+s2t_col*s2t_row,&s2t[0][0]);
                    prog = 6;
                    return true;
                }
                else
                    error_msg = "image size does not match";
            }
            else
                error_msg = "failed to read mapping matrix";
        }
        return false;
    }

    tipl::io::gz_nifti nii;
    tipl::out() << "loading " << file_name;
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
    tipl::par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
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
    error_msg += " is not one of the built-in atlases of ";
    error_msg += name_list;
    return std::shared_ptr<atlas>();
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

    std::vector<std::vector<tipl::vector<3,short> > > buf(std::thread::hardware_concurrency());

    // trigger atlas loading to avoid crash in multi thread
    if(!at->load_from_file())
    {
        error_msg = "cannot read atlas file ";
        error_msg += at->filename;
        return false;
    }
    if(new_geo == dim && to_diffusion_space == tipl::identity_matrix())
    {
        tipl::par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
            [&](const tipl::pixel_index<3>& index,size_t id)
        {
            if (at->is_labeled_as(s2t[index.index()], roi_index))
                buf[id].push_back(tipl::vector<3,short>(index.begin()));
        });
    }
    else
    {
        tipl::par_for(tipl::begin_index(new_geo),tipl::end_index(new_geo),
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
                                 std::vector<std::string>& labels)
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


    std::vector<std::vector<std::vector<tipl::vector<3,short> > > > region_voxels(std::thread::hardware_concurrency());
    for(auto& region : region_voxels)
    {
        region.clear();
        region.resize(at->get_list().size());
    }

    bool need_trans = (new_geo != dim || to_diffusion_space != tipl::identity_matrix());
    auto shape = need_trans ? new_geo : dim;
    tipl::par_for(tipl::begin_index(shape),tipl::end_index(shape),
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
    });

    // aggregating results from all threads
    labels.resize(at->get_list().size());
    tipl::par_for(at->get_list().size(),[&](size_t i)
    {
        labels[i] = at->get_list()[i];
        for(size_t j = 1;j < region_voxels.size();++j)
            region_voxels[0][i].insert(region_voxels[0][i].end(),
                    std::make_move_iterator(region_voxels[j][i].begin()),std::make_move_iterator(region_voxels[j][i].end()));
        std::sort(region_voxels[0][i].begin(),region_voxels[0][i].end());
    });
    region_voxels[0].swap(points);
    return true;
}

const tipl::image<3,tipl::vector<3,float> >& fib_data::get_sub2temp_mapping(void)
{
    if(s2t.empty() &&
       map_to_mni(false) &&
       is_mni && template_id == matched_template_id)
    {
        s2t.resize(dim);
        tipl::par_for(tipl::begin_index(s2t.shape()),tipl::end_index(s2t.shape()),
                      [&](const tipl::pixel_index<3>& index)
        {
            s2t[index.index()] = index.begin();
        });
    }
    return s2t;
}

