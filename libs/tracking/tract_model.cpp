//---------------------------------------------------------------------------
#include <QString>
#include <QFileInfo>
#include <QImage>
#include <fstream>
#include <sstream>
#include <array>
#include <iterator>
#include <tuple>
#include <unordered_set>
#include <map>
#include <cmath>
#include "roi.hpp"
#include "tract_model.hpp"
#include "fib_data.hpp"
#include "mapping/atlas.hpp"
#include "tract_cluster.hpp"
#include "../../tracking/region/Regions.h"
#include "tracking_method.hpp"
#include "reg.hpp"
#include <filesystem>

void prepare_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
void save_idx(const std::string& file_name,std::shared_ptr<tipl::io::gz_istream> in);
const tipl::rgb default_tract_color(255,160,60);
void smoothed_tracks(const std::vector<float>& track,std::vector<float>& smoothed)
{
    smoothed.clear();
    smoothed.resize(track.size());
    float w[5] = {1.0,2.0,4.0,2.0,1.0};
    int shift[5] = {-6, -3, 0, 3, 6};
    for(int index = 0;index < int(track.size());++index)
    {
        float sum_w = 0.0;
        float sum = 0.0;
        for(int i = 0;i < 5;++i)
        {
            int cur_index = index + shift[i];
            if(cur_index < 0 || cur_index >= int(track.size()))
                continue;
            sum += w[i]*track[size_t(cur_index)];
            sum_w += w[i];
        }
        if(sum_w != 0.0f)
            smoothed[size_t(index)] = sum/sum_w;
    }
}

/* 1. spatial resolution of 1/32 voxel spacing.
 * 2. step size between (-127/32 to 128/32) voxels for x,y,z, direction
 */
class TinyTrack{

    union tract_header{
        char buf[16];
        struct {
        uint32_t count; // number of coordinates
        int32_t x; // first coordinate (x*32,y*32,z*32)
        int32_t y;
        int32_t z;
        } h;
    };

    public:
    static bool save_to_file(const char* file_name,
                             tipl::shape<3> geo,
                             tipl::vector<3> vs,
                             const tipl::matrix<4,4>& trans_to_mni,
                             const std::vector<std::vector<float> >& tract_data,
                             const std::vector<uint16_t>& cluster,
                             const std::string& report,
                             const std::string& parameter_id,
                             const std::vector<unsigned int>& color)
    {
        tipl::progress prog0("saving ",std::filesystem::path(file_name).filename().u8string().c_str());
        tipl::io::gz_mat_write out(file_name);
        if (!out)
            return false;
        out.write("dimension",geo);
        out.write("voxel_size",vs);
        out.write("trans_to_mni",trans_to_mni);
        out.write("report",report);
        out.write("parameter_id",parameter_id);
        out.write("color",color);
        out.write("cluster",cluster);
        std::vector<std::vector<int32_t> > track32(tract_data.size());
        std::vector<size_t> buf_size(track32.size());

        {
            tipl::progress prog1("compressing trajectories");
            size_t p = 0;
            tipl::adaptive_par_for(track32.size(),[&](size_t i)
            {
                prog1(p++,tract_data.size());
                auto& t32 = track32[i];
                t32.resize(tract_data[i].size());
                // all coordinates multiply by 32 and convert to integer
                for(size_t j = 0;j < t32.size();j++)
                    t32[j] = int(std::round(std::ldexp(tract_data[i][j],5)));
                // Calculate coordinate displacement, skipping the first coordinate
                for(size_t j = t32.size()-1;j >= 3;j--)
                    t32[j] -= t32[j-3];

                // check if there is a leap, skipping the first coordinate
                bool has_leap = false;
                for(size_t j = 3;j < t32.size();j++)
                    if(t32[j] < -127 || t32[j] > 127)
                    {
                        has_leap = true;
                        break;
                    }
                // if there is a leap, interpolate it
                if(has_leap)
                {
                    std::vector<int32_t> new_t32;
                    new_t32.reserve(t32.size());
                    for(size_t j = 0;j < t32.size();j += 3)
                    {
                        int32_t x = t32[j];
                        int32_t y = t32[j+1];
                        int32_t z = t32[j+2];
                        bool interpolated = false;
                        while(j && (x < -127 || x > 127 || y < -127 || y > 127 || z < -127 || z > 127))
                        {
                            x /= 2;
                            y /= 2;
                            z /= 2;
                            interpolated = true;
                        }
                        if(interpolated)
                        {
                            t32[j] -= x;
                            t32[j+1] -= y;
                            t32[j+2] -= z;
                            j -= 3;
                        }
                        new_t32.push_back(x);
                        new_t32.push_back(y);
                        new_t32.push_back(z);
                    }
                    new_t32.swap(t32);
                }
                buf_size[i] = sizeof(tract_header)+t32.size()-3;
            });
            if(prog1.aborted())
                return false;
        }
        {
            for(size_t block = 0,cur_track_block = 0;prog0(cur_track_block,track32.size());++block)
            {
                // record write position for each track
                size_t total_size = 0;
                std::vector<size_t> pos;
                for(size_t i = cur_track_block;i < track32.size();++i)
                {
                    pos.push_back(total_size);
                    total_size += buf_size[i];
                    if(total_size > 134217728) // 128 mb
                        break;
                }

                std::vector<char> out_buf(total_size);
                tipl::adaptive_par_for(pos.size(),[&](size_t i)
                {
                    auto& t32 = track32[cur_track_block+i];
                    auto out = &out_buf[pos[i]];
                    tract_header hr;
                    hr.h.count = uint32_t(t32.size());
                    hr.h.x = t32[0];
                    hr.h.y = t32[1];
                    hr.h.z = t32[2];
                    std::copy(hr.buf,hr.buf+16,out);
                    out += sizeof(tract_header)-3;
                    for(size_t j = 3;j < t32.size();j++)
                        out[j] = char(t32[j]);
                });

                if(block == 0)
                    out.write("track",&out_buf[0],total_size,1);
                else
                    out.write((std::string("track")+std::to_string(block)).c_str(),&out_buf[0],total_size,1);
                cur_track_block += pos.size();
            }
        }
        return !prog0.aborted();
    }
    static bool load_from_file(const char* file_name,
                               std::vector<std::vector<float> >& tract_data,
                               std::vector<uint16_t>& tract_cluster,
                               tipl::shape<3>& geo,tipl::vector<3>& vs,
                               tipl::matrix<4,4>& trans_to_mni,
                               std::string& report,std::string& parameter_id,
                               std::vector<unsigned int>& color)
    {
        tipl::progress prog("opening ",std::filesystem::path(file_name).filename().c_str());
        tipl::io::gz_mat_read in;
        prepare_idx(file_name,in.in);
        if (!in.load_from_file(file_name))
            return false;
        in.read("dimension",geo);
        in.read("voxel_size",vs);
        in.read("trans_to_mni",trans_to_mni);
        in.read("report",report);
        in.read("parameter_id",parameter_id);
        color = in.read_as_vector<unsigned int>("color");
        tract_cluster = in.read_as_vector<uint16_t>("cluster");
        unsigned int row,col;
        for(unsigned int block = 0;1;block++)
        {
            const char* track_buf = nullptr;
            if(block == 0)
            {
                if(!in.read("track",row,col,track_buf))
                    return false;
            }
            else
            {
                auto name = std::string("track")+std::to_string(block);
                if(!in.has(name.c_str()))
                    break;
                if(!in.read(name.c_str(),row,col,track_buf))
                    return false;
            }
            size_t buf_size = size_t(row)*size_t(col);
            std::vector<size_t> pos;
            for(size_t i = 0;i < buf_size;)
            {
                pos.push_back(i);
                i += *reinterpret_cast<const uint32_t*>(track_buf+i);
                i += sizeof(tract_header)-3;
            }
            size_t add_tract_index = tract_data.size();
            tract_data.resize(add_tract_index+pos.size());
            tipl::adaptive_par_for(pos.size(),[&](size_t i)
            {
                auto& cur_tract = tract_data[i+add_tract_index];
                tract_header hr;
                std::copy(&track_buf[pos[i]],&track_buf[pos[i]]+16,hr.buf);
                if(hr.h.count > buf_size)
                    return;
                cur_tract.resize(hr.h.count);
                cur_tract[0] = hr.h.x;
                cur_tract[1] = hr.h.y;
                cur_tract[2] = hr.h.z;
                size_t shift = pos[i]+sizeof(tract_header)-3;
                for(size_t j = 3;j < cur_tract.size();++j)
                    cur_tract[j] = (cur_tract[j-3] + track_buf[shift+j]);
                for(size_t j = 0;j < cur_tract.size();++j)
                    cur_tract[j] = std::ldexp(cur_tract[j],-5);
            });
        }

        save_idx(file_name,in.in);
        return true;
    }
};

struct TrackVis
{
    char id_string[6] = {'T','R','A','C','K',0};//ID string for track file. The first 5 characters must be "TRACK".
    short int dim[3];//Dimension of the image volume.
    float voxel_size[3];//Voxel size of the image volume.
    float origin[3] = {};//Origin of the image volume. This field is not yet being used by TrackVis. That means the origin is always (0, 0, 0).
    short int n_scalars = 0;//Number of scalars saved at each track point (besides x, y and z coordinates).
    char scalar_name[10][20] = {};//Name of each scalar. Can not be longer than 20 characters each. Can only store up to 10 names.
    short int n_properties = 0;//Number of properties saved at each track.
    char property_name[10][20] = {};//Name of each property. Can not be longer than 20 characters each. Can only store up to 10 names.
    float vox_to_ras[4][4] = {};
    char reserved[444] = {0};//Reserved space for future version.
    char voxel_order[4] = {'L','P','S',0};//Storing order of the original image data. Explained here.
    char pad2[4] = {};//Paddings.
    float image_orientation_patient[6] = {1,0,0,0,1,0};//Image orientation of the original image. As defined in the DICOM header.
    char pad1[2] = {};//Paddings.
    unsigned char invert[6] = {};//Inversion/rotation flags used to generate this track file. For internal use only.
    int n_count = 0;//Number of tract stored in this track file. 0 means the number was NOT stored.
    int version = 2;//Version number. Current version is 1.
    int hdr_size = 1000;//Size of the header. Used to determine byte swap. Should be 1000.
    void init(tipl::shape<3> geo_,
              tipl::vector<3> voxel_size_,
              const tipl::matrix<4,4>& trans)
    {
        std::copy(geo_.begin(),geo_.end(),dim);
        std::copy(voxel_size_.begin(),voxel_size_.end(),voxel_size);
        std::copy(trans.begin(),trans.end(),&vox_to_ras[0][0]);
        std::copy(voxel_order,voxel_order+4,pad2);
    }
    bool load_from_file(const char* file_name,
                std::vector<std::vector<float> >& loaded_tract_data,
                std::vector<unsigned int>& loaded_tract_cluster,
                tipl::shape<3>& geo,
                tipl::vector<3>& vs,
                tipl::matrix<4,4>& trans_to_mni,
                std::string& info)
    {
        tipl::progress prog("opening ",std::filesystem::path(file_name).filename().u8string().c_str());
        tipl::io::gz_istream in;
        if (!in.open(file_name))
            return false;
        in.read((char*)this,1000);
        std::copy(dim,dim+3,geo.begin());
        std::copy(voxel_size,voxel_size+3,vs.begin());
        std::copy(&vox_to_ras[0][0],&vox_to_ras[0][0]+16,trans_to_mni.begin());
        unsigned int track_number = n_count;
        info = reserved;
        if(info.find(' ') != std::string::npos)
            info.clear();
        if(!track_number) // number is not stored
            track_number = 100000000;
        for (unsigned int index = 0;!(!in) && prog(index,track_number);++index)
        {
            unsigned int n_point;
            in.read((char*)&n_point,sizeof(int));
            unsigned int index_shift = 3 + n_scalars;
            std::vector<float> tract(index_shift*n_point + n_properties);
            if(!in.read((char*)&*tract.begin(),sizeof(float)*tract.size()))
                break;

            loaded_tract_data.push_back(std::vector<float>(n_point*3));
            const float *from = &*tract.begin();
            float *to = &*loaded_tract_data.back().begin();
            for (unsigned int i = 0;i < n_point;++i,from += index_shift,to += 3)
            {
                float x = from[0]/voxel_size[0];
                float y = from[1]/voxel_size[1];
                float z = from[2]/voxel_size[2];
                if(voxel_order[1] == 'R')
                    to[0] = dim[0]-x-1;
                else
                    to[0] = x;
                if(voxel_order[1] == 'A')
                    to[1] = dim[1]-y-1;
                else
                    to[1] = y;
                to[2] = z;
            }
            if(n_properties == 1)
                loaded_tract_cluster.push_back(from[0]);
        }
        return !prog.aborted();
    }
    static bool save_to_file(const char* file_name,
                             tipl::shape<3> geo,
                             tipl::vector<3> vs,
                             const tipl::matrix<4,4>& trans_to_mni,
                             const std::vector<std::vector<float> >& tract_data,
                             const std::vector<std::vector<float> >& scalar,
                             const std::string& info,
                             unsigned int color)
    {
        tipl::progress prog("saving ",std::filesystem::path(file_name).filename().u8string().c_str());
        tipl::io::gz_ostream out;
        if (!out.open(file_name))
            return false;
        TrackVis trk;
        trk.init(geo,vs,trans_to_mni);
        trk.n_count = tract_data.size();
        *(uint32_t*)(trk.reserved+440) = color;
        if(!scalar.empty())
            trk.n_scalars = 1;
        if(info.length())
            std::copy(info.begin(),info.begin()+std::min<int>(439,info.length()),trk.reserved);
        out.write((const char*)&trk,1000);
        for (unsigned int i = 0;prog(i,tract_data.size());++i)
        {
            int n_point = tract_data[i].size()/3;
            std::vector<float> buffer(trk.n_scalars ? tract_data[i].size()+scalar[i].size() : tract_data[i].size());
            float* to = &*buffer.begin();
            for (unsigned int flag = 0,j = 0,k = 0;j < tract_data[i].size();++j,++to)
            {
                *to = tract_data[i][j]*vs[flag];
                ++flag;
                if (flag == 3)
                {
                    flag = 0;
                    if(trk.n_scalars)
                    {
                        ++to;
                        *to = scalar[i][k];
                        ++k;
                    }
                }
            }
            out.write((const char*)&n_point,sizeof(int));
            out.write((const char*)&*buffer.begin(),sizeof(float)*buffer.size());
        }
        if(prog.aborted())
            return false;
        return true;
    }
};

struct Tck{
    tipl::vector<3> vs;
    tipl::shape<3> geo;
    bool load_from_file(const char* file_name,
                        std::vector<std::vector<float> >& loaded_tract_data)
    {
        unsigned int offset = 0;
        {
            std::ifstream in(file_name);
            if(!in)
                return false;
            std::string line;
            while(std::getline(in,line))
            {
                if(line.size() <= 4)
                    continue;
                std::istringstream str(line);
                std::string s1,s2;
                str >> s1 >> s2;
                if(line.substr(0,7) == std::string("file: ."))
                {
                    str >> offset;
                    break;
                }
                std::replace(s2.begin(),s2.end(),',',' ');
                if(s1 == "dim:")
                {
                    std::istringstream str(s2);
                    str >> geo[0] >> geo[1] >> geo[2];
                }
                if(s1 == "vox:")
                {
                    std::istringstream str(s2);
                    str >> vs[0] >> vs[1] >> vs[2];
                }
            }
        }

        std::ifstream in(file_name,std::ios::binary);
        if(!in)
            return false;
        in.seekg(0,std::ios::end);
        unsigned int total_size = uint32_t(in.tellg());
        in.seekg(offset,std::ios::beg);
        std::vector<unsigned int> buf((total_size-offset)/4);
        in.read((char*)&*buf.begin(),total_size-offset-16);// 16 skip the final inf
        for(unsigned int index = 0;index < buf.size();)
        {
            unsigned int end = std::find(buf.begin()+index,buf.end(),0x7FC00000)-buf.begin(); // NaN
            if(end-index > 3)
            {
                std::vector<float> track(end-index);
                std::copy((const float*)&*buf.begin() + index,
                          (const float*)&*buf.begin() + end,
                          track.begin());
                loaded_tract_data.push_back(std::move(track));
                tipl::divide_constant(loaded_tract_data.back().begin(),loaded_tract_data.back().end(),vs[0]);
            }
            index = end+3;
        }
        return true;
    }

};



bool tt2trk(const char* tt_file,const char* trk_file)
{
    std::vector<std::vector<float> > tract_data;
    std::vector<uint16_t> cluster;
    std::string report,pid;
    tipl::vector<3> vs;
    tipl::shape<3> geo;
    tipl::matrix<4,4> trans_to_mni;
    std::vector<unsigned int> color;
    if(!TinyTrack::load_from_file(tt_file,tract_data,cluster,geo,vs,trans_to_mni,report,pid,color))
    {
        std::cout << "cannot read " << tt_file << std::endl;
        return false;
    }
    std::vector<std::vector<float> > scalar;
    return TrackVis::save_to_file(trk_file,geo,vs,trans_to_mni,tract_data,scalar,report,color.empty() ? 0 : color[0]);
}

bool trk2tt(const char* trk_file,const char* tt_file)
{
    TrackVis vis;
    std::vector<std::vector<float> > loaded_tract_data;
    std::vector<unsigned int> loaded_tract_cluster;
    std::string info;
    tipl::vector<3> vs;
    tipl::shape<3> geo;
    tipl::matrix<4,4> trans_to_mni;
    if(!vis.load_from_file(trk_file,loaded_tract_data,loaded_tract_cluster,geo,vs,trans_to_mni,info))
    {
        std::cout << "cannot read " << trk_file << std::endl;
        return false;
    }
    unsigned int color = default_tract_color;
    unsigned int new_color = *(uint32_t*)(vis.reserved+440);
    if(new_color)
        color = new_color;

    for(size_t index = 0;index < loaded_tract_data.size();++index)
        for(size_t i = 0;i < loaded_tract_data[index].size();i += 3)
        {
            loaded_tract_data[index][i] /= vs[0];
            loaded_tract_data[index][i+1] /= vs[1];
            loaded_tract_data[index][i+2] /= vs[2];
        }
    std::string p_id;
    return TinyTrack::save_to_file(tt_file,geo,vs,trans_to_mni,loaded_tract_data,
                                   std::vector<uint16_t>(loaded_tract_cluster.begin(),loaded_tract_cluster.end()),info,p_id,std::vector<unsigned int>({color}));
}
//---------------------------------------------------------------------------
void shift_track_for_tck(std::vector<std::vector<float> >& loaded_tract_data,tipl::shape<3>& geo)
{
    tipl::vector<3> min_xyz(0.0f,0.0f,0.0f),max_xyz(0.0f,0.0f,0.0f);
    tipl::adaptive_par_for(loaded_tract_data.size(),[&](size_t i)
    {
        for(unsigned int k = 0;k < 3;++k)
        for(size_t j = k;j < loaded_tract_data[i].size();j += 3)
        {
            if(loaded_tract_data[i][j] < min_xyz[k])
                min_xyz[k] = loaded_tract_data[i][j];
            if(loaded_tract_data[i][j] > max_xyz[k])
                max_xyz[k] = loaded_tract_data[i][j];
        }
    });
    for(unsigned int k = 0;k < 3;++k)
    {
        geo[k] = uint32_t(max_xyz[k]-min_xyz[k]+2);
        min_xyz[k] -= 1;
    }
    tipl::adaptive_par_for(loaded_tract_data.size(),[&](size_t i)
    {
        for(unsigned int k = 0;k < 2;++k)
        for(size_t j = k;j < loaded_tract_data[i].size();j += 3)
            loaded_tract_data[i][j] = max_xyz[k] - loaded_tract_data[i][j];
        for(size_t j = 2;j < loaded_tract_data[i].size();j += 3)
            loaded_tract_data[i][j] -= min_xyz[2];
    });
}
bool load_fib_from_tracks(const char* file_name,
                          tipl::image<3>& I,
                          tipl::vector<3>& vs,
                          tipl::matrix<4,4>& trans_to_mni)
{
    tipl::shape<3> geo;
    std::vector<std::vector<float> > loaded_tract_data;
    if(QString(file_name).endsWith("tck"))
    {
        Tck tck;
        tck.vs = tipl::vector<3>(1.0f,1.0f,1.0f);
        if(!tck.load_from_file(file_name,loaded_tract_data))
        {
            std::cout << "cannot read " << file_name << std::endl;
            return false;
        }
        shift_track_for_tck(loaded_tract_data,geo);
    }
    else
    if(QString(file_name).endsWith("trk.gz") || QString(file_name).endsWith("trk"))
    {
        TrackVis vis;
        std::vector<unsigned int> loaded_tract_cluster;
        std::string info;
        if(!vis.load_from_file(file_name,loaded_tract_data,loaded_tract_cluster,geo,vs,trans_to_mni,info))
        {
            std::cout << "cannot read " << file_name << std::endl;
            return false;
        }
        std::copy(vis.voxel_size,vis.voxel_size+3,vs.begin());
        std::copy(vis.dim,vis.dim+3,geo.begin());
    }
    else
        if(QString(file_name).endsWith("tt.gz"))
        {
            std::vector<unsigned short> loaded_tract_cluster;
            std::string report,pid;
            std::vector<unsigned int> color;
            if(!TinyTrack::load_from_file(file_name,loaded_tract_data,loaded_tract_cluster,geo,vs,trans_to_mni,report,pid,color))
            {
                std::cout << "cannot read " << file_name << std::endl;
                return false;
            }
        }
    else
        return false;
    I.clear();
    I.resize(geo);
    tipl::adaptive_par_for(loaded_tract_data.size(),[&](size_t i)
    {
        for(size_t j = 0;j < loaded_tract_data[i].size();j += 3)
        {
            int x = int(std::round(loaded_tract_data[i][j]));
            int y = int(std::round(loaded_tract_data[i][j+1]));
            int z = int(std::round(loaded_tract_data[i][j+2]));
            if(geo.is_valid(x,y,z))
                I[tipl::pixel_index<3>(x,y,z,geo).index()]++;
        }
    });
    return true;
}
//---------------------------------------------------------------------------
void TractModel::add(const TractModel& rhs)
{
    for(unsigned int index = 0;index < rhs.redo_size.size();++index)
        redo_size.push_back(std::make_pair(rhs.redo_size[index].first + uint32_t(tract_data.size()),
                                           rhs.redo_size[index].second));
    tract_data.insert(tract_data.end(),rhs.tract_data.begin(),rhs.tract_data.end());
    tract_color.insert(tract_color.end(),rhs.tract_color.begin(),rhs.tract_color.end());
    tract_tag.insert(tract_tag.end(),rhs.tract_tag.begin(),rhs.tract_tag.end());
    deleted_tract_data.insert(deleted_tract_data.end(),
                              rhs.deleted_tract_data.begin(),
                              rhs.deleted_tract_data.end());
    deleted_tract_color.insert(deleted_tract_color.end(),
                               rhs.deleted_tract_color.begin(),
                               rhs.deleted_tract_color.end());
    deleted_tract_tag.insert(deleted_tract_tag.end(),
                               rhs.deleted_tract_tag.begin(),
                               rhs.deleted_tract_tag.end());
    deleted_count.insert(deleted_count.begin(),
                         rhs.deleted_count.begin(),
                         rhs.deleted_count.end());
    is_cut.insert(is_cut.end(),rhs.is_cut.begin(),rhs.is_cut.end());
}
//---------------------------------------------------------------------------
template<>
bool dual_reg<3>::apply_warping_tt(const char* from,const char* to) const
{
    std::vector<std::vector<float> > loaded_tract_data;
    std::vector<uint16_t> cluster;
    std::vector<unsigned int> color;
    std::string report, parameter_id;

    tipl::shape<3> geo;
    tipl::vector<3> vs;
    tipl::matrix<4,4,float> trans_to_mni;
    if(!TinyTrack::load_from_file(from,loaded_tract_data,cluster,geo,vs,trans_to_mni,report,parameter_id,color))
    {
        error_msg = "Failed to read file";
        return false;
    }

    tipl::vector<3> max_pos(from2to.shape());
    max_pos -= 1.0f;
    auto T = tipl::from_space(ItR).to(IR);
    tipl::adaptive_par_for(loaded_tract_data.size(),[&](size_t i)
    {
        for(size_t j = 0;j < loaded_tract_data[i].size();j += 3)
        {
            tipl::vector<3> pos(&loaded_tract_data[i][j]);
            pos.to(T);
            pos[0] = std::min<float>(std::max<float>(pos[0],0.0f),max_pos[0]);
            pos[1] = std::min<float>(std::max<float>(pos[1],0.0f),max_pos[1]);
            pos[2] = std::min<float>(std::max<float>(pos[2],0.0f),max_pos[2]);
            tipl::vector<3> new_pos;
            tipl::estimate(from2to,pos,new_pos);
            loaded_tract_data[i][j] = new_pos[0];
            loaded_tract_data[i][j+1] = new_pos[1];
            loaded_tract_data[i][j+2] = new_pos[2];
        }
    });
    tipl::out() << "saving " << to;
    if(!TinyTrack::save_to_file(to,to2from.shape(),Itvs,ItR,loaded_tract_data,cluster,report,parameter_id,color))
    {
        error_msg = "Failed to save file";
        return false;
    }
    return true;
}
bool TractModel::load_tracts_from_file(const char* file_name_,fib_data* handle,bool tract_is_mni)
{
    std::string file_name(file_name_);
    std::vector<std::vector<float> > loaded_tract_data;
    std::vector<unsigned int> loaded_tract_cluster;
    std::vector<unsigned int> colors;
    unsigned int color = default_tract_color;
    if(file_name.find(".dec") != std::string::npos)
        color = 0x004040F0;
    if(file_name.find(".inc") != std::string::npos)
        color = 0x00F04040;

    tipl::matrix<4,4> source_trans_to_mni(trans_to_mni);

    if(QString(file_name_).endsWith("tt.gz"))
    {
        unsigned int old_color = color;
        std::vector<uint16_t> cluster;
        if(!TinyTrack::load_from_file(file_name_,loaded_tract_data,cluster,geo,vs,source_trans_to_mni,report,parameter_id,colors))
            return false;
        if(geo == handle->dim && vs == handle->vs && !tract_is_mni && source_trans_to_mni != handle->trans_to_mni)
        {
            tipl::out() << "identical dimension: overwriting tractography transformation matrix." << std::endl;
            source_trans_to_mni = handle->trans_to_mni;
        }
        std::copy(cluster.begin(),cluster.end(),std::back_inserter(loaded_tract_cluster));
        if(!colors.empty())
            color = colors[0];
    }
    if(QString(file_name_).endsWith("trk.gz") || QString(file_name_).endsWith("trk"))
    {
        TrackVis trk;
        if(!trk.load_from_file(file_name_,loaded_tract_data,loaded_tract_cluster,geo,vs,source_trans_to_mni,parameter_id))
            return false;
        if(geo == handle->dim && vs == handle->vs && !tract_is_mni && source_trans_to_mni != handle->trans_to_mni)
        {
            tipl::out() << "identical dimension: overwriting tractography transformation matrix." << std::endl;
            source_trans_to_mni = handle->trans_to_mni;
        }
        unsigned int new_color = *(uint32_t*)(trk.reserved+440);
        if(new_color)
            color = new_color;
        if(!parameter_id.empty())
        {
            report = "\nThis tractography was generated using the following parameters: ";
            TrackingParam param;
            if(param.set_code(parameter_id))
                report += param.get_report();
        }
    }

    if (QString(file_name_).endsWith(".txt"))
    {
        std::ifstream in(file_name_);
        if (!in)
            return false;
        std::string line;
        in.seekg(0,std::ios::end);
        in.seekg(0,std::ios::beg);
        while (std::getline(in,line))
        {
            loaded_tract_data.push_back(std::vector<float>());
            std::istringstream in(line);
            std::copy(std::istream_iterator<float>(in),
                      std::istream_iterator<float>(),std::back_inserter(loaded_tract_data.back()));

            if(loaded_tract_data.back().size() == 1)// cluster info
                loaded_tract_cluster.push_back(uint32_t(loaded_tract_data.back()[0]));
        }
    }

    if (QString(file_name_).endsWith(".mat"))
    {
        tipl::io::gz_mat_read in;
        if(!in.load_from_file(file_name_))
            return false;
        const float* buf = nullptr;
        const unsigned int* length = nullptr;
        const unsigned int* cluster = nullptr;
        unsigned int row,col;
        if(!in.read("tracts",row,col,buf))
            return false;
        if(!in.read("length",row,col,length))
            return false;
        loaded_tract_data.resize(col);
        in.read("cluster",row,col,cluster);
        for(unsigned int index = 0;index < loaded_tract_data.size();++index)
        {
            loaded_tract_data[index].resize(length[index]*3);
            if(cluster)
                loaded_tract_cluster.push_back(cluster[index]);
            std::copy(buf,buf + loaded_tract_data[index].size(),loaded_tract_data[index].begin());
            buf += loaded_tract_data[index].size();
        }
        in.read("trans",source_trans_to_mni);
    }

    if (QString(file_name_).endsWith("tck"))
    {
        Tck tck;
        tck.vs = vs;
        if(!tck.load_from_file(file_name_,loaded_tract_data))
            return false;
    }





    if (loaded_tract_data.empty())
        return false;
    if(loaded_tract_cluster.size() == loaded_tract_data.size())
    {
        tipl::out() << "cluster information loaded";
        loaded_tract_cluster.swap(tract_cluster);
    }
    else
        tract_cluster.clear();


    // handle trans_to_mni differences
    {
        tipl::out() << ((is_mni) ? "host space (mni): " : "host space (native): ") << std::endl;
        tipl::out() << trans_to_mni << std::endl;
        tipl::out() << ((tract_is_mni) ? "tractography space (mni): " : "tractography space (native): ") << std::endl;
        tipl::out() << source_trans_to_mni << std::endl;

        auto apply_transform = [&](const tipl::matrix<4,4>& T)
        {
            tipl::out() << "apply transform to tracts: " << std::endl;
            tipl::out() << T << std::endl;
            tipl::adaptive_par_for(loaded_tract_data.size(),[&](size_t index)
            {
                auto& tract = loaded_tract_data[index];
                for(size_t i = 0;i < tract.size();i += 3)
                {
                    tipl::vector<3> p(&tract[i]);
                    p.to(T);
                    tract[i] = p[0];
                    tract[i+1] = p[1];
                    tract[i+2] = p[2];
                }
            });
            tipl::out() << "old dim:" << geo << " vs:" << vs;
            geo[0] *= std::abs(T[0]);
            geo[1] *= std::abs(T[5]);
            geo[2] *= std::abs(T[10]);
            vs[0] /= std::abs(T[0]);
            vs[1] /= std::abs(T[5]);
            vs[2] /= std::abs(T[10]);
            tipl::out() << "new dim:" << geo << " vs:" << vs;
        };

        // two conditions to transform tracts

        // 1. QSDR loading MNI space tracts
        if(is_mni && trans_to_mni != source_trans_to_mni)
            apply_transform(tipl::from_space(source_trans_to_mni).to(trans_to_mni));

        // 2. subject FIB loading MNI space tracts
        if(!is_mni && tract_is_mni)
        {
            if(!handle->map_to_mni(tipl::show_prog))
            {
                tipl::out() << "cannot run normalization" << std::endl;;
                return false;
            }
            tipl::out() << "template space: " << std::endl;
            tipl::out() << handle->template_to_mni << std::endl;

            // first transform to template space
            if(handle->template_to_mni != source_trans_to_mni)
                apply_transform(tipl::from_space(source_trans_to_mni).to(handle->template_to_mni));
            // then warp to the native space
            tipl::out() << "warping tract from mni to native space" << std::endl;
            handle->temp2sub(loaded_tract_data);
        }
    }


    loaded_tract_data.swap(tract_data);

    tract_color.resize(tract_data.size(),color);
    for(size_t i = 0;i < tract_cluster.size();++i)
        if(tract_cluster[i] < colors.size())
            tract_color[i] = colors[tract_cluster[i]];

    tract_tag.clear();
    tract_tag.resize(tract_data.size());
    deleted_tract_data.clear();
    deleted_tract_color.clear();
    deleted_tract_tag.clear();
    deleted_count.clear();
    is_cut.clear();
    redo_size.clear();
    return true;
}

//---------------------------------------------------------------------------
bool TractModel::save_data_to_file(std::shared_ptr<fib_data> handle,const char* file_name,const std::string& index_name)
{
    if(get_visible_track_count() == 0)
        return false;

    std::vector<std::vector<float> > data(get_tracts_data(handle,index_name));
    if(data.empty())
        return false;

    std::string file_name_s(file_name);
    if(tipl::ends_with(file_name_s,"tt.gz"))
    {
        bool result = TinyTrack::save_to_file(file_name,geo,vs,trans_to_mni,tract_data,
                                              std::vector<uint16_t>(tract_cluster.begin(),tract_cluster.end()),report,parameter_id,
                                              std::vector<unsigned int>{tract_color.front()});
        return result;
    }
    if(tipl::ends_with(file_name_s,".trk"))
        file_name_s += ".gz";
    if(tipl::ends_with(file_name_s,".trk.gz"))
        return TrackVis::save_to_file(file_name_s.c_str(),geo,vs,trans_to_mni,tract_data,data,parameter_id,tract_color.front());
    if(tipl::ends_with(file_name_s,".txt"))
    {
        std::ofstream out(file_name,std::ios::binary);
        if (!out)
            return false;
        for (unsigned int i = 0;i < data.size();++i)
        {
            std::copy(data[i].begin(),data[i].end(),std::ostream_iterator<float>(out," "));
            out << std::endl;
        }
        return true;
    }
    if (tipl::ends_with(file_name_s,".mat"))
    {
        tipl::io::mat_write out(file_name);
        if(!out)
            return false;
        std::vector<float> buf;
        std::vector<unsigned int> length;
        for(unsigned int index = 0;index < data.size();++index)
        {
            length.push_back((unsigned int)data[index].size());
            std::copy(data[index].begin(),data[index].end(),std::back_inserter(buf));
        }
        out.write("data",buf);
        out.write("length",length);
        out.write("trans",trans_to_mni);
        return true;
    }

    return false;
}
//---------------------------------------------------------------------------
// Native space FIB save tracts to the template space
bool TractModel::save_tracts_in_template_space(std::shared_ptr<fib_data> handle,const char* file_name,bool output_mni)
{
    if(!handle->map_to_mni(tipl::show_prog))
        return false;
    std::shared_ptr<TractModel> tract_in_template(
                new TractModel(handle->template_I.shape(),handle->template_vs,handle->template_to_mni));
    std::vector<std::vector<float> > new_tract_data(tract_data.size());
    tipl::adaptive_par_for(tract_data.size(),[&](unsigned int i)
    {
        new_tract_data[i].resize(tract_data[i].size());
        for(unsigned int j = 0;j < tract_data[i].size();j += 3)
        {
            tipl::vector<3> v(&(tract_data[i][j]));
            if(output_mni)
                handle->sub2mni(v);
            else
                handle->sub2temp(v);
            new_tract_data[i][j] = v[0];
            new_tract_data[i][j+1] = v[1];
            new_tract_data[i][j+2] = v[2];
        }
    });
    tract_in_template->add_tracts(new_tract_data);
    tract_in_template->resample(0.5f);

    return tract_in_template->save_tracts_to_file(file_name);
}

//---------------------------------------------------------------------------
bool TractModel::save_transformed_tracts_to_file(const char* file_name,tipl::shape<3> new_dim,
                                                 tipl::vector<3> new_vs,
                                                 const tipl::matrix<4,4>& trans_to_mni,
                                                 const tipl::matrix<4,4>& T,bool end_point)
{
    std::shared_ptr<TractModel> tract_in_other_space(new TractModel(new_dim,new_vs,trans_to_mni));
    std::vector<std::vector<float> > new_tract_data(tract_data);
    for(unsigned int i = 0;i < tract_data.size();++i)
        for(unsigned int j = 0;j < tract_data[i].size();j += 3)
        tipl::vector_transformation(&(tract_data[i][j]),&(new_tract_data[i][j]),&T[0],tipl::vdim<3>());
    tract_in_other_space->add_tracts(new_tract_data);
    tract_in_other_space->resample(0.5f);
    if(end_point)
        return tract_in_other_space->save_end_points(file_name);
    else
        return tract_in_other_space->save_tracts_to_file(file_name);
}

//---------------------------------------------------------------------------
bool TractModel::save_tracts_to_file(const char* file_name_)
{
    std::string file_name(file_name_);
    saved = true;
    if(get_visible_track_count() == 0)
        return false;
    if(tipl::ends_with(file_name,"tt.gz"))
    {
        return TinyTrack::save_to_file(file_name.c_str(),geo,vs,trans_to_mni,
                                       tract_data,std::vector<uint16_t>(tract_cluster.begin(),tract_cluster.end()),report,parameter_id,
                                       std::vector<unsigned int>{tract_color.front()});
    }
    if(tipl::ends_with(file_name,".trk") || tipl::ends_with(file_name,".trk.gz"))
    {
        return TrackVis::save_to_file(file_name.c_str(),geo,vs,trans_to_mni,
                tract_data,std::vector<std::vector<float> >(),parameter_id,tract_color.front());
    }
    if(tipl::ends_with(file_name,".tck"))
    {
        std::ofstream out(file_name.c_str(), std::ios::binary);
        if(!out)
            return false;
        std::array<char, 200> header;
        std::sprintf(header.data(), "mrtrix tracks\ndatatype: Float32LE\ndim: %d,%d,%d\nvox: %f,%f,%f\ndatatype: Float32LE\nfile: . 200\ncount: %d\nEND\n",
                     geo[0], geo[1], geo[2], vs[0], vs[1], vs[2], static_cast<int>(tract_data.size()));
        out.write(header.data(), header.size());

        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        for (const auto& t : tract_data) {
            std::vector<float> buf(t);
            tipl::multiply_constant(buf, vs[0]);
            out.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(float));
            out.write(reinterpret_cast<const char*>(&nan),sizeof(nan));
            out.write(reinterpret_cast<const char*>(&nan),sizeof(nan));
            out.write(reinterpret_cast<const char*>(&nan),sizeof(nan));
        }
        out.write(reinterpret_cast<const char*>(&inf),sizeof(inf));
        out.write(reinterpret_cast<const char*>(&inf),sizeof(inf));
        out.write(reinterpret_cast<const char*>(&inf),sizeof(inf));
        return true;
    }

    if (tipl::ends_with(file_name,".txt"))
    {
        std::ofstream out(file_name_,std::ios::binary);
        if (!out)
            return false;
        for (unsigned int i = 0;i < tract_data.size();++i)
        {
            std::copy(tract_data[i].begin(),
                      tract_data[i].end(),
                      std::ostream_iterator<float>(out," "));
            out << std::endl;
        }
        return true;
    }
    if (tipl::ends_with(file_name,".mat"))
    {
        tipl::io::mat_write out(file_name.c_str());
        if(!out)
            return false;
        std::vector<float> buf;
        std::vector<unsigned int> length;
        for(unsigned int index = 0;index < tract_data.size();++index)
        {
            length.push_back((unsigned int)tract_data[index].size()/3);
            std::copy(tract_data[index].begin(),tract_data[index].end(),std::back_inserter(buf));
        }
        out.write("tracts",buf,3);
        out.write("length",length);
        return true;
    }
    if (tipl::ends_with(file_name,".nii") || tipl::ends_with(file_name,".nii.gz"))
    {
        std::vector<tipl::vector<3,float> >points;
        get_tract_points(points);
        ROIRegion region(geo,vs,trans_to_mni);
        region.add_points(std::move(points));
        region.save_region_to_file(file_name_);
        return true;
    }
    return save_tracts_to_file((std::string(file_name_) + ".tt.gz").c_str());
}
std::string TractModel::get_obj(unsigned int& coordinate_count,
                         unsigned char tract_style,
                         float tube_diameter,
                         unsigned char tract_tube_detail)
{
    std::vector<tipl::vector<3,float> > points(8),previous_points(8);

    const float detail_option[] = {1.0,0.5,0.25,0.0,0.0};
    const unsigned char end_sequence[8] = {4,3,5,2,6,1,7,0};
    const unsigned char end_sequence2[8] = {0,1,7,2,6,3,5,4};

    float tube_detail = tube_diameter*detail_option[tract_tube_detail]*4.0f;


    std::string coordinate,coordinate_index;
    auto push_vertices = [&](const tipl::vector<3,float>& pos)
    {
        coordinate.push_back('v');
        coordinate.push_back(' ');
        coordinate += std::to_string(pos[0]*vs[0]);
        coordinate.pop_back();
        coordinate.pop_back();
        coordinate.pop_back();
        coordinate.back() = ' ';
        coordinate += std::to_string(pos[2]*vs[2]);
        coordinate.pop_back();
        coordinate.pop_back();
        coordinate.pop_back();
        coordinate.back() = ' ';
        coordinate += std::to_string(-pos[1]*vs[1]);
        coordinate.pop_back();
        coordinate.pop_back();
        coordinate.pop_back();
        coordinate.back() = '\n';
    };

    for (unsigned int data_index = 0; data_index < tract_data.size(); ++data_index)
    {
        unsigned int vertex_count = uint32_t(tract_data[data_index].size()/3);
        if (vertex_count <= 1)
            continue;

        const float* data_iter = &tract_data[data_index][0];


        unsigned int prev_coordinate_count = coordinate_count;
        tipl::vector<3,float> last_pos(data_iter),
                vec_a(1,0,0),vec_b(0,1,0),
                vec_n,prev_vec_n,vec_ab,vec_ba;

        for (unsigned int j = 0, index = 0; index < vertex_count;j += 3, data_iter += 3, ++index)
        {
            tipl::vector<3,float> pos(data_iter);
            if (index + 1 < vertex_count)
            {
                vec_n[0] = data_iter[3] - data_iter[0];
                vec_n[1] = data_iter[4] - data_iter[1];
                vec_n[2] = data_iter[5] - data_iter[2];
                vec_n.normalize();
            }
            if (index != 0 && index+1 != vertex_count)
            {
                tipl::vector<3,float> displacement(data_iter+3);
                displacement -= last_pos;
                displacement -= prev_vec_n*(prev_vec_n*displacement);
                if (float(displacement.length()) < tube_detail)
                    continue;
            }
            // add end
            if(tract_style == 0)// line
            {
                push_vertices(pos);
                prev_vec_n = vec_n;
                last_pos = pos;
                ++coordinate_count;
                continue;
            }

            if (index == 0 && std::fabs(vec_a*vec_n) > 0.5f)
                std::swap(vec_a,vec_b);

            vec_b = vec_a.cross_product(vec_n);
            vec_a = vec_n.cross_product(vec_b);
            vec_a.normalize();
            vec_b.normalize();
            vec_ba = vec_ab = vec_a;
            vec_ab += vec_b;
            vec_ba -= vec_b;
            vec_ab.normalize();
            vec_ba.normalize();
            vec_ba *= tube_diameter;
            vec_ab *= tube_diameter;
            vec_a *= tube_diameter;
            vec_b *= tube_diameter;

            // add point
            {
                std::fill(points.begin(),points.end(),pos);
                points[0] += vec_a;
                points[1] += vec_ab;
                points[2] += vec_b;
                points[3] -= vec_ba;
                points[4] -= vec_a;
                points[5] -= vec_ab;
                points[6] -= vec_b;
                points[7] += vec_ba;
            }

            if (index == 0)
            {
                for (unsigned int k = 0;k < 8;++k)
                {
                    tipl::vector<3,float> pos(points[end_sequence[k]][0],points[end_sequence[k]][1],points[end_sequence[k]][2]);
                    push_vertices(pos);
                }
                coordinate_count+=8;
            }
            else
            // add tube
            {
                push_vertices(points[0]);
                for (unsigned int k = 1;k < 8;++k)
                {
                    push_vertices(previous_points[k]);
                    push_vertices(points[k]);
                }
                push_vertices(points[0]);
                coordinate_count+=16;
                if(index +1 == vertex_count)
                {
                    for (unsigned int k = 0;k < 8;++k)
                    {
                        tipl::vector<3,float> pos(points[end_sequence2[k]][0],points[end_sequence2[k]][1],points[end_sequence2[k]][2]);
                        push_vertices(pos);
                    }
                    coordinate_count+=8;
                }
            }
            previous_points.swap(points);
            prev_vec_n = vec_n;
            last_pos = pos;
        }
        if(tract_style == 0)// line
        {
            coordinate_index.push_back('l');
            for (unsigned int j = prev_coordinate_count+1;j+2 <= coordinate_count;++j)
            {
                coordinate_index.push_back(' ');
                coordinate_index += std::to_string(j);
            }
            coordinate_index.push_back('\n');
        }
        else
        for (unsigned int j = prev_coordinate_count,k = 0;j+2 < coordinate_count;++j)
        {
            coordinate_index.push_back('f');
            coordinate_index.push_back(' ');
            coordinate_index += std::to_string(j+1);
            coordinate_index.push_back(' ');
            coordinate_index += std::to_string(j+ (k ? 2 : 3));
            coordinate_index.push_back(' ');
            coordinate_index += std::to_string(j+ (k ? 3 : 2));
            coordinate_index.push_back('\n');
            k = (k ? 0 :1);
        }
    }
    return coordinate+coordinate_index;
}

//---------------------------------------------------------------------------
bool TractModel::save_all(const char* file_name,
                          const std::vector<std::shared_ptr<TractModel> >& all,
                          const std::vector<std::string>& name_list)
{    
    if(all.empty())
        return false;
    tipl::progress prog("saving ",std::filesystem::path(file_name).filename().u8string().c_str());
    for(unsigned int index = 0;index < all.size();++index)
        all[index]->saved = true;
    if (tipl::ends_with(file_name,".tt.gz"))
    {
        std::vector<size_t> tract_size(all.size());
        for(size_t i = 0;i < all.size();++i)
            tract_size[i] = all[i]->tract_data.size();
        size_t total_size = std::accumulate(tract_size.begin(),tract_size.end(),size_t(0));

        // collect all tract together
        std::vector<std::vector<float> > all_tract(total_size);
        std::vector<uint16_t> cluster(total_size);
        std::vector<unsigned int> colors(all.size());
        for(size_t cluster_index = 0,pos = 0;cluster_index < all.size();++cluster_index)
        {
            auto& tract = all[cluster_index]->tract_data;
            if(tract.empty())
                continue;
            for (size_t j = 0;j < tract.size();++j,++pos)
            {
                all_tract[pos].swap(tract[j]);
                cluster[pos] = uint16_t(cluster_index);
            }
            colors[cluster_index] = all[cluster_index]->tract_color.front();
        }
        // save file
        bool result = TinyTrack::save_to_file(file_name,all[0]->geo,all[0]->vs,all[0]->trans_to_mni,
                    all_tract,cluster,all[0]->report,all[0]->parameter_id,colors);
        // restore tracts
        for(size_t i = 0,pos = 0;i < all.size();++i)
        {
            auto& tract = all[i]->tract_data;
            for (size_t j = 0;j < tract.size();++j,++pos)
                all_tract[pos].swap(tract[j]);
        }
        if(!result)
            return false;
    }
    if (tipl::ends_with(file_name,".txt"))
    {
        std::ofstream out(file_name,std::ios::binary);
        if (!out)
            return false;
        for(unsigned int index = 0;index < all.size();++index)
        for (unsigned int i = 0;i < all[index]->tract_data.size();++i)
        {
            std::copy(all[index]->tract_data[i].begin(),
                      all[index]->tract_data[i].end(),
                      std::ostream_iterator<float>(out," "));
            out << std::endl;
            out << index << std::endl;
        }
        return true;
    }
    if (tipl::ends_with(file_name,".trk") || tipl::ends_with(file_name,".trk.gz"))
    {
        tipl::io::gz_ostream out;
        if (!out.open(file_name))
            return false;
        {
            TrackVis trk;
            trk.init(all[0]->geo,all[0]->vs,all[0]->trans_to_mni);
            trk.n_count = 0;
            trk.n_properties = 1;
            std::copy(all[0]->report.begin(),all[0]->report.begin()+
                    std::min<int>(439,all[0]->report.length()),trk.reserved);
            for(unsigned int index = 0;index < all.size();++index)
                trk.n_count += all[index]->tract_data.size();
            out.write((const char*)&trk,1000);

        }
        for(unsigned int index = 0;prog(index,all.size());++index)
        for (unsigned int i = 0;i < all[index]->tract_data.size();++i)
        {
            int n_point = all[index]->tract_data[i].size()/3;
            std::vector<float> buffer(all[index]->tract_data[i].size()+1);
            const float *from = &*all[index]->tract_data[i].begin();
            const float *end = from + all[index]->tract_data[i].size();
            float* to = &*buffer.begin();
            for (unsigned int flag = 0;from != end;++from,++to)
            {
                *to = (*from)*all[index]->vs[flag];
                ++flag;
                if (flag == 3)
                    flag = 0;
            }
            buffer.back() = index;
            out.write((const char*)&n_point,sizeof(int));
            out.write((const char*)&*buffer.begin(),sizeof(float)*buffer.size());
        }
    }
    if (tipl::ends_with(file_name,".nii.gz"))
    {
        return TractModel::export_pdi(file_name,all);
    }
    if (tipl::ends_with(file_name,".mat"))
    {
        tipl::io::mat_write out(file_name);
        if(!out)
            return false;
        std::vector<float> buf;
        std::vector<unsigned int> length;
        std::vector<unsigned int> cluster;
        for(unsigned int index = 0; index < all.size();++index)
        for (unsigned int i = 0;i < all[index]->tract_data.size();++i)
        {
            cluster.push_back(index);
            length.push_back(all[index]->tract_data[i].size()/3);
            std::copy(all[index]->tract_data[i].begin(),all[index]->tract_data[i].end(),std::back_inserter(buf));
        }
        out.write("tracts",buf,3);
        out.write("length",length);
        out.write("cluster",cluster);
        return true;
    }
    if(prog.aborted())
        return false;
    // output label file
    std::ofstream out(std::string(file_name)+".txt");
    for(int i = 0;i < name_list.size();++i)
        out << name_list[i] << std::endl;
    return true;
}
//---------------------------------------------------------------------------
bool TractModel::load_tracts_color_from_file(const char* file_name)
{
    std::ifstream in(file_name);
    if (!in)
        return false;
    std::vector<float> colors;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),
              std::back_inserter(colors));
    if(colors.size() <= tract_color.size())
        std::copy(colors.begin(),colors.begin()+colors.size(),tract_color.begin());
    if(colors.size()/3 <= tract_color.size())
        for(unsigned int index = 0,pos = 0;pos+2 < colors.size();++index,pos += 3)
            tract_color[index] = tipl::rgb(std::min<int>(colors[pos],255),
                                                  std::min<int>(colors[pos+1],255),
                                                  std::min<int>(colors[pos+2],255));
    return true;
}
//---------------------------------------------------------------------------
bool TractModel::save_tracts_color_to_file(const char* file_name)
{
    std::ofstream out(file_name);
    if (!out)
        return false;
    for(unsigned int index = 0;index < tract_color.size();++index)
    {
        tipl::rgb color;
        color.color = tract_color[index];
        out << (int)color.r << " " << (int)color.g << " " << (int)color.b << std::endl;
    }
    return out.good();
}

//---------------------------------------------------------------------------
/*
bool TractModel::save_data_to_mat(const char* file_name,int index,const char* data_name)
{
    MatWriter mat_writer(file_name);
    if(!mat_writer.opened())
        return false;
    tipl::progress prog_("saving");
        for (unsigned int i = 0;prog(i,tract_data.size());++i)
    {
        unsigned int count;
        const float* ptr = get_data(tract_data[i],index,count);
        if (!ptr)
            continue;
		std::ostringstream out;
		out << data_name << i;
        mat_writer.add_matrix(out.str().c_str(),ptr,
                              (index != -2 ) ? 1 : 3,
                              (index != -2 ) ? count : count /3);
    }
        return true;
}
*/
//---------------------------------------------------------------------------
bool TractModel::save_end_points(const char* file_name_) const
{

    std::vector<float> buffer;
    buffer.reserve(tract_data.size() * 6);
    for (unsigned int index = 0;index < tract_data.size();++index)
    {
        size_t length = tract_data[index].size();
        buffer.push_back(tract_data[index][0]);
        buffer.push_back(tract_data[index][1]);
        buffer.push_back(tract_data[index][2]);
        buffer.push_back(tract_data[index][length-3]);
        buffer.push_back(tract_data[index][length-2]);
        buffer.push_back(tract_data[index][length-1]);
    }

    std::string file_name(file_name_);
    if (file_name.find(".txt") != std::string::npos)
    {
        std::ofstream out(file_name_,std::ios::out);
        if (!out)
            return false;
        std::copy(buffer.begin(),buffer.end(),std::ostream_iterator<float>(out," "));
    }
    if (file_name.find(".mat") != std::string::npos)
    {
        tipl::io::mat_write out(file_name_);
        if(!out)
            return false;
        out.write("end_points",buffer,3);
    }
    return true;
}
//---------------------------------------------------------------------------
void TractModel::resample(float new_step)
{
    tipl::adaptive_par_for(tract_data.size(),[&](size_t i)
    {
        if(tract_data[i].size() <= 6)
            return;
        std::vector<float> new_tracts;
        float d = 0.0;
        new_tracts.push_back(tract_data[i][0]);
        new_tracts.push_back(tract_data[i][1]);
        new_tracts.push_back(tract_data[i][2]);
        for (unsigned int j = 3;j < tract_data[i].size();j += 3)
        {
            tipl::vector<3> p(&tract_data[i][j-3]),dis(&tract_data[i][j]);
            dis -= p;
            float step = float(dis.length());
            dis *= new_step/step;
            while(d+new_step < step)
            {
                p += dis;
                d += new_step;
                new_tracts.push_back(p[0]);
                new_tracts.push_back(p[1]);
                new_tracts.push_back(p[2]);
            }
            d -= step;
        }
        new_tracts.push_back(tract_data[i][tract_data[i].size()-3]);
        new_tracts.push_back(tract_data[i][tract_data[i].size()-2]);
        new_tracts.push_back(tract_data[i][tract_data[i].size()-1]);
        new_tracts.swap(tract_data[i]);
    });
}
//---------------------------------------------------------------------------
void TractModel::get_tract_points(std::vector<tipl::vector<3,float> >& points)
{
    for (unsigned int index = 0;index < tract_data.size();++index)
        for (unsigned int j = 0;j < tract_data[index].size();j += 3)
        {
            tipl::vector<3,float> point(&tract_data[index][j]);
            points.push_back(point);
        }
}
//---------------------------------------------------------------------------
void TractModel::get_in_slice_tracts(unsigned char dim,int pos,
                                     tipl::matrix<4,4>* pT,
                                     std::vector<std::vector<tipl::vector<2,float> > >& lines,
                                     std::vector<std::vector<unsigned int> >& colors,
                                     unsigned int max_count,
                                     int track_color_style,
                                     bool& terminated)
{

    std::vector<tipl::vector<2,float> > line;
    std::vector<unsigned int> color;

    auto add_line = [&](unsigned int index)
    {
        if(line.empty() || index >= tract_color.size())
            return;
        lines.push_back(std::move(line));
        colors.push_back(std::move(color));
        line.clear();
    };

    auto add_color = [&](tipl::vector<3>&& pos,tipl::vector<3>& prev_pos)
    {
        prev_pos -= pos;
        prev_pos.abs();
        prev_pos *= 200.0f/float(prev_pos.length());
        color.push_back(uint32_t(tipl::rgb(uint8_t(prev_pos[0]),uint8_t(prev_pos[1]),uint8_t(prev_pos[2]))));
        prev_pos = pos;
    };

    unsigned int skip = std::max<unsigned int>(1,uint32_t(tract_data.size())/max_count);
    for (unsigned int index = 0;!terminated && index < tract_data.size();add_line(index),index += skip)
    {
        const auto& tract = tract_data[index];
        if(tract.size() < 6)
            continue;
        tipl::vector<3> prev_pos(&tract[3]);
        for (unsigned int j = 0;j < tract.size();j += 3)
        {
            tipl::vector<3> t(&tract[j]);
            if(pT)
                t.to(*pT);
            if(int(std::round(t[dim])) == pos)
            {
                auto p = tipl::space2slice<tipl::vector<2,float> >(dim,t);
                line.push_back(p);
                if(track_color_style)
                    color.push_back(tract_color[index]);
                else
                    add_color(std::move(t),prev_pos);
            }
            else
                add_line(index);
        }
    }
}
//---------------------------------------------------------------------------
void TractModel::select(float select_angle,
                        const std::vector<tipl::vector<3,float> >& dirs,
                        const tipl::vector<3,float>& from_pos,std::vector<unsigned int>& selected)
{
    selected.resize(tract_data.size());
    std::fill(selected.begin(),selected.end(),0);
    for(int i = 1;i < dirs.size();++i)
    {
        tipl::vector<3,float> from_dir = dirs[i-1];
        tipl::vector<3,float> to_dir = (i+1 < dirs.size() ? dirs[i+1] : dirs[i]);
        tipl::vector<3,float> z_axis = from_dir.cross_product(to_dir);
        z_axis.normalize();
        float view_angle = from_dir*to_dir;

        float select_angle_cos = std::cos(select_angle*3.141592654/180);
        unsigned int total_track_number = tract_data.size();
        for (unsigned int index = 0;index < total_track_number;++index)
        {
            float angle = 0.0;
            const float* ptr = &*tract_data[index].begin();
            const float* end = ptr + tract_data[index].size();
            for (;ptr < end;ptr += 3)
            {
                tipl::vector<3,float> p(ptr);
                p -= from_pos;
                float next_angle = z_axis*p;
                if ((angle < 0.0 && next_angle >= 0.0) ||
                        (angle > 0.0 && next_angle <= 0.0))
                {

                    p.normalize();
                    if (p*from_dir > view_angle &&
                            p*to_dir > view_angle)
                    {
                        if(select_angle != 0.0)
                        {
                            tipl::vector<3,float> p1(ptr),p2(ptr-3);
                            p1 -= p2;
                            p1.normalize();
                            if(std::abs(p1*z_axis) < select_angle_cos)
                                continue;
                        }
                        selected[index] = ptr - &*tract_data[index].begin();
                        break;
                    }

                }
                angle = next_angle;
            }
        }
    }
}
//---------------------------------------------------------------------------
void TractModel::release_tracts(std::vector<std::vector<float> >& released_tracks)
{
    released_tracks.clear();
    released_tracks.swap(tract_data);
    clear();
}
//---------------------------------------------------------------------------
void TractModel::clear(void)
{
    tract_data.clear();
    tract_color.clear();
    tract_tag.clear();
    redo_size.clear();
}
//---------------------------------------------------------------------------
void TractModel::erase_empty(void)
{
    tract_color.erase(std::remove_if(tract_color.begin(),tract_color.end(),
                        [&](const unsigned int& data){return tract_data[&data-&tract_color[0]].empty();}), tract_color.end());
    tract_tag.erase(std::remove_if(tract_tag.begin(),tract_tag.end(),
                        [&](const unsigned int& data){return tract_data[&data-&tract_tag[0]].empty();}), tract_tag.end());
    tract_data.erase(std::remove_if(tract_data.begin(),tract_data.end(),
                        [&](const std::vector<float>& data){return data.empty();}), tract_data.end() );
}
//---------------------------------------------------------------------------
bool TractModel::delete_tracts(const std::vector<unsigned int>& tracts_to_delete)
{
    if (tracts_to_delete.empty())
        return false;
    for (unsigned int index = 0;index < tracts_to_delete.size();++index)
    {
        deleted_tract_data.push_back(std::move(tract_data[tracts_to_delete[index]]));
        deleted_tract_color.push_back(tract_color[tracts_to_delete[index]]);
        deleted_tract_tag.push_back(tract_tag[tracts_to_delete[index]]);
    }
    erase_empty();
    deleted_count.push_back(tracts_to_delete.size());
    is_cut.push_back(0);
    // no redo once track deleted
    redo_size.clear();
    saved = tract_data.empty();
    return true;
}
//---------------------------------------------------------------------------
bool TractModel::select_tracts(const std::vector<unsigned int>& tracts_to_select)
{
    std::vector<unsigned int> selected(tract_data.size());
    for (unsigned int index = 0;index < tracts_to_select.size();++index)
        selected[tracts_to_select[index]] = 1;

    std::vector<unsigned int> not_selected;
    not_selected.reserve(tract_data.size());

    for (unsigned int index = 0;index < selected.size();++index)
        if (!selected[index])
            not_selected.push_back(index);
    return delete_tracts(not_selected);
}
//---------------------------------------------------------------------------
bool TractModel::delete_repeated(float d)
{   
    std::vector<std::vector<size_t> > x_reg;
    std::vector<size_t> track_reg;
    if(tract_data.size() > 50000)
    {
        x_reg.resize(geo.plane_size());
        track_reg.resize(tract_data.size());
        for(size_t i = 0; i < tract_data.size();++i)
        {
            int x = int(std::round(tract_data[i][0]));
            int y = int(std::round(tract_data[i][1]));
            if(x < 0)
                x = 0;
            if(y < 0)
                y = 0;
            if(x >= geo[0])
                x = geo[0]-1;
            if(y >= geo[1])
                y = geo[1]-1;
            x_reg[track_reg[i] = size_t(x + y*geo[0])].push_back(i);
        }
    }
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
    std::vector<bool> repeated(tract_data.size());
    tipl::adaptive_par_for(tract_data.size(),[&](size_t i)
    {
        if(repeated[i])
            return;
        size_t max_k = x_reg.empty() ? tract_data.size():x_reg[track_reg[i]].size();
        for(size_t k = x_reg.empty() ? i+1:0;k < max_k;++k)
        {
            size_t j = x_reg.empty() ? k :  x_reg[track_reg[i]][k];
            if(j <= i || repeated[j] ||
               min_min_fun(d,&tract_data[i][0],&tract_data[j][0]) >= d ||
               min_min_fun(d,&tract_data[i][tract_data[i].size()-3],&tract_data[j][tract_data[j].size()-3]) >= d)
                continue;
            bool not_repeated = false;
            for(size_t m = 0;m < tract_data[i].size();m += 3)
            {
                float min_dis = norm1(&tract_data[i][m],&tract_data[j][0]);
                for(size_t n = 3;n < tract_data[j].size();n += 3)
                    min_dis = min_min_fun(min_dis,&tract_data[i][m],&tract_data[j][n]);
                if(min_dis > d)
                {
                    not_repeated = true;
                    break;
                }
            }
            if(!not_repeated)
            for(size_t m = 0;m < tract_data[j].size();m += 3)
            {
                float min_dis = norm1(&tract_data[j][m],&tract_data[i][0]);
                for(size_t n = 0;n < tract_data[i].size();n += 3)
                    min_dis = min_min_fun(min_dis,&tract_data[j][m],&tract_data[i][n]);
                if(min_dis > d)
                {
                    not_repeated = true;
                    break;
                }
            }
            if(!not_repeated)
                repeated[j] = true;
        }
    });
    std::vector<unsigned int> track_to_delete;
    for(size_t i = 0;i < tract_data.size();++i)
        if(repeated[i])
            track_to_delete.push_back(uint32_t(i));
    return delete_tracts(track_to_delete);
}
bool TractModel::delete_branch(void)
{
    std::vector<tipl::vector<3,short> > p1,p2;
    to_end_point_voxels(p1,p2);
    tipl::image<3,unsigned char>mask;
    ROIRegion r1(geo,vs,trans_to_mni),r2(geo,vs,trans_to_mni);
    r1.add_points(std::move(p1));
    r2.add_points(std::move(p2));

    r1.save_region_to_buffer(mask);
    tipl::morphology::defragment(mask);
    r1.load_region_from_buffer(mask);

    r2.save_region_to_buffer(mask);
    tipl::morphology::defragment(mask);
    r2.load_region_from_buffer(mask);

    std::shared_ptr<fib_data> handle(new fib_data(geo,vs,trans_to_mni));
    std::shared_ptr<RoiMgr> roi_mgr(new RoiMgr(handle));
    roi_mgr->setRegions(r1.region,end_id,"end1");
    roi_mgr->setRegions(r2.region,end_id,"end2");
    return filter_by_roi(roi_mgr);
}
//---------------------------------------------------------------------------
bool TractModel::delete_by_length(float length)
{
    std::vector<unsigned int> track_to_delete;
    for(unsigned int i = 0;i < tract_data.size();++i)
    {
        if(tract_data[i].size() <= 6)
        {
            track_to_delete.push_back(i);
            continue;
        }
        if((((tract_data[i].size()/3)-1)*(get_tract_point(i,0)-get_tract_point(i,1)).length()) < length)
            track_to_delete.push_back(i);
    }
    return delete_tracts(track_to_delete);
}
//---------------------------------------------------------------------------
void TractModel::cut(const std::vector<unsigned int>& tract_to_delete,
         const std::vector<std::vector<float> >& new_tract,
         const std::vector<unsigned int>& new_tract_color)
{
    delete_tracts(tract_to_delete);
    is_cut.back() = cur_cut_id;
    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        tract_data.push_back(std::move(new_tract[index]));
        tract_color.push_back(new_tract_color[index]);
        tract_tag.push_back(cur_cut_id);
    }
    ++cur_cut_id;
    redo_size.clear();
}
bool TractModel::cut(float select_angle,const std::vector<tipl::vector<3,float> >& dirs,
                     const tipl::vector<3,float>& from_pos)
{
    std::vector<unsigned int> selected;
    select(select_angle,dirs,from_pos,selected);
    std::vector<std::vector<float> > new_tract;
    std::vector<unsigned int> new_tract_color;

    std::vector<unsigned int> tract_to_delete;
    for (unsigned int index = 0;index < selected.size();++index)
        if (selected[index] && selected[index] < tract_data[index].size() &&
            tract_data[index].size() > 6)
        {
            new_tract.push_back(std::vector<float>(tract_data[index].begin(),tract_data[index].begin()+selected[index]));
            new_tract_color.push_back(tract_color[index]);
            new_tract.push_back(std::vector<float>(tract_data[index].begin() + selected[index],tract_data[index].end()));
            new_tract_color.push_back(tract_color[index]);
            tract_to_delete.push_back(index);
        }
    if(tract_to_delete.empty())
        return false;
    cut(tract_to_delete,new_tract,new_tract_color);
    return true;

}

void get_cut_points(const std::vector<std::vector<float> >& tract_data,
                    unsigned int dim, unsigned int pos,bool greater,
                    std::vector<std::vector<bool> >& has_cut)
{
    has_cut.resize(tract_data.size());
    tipl::adaptive_par_for(tract_data.size(),[&](unsigned int i)
    {
        has_cut[i].resize(tract_data[i].size()/3);
        for(unsigned int j = 0,t = 0;j < tract_data[i].size();j += 3,++t)
        {
            has_cut[i][t] = ((tract_data[i][j+dim] < pos) ^ greater);
        }
    });
}

void get_cut_points(const std::vector<std::vector<float> >& tract_data,
                    unsigned int dim, unsigned int pos,bool greater,
                    const tipl::matrix<4,4>& T,
                    std::vector<std::vector<bool> >& has_cut)
{
    has_cut.resize(tract_data.size());
    tipl::vector<3> sr(T.begin()+dim*4);
    float shift = T[3+dim*4];
    tipl::adaptive_par_for(tract_data.size(),[&](unsigned int i)
    {
        has_cut[i].resize(tract_data[i].size()/3);
        for(unsigned int j = 0,t = 0;j < tract_data[i].size();j += 3,++t)
        {
            tipl::vector<3> v(&tract_data[i][j]);
            float p = v*sr;
            p += shift;
            has_cut[i][t] = ((p < pos) ^ greater);
        }
    });
}
tipl::vector<3> get_tract_dir(const std::vector<std::vector<float> >& tract_data,
                   std::vector<char>& dir);
void TractModel::cut_end_portion(float from,float to)
{
    tipl::vector<3,double> from_point,to_point;
    std::vector<char> dir;
    get_tract_dir(tract_data,dir);
    for(unsigned int i = 0;i < tract_data.size();++i)
    {
        if(tract_data[i].size() <= 6)
            continue;
        size_t from_length = size_t(from*size_t((tract_data[i].size()-3)/3))*3;
        size_t to_length = size_t(to*size_t((tract_data[i].size()-3)/3))*3;
        if(!dir[i])
            std::swap(from_length,to_length);
        from_point += tipl::vector<3>(tract_data[i].data()+from_length);
        to_point += tipl::vector<3>(tract_data[i].data()+to_length);
    }
    from_point /= tract_data.size();
    to_point /= tract_data.size();

    auto find_location = [&](const std::vector<float>& tract,const tipl::vector<3>& point)
    {
        float best_dis2 = (tipl::vector<3>(tract.data())-point).length2();
        float best_dis = std::sqrt(best_dis2);
        size_t best_pos = 0;
        for(size_t pos = 3;pos < tract.size();pos += 3)
        {
            float dx = std::fabs(tract[pos]-point[0]);
            if(dx > best_dis)
                continue;
            float dy = std::fabs(tract[pos+1]-point[1]);
            if(dy > best_dis)
                continue;
            float dz = std::fabs(tract[pos+2]-point[2]);
            if(dz > best_dis)
                continue;
            auto dis2 = dx*dx + dy*dy + dz*dz;
            if(dis2 > best_dis2)
                continue;
            best_dis2 = dis2;
            best_dis = std::sqrt(best_dis2);
            best_pos = pos;
        }
        return best_pos;
    };

    tipl::vector<3> from_point16(from_point),to_point16(to_point);
    std::vector<std::vector<float> > new_tract(tract_data);
    std::vector<unsigned int> new_tract_color(tract_color);
    std::vector<unsigned int> tract_to_delete(tract_data.size());

    tipl::adaptive_par_for(tract_data.size(),[&](size_t i)
    {
        tract_to_delete[i] = i;
        auto from = tract_data[i].data()+find_location(tract_data[i],dir[i] ? from_point16 : to_point16);
        auto to = tract_data[i].data()+find_location(tract_data[i],dir[i] ? to_point16 : from_point16);
        if(from > to)
            std::swap(from,to);
        new_tract[i] = std::vector<float>(from,to+3);
    });
    cut(tract_to_delete,new_tract,new_tract_color);
}
bool TractModel::cut_by_slice(unsigned int dim, unsigned int pos,bool greater,const tipl::matrix<4,4>* T)
{
    std::vector<std::vector<bool> > has_cut;
    if(T == nullptr)
        get_cut_points(tract_data,dim,pos,greater,has_cut);
    else
        get_cut_points(tract_data,dim,pos,greater,*T,has_cut);
    std::vector<std::vector<float> > new_tract;
    std::vector<unsigned int> new_tract_color;
    std::vector<unsigned int> tract_to_delete;
    bool modified = false;
    for(unsigned int i = 0;i < tract_data.size();++i)
    {
        bool adding = false;
        for(unsigned int j = 0,t = 0;j < tract_data[i].size();j += 3,++t)
        {
            if(has_cut[i][t])
            {
                modified = true;
                if(!adding)
                    continue;
                adding = false;
            }
            if(!adding)
            {
                new_tract.push_back(std::vector<float>());
                new_tract_color.push_back(tract_color[i]);
                adding = true;
            }
            new_tract.back().push_back(tract_data[i][j]);
            new_tract.back().push_back(tract_data[i][j+1]);
            new_tract.back().push_back(tract_data[i][j+2]);
        }
        tract_to_delete.push_back(i);
    }
    cut(tract_to_delete,new_tract,new_tract_color);
    return modified;
}
//---------------------------------------------------------------------------
bool TractModel::filter_by_roi(std::shared_ptr<RoiMgr> roi_mgr)
{
    std::vector<unsigned int> tracts_to_delete;
    for (unsigned int index = 0;index < tract_data.size();++index)
    if(tract_data[index].size() >= 6)
    {
        if(!roi_mgr->within_roi(&(tract_data[index][0]),tract_data[index].size()) ||
           !roi_mgr->fulfill_end_point(tipl::vector<3,float>(tract_data[index][0],
                                                             tract_data[index][1],
                                                             tract_data[index][2]),
                                      tipl::vector<3,float>(tract_data[index][tract_data[index].size()-3],
                                                             tract_data[index][tract_data[index].size()-2],
                                                             tract_data[index][tract_data[index].size()-1])))
        {
            tracts_to_delete.push_back(index);
            continue;
        }
        if(!roi_mgr->roa.empty() || !roi_mgr->limiting.empty())
        {
            for(unsigned int i = 0;i < tract_data[index].size();i+=3)
            {
                auto v = tipl::v(tract_data[index][i],tract_data[index][i+1],tract_data[index][i+2]);
                if(roi_mgr->within_roa(v) || !roi_mgr->within_limiting(v))
                {
                    tracts_to_delete.push_back(index);
                    break;
                }
            }
        }
    }
    return delete_tracts(tracts_to_delete);
}
//---------------------------------------------------------------------------
bool TractModel::reconnect_track(float distance,float angular_threshold)
{
    if(distance >= 2.0f)
        return reconnect_track(distance*0.5f,angular_threshold);
    bool has_merged = false;
    std::vector<std::vector<uint32_t> > endpoint_map(geo.size());
    for (unsigned int index = 0;index < tract_data.size();++index)
        if(tract_data[index].size() > 6)
        {
            tipl::vector<3,float> end1(&tract_data[index][0]);
            tipl::vector<3,float> end2(&tract_data[index][tract_data[index].size()-3]);
            end1 /= distance;
            end2 /= distance;
            end1.round();
            end2.round();
            if(geo.is_valid(end1))
                endpoint_map[tipl::pixel_index<3>(end1[0],end1[1],end1[2],geo).index()].push_back(index);
            if(geo.is_valid(end2))
                endpoint_map[tipl::pixel_index<3>(end2[0],end2[1],end2[2],geo).index()].push_back(index);
        }
    tipl::adaptive_par_for(endpoint_map.size(),[&](size_t index)
    {
        if(endpoint_map[index].size() <= 1)
            return;
        std::set<uint32_t> sorted_list(endpoint_map[index].begin(),endpoint_map[index].end());
        endpoint_map[index] = std::vector<uint32_t>(sorted_list.begin(),sorted_list.end());
    });
    // 0: track1 beg
    // 1: track1 end
    // 2: track2 beg
    // 3: track2 end
    const int compare_pair[4][2] = {{0,2},{0,3},{1,2},{1,3}}; // for checking which end to connect
    for (size_t pos = 0;pos < geo.size();++pos)
        if(endpoint_map[pos].size() >= 2)
        {
            std::vector<uint32_t>& track_list = endpoint_map[pos];
            std::map<float,std::tuple<uint32_t,uint32_t,uint32_t> > merge_list;
            for(uint32_t i = 0;i < track_list.size();++i)
                for(uint32_t j = i+1;j < track_list.size();++j)
                {
                    uint32_t t1 = track_list[i];
                    uint32_t t2 = track_list[j];
                    if(tract_data[t1].size() <= 6 || tract_data[t2].size() <= 6)
                        continue;
                    tipl::vector<3,float> end[4],dir[4];
                    end[0] = &tract_data[t1][0];  // 0: track1 beg
                    end[1] = &tract_data[t1][tract_data[t1].size()-3]; // 1: track1 end
                    end[2] = &tract_data[t2][0];  // 2: track2 beg
                    end[3] = &tract_data[t2][tract_data[t2].size()-3]; // 3: track2 end
                    dir[0] = &tract_data[t1][3];
                    dir[1] = &tract_data[t1][tract_data[t1].size()-6];
                    dir[2] = &tract_data[t2][3];
                    dir[3] = &tract_data[t2][tract_data[t2].size()-6];
                    for(uint32_t k = 0;k < 4;++k)
                    {
                        dir[k] -= end[k];
                        dir[k].normalize();
                    }
                    for(uint32_t k = 0;k < 4;++k)
                    {
                        int p1 = compare_pair[k][0];
                        int p2 = compare_pair[k][1];
                        if(std::fabs(end[p1][0]-end[p2][0]) > distance)
                            continue;
                        tipl::vector<3> dis_end = end[p1]-end[p2];
                        float dis = float(dis_end.length());
                        if(dis > distance)
                            continue;
                        float angle = -(dir[p1]*dir[p2]);
                        if(angle < angular_threshold)
                            continue;
                        dis_end.normalize();
                        float angle1 = dis_end*dir[p1];
                        if(angle1 < angular_threshold)
                            continue;
                        float angle2 = -dis_end*dir[p2];
                        if(angle2 < angular_threshold)
                            continue;
                        merge_list[dis*angle*angle1*angle2] = std::make_tuple(i,j,k);
                        break;
                    }
                }
            std::vector<char> merged(track_list.size());
            for(auto& iter : merge_list)
            {
                uint32_t i = std::get<0>(iter.second);
                uint32_t j = std::get<1>(iter.second);
                uint32_t k = std::get<2>(iter.second);
                if(merged[i] || merged[j])
                    continue;
                merged[i] = 1;
                merged[j] = 1;
                uint32_t t1 = track_list[i];
                uint32_t t2 = track_list[j];
                // reverse track
                if(k == 1)// k = 1: track1 beg connects tract2 end
                {
                    tract_data[t2].insert(tract_data[t2].end(),tract_data[t1].begin(),tract_data[t1].end());
                    tract_data[t1].clear();
                    has_merged = true;
                    continue;
                }

                if(k == 0 || k == 3)
                {
                    auto rt = ((k == 0) ? t1 : t2);
                    float* beg1 = &tract_data[rt][0];
                    float* end1 = &tract_data[rt][tract_data[rt].size()-3];
                    while(beg1 < end1)
                    {
                        std::swap(beg1[0],end1[0]);
                        std::swap(beg1[1],end1[1]);
                        std::swap(beg1[2],end1[2]);
                        beg1 += 3;
                        end1 -= 3;
                    }
                }
                // k = 0: track1 beg connects tract2 beg (track1 reversed)
                // k = 2: track1 end connects tract2 beg
                // k = 3: track1 end connects tract2 end (track2 reversed)
                tract_data[t1].insert(tract_data[t1].end(),tract_data[t2].begin(),tract_data[t2].end());
                tract_data[t2].clear();
                has_merged = true;
            }
        }
    erase_empty();
    return has_merged;
}
//---------------------------------------------------------------------------
bool TractModel::cull(float select_angle,
                      const std::vector<tipl::vector<3,float> >& dirs,
                      const tipl::vector<3,float>& from_pos,
                      bool delete_track)
{
    std::vector<unsigned int> selected;
    select(select_angle,dirs,from_pos,selected);
    std::vector<unsigned int> tracts_to_delete;
    tracts_to_delete.reserve(100 + (selected.size() >> 4));
    for (unsigned int index = 0;index < selected.size();++index)
        if (!((selected[index] > 0) ^ delete_track))
            tracts_to_delete.push_back(index);
    return delete_tracts(tracts_to_delete);
}
//---------------------------------------------------------------------------
bool TractModel::paint(float select_angle,
                       const std::vector<tipl::vector<3,float> >& dirs,
                       const tipl::vector<3,float>& from_pos,
                       unsigned int color)
{
    std::vector<unsigned int> selected;
    select(select_angle,dirs,from_pos,selected);
    if(selected.empty())
        return false;
    for (unsigned int index = 0;index < selected.size();++index)
        if (selected[index] > 0)
            tract_color[index] = color;
    return true;
}

//---------------------------------------------------------------------------
void TractModel::cut_by_mask(const char*)
{
    /*
    std::ifstream in(file_name,std::ios::in);
    if(!in)
        return;
    std::set<tipl::vector<3,short> > mask(
                  (std::istream_iterator<tipl::vector<3,short> > (in)),
                  (std::istream_iterator<tipl::vector<3,short> > ()));
    std::vector<std::vector<float> > new_data;
    for (unsigned int index = 0;prog(index,tract_data.size());++index)
    {
        bool on = false;
        std::vector<float>::const_iterator iter = tract_data[index].begin();
        std::vector<float>::const_iterator end = tract_data[index].end();
        for (;iter < end;iter += 3)
        {
            tipl::vector<3,short> p(std::round(iter[0]),
                                  std::round(iter[1]),std::round(iter[2]));

            if (mask.find(p) == mask.end())
            {
                if (on)
                {
                    on = false;
                    if (new_data.back().size() == 3)
                        new_data.pop_back();
                }
                continue;
            }
            else
            {
                if (!on)
                    new_data.push_back(std::vector<float>());
                new_data.back().push_back(iter[0]);
                new_data.back().push_back(iter[1]);
                new_data.back().push_back(iter[2]);
                on = true;
            }
        }
    }
    tract_data.swap(new_data);*/
}
//---------------------------------------------------------------------------
void TractModel::flip(char dim)
{
    auto w = geo[dim];
    tipl::par_for (tract_data.size(),[&](size_t index)
    {
        auto& tract = tract_data[index];
        for(size_t j = dim;j < tract.size();j+=3)
            tract[j] = w-tract[j];
    });
}

bool TractModel::trim(void)
{
    /*
    std::vector<char> continuous(tract_data.size());
        float epsilon = 2.0f;
        tipl::adaptive_par_for(tract_data.size(),[&](int i)
        {
            if(tract_data[i].empty())
                return;
            const float* t1 = &tract_data[i][0];
            const float* t1_end = &tract_data[i][tract_data[i].size()-3];
            for(int j = 1;j < tract_data.size();++j)
            {
                if(i == j)
                    continue;
                if(tract_data[j].empty())
                    continue;
                const float* t2 = &tract_data[j][0];
                const float* t2_end = &tract_data[j][tract_data[j].size()-3];
                float d1 = std::fabs(t1[0]-t2[0])+std::fabs(t1[1]-t2[1])+std::fabs(t1[2]-t2[2]);
                float d2 = std::fabs(t1[0]-t2_end[0])+std::fabs(t1[1]-t2_end[1])+std::fabs(t1[2]-t2_end[2]);
                if(d1 > epsilon && d2 > epsilon)
                    continue;
                float d3 = std::fabs(t1_end[0]-t2[0])+std::fabs(t1_end[1]-t2[1])+std::fabs(t1_end[2]-t2[2]);
                float d4 = std::fabs(t1_end[0]-t2_end[0])+std::fabs(t1_end[1]-t2_end[1])+std::fabs(t1_end[2]-t2_end[2]);
                if(d1 <= epsilon && d4 > epsilon)
                    continue;
                if(d2 <= epsilon && d3 > epsilon)
                    continue;

                unsigned int length1 = tract_data[i].size();
                unsigned int length2 = tract_data[j].size();

                bool con = true;
                for(int m = 0;m < length1;m += 3)
                {
                    bool has_c = false;
                    for(int n = 0;n < length2;n += 3)
                        if(t1[m]-t2[n] < epsilon &&
                           t1[m+1]-t2[n+1] < epsilon &&
                           t1[m+2]-t2[n+2] < epsilon)
                        {
                            has_c = true;
                            break;
                        }
                    if(!has_c)
                    {
                        con = false;
                        break;
                    }
                }
                if(!con)
                    continue;
                for(int n = 0;n < length2;n += 3)
                {
                    bool has_c = false;
                    for(int m = 0;m < length1;m += 3)
                        if(t1[m]-t2[n] < epsilon &&
                           t1[m+1]-t2[n+1] < epsilon &&
                           t1[m+2]-t2[n+2] < epsilon)
                        {
                            has_c = true;
                            break;
                        }
                    if(!has_c)
                    {
                        con = false;
                        break;
                    }
                }
                if(con)
                {
                    ++continuous[i];
                    ++continuous[j];
                }
            }
        });
        std::vector<unsigned int> tracts_to_delete;
        for (unsigned int index = 0;index < continuous.size();++index)
            if (continuous[index] < 2)
                tracts_to_delete.push_back(index);
        if(tracts_to_delete.empty())
            return false;
        delete_tracts(tracts_to_delete);
    */

    tipl::image<3,unsigned int> label(geo),label2(geo),label3(geo);
    int total_track_number = tract_data.size();
    int no_fiber_label = total_track_number;
    int have_multiple_fiber_label = total_track_number+1;

    int width = label.width();
    int height = label.height();
    int depth = label.depth();
    int wh = width*height;

    std::fill(label.begin(),label.end(),no_fiber_label);
    std::fill(label2.begin(),label2.end(),no_fiber_label);
    std::fill(label3.begin(),label3.end(),no_fiber_label);

    int shift[8] = {0,1,width,wh,1+width,1+wh,width+wh,1+width+wh};
    tipl::adaptive_par_for(total_track_number,[&](int index)
    {
        const float* ptr = &*tract_data[index].begin();
        const float* end = ptr + tract_data[index].size();
        for (;ptr < end;ptr += 3)
        {
            int x = *ptr;
            if (x <= 0 || x >= width)
                continue;
            int y = *(ptr+1);
            if (y <= 0 || y >= height)
                continue;
            int z = *(ptr+2);
            if (z <= 0 || z >= depth)
                continue;
            for(unsigned int i = 0;i < 8;++i)
            {
                unsigned int pixel_index = z*wh+y*width+x+shift[i];
                if (pixel_index >= label.size())
                    continue;
                unsigned int cur_label = label[pixel_index];
                if (cur_label == have_multiple_fiber_label || cur_label == index)
                    continue;
                if (cur_label == no_fiber_label)
                {
                    label[pixel_index] = index;
                    continue;
                }

                unsigned int cur_label2 = label2[pixel_index];
                if (cur_label2 == index)
                    continue;
                if (cur_label2 == no_fiber_label)
                {
                    label2[pixel_index] = index;
                    continue;
                }

                unsigned int cur_label3 = label3[pixel_index];
                if (cur_label3 == index)
                    continue;
                if (cur_label3 == no_fiber_label)
                {
                    label3[pixel_index] = index;
                    continue;
                }


                label[pixel_index] = have_multiple_fiber_label;
                label2[pixel_index] = have_multiple_fiber_label;
                label3[pixel_index] = have_multiple_fiber_label;
            }
        }
    });

    std::set<unsigned int> tracts_to_delete;
    for (unsigned int index = 0;index < label.size();++index)
    {
        if (label[index] < total_track_number)
            tracts_to_delete.insert(label[index]);
        if (label2[index] < total_track_number)
            tracts_to_delete.insert(label2[index]);
        if (label3[index] < total_track_number)
            tracts_to_delete.insert(label3[index]);
    }

    return delete_tracts(std::vector<unsigned int>(tracts_to_delete.begin(),tracts_to_delete.end()));
}
//---------------------------------------------------------------------------
void TractModel::trim(unsigned int tip_iteration)
{
    if(!tip_iteration)
        return;
    auto last = get_deleted_track_count();
    for(size_t i = 0;i < tip_iteration && get_visible_track_count() && trim();++i)
                ;
    tipl::out() << get_deleted_track_count()-last << " tracts removed by pruning.";
    tipl::out() << "tract count after pruning: " << get_visible_track_count();
}
//---------------------------------------------------------------------------
void TractModel::clear_deleted(void)
{
    deleted_count.clear();
    deleted_tract_data.clear();
    deleted_tract_color.clear();
    deleted_tract_tag.clear();
    redo_size.clear();
}

bool TractModel::undo(void)
{
    if (deleted_count.empty())
        return false;
    redo_size.push_back(std::make_pair((unsigned int)tract_data.size(),deleted_count.back()));
    for (unsigned int index = 0;index < deleted_count.back();++index)
    {
        tract_data.push_back(std::move(deleted_tract_data.back()));
        tract_color.push_back(deleted_tract_color.back());
        tract_tag.push_back(deleted_tract_tag.back());
        deleted_tract_data.pop_back();
        deleted_tract_color.pop_back();
        deleted_tract_tag.pop_back();
    }
    // handle the cut situation
    if(is_cut.back())
    {
        for(int i = 0;i < tract_tag.size();++i)
            if(tract_tag[i] == is_cut.back())
                tract_data[i].clear();
        erase_empty();
    }
    is_cut.pop_back();
    deleted_count.pop_back();
    saved = false;
    return true;
}


//---------------------------------------------------------------------------
bool TractModel::redo(void)
{
    if(redo_size.empty())
        return false;
    std::vector<unsigned int> redo_tracts(redo_size.back().second);
    for (unsigned int index = 0;index < redo_tracts.size();++index)
        redo_tracts[index] = redo_size.back().first + index;
    redo_size.pop_back();
    // keep redo because delete tracts will clear redo
    std::vector<std::pair<unsigned int,unsigned int> > keep_redo;
    keep_redo.swap(redo_size);
    delete_tracts(redo_tracts);
    keep_redo.swap(redo_size);
    return true;
}
//---------------------------------------------------------------------------
void TractModel::add_tracts(std::vector<std::vector<float> >& new_tracks)
{
    add_tracts(new_tracks,tract_color.empty() ? default_tract_color : tipl::rgb(tract_color.back()));
}
//---------------------------------------------------------------------------
void TractModel::add_tracts(std::vector<std::vector<float> >& new_tract,tipl::rgb color)
{
    tract_data.reserve(tract_data.size()+new_tract.size());

    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        if (new_tract[index].empty())
            continue;
        tract_data.push_back(std::move(new_tract[index]));
        tract_color.push_back(color);
        tract_tag.push_back(0);
    }
    saved = false;
}

void TractModel::add_tracts(std::vector<std::vector<float> >& new_tract, unsigned int length_threshold,tipl::rgb color)
{
    tract_data.reserve(tract_data.size()+new_tract.size()/2.0);
    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        if (new_tract[index].size()/3-1 < length_threshold)
            continue;
        tract_data.push_back(std::move(new_tract[index]));
        tract_color.push_back(color);
        tract_tag.push_back(0);
    }
    saved = false;
}
//---------------------------------------------------------------------------
void TractModel::get_density_map(tipl::image<3,unsigned int>& mapping,
                                 const tipl::matrix<4,4>& to_t1t2,bool endpoint)
{
    tipl::shape<3> s = mapping.shape();
    std::vector<tipl::image<3,unsigned int> > maps(std::thread::hardware_concurrency());
    tipl::adaptive_par_for<tipl::sequential_with_id>(tract_data.size(),[&](unsigned int i,unsigned int id)
    {
        auto& m = maps[id];
        if(m.empty())
            m.resize(s);
        std::unordered_set<size_t> point_set;
        for (unsigned int j = 0;j < tract_data[i].size();j+=3)
        {
            if(j && endpoint)
                j = uint32_t(tract_data[i].size())-3;
            tipl::vector<3,float> pos(tract_data[i].begin()+j);
            pos.to(to_t1t2);
            pos.round();
            tipl::vector<3,int> ipos(pos);
            if (s.is_valid(ipos))
                point_set.insert(tipl::voxel2index(ipos.begin(),mapping.shape()));
        }
        for(auto pos : point_set)
            ++m[pos];
    });

    while(maps.back().empty() && !maps.empty())
        maps.pop_back();

    tipl::adaptive_par_for(s.size(),[&](unsigned int i)
    {
        for(const auto& each : maps)
            mapping[i] += each[i];
    });

}
//---------------------------------------------------------------------------
void TractModel::get_density_map(
        tipl::image<3,tipl::rgb>& mapping,
        const tipl::matrix<4,4>& to_t1t2,bool endpoint)
{
    tipl::shape<3> geo = mapping.shape();
    tipl::image<3> map_r(geo),map_g(geo),map_b(geo);
    std::cout << "aggregating tracts to voxels" << std::endl;
    tipl::par_for (tract_data.size(),[&](unsigned int i)
    {
        const float* buf = &*tract_data[i].begin();
        for (unsigned int j = 3;j < tract_data[i].size();j+=3)
        {
            if(j > 3 && endpoint)
                j = uint32_t(tract_data[i].size()-3);
            tipl::vector<3,float>  pos(buf+j),dir(buf+j-3);
            pos.to(to_t1t2);
            dir.to(to_t1t2);
            dir -= pos;
            dir.normalize();
            pos.round();
            tipl::vector<3,int> ipos(pos);
            if (!geo.is_valid(ipos))
                continue;
            size_t ptr = tipl::voxel2index(ipos.begin(),mapping.shape());
            map_r[ptr] += std::fabs(dir[0]);
            map_g[ptr] += std::fabs(dir[1]);
            map_b[ptr] += std::fabs(dir[2]);
        }
    });
    std::cout << "generating rgb maps" << std::endl;
    float max_value = 0.0f;
    for(size_t index = 0;index < mapping.size();++index)
        max_value = std::max<float>(max_value,map_r[index]+map_g[index]+map_b[index]);

    tipl::adaptive_par_for(mapping.size(),[&](size_t index)
    {
        float sum = map_r[index]+map_g[index]+map_b[index];
        if(sum == 0.0f)
            return;
        tipl::vector<3> v(map_r[index],map_g[index],map_b[index]);
        sum = v.normalize();
        v*=255.0f*std::log(200.0f*sum/max_value+1)/2.303f;
        mapping[index] = tipl::rgb(uint8_t(std::min<float>(255,v[0])),uint8_t(std::min<float>(255,v[1])),uint8_t(std::min<float>(255,v[2])));
    });
}
bool TractModel::export_end_pdi(
                       const char* file_name,
                       const std::vector<std::shared_ptr<TractModel> >& tract_models,float end_distance)
{
    if(tract_models.empty())
        return false;
    auto dim = tract_models.front()->geo;
    auto vs = tract_models.front()->vs;
    auto trans_to_mni = tract_models.front()->trans_to_mni;
    auto is_mni = tract_models.front()->is_mni;
    tipl::image<3,uint32_t> p1_map(dim),p2_map(dim);
    for(size_t index = 0;index < tract_models.size();++index)
    {
        std::vector<tipl::vector<3,short> > p1,p2;
        tract_models[index]->to_end_point_voxels(p1,p2,tipl::identity_matrix(),end_distance);
        tipl::adaptive_par_for(p1.size(),[&](size_t j)
        {
            tipl::vector<3,short> p = p1[j];
            if(dim.is_valid(p))
                p1_map[tipl::pixel_index<3>(p[0],p[1],p[2],dim).index()]++;
        });
        tipl::adaptive_par_for(p2.size(),[&](size_t j)
        {
            tipl::vector<3,short> p = p2[j];
            if(dim.is_valid(p))
                p2_map[tipl::pixel_index<3>(p[0],p[1],p[2],dim).index()]++;
        });
    }
    tipl::image<3> pdi1(p1_map),pdi2(p2_map);
    if(tract_models.size() > 1)
    {
        tipl::multiply_constant(pdi1,1.0f/float(tract_models.size()));
        tipl::multiply_constant(pdi2,1.0f/float(tract_models.size()));
    }
    QString f1 = QFileInfo(file_name).absolutePath() + "/"+ QFileInfo(file_name).baseName() + "_1.nii.gz";
    QString f2 = QFileInfo(file_name).absolutePath() + "/"+ QFileInfo(file_name).baseName() + "_2.nii.gz";
    return tipl::io::gz_nifti::save_to_file(f1.toStdString().c_str(),pdi1,vs,trans_to_mni,is_mni) &&
           tipl::io::gz_nifti::save_to_file(f2.toStdString().c_str(),pdi2,vs,trans_to_mni,is_mni);
}
bool TractModel::export_pdi(const char* file_name,
                            const std::vector<std::shared_ptr<TractModel> >& tract_models)
{
    if(tract_models.empty())
        return false;
    auto dim = tract_models.front()->geo;
    auto vs = tract_models.front()->vs;
    auto trans_to_mni = tract_models.front()->trans_to_mni;
    auto is_mni = tract_models.front()->is_mni;
    tipl::image<3,uint32_t> accumulate_map(dim);
    for(size_t index = 0;index < tract_models.size();++index)
    {
        std::vector<tipl::vector<3,short> > points;
        tract_models[index]->to_voxel(points);
        tipl::adaptive_par_for(points.size(),[&](size_t j)
        {
            tipl::vector<3,short> p = points[j];
            if(dim.is_valid(p))
                accumulate_map[tipl::pixel_index<3>(p[0],p[1],p[2],dim).index()]++;
        });
    }
    tipl::image<3> pdi(accumulate_map);
    if(tract_models.size() > 1)
        tipl::multiply_constant(pdi,1.0f/float(tract_models.size()));
    return tipl::io::gz_nifti::save_to_file(file_name,pdi,vs,trans_to_mni,is_mni);
}
bool TractModel::export_tdi(const char* filename,
                  std::vector<std::shared_ptr<TractModel> > tract_models,
                  tipl::shape<3>& dim,
                  tipl::vector<3,float> vs,
                  const tipl::matrix<4,4>& trans_to_mni,
                  const tipl::matrix<4,4>& to_t1t2,
                  bool color,bool end_point)
{
    if(!QFileInfo(filename).fileName().endsWith(".nii") &&
       !QFileInfo(filename).fileName().endsWith(".nii.gz"))
        return false;
    if(color)
    {
        tipl::image<3,tipl::rgb> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
            tract_models[index]->get_density_map(tdi,to_t1t2,end_point);
        return tipl::io::gz_nifti::save_to_file(filename,tdi,vs,trans_to_mni,tract_models[0]->is_mni);
    }
    else
    {
        tipl::image<3,unsigned int> tdi(dim);
        for(unsigned int index = 0;index < tract_models.size();++index)
            tract_models[index]->get_density_map(tdi,to_t1t2,end_point);
        return tipl::io::gz_nifti::save_to_file(filename,tdi,vs,trans_to_mni,tract_models[0]->is_mni);
    }
}
void TractModel::to_voxel(std::vector<tipl::vector<3,short> >& points,const tipl::matrix<4,4>& trans,int id)
{
    bool need_trans = (trans != tipl::identity_matrix());
    float voxel_length_2 = 0.5f;

    if(need_trans)
    {
        tipl::transformation_matrix<float> m(tipl::matrix<4,4>(tipl::inverse(trans)));
        tipl::vector<3> L(0.5f,0.0f,0.0f); // in the point space.
        L.rotate(m.sr);
        voxel_length_2 = float(L.length());
    }

    std::vector<std::set<tipl::vector<3,short> > > pass_map(tipl::max_thread_count);
    tipl::par_for<tipl::sequential_with_id>(tract_data.size(),[&](size_t i,size_t thread)
    {
        if(tract_data[i].size() < 6)
            return;
        if(id != -1 && int(tract_cluster[i]) != id)
            return;
        float step_size = float((tipl::vector<3>(&tract_data[i][0])-tipl::vector<3>(&tract_data[i][3])).length());
        for (size_t j = 3;j < tract_data[i].size();j += 3)
        {
            tipl::vector<3> dir(&tract_data[i][j]);
            dir -= tipl::vector<3>(&tract_data[i][j-3]);
            for(float d = 0.0;d < step_size;d += voxel_length_2)
            {
                tipl::vector<3> cur(dir);
                cur *= d/step_size;
                cur += tipl::vector<3>(&tract_data[i][j-3]);
                if(need_trans)
                    cur.to(trans);
                cur.round();
                pass_map[thread].insert(tipl::vector<3,short>(cur));
            }

        }
    });
    for(size_t i = 1;i < pass_map.size();++i)
    {
        std::set<tipl::vector<3,short> > new_set;
        std::merge(pass_map[0].begin(),pass_map[0].end(),
                   pass_map[i].begin(),pass_map[i].end(),
                    std::inserter(new_set,std::begin(new_set)));
        new_set.swap(pass_map[0]);
    }
    points = std::vector<tipl::vector<3,short> >(pass_map[0].begin(),pass_map[0].end());
}

tipl::vector<3> get_tract_dir(const std::vector<std::vector<float> >& tract_data,
                   std::vector<char>& dir)
{
    // estimate the average mid-point direction
    tipl::vector<3> total_dis;
    for(size_t i = 0;i < tract_data.size();++i)
    {
        if(tract_data[i].size() < 6)
            continue;
        uint32_t mid_pos = uint32_t(tract_data[i].size()/6)*3;
        tipl::vector<3> dis(&tract_data[i][mid_pos]);
        dis -= tipl::vector<3>(&tract_data[i][mid_pos+3]);
        if(dis*total_dis < 0.0f)
            total_dis -= dis;
        else
            total_dis += dis;
    }
    // categorize endpoints using the mid point direction
    total_dis.normalize();
    dir.resize(tract_data.size());
    tipl::adaptive_par_for(tract_data.size(),[&](size_t i)
    {
        if(tract_data[i].size() < 6)
            return;
        uint32_t mid_pos = uint32_t(tract_data[i].size()/6)*3;
        uint32_t q1_pos = uint32_t(tract_data[i].size()/12)*3;
        uint32_t q3_pos = uint32_t(tract_data[i].size()/4)*3;
        tipl::vector<3> mid_dis(&tract_data[i][mid_pos]);
        mid_dis -= tipl::vector<3>(&tract_data[i][mid_pos+3]);
        tipl::vector<3> q1_dis(&tract_data[i][q1_pos]);
        q1_dis -= tipl::vector<3>(&tract_data[i][q1_pos+3]);
        tipl::vector<3> q3_dis(&tract_data[i][q3_pos]);
        q3_dis -= tipl::vector<3>(&tract_data[i][q3_pos+3]);
        mid_dis += q1_dis;
        mid_dis += q3_dis;
        if(total_dis*mid_dis > 0.0f)
            dir[i] = 1;
    });
    return total_dis;
}

void check_order(tipl::shape<3> geo,
                 std::vector<tipl::vector<3,short> >& s1,
                 std::vector<tipl::vector<3,short> >& s2)
{
    // use end surface central point to determine
    // end surface 1 is located at larger axis value
    tipl::vector<3,float> sum_s1 = std::accumulate(s1.begin(),s1.end(),tipl::vector<3,float>(0,0,0));
    tipl::vector<3,float> sum_s2 = std::accumulate(s2.begin(),s2.end(),tipl::vector<3,float>(0,0,0));
    sum_s1 -= sum_s2;
    sum_s1[0] *= geo[0];
    sum_s1[1] *= geo[1];
    sum_s1[2] *= geo[2];
    auto dir = sum_s1;
    dir.abs();
    auto max_sum_dim = std::max_element(dir.begin(),dir.end())-dir.begin();
    if(sum_s1[uint32_t(max_sum_dim)] < 0.0f)
        s1.swap(s2);

}
inline tipl::vector<3> get_rounded_voxel(const float* ptr,bool need_trans,const tipl::matrix<4,4>& trans)
{
    tipl::vector<3> p(ptr);
    if(need_trans)
        p.to(trans);
    p.round();
    return p;
}
void TractModel::to_end_point_voxels(std::vector<tipl::vector<3,short> >& points1,
                               std::vector<tipl::vector<3,short> >& points2,const tipl::matrix<4,4>& trans)
{
    bool need_trans = (trans != tipl::identity_matrix());
    std::vector<char> dir;
    get_tract_dir(tract_data,dir);

    // categorize endpoints using the mid point direction
    std::vector<tipl::vector<3,short> > s1,s2;
    for(size_t i = 0;i < tract_data.size();++i)
    {
        if(tract_data[i].size() < 6)
            continue;
        tipl::vector<3,short> p1(get_rounded_voxel(&tract_data[i][0],need_trans,trans));
        tipl::vector<3,short> p2(get_rounded_voxel(&tract_data[i][tract_data[i].size()-3],need_trans,trans));
        if(dir[i])
        {
            s1.push_back(p1);
            s2.push_back(p2);
        }
        else
        {
            s2.push_back(p1);
            s1.push_back(p2);
        }
    }

    check_order(geo,s1,s2);

    std::sort(s1.begin(),s1.end());
    std::sort(s2.begin(),s2.end());
    std::unique_copy(s1.begin(),s1.end(),std::back_inserter(points1));
    std::unique_copy(s2.begin(),s2.end(),std::back_inserter(points2));
}

void TractModel::to_end_point_voxels(std::vector<tipl::vector<3,short> >& points1,
                                     std::vector<tipl::vector<3,short> >& points2,
                                     const tipl::matrix<4,4>& trans,float end_dis)
{
    bool need_trans = (trans != tipl::identity_matrix());
    std::vector<char> dir;
    get_tract_dir(tract_data,dir);

    // categorize endpoints using the mid point direction
    std::vector<tipl::vector<3,short> > s1,s2;
    for(size_t i = 0;i < tract_data.size();++i)
    {
        if(tract_data[i].size() < 6)
            continue;

        std::vector<tipl::vector<3,short> > p1,p2;
        // get end points at first end
        {
            size_t j = 0;
            for(float dis = 0.0f;dis < end_dis;)
            {
                p1.push_back(get_rounded_voxel(&tract_data[i][j],need_trans,trans));
                j += 3;
                if(j >= tract_data[i].size())
                    break;
                tipl::vector<3> u(&tract_data[i][j]),v(&tract_data[i][j-3]);
                u -= v;
                dis += float(u.length());
            }
        }
        // get end points at the second end
        {
            size_t j = tract_data[i].size()-3;
            for(float dis = 0.0f;dis < end_dis;)
            {
                p2.push_back(get_rounded_voxel(&tract_data[i][j],need_trans,trans));
                if(j < 3)
                    break;
                j -= 3;
                tipl::vector<3> u(&tract_data[i][j]),v(&tract_data[i][j+3]);
                u -= v;
                dis += float(u.length());
            }
        }

        if(dir[i])
        {
            s1.insert(s1.end(),p1.begin(),p1.end());
            s2.insert(s2.end(),p2.begin(),p2.end());
        }
        else
        {
            s2.insert(s2.end(),p1.begin(),p1.end());
            s1.insert(s1.end(),p2.begin(),p2.end());
        }
    }

    check_order(geo,s1,s2);

    std::sort(s1.begin(),s1.end());
    std::sort(s2.begin(),s2.end());
    std::unique_copy(s1.begin(),s1.end(),std::back_inserter(points1));
    std::unique_copy(s2.begin(),s2.end(),std::back_inserter(points2));
}

float TractModel::get_tract_length_in_mm(unsigned int index) const
{
    double length = 0.0;
    for (unsigned int j = 3;j < tract_data[index].size();j += 3)
    {
        length += tipl::vector<3,float>(
            vs[0]*(tract_data[index][j]-tract_data[index][j-3]),
            vs[1]*(tract_data[index][j+1]-tract_data[index][j-2]),
            vs[2]*(tract_data[index][j+2]-tract_data[index][j-1])).length();

    }
    return float(length);
}
void TractModel::get_quantitative_info(std::shared_ptr<fib_data> handle,std::string& result)
{
    if(tract_data.empty())
    {
        result = "number of tracts\t0";
        return;
    }
    std::ostringstream out;
    std::vector<std::string> titles;
    std::vector<float> data;
    {
        const float resolution_ratio = 2.0f;
        tipl::matrix<4,4> resolution_trans((tipl::identity_matrix()));
        resolution_trans[0] = resolution_trans[5] = resolution_trans[10] = 2.0f;
        float voxel_volume = vs[0]*vs[1]*vs[2];
        const float PI = 3.14159265358979323846f;
        float tract_volume, branch_volume1, branch_volume2, tract_area, tract_length, span, curl, bundle_diameter;
        tipl::image<3,unsigned char> volume;


        titles.push_back("number of tracts");
        data.push_back(tract_data.size());

        // mean length
        {
            std::vector<float> length_each(tract_data.size());
            std::vector<float> end_dis_each(tract_data.size());
            tipl::par_for (tract_data.size(),[&](unsigned int i)
            {
                length_each[i] = get_tract_length_in_mm(i);
                end_dis_each[i] = float((tipl::vector<3,float>(&tract_data[i][0])-
                                    tipl::vector<3,float>(&tract_data[i][tract_data[i].size()-3])).length());
            });
            float sum_length = tipl::sum(length_each);
            float sum_end_dis = tipl::sum(end_dis_each);

            tract_length = sum_length/float(tract_data.size());
            span = sum_end_dis/float(tract_data.size());
            curl = sum_length/sum_end_dis;

        }


        {
            std::vector<tipl::vector<3,short> > points;
            to_voxel(points,resolution_trans);
            tract_volume = points.size()*voxel_volume/resolution_ratio/resolution_ratio/resolution_ratio;
            bundle_diameter = 2.0f*float(std::sqrt(tract_volume/tract_length/PI));

            // now next convert point list to volume
            tipl::vector<3,short> max_value(points[0]), min_value(points[0]);
            tipl::bounding_box(points,max_value,min_value);

            max_value += tipl::vector<3,short>(1, 1, 1);
            min_value -= tipl::vector<3,short>(1, 1, 1);

            tipl::shape<3> geo(max_value[0] - min_value[0],
                                  max_value[1] - min_value[1],
                                  max_value[2] - min_value[2]);

            volume.resize(geo);
            tipl::adaptive_par_for(points.size(),[&](unsigned int index)
            {
                tipl::vector<3,short> point(points[index]);
                point -= min_value;
                volume[tipl::pixel_index<3>(point[0], point[1], point[2],geo).index()] = 1;
            });
        }
        // surface area
        {
            tipl::image<3,unsigned char> edge;
            tipl::morphology::edge(volume,edge);
            size_t num = 0;
            for(size_t i = 0;i < edge.size();++i)
                if(edge[i])
                    ++num;
            tract_area = float(num)*vs[0]*vs[1]/resolution_ratio/resolution_ratio;

        }
        // end points
        float end_area1,end_area2,radius1,radius2;
        {
            std::vector<tipl::vector<3,short> > endpoint1,endpoint2;
            to_end_point_voxels(endpoint1,endpoint2,resolution_trans);
            // end point surface 1 and 2
            end_area1 = float(endpoint1.size())*vs[0]*vs[1]/resolution_ratio/resolution_ratio;
            end_area2 = float(endpoint2.size())*vs[0]*vs[1]/resolution_ratio/resolution_ratio;

            // radius
            auto c1 = std::accumulate(endpoint1.begin(),endpoint1.end(),tipl::vector<3,float>(0.0f,0.0f,0.0f))/float(endpoint1.size());
            float mean_dis1 = 0.0f;
            for(size_t i = 0;i < endpoint1.size();++i)
            {
                auto dis(c1);
                dis -= endpoint1[i];
                mean_dis1 += float(dis.length());
            }
            auto c2 = std::accumulate(endpoint2.begin(),endpoint2.end(),tipl::vector<3,float>(0.0f,0.0f,0.0f))/float(endpoint2.size());
            float mean_dis2 = 0.0f;
            for(size_t i = 0;i < endpoint2.size();++i)
            {
                auto dis(c2);
                dis -= endpoint2[i];
                mean_dis2 += float(dis.length());
            }
            mean_dis1 /= float(endpoint1.size());
            mean_dis2 /= float(endpoint2.size());
            // the average distance of a point in a circle to the center is 2R/3, where R is the radius
            radius1 = 1.5f*mean_dis1/resolution_ratio;
            radius2 = 1.5f*mean_dis2/resolution_ratio;

        }
        // mid_portion as the trunk
        {
            std::vector<tipl::vector<3,short> > branch1,branch2;
            cut_end_portion(0.0f,0.25f);
            to_voxel(branch1,resolution_trans);
            undo();
            cut_end_portion(0.75f,1.0f);
            to_voxel(branch2,resolution_trans);
            undo();

            branch_volume1 = branch1.size()*voxel_volume/resolution_ratio/resolution_ratio/resolution_ratio;
            branch_volume2 = branch2.size()*voxel_volume/resolution_ratio/resolution_ratio/resolution_ratio;

        }
        data.push_back(tract_length);   titles.push_back("mean length(mm)");
        data.push_back(span);           titles.push_back("span(mm)");
        data.push_back(curl);           titles.push_back("curl");
        data.push_back(tract_length/bundle_diameter);titles.push_back("elongation");

        data.push_back(tract_volume);   titles.push_back("total volume(mm^3)");
        data.push_back(branch_volume1);   titles.push_back("1st quarter volume(mm^3)");
        data.push_back(tract_volume-branch_volume1-branch_volume2);   titles.push_back("2nd and 3rd quarter volume(mm^3)");
        data.push_back(branch_volume2);   titles.push_back("4th quarter volume(mm^3)");

        data.push_back(tract_area);     titles.push_back("total surface area(mm^2)");
        data.push_back(radius1+radius2);titles.push_back("total radius of end regions(mm)");
        data.push_back(end_area1+end_area2);      titles.push_back("total area of end regions(mm^2)");
        data.push_back(float(tract_area/PI/bundle_diameter/tract_length));  titles.push_back("irregularity");

        data.push_back(end_area1);      titles.push_back("area of end region 1(mm^2)");
        data.push_back(radius1);        titles.push_back("radius of end region 1(mm)");
        data.push_back(branch_volume1);        titles.push_back("volume of end branches 1");

        data.push_back(end_area2);      titles.push_back("area of end region 2(mm^2)");
        data.push_back(radius2);        titles.push_back("radius of end region 2(mm)");
        data.push_back(branch_volume2);        titles.push_back("volume of end branches 2");
    }

    for(size_t data_index = 0;data_index < handle->slices.size();++data_index)
    {
        if(handle->slices[data_index]->optional())
            break;
        data.push_back(get_tracts_mean(handle,data_index));
        titles.push_back(handle->slices[data_index]->name);
    }


    for(unsigned int index = 0;index < data.size() && index < titles.size();++index)
        out << titles[index] << "\t" << data[index] << std::endl;

    if(handle->db.has_db()) // connectometry database
    {
        std::vector<const float*> old_index_data(handle->dir.index_data[0]);
        {
            for(unsigned int i = 0;i < handle->db.num_subjects;++i)
            {
                std::vector<std::vector<float> > fa_data;
                handle->db.get_subject_fa(i,fa_data);
                for(unsigned int j = 0;j < fa_data.size();++j)
                    handle->dir.index_data[0][j] = &fa_data[j][0];
                out << handle->db.subject_names[i] << " mean_" <<
                       handle->db.index_name << "\t" << get_tracts_mean(handle,0) << std::endl;
            }
        }
        handle->dir.index_data[0] = old_index_data;
    }
    result = out.str();
}

tipl::vector<3> TractModel::get_report(std::shared_ptr<fib_data> handle,
                            unsigned int profile_dir,float band_width,const std::string& index_name,
                            std::vector<float>& values,
                            std::vector<float>& data_profile,
                            std::vector<float>& data_ci1,
                            std::vector<float>& data_ci2)
{
    tipl::vector<3> avg_dir;
    if(tract_data.empty())
        return avg_dir;
    unsigned int profile_on_length = 0;// 1 :along tract 2: mean value
    if(profile_dir > 2)
    {
        profile_on_length = profile_dir-2;
        profile_dir = 0;
    }
    float detail = profile_on_length ? 1.0 : 2.0;
    size_t profile_width = size_t((geo[profile_dir]+1)*detail);


    std::vector<float> weighting(uint32_t(1.0f+band_width*3.0f));
    for(size_t index = 0;index < weighting.size();++index)
    {
        float x = index;
        weighting[index] = std::exp(-x*x/2.0f/band_width/band_width);
    }
    // along tract profile
    if(profile_on_length == 1)
        profile_width = 100;
    // mean value of each tract
    if(profile_on_length == 2)
        profile_width = tract_data.size();

    values.resize(profile_width);
    data_profile.resize(profile_width);

    {
        std::vector<std::vector<float> > data(get_tracts_data(handle,index_name));


        if(profile_on_length == 2)// list the mean fa value of each tract
        {
            data_profile.resize(data.size());
            for(unsigned int index = 0;index < data_profile.size();++index)
                data_profile[index] = float(tipl::mean(data[index].begin(),data[index].end()));
        }
        else
        // along tract profile
        {
            std::vector<char> dir;
            avg_dir = get_tract_dir(tract_data,dir);
            std::vector<std::set<float> > profile_upper_ci(profile_width),profile_lower_ci(profile_width);
            size_t ci_size = std::max<size_t>(1,size_t(float(data.size())*0.025f));

            for(size_t i = 0;i < data.size();++i)
            {
                std::vector<float> line_profile(profile_width);
                std::vector<float> line_profile_w(profile_width);
                if(profile_on_length == 1 && !dir[i])
                    std::reverse(data[i].begin(),data[i].end());
                for(size_t j = 0;j < data[i].size();++j)
                {
                    size_t pos = profile_on_length ?
                              size_t(j*profile_width/data[i].size()):
                              size_t(std::max<int>(0,int(std::round(tract_data[i][j + j + j + profile_dir]*detail))));
                    if(pos >= profile_width)
                        pos = profile_width-1;

                    for(size_t k = 0;k < weighting.size();++k)
                    {
                        float dw = data[i][j]*weighting[k];
                        float w = weighting[k];
                        if(pos > k && k != 0)
                        {
                            line_profile[pos-k] += dw;
                            line_profile_w[pos-k] += w;
                        }
                        if(pos+k < data_profile.size())
                        {
                            line_profile[pos+k] += dw;
                            line_profile_w[pos+k] += w;
                        }
                    }
                }
                for(unsigned int j = 0;j < line_profile.size();++j)
                {
                    float value = (line_profile_w[j] == 0.0f ? 0.0f : line_profile[j] / line_profile_w[j]);

                    if(profile_upper_ci[j].size() < ci_size || value < *profile_upper_ci[j].begin())
                        profile_upper_ci[j].insert(value);
                    if(profile_lower_ci[j].size() < ci_size || value > *std::prev(profile_lower_ci[j].end()))
                        profile_lower_ci[j].insert(value);

                    if(profile_upper_ci[j].size() > ci_size)
                        profile_upper_ci[j].erase(profile_upper_ci[j].begin());

                    if(profile_lower_ci[j].size() > ci_size)
                        profile_lower_ci[j].erase(std::prev(profile_lower_ci[j].end()));

                    data_profile[j] += value;
                }
            }
            data_ci1.resize(data_profile.size());
            data_ci2.resize(data_profile.size());
            for(unsigned int j = 0;j < data_profile.size();++j)
            {
                values[j] = float(j)/detail;
                data_profile[j] /= data.size();
                data_ci1[j] = *profile_upper_ci[j].begin();
                data_ci2[j] = *std::prev(profile_upper_ci[j].end());
            }
        }
    }
    return avg_dir;
}




template<class input_iterator,class output_iterator>
void gradient(input_iterator from,input_iterator to,output_iterator out)
{
    if(from == to)
        return;
    --to;
    if(from == to)
        return;
    *out = *(from+1);
    *out -= *(from);
    output_iterator last = out + (to-from);
    *last = *to;
    *last -= *(to-1);
    input_iterator pre_from = from;
    ++from;
    ++out;
    input_iterator next_from = from;
    ++next_from;
    for(;from != to;++out)
    {
        *out = *(next_from);
        *out -= *(pre_from);
        *out /= 2.0;
        pre_from = from;
        from = next_from;
        ++next_from;
    }
}

std::vector<float> TractModel::get_tract_data(std::shared_ptr<fib_data> handle,size_t fiber_index,size_t index_num) const
{
    if(tract_data[fiber_index].empty())
        return std::vector<float>();
    auto count =  tract_data[fiber_index].size()/3;
    std::vector<float> data(count);
    // track specific index
    if(index_num < handle->dir.index_data.size())
    {
        auto base_image = tipl::make_image(handle->dir.index_data[index_num][0],handle->dim);
        std::vector<tipl::vector<3,float> > gradient(count);
        auto tract_ptr = reinterpret_cast<const float (*)[3]>(&(tract_data[fiber_index][0]));
        ::gradient(tract_ptr,tract_ptr+count,gradient.begin());
        for (unsigned int point_index = 0,tract_index = 0;
             point_index < count;++point_index,tract_index += 3)
        {
            tipl::interpolator::linear<3> tri_interpo;
            gradient[point_index].normalize();
            if (tri_interpo.get_location(handle->dim,&(tract_data[fiber_index][tract_index])))
            {
                float value,average_value = 0.0f;
                float sum_value = 0.0f;
                for (unsigned int index = 0;index < 8;++index)
                {
                    if ((value = handle->dir.get_track_specific_metrics(tri_interpo.dindex[index],
                                                               handle->dir.index_data[index_num],gradient[point_index])) == 0.0f)
                        continue;
                    average_value += value*tri_interpo.ratio[index];
                    sum_value += tri_interpo.ratio[index];
                }
                if (sum_value > 0.5f)
                    data[point_index] = average_value/sum_value;
                else
                    tipl::estimate(base_image,&(tract_data[fiber_index][tract_index]),data[point_index]);
            }
            else
                tipl::estimate(base_image,&(tract_data[fiber_index][tract_index]),data[point_index]);
        }
    }
    else
    // voxel-based index
    {
        auto I = handle->slices[index_num]->get_image();
        if(I.shape() != handle->dim) // other slices
        {
            for (unsigned int data_index = 0,index = 0;index < tract_data[fiber_index].size();index += 3,++data_index)
            {
                tipl::vector<3> pos(&(tract_data[fiber_index][index]));
                pos.to(handle->slices[index_num]->iT);
                tipl::estimate(I,pos,data[data_index]);
            }
        }
        else
        for (unsigned int data_index = 0,index = 0;index < tract_data[fiber_index].size();index += 3,++data_index)
            tipl::estimate(I,&(tract_data[fiber_index][index]),data[data_index]);
    }
    return data;
}

std::vector<std::vector<float> > TractModel::get_tracts_data(std::shared_ptr<fib_data> handle,const std::string& index_name) const
{
    std::vector<std::vector<float> > data;
    unsigned int data_index = handle->get_name_index(index_name);
    if(data_index < handle->slices.size())
    {
        data.resize(tract_data.size());
        tipl::adaptive_par_for(tract_data.size(),[&](unsigned int i)
        {
             data[i] = std::move(get_tract_data(handle,i,data_index));
        });
    }
    return data;
}
float TractModel::get_tracts_mean(std::shared_ptr<fib_data> handle,unsigned int data_index) const
{
    if(handle->slices[data_index]->optional() || tract_data.empty())
        return 0.0f;
    std::vector<double> mean(tract_data.size());
    tipl::adaptive_par_for(tract_data.size(),[&](unsigned int i)
    {
        mean[i] = tipl::mean(get_tract_data(handle,i,data_index));
    });
    return tipl::mean(mean);
}

void TractModel::get_passing_list(const tipl::image<3,std::vector<short> >& region_map,
                                  unsigned int region_count,
                                  std::vector<std::vector<short> >& passing_list1,
                                  std::vector<std::vector<short> >& passing_list2) const
{
    passing_list1.clear();
    passing_list1.resize(tract_data.size());
    passing_list2.clear();
    passing_list2.resize(tract_data.size());
    // create regions maps

    tipl::adaptive_par_for(tract_data.size(),[&](unsigned int index)
    {
        if(tract_data[index].size() < 6)
            return;
        std::vector<unsigned char> has_region(region_count);
        for(unsigned int ptr = 0;ptr < tract_data[index].size();ptr += 3)
        {
            tipl::pixel_index<3> pos(std::round(tract_data[index][ptr]),
                                        std::round(tract_data[index][ptr+1]),
                                        std::round(tract_data[index][ptr+2]),geo);
            if(!geo.is_valid(pos))
                continue;
            unsigned int pos_index = uint32_t(pos.index());
            for(unsigned int j = 0;j < region_map[pos_index].size();++j)
                has_region[uint32_t(region_map[pos_index][j])] = 1;
        }
        for(unsigned int i = 0;i < has_region.size();++i)
            if(has_region[i])
            {
                passing_list1[index].push_back(short(i));
                passing_list2[index].push_back(short(i));
            }
    });
}

void TractModel::get_end_list(const tipl::image<3,std::vector<short> >& region_map,
                              std::vector<std::vector<short> >& end_pair1,
                              std::vector<std::vector<short> >& end_pair2) const
{
    end_pair1.clear();
    end_pair1.resize(tract_data.size());
    end_pair2.clear();
    end_pair2.resize(tract_data.size());
    tipl::adaptive_par_for(tract_data.size(),[&](unsigned int index)
    {
        if(tract_data[index].size() < 6)
            return;
        tipl::pixel_index<3> end1(std::round(tract_data[index][0]),
                                    std::round(tract_data[index][1]),
                                    std::round(tract_data[index][2]),geo);
        tipl::pixel_index<3> end2(std::round(tract_data[index][tract_data[index].size()-3]),
                                    std::round(tract_data[index][tract_data[index].size()-2]),
                                    std::round(tract_data[index][tract_data[index].size()-1]),geo);
        if(!geo.is_valid(end1) || !geo.is_valid(end2))
            return;
        end_pair1[index] = region_map[end1.index()];
        end_pair2[index] = region_map[end2.index()];
    });
}


void TractModel::run_clustering(unsigned char method_id,unsigned int cluster_count,float detail)
{
    float param[4] = {0};
    if(method_id)// k-means or EM
        param[0] = cluster_count;
    else
    {
        std::copy(geo.begin(),geo.end(),param);
        param[3] = detail;
    }
    std::unique_ptr<BasicCluster> c;
    switch (method_id)
    {
    case 0:
        c.reset(new TractCluster(param));
        break;
    case 1:
        c.reset(new FeatureBasedClutering<tipl::ml::k_means<double,unsigned char> >(param));
        break;
    case 2:
        c.reset(new FeatureBasedClutering<tipl::ml::expectation_maximization<double,unsigned char> >(param));
        break;
    }

    c->add_tracts(tract_data);
    c->run_clustering();
    {
        cluster_count = method_id ? c->get_cluster_count() : std::min<float>(c->get_cluster_count(),cluster_count);
        tract_cluster.resize(tract_data.size());
        std::fill(tract_cluster.begin(),tract_cluster.end(),cluster_count);
        for(int index = 0;index < cluster_count;++index)
        {
            unsigned int cluster_size;
            const unsigned int* data = c->get_cluster(index,cluster_size);
            for(int i = 0;i < cluster_size;++i)
                tract_cluster[data[i]] = index;
        }
    }
}

void ConnectivityMatrix::save_to_image(tipl::color_image& cm)
{
    if(matrix_value.empty())
        return;
    cm.resize(matrix_value.shape());
    std::vector<float> values(matrix_value.size());
    std::copy(matrix_value.begin(),matrix_value.end(),values.begin());
    tipl::normalize(values,255.9f);
    for(unsigned int index = 0;index < values.size();++index)
    {
        cm[index] = tipl::rgb((unsigned char)values[index],(unsigned char)values[index],(unsigned char)values[index]);
    }
}

void ConnectivityMatrix::save_to_file(const char* file_name)
{
    tipl::io::mat_write mat_header(file_name);
    mat_header.write("connectivity",matrix_value,matrix_value.width());
    std::ostringstream out;
    std::copy(region_name.begin(),region_name.end(),std::ostream_iterator<std::string>(out,"\n"));
    std::string result(out.str());
    mat_header.write("name",result);
    mat_header.write("atlas",atlas_name);
}

void ConnectivityMatrix::save_to_text(std::string& text)
{
    std::ostringstream out;
    int w = matrix_value.width();
    for(int i = 0;i < w;++i)
    {
        for(int j = 0;j < w;++j)
            out << matrix_value[i*w+j] << "\t";
        out << std::endl;
    }
    text = out.str();
}

void ConnectivityMatrix::save_to_connectogram(const char* file_name)
{
    std::ofstream out(file_name);
    unsigned int w = uint32_t(matrix_value.width());
    std::vector<float> sum(w);
    out << "data\tdata\t";
    for(unsigned int i = 0;i < w;++i)
    {
        sum[i] = std::max(1.0f,std::accumulate(matrix_value.begin()+int64_t(i)*w,
                                                          matrix_value.begin()+int64_t(i)*w+w,0.0f)*2.0f);
        out << sum[i] << "\t";
    }
    out << std::endl;
    out << "data\tdata\t";
    for(unsigned int i = 0;i < region_name.size();++i)
        std::replace(region_name[i].begin(),region_name[i].end(),' ','_');
    for(unsigned int i = 0;i < w;++i)
        out << region_name[i] << "\t";
    out << std::endl;

    for(unsigned int i = 0;i < w;++i)
    {
        out << sum[i] << "\t" << region_name[i] << "\t";
        for(unsigned int j = 0;j < w;++j)
            out << matrix_value[i*w+j] << "\t";
        out << std::endl;
    }
}

void ConnectivityMatrix::set_regions(const tipl::shape<3>& geo,
                                     const std::vector<std::vector<tipl::vector<3,short> > >& points,
                                     const std::vector<std::string>& labels)
{
    region_count = points.size();
    region_name = labels;
    region_map.clear();
    region_map.resize(geo);
    for(size_t roi = 0;roi < points.size();++roi)
        for(auto& pos : points[roi])
            if(geo.is_valid(pos))
                region_map.at(pos).push_back(uint16_t(roi));
    atlas_name = "roi";
}


template<class m_type>
void init_matrix(m_type& m,unsigned int size)
{
    m.resize(size);
    for(unsigned int i = 0;i < size;++i)
        m[i].resize(size);
}

template<class T,class fun_type>
void for_each_connectivity(const T& end_list1,
                           const T& end_list2,
                           fun_type lambda_fun)
{
    for(unsigned int index = 0;index < end_list1.size();++index)
    {
        const auto& r1 = end_list1[index];
        const auto& r2 = end_list2[index];
        std::vector<std::pair<uint32_t,uint32_t> > region_pair;
        for(unsigned int i = 0;i < r1.size();++i)
            for(unsigned int j = 0;j < r2.size();++j)
                if(r1[i] != r2[j])
                {
                    region_pair.push_back(std::make_pair(uint32_t(r1[i]),uint32_t(r2[j])));
                    region_pair.push_back(std::make_pair(uint32_t(r2[j]),uint32_t(r1[i])));
                }
        // remove duplicates
        std::sort(region_pair.begin(), region_pair.end());
        region_pair.erase(std::unique(region_pair.begin(), region_pair.end()), region_pair.end());

        for(const auto& pair : region_pair)
            lambda_fun(index,pair.first,pair.second);
    }
}

bool ConnectivityMatrix::calculate(std::shared_ptr<fib_data> handle,
                                   TractModel& tract_model,std::string matrix_value_type,bool use_end_only,float threshold)
{
    tipl::progress p("calculating connectivity matrix");
    tipl::out() << "tract count: " << tract_model.get_visible_track_count();
    tipl::out() << "value: " << matrix_value_type;
    tipl::out() << "use_end_only: " << (use_end_only ? "yes":"no");
    if(!atlas_name.empty())
        tipl::out() << "atlas_name: " << atlas_name;

    if(region_count == 0)
    {
        error_msg = "No region information. Please assign regions";
        return false;
    }

    std::vector<std::vector<short> > end_list1,end_list2;
    if(use_end_only)
        tract_model.get_end_list(region_map,end_list1,end_list2);
    else
        tract_model.get_passing_list(region_map,uint32_t(region_count),end_list1,end_list2);

    matrix_value.clear();
    matrix_value.resize(tipl::shape<2>(uint32_t(region_count),uint32_t(region_count)));

    if(tipl::begins_with(matrix_value_type,"trk"))
    {
        std::vector<std::vector<std::vector<unsigned int> > > region_passing_list;
        init_matrix(region_passing_list,uint32_t(region_count));

        for_each_connectivity(end_list1,end_list2,
                              [&](unsigned int index,unsigned int i,unsigned int j){
            region_passing_list[i][j].push_back(index);
        });

        const float resolution_ratio = 2.0f;
        tipl::matrix<4,4> resolution_trans((tipl::identity_matrix()));
        resolution_trans[0] = resolution_trans[5] = resolution_trans[10] = 2.0f;

        std::vector<std::pair<size_t,size_t> > ij_pair;
        for(unsigned int i = 0;i < region_passing_list.size();++i)
            for(unsigned int j = i+1;j < region_passing_list.size();++j)
                if(!region_passing_list[i][j].empty())
                    ij_pair.push_back(std::make_pair(i,j));

        bool return_value = true;
        tipl::adaptive_par_for(ij_pair.size(),[&](size_t index)
        {
            auto i = ij_pair[index].first;
            auto j = ij_pair[index].second;
            TractModel tm(tract_model.geo,tract_model.vs);
            tm.report = tract_model.report;
            tm.trans_to_mni = tract_model.trans_to_mni;
            tm.is_mni = tract_model.is_mni;

            std::vector<std::vector<float> > new_tracts;
            for (unsigned int k = 0;k < region_passing_list[i][j].size();++k)
                new_tracts.push_back(tract_model.get_tract(region_passing_list[i][j][k]));
            tm.add_tracts(new_tracts);
            if(matrix_value_type == "trk")
            {
                auto file_name = region_name[i]+"_"+region_name[j]+".tt.gz";
                if(!tm.save_tracts_to_file(file_name.c_str()))
                {
                    tipl::error() << "cannot write to file: " << file_name;
                    return_value = false;
                    return;
                }
                matrix_value[i+j*region_count] = matrix_value[j+i*region_count] = tm.get_visible_track_count();
            }
            if(tipl::ends_with(matrix_value_type,"area"))
            {
                std::vector<tipl::vector<3,short> > endpoint1,endpoint2;
                tm.to_end_point_voxels(endpoint1,endpoint2,resolution_trans);
                // end point surface 1 and 2
                matrix_value[i+j*region_count] = matrix_value[j+i*region_count] =
                    float(endpoint1.size()+endpoint2.size())*tract_model.vs[0]*tract_model.vs[1]/resolution_ratio/resolution_ratio;
            }
        });
        return return_value;
    }

    std::vector<std::vector<unsigned int> > count;
    init_matrix(count,uint32_t(region_count));
    for_each_connectivity(end_list1,end_list2,
                          [&](unsigned int,unsigned int i,unsigned int j){
        ++count[i][j];
    });

    // determine the threshold for counting the connectivity
    tipl::out() << "threshold: " << threshold;
    unsigned int threshold_count = 0;
    for (const auto& inner : count)
        for (const auto& val : inner)
            threshold_count = std::max(threshold_count, val);
    threshold_count *= threshold;

    if(matrix_value_type == "count")
    {
        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                matrix_value[index] = (count[i][j] > threshold_count ? count[i][j] : 0);
        return true;
    }
    if(matrix_value_type == "ncount" || matrix_value_type == "ncount2")
    {
        std::vector<std::vector<std::vector<unsigned int> > > length_matrix;
        init_matrix(length_matrix,uint32_t(region_count));

        for_each_connectivity(end_list1,end_list2,
                              [&](unsigned int index,unsigned int i,unsigned int j){
            length_matrix[i][j].push_back(uint32_t(tract_model.get_tract(index).size()));
        });

        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                if(!length_matrix[i][j].empty() && count[i][j] > threshold_count)
                {
                    float length = 0.0;
                    if(matrix_value_type == "ncount")
                        length = 1.0f/tipl::median(length_matrix[i][j].begin(),length_matrix[i][j].end());
                    else
                    {
                        for(unsigned int k = 0;k < length_matrix[i][j].size();++k)
                            length += 1.0f/length_matrix[i][j][k];
                    }
                    matrix_value[index] = count[i][j]*length;
                }
                else
                    matrix_value[index] = 0;

        return true;
    }


    if(matrix_value_type == "mean_length")
    {
        std::vector<std::vector<float> > sum_length;
        std::vector<std::vector<unsigned int> > sum_n;
        init_matrix(sum_length,uint32_t(region_count));
        init_matrix(sum_n,uint32_t(region_count));

        for_each_connectivity(end_list1,end_list2,
                              [&](unsigned int index,unsigned int i,unsigned int j){
            auto num_steps = tract_model.get_tract(index).size();
            if(num_steps >= 6)
            {
                auto dis = tract_model.get_tract_point(index,0)-tract_model.get_tract_point(index,1);
                tipl::multiply(dis,handle->vs);
                sum_length[i][j] += dis.length()*num_steps;
                ++sum_n[i][j];
            }

        });

        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                if(sum_n[i][j] && count[i][j] > threshold_count)
                    matrix_value[index] = float(sum_length[i][j])/float(sum_n[i][j])/3.0f;
        return true;
    }
    std::vector<std::vector<float> > data(tract_model.get_tracts_data(handle,matrix_value_type));
    if(data.empty())
    {
        error_msg = "Cannot quantify matrix value using ";
        error_msg += matrix_value_type;
        return false;
    }
    std::vector<std::vector<float> > sum;
    init_matrix(sum,uint32_t(region_count));

    std::vector<float> m(data.size());
    for(unsigned int index = 0;index < data.size();++index)
        if(!data[index].empty())
            m[index] = float(tipl::mean(data[index].begin(),data[index].end()));

    for_each_connectivity(end_list1,end_list2,
                          [&](unsigned int index,unsigned int i,unsigned int j){
        sum[i][j] += m[index];
    });


    for(unsigned int i = 0,index = 0;i < count.size();++i)
        for(unsigned int j = 0;j < count[i].size();++j,++index)
            matrix_value[index] = (count[i][j] > threshold_count ? sum[i][j]/float(count[i][j]) : 0.0f);
    return true;

}
template<class matrix_type>
void distance_bin(const matrix_type& bin,tipl::image<2,float>& D)
{
    unsigned int n = bin.width();
    tipl::image<2,unsigned int> A,Lpath;
    A = bin;
    Lpath = bin;
    D = bin;
    for(unsigned int l = 2;1;++l)
    {
        tipl::image<2,unsigned int> t(A.shape());
        tipl::mat::product(Lpath.begin(),A.begin(),t.begin(),tipl::shape<2>(n,n),tipl::shape<2>(n,n));
        std::swap(Lpath,t);
        bool con = false;
        for(unsigned int i = 0;i < D.size();++i)
            if(Lpath[i] != 0 && D[i] == 0)
            {
                D[i] = l;
                con = true;
            }
        if(!con)
            break;
    }
    std::replace(D.begin(),D.end(),(float)0,std::numeric_limits<float>::max());
}
template<class matrix_type>
void distance_wei(const matrix_type& W_,tipl::image<2,float>& D)
{
    tipl::image<2,float> W(W_);
    for(unsigned int i = 0;i < W.size();++i)
        W[i] = (W[i] != 0) ? 1.0/W[i]:0;
    unsigned int n = W.width();
    D.clear();
    D.resize(W.shape());
    std::fill(D.begin(),D.end(),std::numeric_limits<float>::max());
    for(unsigned int i = 0,dg = 0;i < n;++i,dg += n + 1)
        D[dg] = 0;
    for(unsigned int i = 0,in = 0;i < n;++i,in += n)
    {
        std::vector<unsigned char> S(n);
        tipl::image<2,float> W1(W);

        std::vector<unsigned int> V;
        V.push_back(i);
        while(1)
        {
            for(unsigned int j = 0;j < V.size();++j)
            {
                S[V[j]] = 1;
                for(unsigned int k = V[j];k < W1.size();k += n)
                    W1[k] = 0;
            }
            for(unsigned int j = 0;j < V.size();++j)
            {
                unsigned int v = V[j];
                unsigned int vn = v*n;
                for(unsigned int k = 0;k < n;++k)
                if(W1[vn+k] > 0)
                    D[in+k] = std::min<float>(D[in+k],D[in+v]+W1[vn+k]);
            }
            float minD = std::numeric_limits<float>::max();
            for(unsigned int j = 0;j < n;++j)
                if(S[j] == 0 && minD > D[in+j])
                    minD = D[in+j];
            if(minD == std::numeric_limits<float>::max())
                break;
            V.clear();
            for(unsigned int j = 0;j < n;++j)
                if(D[in+j]  == minD)
                    V.push_back(j);
        }
    }
    std::replace(D.begin(),D.end(),(float)0.0,std::numeric_limits<float>::max());
}
template<class matrix_type>
void inv_dis(const matrix_type& D,matrix_type& e)
{
    e = D;
    unsigned int n = D.width();
    for(unsigned int i = 0;i < e.size();++i)
        e[i] = ((e[i] == 0 || e[i] == std::numeric_limits<float>::max()) ? 0:1.0/e[i]);
    for(unsigned int i = 0,pos = 0;i < n;++i,pos += n+1)
        e[pos] = 0;
}

template<class vec_type>
void output_node_measures(std::ostream& out,const char* name,const vec_type& data)
{
    out << name << "\t";
    for(unsigned int i = 0;i < data.size();++i)
        out << data[i] << "\t";
    out << std::endl;
}

void ConnectivityMatrix::network_property(std::string& report)
{
    std::ostringstream out;
    size_t n = matrix_value.width();
    tipl::image<2,unsigned char> binary_matrix(matrix_value.shape());
    tipl::image<2,float> norm_matrix(matrix_value.shape());

    float max_value = tipl::max_value(matrix_value);
    for(unsigned int i = 0;i < binary_matrix.size();++i)
    {
        binary_matrix[i] = matrix_value[i] > 0 ? 1 : 0;
        norm_matrix[i] = matrix_value[i]/max_value;
    }
    // density
    size_t edge = std::accumulate(binary_matrix.begin(),binary_matrix.end(),size_t(0))/2;
    out << "density\t" << (float)edge*2.0/(float)(n*n-n) << std::endl;

    // calculate degree
    std::vector<float> degree(n);
    for(unsigned int i = 0;i < n;++i)
        degree[i] = std::accumulate(binary_matrix.begin()+i*n,binary_matrix.begin()+(i+1)*n,0.0);
    // calculate strength
    std::vector<float> strength(n);
    for(unsigned int i = 0;i < n;++i)
        strength[i] = std::accumulate(norm_matrix.begin()+i*n,norm_matrix.begin()+(i+1)*n,0.0);
    // calculate clustering coefficient
    std::vector<float> cluster_co(n);
    for(unsigned int i = 0,posi = 0;i < n;++i,posi += n)
    if(degree[i] >= 2)
    {
        for(unsigned int j = 0,index = 0;j < n;++j)
            for(unsigned int k = 0;k < n;++k,++index)
                if(binary_matrix[posi + j] && binary_matrix[posi + k])
                    cluster_co[i] += binary_matrix[index];
        float d = degree[i];
        cluster_co[i] /= (d*d-d);
    }
    float cc_bin = tipl::mean(cluster_co.begin(),cluster_co.end());
    out << "clustering_coeff_average(binary)\t" << cc_bin << std::endl;

    // calculate weighted clustering coefficient
    tipl::image<2,float> cyc3(norm_matrix.shape());
    std::vector<float> wcluster_co(n);
    {
        tipl::image<2,float> root(norm_matrix);
        // root = W.^ 1/3
        for(unsigned int j = 0;j < root.size();++j)
            root[j] = std::pow(root[j],(float)(1.0/3.0));
        // cyc3 = (W.^1/3)^3
        tipl::image<2,float> t(root.shape());
        tipl::mat::product(root.begin(),root.begin(),t.begin(),tipl::shape<2>(n,n),tipl::shape<2>(n,n));
        tipl::mat::product(t.begin(),root.begin(),cyc3.begin(),tipl::shape<2>(n,n),tipl::shape<2>(n,n));
        // wcc = diag(cyc3)/(K.*(K-1));
        for(unsigned int i = 0;i < n;++i)
        if(degree[i] >= 2)
        {
            float d = degree[i];
            wcluster_co[i] = cyc3[i*(n+1)]/(d*d-d);
        }
    }
    float cc_wei = tipl::mean(wcluster_co.begin(),wcluster_co.end());
    out << "clustering_coeff_average(weighted)\t" << cc_wei << std::endl;


    // transitivity
    {
        tipl::image<2,float> norm_matrix2(norm_matrix.shape());
        tipl::image<2,float> norm_matrix3(norm_matrix.shape());
        tipl::mat::product(norm_matrix.begin(),norm_matrix.begin(),norm_matrix2.begin(),tipl::shape<2>(n,n),tipl::shape<2>(n,n));
        tipl::mat::product(norm_matrix2.begin(),norm_matrix.begin(),norm_matrix3.begin(),tipl::shape<2>(n,n),tipl::shape<2>(n,n));
        out << "transitivity(binary)\t" << tipl::mat::trace(norm_matrix3.begin(),tipl::shape<2>(n,n)) /
                (std::accumulate(norm_matrix2.begin(),norm_matrix2.end(),0.0) - tipl::mat::trace(norm_matrix2.begin(),tipl::shape<2>(n,n))) << std::endl;
        float k = 0;
        for(unsigned int i = 0;i < n;++i)
            k += degree[i]*(degree[i]-1);
        out << "transitivity(weighted)\t" << (k == 0 ? 0 : tipl::mat::trace(cyc3.begin(),tipl::shape<2>(n,n))/k) << std::endl;
    }

    std::vector<float> eccentricity_bin(n),eccentricity_wei(n);

    {
        tipl::image<2,float> dis_bin,dis_wei;
        distance_bin(binary_matrix,dis_bin);
        distance_wei(norm_matrix,dis_wei);
        unsigned int inf_count_bin = std::count(dis_bin.begin(),dis_bin.end(),std::numeric_limits<float>::max());
        unsigned int inf_count_wei = std::count(dis_wei.begin(),dis_wei.end(),std::numeric_limits<float>::max());
        std::replace(dis_bin.begin(),dis_bin.end(),std::numeric_limits<float>::max(),(float)0);
        std::replace(dis_wei.begin(),dis_wei.end(),std::numeric_limits<float>::max(),(float)0);
        float ncpl_bin = std::accumulate(dis_bin.begin(),dis_bin.end(),0.0)/(n*n-inf_count_bin);
        float ncpl_wei = std::accumulate(dis_wei.begin(),dis_wei.end(),0.0)/(n*n-inf_count_wei);
        out << "network_characteristic_path_length(binary)\t" << ncpl_bin << std::endl;
        out << "network_characteristic_path_length(weighted)\t" << ncpl_wei << std::endl;
        out << "small-worldness(binary)\t" << (ncpl_bin == 0.0 ? 0.0:cc_bin/ncpl_bin) << std::endl;
        out << "small-worldness(weighted)\t" << (ncpl_wei == 0.0 ? 0.0:cc_wei/ncpl_wei) << std::endl;
        tipl::image<2,float> invD;
        inv_dis(dis_bin,invD);
        out << "global_efficiency(binary)\t" << std::accumulate(invD.begin(),invD.end(),0.0)/(n*n-inf_count_bin) << std::endl;
        inv_dis(dis_wei,invD);
        out << "global_efficiency(weighted)\t" << std::accumulate(invD.begin(),invD.end(),0.0)/(n*n-inf_count_wei) << std::endl;

        for(unsigned int i = 0,ipos = 0;i < n;++i,ipos += n)
        {
            eccentricity_bin[i] = tipl::max_value(dis_bin.begin()+ipos,dis_bin.begin()+ipos+n);
            eccentricity_wei[i] = tipl::max_value(dis_wei.begin()+ipos,dis_wei.begin()+ipos+n);

        }
        out << "diameter_of_graph(binary)\t" << *std::max_element(eccentricity_bin.begin(),eccentricity_bin.end()) <<std::endl;
        out << "diameter_of_graph(weighted)\t" << *std::max_element(eccentricity_wei.begin(),eccentricity_wei.end()) <<std::endl;


        std::replace(eccentricity_bin.begin(),eccentricity_bin.end(),(float)0,std::numeric_limits<float>::max());
        std::replace(eccentricity_wei.begin(),eccentricity_wei.end(),(float)0,std::numeric_limits<float>::max());
        out << "radius_of_graph(binary)\t" << tipl::min_value(eccentricity_bin) <<std::endl;
        out << "radius_of_graph(weighted)\t" << tipl::min_value(eccentricity_wei) <<std::endl;
        std::replace(eccentricity_bin.begin(),eccentricity_bin.end(),std::numeric_limits<float>::max(),(float)0);
        std::replace(eccentricity_wei.begin(),eccentricity_wei.end(),std::numeric_limits<float>::max(),(float)0);
    }

    std::vector<float> local_efficiency_bin(n);
    //calculate local efficiency
    {
        for(unsigned int i = 0,ipos = 0;i < n;++i,ipos += n)
        {
            unsigned int new_n = std::accumulate(binary_matrix.begin()+ipos,
                                                 binary_matrix.begin()+ipos+n,0);
            if(new_n < 2)
                continue;
            tipl::image<2,float> newA(tipl::shape<2>(new_n,new_n));
            unsigned int pos = 0;
            for(unsigned int j = 0,index = 0;j < n;++j)
                for(unsigned int k = 0;k < n;++k,++index)
                    if(binary_matrix[ipos+j] && binary_matrix[ipos+k])
                    {
                        if(pos < newA.size())
                            newA[pos] = binary_matrix[index];
                        ++pos;
                    }
            tipl::image<2,float> invD;
            distance_bin(newA,invD);
            inv_dis(invD,invD);
            local_efficiency_bin[i] = std::accumulate(invD.begin(),invD.end(),0.0)/(new_n*new_n-new_n);
        }
    }

    std::vector<float> local_efficiency_wei(n);
    {

        for(unsigned int i = 0,ipos = 0;i < n;++i,ipos += n)
        {
            unsigned int new_n = std::accumulate(binary_matrix.begin()+ipos,
                                                 binary_matrix.begin()+ipos+n,0);
            if(new_n < 2)
                continue;
            tipl::image<2,float> newA(tipl::shape<2>(new_n,new_n));
            unsigned int pos = 0;
            for(unsigned int j = 0,index = 0;j < n;++j)
                for(unsigned int k = 0;k < n;++k,++index)
                    if(binary_matrix[ipos+j] && binary_matrix[ipos+k])
                    {
                        if(pos < newA.size())
                            newA[pos] = norm_matrix[index];
                        ++pos;
                    }
            std::vector<float> sw;
            for(unsigned int j = 0;j < n;++j)
                if(binary_matrix[ipos+j])
                    sw.push_back(std::pow(norm_matrix[ipos+j],(float)(1.0/3.0)));
            tipl::image<2,float> invD;
            distance_wei(newA,invD);
            inv_dis(invD,invD);
            float numer = 0.0;
            for(unsigned int j = 0,index = 0;j < new_n;++j)
                for(unsigned int k = 0;k < new_n;++k,++index)
                    numer += std::pow(invD[index],(float)(1.0/3.0))*sw[j]*sw[k];
            local_efficiency_wei[i] = numer/(new_n*new_n-new_n);
        }
    }


    // calculate assortativity
    {
        std::vector<float> degi,degj;
        for(unsigned int i = 0,index = 0;i < n;++i)
            for(unsigned int j = 0;j < n;++j,++index)
                if(j > i && binary_matrix[index])
                {
                    degi.push_back(degree[i]);
                    degj.push_back(degree[j]);
                }
        float a = (std::accumulate(degi.begin(),degi.end(),0.0)+
                   std::accumulate(degj.begin(),degj.end(),0.0))/2.0/degi.size();
        float sum = tipl::vec::dot(degi.begin(),degi.end(),degj.begin())/degi.size();
        tipl::square(degi);
        tipl::square(degj);
        float b = (std::accumulate(degi.begin(),degi.end(),0.0)+
                   std::accumulate(degj.begin(),degj.end(),0.0))/2.0/degi.size();
        a = a*a;
        out << "assortativity_coefficient(binary)\t" << (b == a ? 0 : ( sum  - a)/ ( b - a )) << std::endl;
    }




    // calculate assortativity
    {
        std::vector<float> degi,degj;
        for(unsigned int i = 0,index = 0;i < n;++i)
            for(unsigned int j = 0;j < n;++j,++index)
                if(j > i && binary_matrix[index])
                {
                    degi.push_back(strength[i]);
                    degj.push_back(strength[j]);
                }
        float a = (std::accumulate(degi.begin(),degi.end(),0.0)+
                   std::accumulate(degj.begin(),degj.end(),0.0))/2.0/degi.size();
        float sum = tipl::vec::dot(degi.begin(),degi.end(),degj.begin())/degi.size();
        tipl::square(degi);
        tipl::square(degj);
        float b = (std::accumulate(degi.begin(),degi.end(),0.0)+
                   std::accumulate(degj.begin(),degj.end(),0.0))/2.0/degi.size();
        out << "assortativity_coefficient(weighted)\t" << ( sum  - a*a)/ ( b - a*a ) << std::endl;
    }

    //rich club binary
    {
        for(int k = 5;k <= 25;k += 5)
        {
            float nk = 0.0f,ek = 0.0f;
            for(int i = 0;i < binary_matrix.height();++i)
            {
                if(degree[i] <= k)
                    continue;
                nk += 1.0f;
                int pos = i*binary_matrix.width();
                for(int j = 0;j < binary_matrix.width();++j)
                {
                    if(degree[j] <= k || i == j)
                        continue;
                    ek += binary_matrix[pos + j];
                }
            }
            float nk_nk1 = nk*(nk-1.0f);
            out << "rich_club_k" << k << "(binary)\t" << (nk_nk1 == 0.0f ? 0.0f: ek/nk_nk1) << std::endl;
        }
    }

    //rich club weighted
    {
        std::vector<float> wrant(norm_matrix.begin(),norm_matrix.end());
        std::sort(wrant.begin(),wrant.end(),std::greater<float>());
        for(int k = 5;k <= 25;k += 5)
        {
            float wr = 0.0f;
            int er = 0;
            for(int i = 0;i < norm_matrix.height();++i)
            {
                if(degree[i] <= k)
                    continue;
                int pos = i*norm_matrix.width();
                for(int j = 0;j < norm_matrix.width();++j)
                {
                    if(degree[j] <= k || i == j)
                        continue;
                    wr += norm_matrix[pos + j];
                    ++er;
                }
            }
            float wrank_r = std::accumulate(wrant.begin(),wrant.begin()+er,0.0f);
            out << "rich_club_k" << k << "(weighted)\t" << (wrank_r == 0.0f ? 0.0f: wr/wrank_r) << std::endl;
        }
    }

    // betweenness
    std::vector<float> betweenness_bin(n);
    {

        tipl::image<2,unsigned int> NPd(binary_matrix),NSPd(binary_matrix),NSP(binary_matrix);
        for(unsigned int i = 0,dg = 0;i < n;++i,dg += n+1)
            NSP[dg] = 1;
        tipl::image<2,unsigned int> L(NSP);
        unsigned int d = 2;
        for(;std::find(NSPd.begin(),NSPd.end(),1) != NSPd.end();++d)
        {
            tipl::image<2,unsigned int> t(binary_matrix.shape());
            tipl::mat::product(NPd.begin(),binary_matrix.begin(),t.begin(),tipl::shape<2>(n,n),tipl::shape<2>(n,n));
            t.swap(NPd);
            for(unsigned int i = 0;i < L.size();++i)
            {
                NSPd[i] = (L[i] == 0) ? NPd[i]:0;
                NSP[i] += NSPd[i];
                L[i] += (NSPd[i] == 0) ? 0:d;
            }
        }

        for(unsigned int i = 0,dg = 0;i < n;++i,dg += n+1)
            L[dg] = 0;
        std::replace(NSP.begin(),NSP.end(),0,1);
        tipl::image<2,float> DP(binary_matrix.shape());
        for(--d;d >= 2;--d)
        {
            tipl::image<2,float> t(DP),DPd1(binary_matrix.shape());
            t += 1.0;
            for(unsigned int i = 0;i < t.size();++i)
                if(L[i] != d)
                    t[i] = 0;
                else
                    t[i] /= NSP[i];
            tipl::mat::product(t.begin(),binary_matrix.begin(),DPd1.begin(),tipl::shape<2>(n,n),tipl::shape<2>(n,n));
            for(unsigned int i = 0;i < DPd1.size();++i)
                if(L[i] != d-1)
                    DPd1[i] = 0;
                else
                    DPd1[i] *= NSP[i];
            DP += DPd1;
        }

        for(unsigned int i = 0,index = 0;i < n;++i)
            for(unsigned int j = 0;j < n;++j,++index)
                betweenness_bin[j] += DP[index];
    }
    std::vector<float> betweenness_wei(n);
    {
        for(unsigned int i = 0;i < n;++i)
        {
            std::vector<float> D(n),NP(n);
            std::fill(D.begin(),D.end(),std::numeric_limits<float>::max());
            D[i] = 0;
            NP[i] = 1;
            std::vector<unsigned char> S(n),Q(n);
            int q = n-1;
            std::fill(S.begin(),S.end(),1);
            tipl::image<2,unsigned char> P(binary_matrix.shape());
            tipl::image<2,float> G1(norm_matrix);
            // per suggestion from Mikail Rubinov, the matrix has to be "granulated"
            {
                float eps = max_value*0.001f;
                for(size_t i = 0;i < G1.size();++i)
                    if(G1[i] > 0.0f && G1[i] < eps)
                        G1[i] = eps;
            }
            std::vector<unsigned int> V;
            V.push_back(i);
            while(q >= V.size())
            {
                for(unsigned int j = 0,jpos = 0;j < n;++j,jpos += n)
                    for(unsigned int k = 0;k < V.size();++k)
                        G1[jpos+V[k]] = 0;
                for(unsigned int k = 0;k < V.size();++k)
                {
                    S[V[k]] = 0;
                    Q[q--]=V[k];
                    unsigned int v_rowk = V[k]*n;
                    for(unsigned int w = 0,w_row = 0;w < n;++w, w_row += n)
                        if(G1[v_rowk+w] > 0)
                        {
                            float Duw=D[V[k]]+G1[v_rowk+w];
                            if(Duw < D[w])
                            {
                                D[w]=Duw;
                                NP[w]=NP[V[k]];
                                std::fill(P.begin()+w_row,P.begin()+w_row+n,0);
                                P[w_row + V[k]] = 1;
                            }
                            else
                            if(Duw==D[w])
                            {
                                NP[w]+=NP[V[k]];
                                P[w_row+V[k]]=1;
                            }
                        }
                }
                if(std::find(S.begin(),S.end(),1) == S.end())
                    break;
                float minD = std::numeric_limits<float>::max();
                for(unsigned int j = 0;j < n;++j)
                    if(S[j] && minD > D[j])
                        minD = D[j];
                if(minD == std::numeric_limits<float>::max())
                {
                    for(unsigned int j = 0,k = 0;j < n;++j)
                        if(D[j] == std::numeric_limits<float>::max())
                            Q[k++] = j;
                    break;
                }
                V.clear();
                for(unsigned int j = 0;j < n;++j)
                    if(D[j] == minD)
                        V.push_back(j);
            }

            std::vector<float> DP(n);
            for(unsigned int j = 0;j < n-1;++j)
            {
                unsigned int w=Q[j];
                unsigned int w_row = w*n;
                betweenness_wei[w] += DP[w];
                for(unsigned int k = 0;k < n;++k)
                    if(P[w_row+k])
                        DP[k] += (1.0+DP[w])*NP[k]/NP[w];
            }
        }
    }


    std::vector<float> eigenvector_centrality_bin(n),eigenvector_centrality_wei(n);
    {
        tipl::image<2,float> bin;
        bin = binary_matrix;
        std::vector<float> V(binary_matrix.size()),d(n);
        tipl::mat::eigen_decomposition_sym(bin.begin(),V.begin(),d.begin(),tipl::shape<2>(n,n));
        std::copy(V.begin(),V.begin()+n,eigenvector_centrality_bin.begin());
        tipl::mat::eigen_decomposition_sym(norm_matrix.begin(),V.begin(),d.begin(),tipl::shape<2>(n,n));
        std::copy(V.begin(),V.begin()+n,eigenvector_centrality_wei.begin());
    }

    std::vector<float> pagerank_centrality_bin(n),pagerank_centrality_wei(n);
    {
        float d = 0.85f;
        std::vector<float> deg_bin(degree.begin(),degree.end()),deg_wei(strength.begin(),strength.end());
        std::replace(deg_bin.begin(),deg_bin.end(),0.0f,1.0f);
        std::replace(deg_wei.begin(),deg_wei.end(),0.0f,1.0f);

        tipl::image<2,float> B_bin(binary_matrix.shape()),B_wei(binary_matrix.shape());
        for(unsigned int i = 0,index = 0;i < n;++i)
            for(unsigned int j = 0;j < n;++j,++index)
            {
                B_bin[index] = -d*((float)binary_matrix[index])*1.0/deg_bin[j];
                B_wei[index] = -d*norm_matrix[index]*1.0/deg_wei[j];
                if(i == j)
                {
                    B_bin[index] += 1.0;
                    B_wei[index] += 1.0;
                }
            }
        std::vector<unsigned int> pivot(n);
        std::vector<float> b(n);
        std::fill(b.begin(),b.end(),(1.0-d)/n);
        tipl::mat::lu_decomposition(B_bin.begin(),pivot.begin(),tipl::shape<2>(n,n));
        tipl::mat::lu_solve(B_bin.begin(),pivot.begin(),b.begin(),pagerank_centrality_bin.begin(),tipl::shape<2>(n,n));
        tipl::mat::lu_decomposition(B_wei.begin(),pivot.begin(),tipl::shape<2>(n,n));
        tipl::mat::lu_solve(B_wei.begin(),pivot.begin(),b.begin(),pagerank_centrality_wei.begin(),tipl::shape<2>(n,n));

        float sum_bin = std::accumulate(pagerank_centrality_bin.begin(),pagerank_centrality_bin.end(),0.0);
        float sum_wei = std::accumulate(pagerank_centrality_wei.begin(),pagerank_centrality_wei.end(),0.0);

        if(sum_bin != 0)
            tipl::divide_constant(pagerank_centrality_bin,sum_bin);
        if(sum_wei != 0)
            tipl::divide_constant(pagerank_centrality_wei,sum_wei);
    }
    output_node_measures(out,"network_measures",region_name);
    output_node_measures(out,"degree(binary)",degree);
    output_node_measures(out,"strength(weighted)",strength);
    output_node_measures(out,"cluster_coef(binary)",cluster_co);
    output_node_measures(out,"cluster_coef(weighted)",wcluster_co);
    output_node_measures(out,"local_efficiency(binary)",local_efficiency_bin);
    output_node_measures(out,"local_efficiency(weighted)",local_efficiency_wei);
    output_node_measures(out,"betweenness_centrality(binary)",betweenness_bin);
    output_node_measures(out,"betweenness_centrality(weighted)",betweenness_wei);
    output_node_measures(out,"eigenvector_centrality(binary)",eigenvector_centrality_bin);
    output_node_measures(out,"eigenvector_centrality(weighted)",eigenvector_centrality_wei);
    output_node_measures(out,"pagerank_centrality(binary)",pagerank_centrality_bin);
    output_node_measures(out,"pagerank_centrality(weighted)",pagerank_centrality_wei);
    output_node_measures(out,"eccentricity(binary)",eccentricity_bin);
    output_node_measures(out,"eccentricity(weighted)",eccentricity_wei);

    report = out.str();
}
