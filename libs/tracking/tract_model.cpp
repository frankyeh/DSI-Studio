//---------------------------------------------------------------------------
#include <fstream>
#include <sstream>
#include <iterator>
#include <set>
#include <map>
#include "tract_model.hpp"
#include "tracking_static_link.h"
#include "prog_interface_static_link.h"
#include "libs/tracking/tracking_model.hpp"
#include "gzip_interface.hpp"
#include "mapping/atlas.hpp"


struct TrackVis
{
    char id_string[6];//ID string for track file. The first 5 characters must be "TRACK".
    short int dim[3];//Dimension of the image volume.
    float voxel_size[3];//Voxel size of the image volume.
    float origin[3];//Origin of the image volume. This field is not yet being used by TrackVis. That means the origin is always (0, 0, 0).
    short int n_scalars;//Number of scalars saved at each track point (besides x, y and z coordinates).
    char scalar_name[10][20];//Name of each scalar. Can not be longer than 20 characters each. Can only store up to 10 names.
    short int n_properties	;//Number of properties saved at each track.
    char property_name[10][20];//Name of each property. Can not be longer than 20 characters each. Can only store up to 10 names.
    float vox_to_ras[4][4];
    char reserved[444];//Reserved space for future version.
    char voxel_order[4];//Storing order of the original image data. Explained here.
    char pad2[4];//Paddings.
    float image_orientation_patient[6];//Image orientation of the original image. As defined in the DICOM header.
    char pad1[2];//Paddings.
    unsigned char invert[6];//Inversion/rotation flags used to generate this track file. For internal use only.
    int n_count;//Number of tract stored in this track file. 0 means the number was NOT stored.
    int version;//Version number. Current version is 1.
    int hdr_size;//Size of the header. Used to determine byte swap. Should be 1000.
    void init(image::geometry<3> geometry_,const image::vector<3>& voxel_size_)
    {
        id_string[0] = 'T';
        id_string[1] = 'R';
        id_string[2] = 'A';
        id_string[3] = 'C';
        id_string[4] = 'K';
        id_string[5] = 0;
        std::copy(geometry_.begin(),geometry_.end(),dim);
        std::copy(voxel_size_.begin(),voxel_size_.end(),voxel_size);
        //voxel_size
        origin[0] = origin[1] = origin[2] = 0;
        n_scalars = 0;
        std::fill((char*)scalar_name,(char*)scalar_name+200,0);
        n_properties = 0;
        std::fill((char*)property_name,(char*)property_name+200,0);
        std::fill((float*)vox_to_ras,(float*)vox_to_ras+16,(float)0.0);
        std::fill(reserved,reserved+sizeof(reserved),0);
        voxel_order[0] = 'L';
        voxel_order[1] = 'P';
        voxel_order[2] = 'S';
        voxel_order[3] = 0;
        std::copy(voxel_order,voxel_order+4,pad2);
        image_orientation_patient[0] = 1.0;
        image_orientation_patient[1] = 0.0;
        image_orientation_patient[2] = 0.0;
        image_orientation_patient[3] = 0.0;
        image_orientation_patient[4] = 1.0;
        image_orientation_patient[5] = 0.0;
        std::fill(pad1,pad1+2,0);
        std::fill(invert,invert+6,0);
        n_count = 0;
        version = 2;
        hdr_size = 1000;
    }
};
//---------------------------------------------------------------------------
TractModel::TractModel(ODFModel* handle_):handle(handle_),geometry(handle_->fib_data.dim),vs(handle_->fib_data.vs),fib(new fiber_orientations)
{
    fib->read(handle_->fib_data);
}
//---------------------------------------------------------------------------
void TractModel::add(const TractModel& rhs)
{
    for(unsigned int index = 0;index < rhs.redo_size.size();++index)
        redo_size.push_back(std::make_pair(rhs.redo_size[index].first + tract_data.size(),
                                           rhs.redo_size[index].second));
    tract_data.insert(tract_data.end(),rhs.tract_data.begin(),rhs.tract_data.end());
    tract_color.insert(tract_color.end(),rhs.tract_color.begin(),rhs.tract_color.end());
    deleted_tract_data.insert(deleted_tract_data.end(),
                              rhs.deleted_tract_data.begin(),
                              rhs.deleted_tract_data.end());
    deleted_tract_color.insert(deleted_tract_color.end(),
                               rhs.deleted_tract_color.begin(),
                               rhs.deleted_tract_color.end());
    deleted_count.insert(deleted_count.begin(),
                         rhs.deleted_count.begin(),
                         rhs.deleted_count.end());
}
//---------------------------------------------------------------------------
bool TractModel::load_from_file(const char* file_name_,bool append)
{
    std::string file_name(file_name_);
    std::vector<std::vector<float> > loaded_tract_data;
    std::vector<unsigned int> loaded_tract_cluster;
    if (file_name.find(".txt") != std::string::npos)
    {
        std::ifstream in(file_name_);
        if (!in)
            return false;
        std::string line;
        in.seekg(0,std::ios::end);
        unsigned int total = in.tellg();
        in.seekg(0,std::ios::beg);
        begin_prog("loading");
        while (std::getline(in,line))
        {
            check_prog(in.tellg(),total);
            loaded_tract_data.push_back(std::vector<float>());
            std::istringstream in(line);
            std::copy(std::istream_iterator<float>(in),
                      std::istream_iterator<float>(),std::back_inserter(loaded_tract_data.back()));
            if (loaded_tract_data.back().size() < 6)
            {
                if(loaded_tract_data.back().size() == 1)// cluster info
                    loaded_tract_cluster.push_back(loaded_tract_data.back()[0]);
                loaded_tract_data.pop_back();
                continue;
            }
        }

    }
    else
        //trackvis
        if (file_name.find(".trk") != std::string::npos)
        {
            TrackVis trk;
            std::ifstream in(file_name_,std::ios::binary);
            if (!in)
                return false;
            in.read((char*)&trk,1000);
            //if (geo != geometry)
            //    ShowMessage("Incompatible image dimension. The tracts may not be properly presented");
            //std::copy(trk.voxel_size,trk.voxel_size+3,vs.begin());
            unsigned int track_number = trk.n_count;
            begin_prog("loading");
            for (unsigned int index = 0;check_prog(index,track_number);++index)
            {
                int n_point;
                in.read((char*)&n_point,sizeof(int));
                unsigned int index_shift = 3 + trk.n_scalars;
                std::vector<float> tract(index_shift*n_point + trk.n_properties);
                in.read((char*)&*tract.begin(),sizeof(float)*tract.size());

                loaded_tract_data.push_back(std::vector<float>());
                loaded_tract_data.back().resize(n_point*3);
                const float *from = &*tract.begin();
                float *to = &*loaded_tract_data.back().begin();
                for (unsigned int i = 0;i < n_point;++i,from += index_shift,to += 3)
                {
                    float x = from[0]/vs[0];
                    float y = from[1]/vs[1];
                    float z = from[2]/vs[2];
                    to[0] = x;
                    to[1] = y;
                    to[2] = z;
                }
                if(trk.n_properties == 1)
                    loaded_tract_cluster.push_back(from[0]);
            }
        }
        else
            if (file_name.find(".mat") != std::string::npos)
            {
                gz_mat_read in;
                if(!in.load_from_file(file_name_))
                    return false;
                const float* buf = 0;
                const unsigned int* length = 0;
                const unsigned int* cluster = 0;
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
            }
    else
                if (file_name.find(".tck") != std::string::npos)
                {
                    unsigned int offset = 0;
                    {
                        std::ifstream in(file_name.c_str());
                        if(!in)
                            return false;
                        std::string line;
                        while(std::getline(in,line))
                        {
                            if(line.size() > 4 &&
                                    line.substr(0,7) == std::string("file: ."))
                            {
                                std::istringstream str(line);
                                std::string s1,s2;
                                str >> s1 >> s2 >> offset;
                                break;
                            }
                        }
                        if(!in)
                            return false;
                    }
                    std::ifstream in(file_name.c_str(),std::ios::binary);
                    if(!in)
                        return false;
                    in.seekg(0,std::ios::end);
                    unsigned int total_size = in.tellg();
                    in.seekg(offset,std::ios::beg);
                    std::vector<unsigned int> buf((total_size-offset)/4);
                    in.read((char*)&*buf.begin(),total_size-offset-16);// 16 skip the final inf
                    for(unsigned int index = 0;index < buf.size();)
                    {
                        unsigned int end = std::find(buf.begin()+index,buf.end(),2143289344)-buf.begin(); // NaN
                        loaded_tract_data.push_back(std::vector<float>());
                        loaded_tract_data.back().resize(end-index);
                        std::copy((const float*)&*buf.begin() + index,
                                  (const float*)&*buf.begin() + end,
                                  loaded_tract_data.back().begin());
                        image::divide_constant(loaded_tract_data.back().begin(),loaded_tract_data.back().end(),handle->fib_data.vs[0]);
                        index = end+3;
                    }

                }

    if (loaded_tract_data.empty())
        return false;
    if (append)
    {
        add_tracts(loaded_tract_data);
        return true;
    }
    if(loaded_tract_cluster.size() == loaded_tract_data.size())
        loaded_tract_cluster.swap(tract_cluster);
    else
        tract_cluster.clear();
    loaded_tract_data.swap(tract_data);
    tract_color.resize(tract_data.size());
    std::fill(tract_color.begin(),tract_color.end(),0);
    deleted_tract_data.clear();
    deleted_tract_color.clear();
    redo_size.clear();
    return true;
}
//---------------------------------------------------------------------------
bool TractModel::save_fa_to_file(const char* file_name)
{
    std::vector<std::vector<float> > data;
    get_tracts_fa(data);
    if(data.empty())
        return false;
    std::ofstream out(file_name,std::ios::binary);
    if (!out)
        return false;
    begin_prog("saving");
    for (unsigned int i = 0;check_prog(i,data.size());++i)
    {
        std::copy(data[i].begin(),data[i].end(),std::ostream_iterator<float>(out," "));
        out << std::endl;
    }
    return true;
}

//---------------------------------------------------------------------------
bool TractModel::save_data_to_file(const char* file_name,const std::string& index_name)
{
    std::vector<std::vector<float> > data;
    get_tracts_data(index_name,data);
    if(data.empty())
        return false;
    std::ofstream out(file_name,std::ios::binary);
    if (!out)
        return false;
    begin_prog("saving");
    for (unsigned int i = 0;check_prog(i,data.size());++i)
    {
        std::copy(data[i].begin(),data[i].end(),std::ostream_iterator<float>(out," "));
        out << std::endl;
    }
    return true;
}
//---------------------------------------------------------------------------
bool TractModel::save_tracts_to_file(const char* file_name_)
{
    std::string file_name(file_name_);
    if (file_name.find(".txt") != std::string::npos)
    {
        std::ofstream out(file_name_,std::ios::binary);
        if (!out)
            return false;
        begin_prog("saving");
        for (unsigned int i = 0;check_prog(i,tract_data.size());++i)
        {
            std::copy(tract_data[i].begin(),
                      tract_data[i].end(),
                      std::ostream_iterator<float>(out," "));
            out << std::endl;
        }
        return true;
    }

    if (file_name.find(".trk") != std::string::npos)
    {
        std::ofstream out(file_name_,std::ios::binary);
        if (!out)
            return false;
        {
            TrackVis trk;
            trk.init(geometry,vs);
            trk.n_count = tract_data.size();
            out.write((const char*)&trk,1000);
        }
        for (unsigned int i = 0;i < tract_data.size();++i)
        {
            int n_point = tract_data[i].size()/3;
            std::vector<float> buffer(tract_data[i].size());
            const float *from = &*tract_data[i].begin();
            const float *end = from + tract_data[i].size();
            float* to = &*buffer.begin();
            for (unsigned int flag = 0;from != end;++from,++to)
            {
                *to = (*from)*vs[flag];
                ++flag;
                if (flag == 3)
                    flag = 0;
            }
            out.write((const char*)&n_point,sizeof(int));
            out.write((const char*)&*buffer.begin(),sizeof(float)*buffer.size());
        }
        return true;
    }
    if (file_name.find(".mat") != std::string::npos)
    {
        image::io::mat_write out(file_name.c_str());
        if(!out)
            return false;
        std::vector<float> buf;
        std::vector<unsigned int> length;
        for(unsigned int index = 0;index < tract_data.size();++index)
        {
            length.push_back(tract_data[index].size()/3);
            std::copy(tract_data[index].begin(),tract_data[index].end(),std::back_inserter(buf));
        }
        out.write("tracts",&*buf.begin(),3,buf.size()/3);
        out.write("length",&*length.begin(),1,length.size());
        return true;
    }
    return false;
}

//---------------------------------------------------------------------------
bool TractModel::save_all(const char* file_name_,const std::vector<TractModel*>& all)
{
    if(all.empty())
        return false;
    std::string file_name(file_name_);
    if (file_name.find(".txt") != std::string::npos)
    {
        std::ofstream out(file_name_,std::ios::binary);
        if (!out)
            return false;
        begin_prog("saving");
        for(unsigned int index = 0;check_prog(index,all.size());++index)
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

    if (file_name.find(".trk") != std::string::npos)
    {
        std::ofstream out(file_name_,std::ios::binary);
        if (!out)
            return false;
        {
            TrackVis trk;
            trk.init(all[0]->geometry,all[0]->vs);
            trk.n_count = 0;
            trk.n_properties = 1;
            for(unsigned int index = 0;index < all.size();++index)
                trk.n_count += all[index]->tract_data.size();
            out.write((const char*)&trk,1000);

        }
        begin_prog("saving");
        for(unsigned int index = 0;check_prog(index,all.size());++index)
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
        return true;
    }

    if (file_name.find(".mat") != std::string::npos)
    {
        image::io::mat_write out(file_name.c_str());
        if(!out)
            return false;
        std::vector<float> buf;
        std::vector<unsigned int> length;
        std::vector<unsigned int> cluster;
        for(unsigned int index = 0;check_prog(index,all.size());++index)
        for (unsigned int i = 0;i < all[index]->tract_data.size();++i)
        {
            cluster.push_back(index);
            length.push_back(all[index]->tract_data[i].size()/3);
            std::copy(all[index]->tract_data[i].begin(),all[index]->tract_data[i].end(),std::back_inserter(buf));
        }
        out.write("tracts",&*buf.begin(),3,buf.size()/3);
        out.write("length",&*length.begin(),1,length.size());
        out.write("cluster",&*cluster.begin(),1,cluster.size());
        return true;
    }
    return false;
}
//---------------------------------------------------------------------------
bool TractModel::save_transformed_tracts_to_file(const char* file_name,const float* transform,bool end_point)
{
    std::vector<std::vector<float> > new_tract_data(tract_data);
    for(unsigned int i = 0;i < tract_data.size();++i)
        for(unsigned int j = 0;j < tract_data[i].size();j += 3)
        image::vector_transformation(&(new_tract_data[i][j]),
                                    &(tract_data[i][j]),transform,image::vdim<3>());
    bool result = true;
    if(end_point)
        save_end_points(file_name);
    else
        result = save_tracts_to_file(file_name);
    new_tract_data.swap(tract_data);
    return result;
}
//---------------------------------------------------------------------------
bool TractModel::load_tracts_color_from_file(const char* file_name)
{
    std::ifstream in(file_name);
    if (!in)
        return false;
    std::vector<int> colors;
    std::copy(std::istream_iterator<int>(in),
              std::istream_iterator<int>(),
              std::back_inserter(colors));
    std::copy(colors.begin(),
              colors.begin()+std::min(colors.size(),tract_color.size()),
              tract_color.begin());
    return true;
}
//---------------------------------------------------------------------------
bool TractModel::save_tracts_color_to_file(const char* file_name)
{
    std::ofstream out(file_name);
    if (!out)
        return false;
    std::copy(tract_color.begin(),
              tract_color.end(),
              std::ostream_iterator<int>(out," "));
    return out;
}

//---------------------------------------------------------------------------
/*
bool TractModel::save_data_to_mat(const char* file_name,int index,const char* data_name)
{
    MatWriter mat_writer(file_name);
    if(!mat_writer.opened())
        return false;
    begin_prog("saving");
        for (unsigned int i = 0;check_prog(i,tract_data.size());++i)
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
void TractModel::save_end_points(const char* file_name_) const
{

    std::vector<float> buffer;
    buffer.reserve(tract_data.size() * 6);
    for (unsigned int index = 0;index < tract_data.size();++index)
    {
        unsigned int length = tract_data[index].size();
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
            return;
        std::copy(buffer.begin(),buffer.end(),std::ostream_iterator<float>(out," "));
    }
    if (file_name.find(".mat") != std::string::npos)
    {
        image::io::mat_write out(file_name_);
        if(!out)
            return;
        out.write("end_points",(const float*)&*buffer.begin(),3,buffer.size()/3);
    }

}
//---------------------------------------------------------------------------
void TractModel::get_end_points(std::vector<image::vector<3,short> >& points)
{
    for (unsigned int index = 0;index < tract_data.size();++index)
    {
        if (tract_data[index].size() < 3)
            return;
        image::vector<3,float> p1(&tract_data[index][0]);
        image::vector<3,float> p2(&tract_data[index][tract_data[index].size()-3]);
        p1 += 0.5;
        p2 += 0.5;
        points.push_back(image::vector<3,short>(std::floor(p1[0]),std::floor(p1[1]),std::floor(p1[2])));
        points.push_back(image::vector<3,short>(std::floor(p2[0]),std::floor(p2[1]),std::floor(p2[2])));
    }
}
//---------------------------------------------------------------------------
void TractModel::get_tract_points(std::vector<image::vector<3,short> >& points)
{
    for (unsigned int index = 0;index < tract_data.size();++index)
        for (unsigned int j = 0;j < tract_data[index].size();j += 3)
        {
            image::vector<3,short> point;
            point[0] = std::floor(tract_data[index][j]+0.5);
            point[1] = std::floor(tract_data[index][j+1]+0.5);
            point[2] = std::floor(tract_data[index][j+2]+0.5);
            points.push_back(point);
        }
}
//---------------------------------------------------------------------------
void TractModel::select(float select_angle,
                        const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                        const image::vector<3,float>& from_pos,std::vector<unsigned int>& selected)
{
    image::vector<3,float> z_axis = from_dir.cross_product(to_dir);
    z_axis.normalize();
    float view_angle = from_dir*to_dir;
    selected.resize(tract_data.size());
    std::fill(selected.begin(),selected.end(),0);

    float select_angle_cos = std::cos(select_angle*3.141592654/180);
    unsigned int total_track_number = tract_data.size();
    for (unsigned int index = 0;index < total_track_number;++index)
    {
        float angle = 0.0;
        const float* ptr = &*tract_data[index].begin();
        const float* end = ptr + tract_data[index].size();
        for (;ptr < end;ptr += 3)
        {
            image::vector<3,float> p(ptr);
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
                        image::vector<3,float> p1(ptr),p2(ptr-3);
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
//---------------------------------------------------------------------------
void TractModel::release_tracts(std::vector<std::vector<float> >& released_tracks)
{
    released_tracks.clear();
    released_tracks.swap(tract_data);
    tract_color.clear();
    redo_size.clear();
}
//---------------------------------------------------------------------------
void TractModel::delete_tracts(const std::vector<unsigned int>& tracts_to_delete)
{
    if (tracts_to_delete.empty())
        return;
    for (unsigned int index = 0;index < tracts_to_delete.size();++index)
    {
        deleted_tract_data.push_back(std::vector<float>());
        deleted_tract_data.back().swap(tract_data[tracts_to_delete[index]]);
        deleted_tract_color.push_back(tract_color[tracts_to_delete[index]]);
    }
    deleted_count.push_back(tracts_to_delete.size());
    // delete all blank tract
    std::vector<unsigned int> index_list(tract_data.size()+1);
    unsigned int new_ptr = 0;
    for (unsigned int index = 0;index < tract_data.size();++index)
    {
        index_list[index] = new_ptr;
        if (!tract_data[index].empty())
        {
            if(new_ptr != index)
            {
                tract_data[new_ptr].swap(tract_data[index]);
                std::swap(tract_color[new_ptr],tract_color[index]);
            }
            ++new_ptr;
        }
    }
    index_list.back() = new_ptr;
    tract_data.resize(new_ptr);
    // update redo_size

    for (unsigned int index = 0;index < redo_size.size();)
    {
        unsigned int from = index_list[redo_size[index].first];
        unsigned int size = index_list[redo_size[index].first+redo_size[index].second]-from;
        if(size)
        {
            redo_size[index].first = from;
            redo_size[index].second = size;
            ++index;
        }
        else
            redo_size.erase(redo_size.begin()+index);
    }
}
//---------------------------------------------------------------------------
void TractModel::select_tracts(const std::vector<unsigned int>& tracts_to_select)
{
    std::vector<unsigned> selected(tract_data.size());
    for (unsigned int index = 0;index < tracts_to_select.size();++index)
        selected[tracts_to_select[index]] = 1;

    std::vector<unsigned int> not_selected;
    not_selected.reserve(tract_data.size());

    for (unsigned int index = 0;index < selected.size();++index)
        if (!selected[index])
            not_selected.push_back(index);
    delete_tracts(not_selected);

}
//---------------------------------------------------------------------------
void TractModel::cut(float select_angle,const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                     const image::vector<3,float>& from_pos)
{
    std::vector<unsigned int> selected;
    select(select_angle,from_dir,to_dir,from_pos,selected);
    std::vector<std::vector<float> > new_tract;
    std::vector<unsigned int> new_tract_color;

    for (unsigned int index = 0;index < selected.size();++index)
        if (selected[index] && tract_data[index].size() > 6)
        {
            new_tract.push_back(std::vector<float>(tract_data[index].begin() + selected[index],tract_data[index].end()));
            new_tract_color.push_back(tract_color[index]);
            tract_data[index].resize(selected[index]);
        }
    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        tract_data.push_back(std::vector<float>());
        tract_data.back().swap(new_tract[index]);
        tract_color.push_back(new_tract_color[index]);
    }
    redo_size.clear();
}
//---------------------------------------------------------------------------
void TractModel::cull(float select_angle,
                      const image::vector<3,float>& from_dir,
                      const image::vector<3,float>& to_dir,
                      const image::vector<3,float>& from_pos,
                      bool delete_track)
{
    std::vector<unsigned int> selected;
    select(select_angle,from_dir,to_dir,from_pos,selected);
    std::vector<unsigned int> tracts_to_delete;
    tracts_to_delete.reserve(100 + (selected.size() >> 4));
    for (unsigned int index = 0;index < selected.size();++index)
        if (!((selected[index] > 0) ^ delete_track))
            tracts_to_delete.push_back(index);
    delete_tracts(tracts_to_delete);
}
//---------------------------------------------------------------------------
void TractModel::paint(float select_angle,const image::vector<3,float>& from_dir,const image::vector<3,float>& to_dir,
                       const image::vector<3,float>& from_pos,
                       unsigned int color)
{
    std::vector<unsigned int> selected;
    select(select_angle,from_dir,to_dir,from_pos,selected);
    for (unsigned int index = 0;index < selected.size();++index)
        if (selected[index] > 0)
            tract_color[index] = color;
}

//---------------------------------------------------------------------------
void TractModel::cut_by_mask(const char* file_name)
{
    /*
    std::ifstream in(file_name,std::ios::in);
    if(!in)
        return;
    std::set<image::vector<3,short> > mask(
                  (std::istream_iterator<image::vector<3,short> > (in)),
                  (std::istream_iterator<image::vector<3,short> > ()));
    std::vector<std::vector<float> > new_data;
    for (unsigned int index = 0;check_prog(index,tract_data.size());++index)
    {
        bool on = false;
        std::vector<float>::const_iterator iter = tract_data[index].begin();
        std::vector<float>::const_iterator end = tract_data[index].end();
        for (;iter < end;iter += 3)
        {
            image::vector<3,short> p(std::floor(iter[0]+0.5),
                                  std::floor(iter[1]+0.5),std::floor(iter[2]+0.5));

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
void TractModel::trim(void)
{
    image::basic_image<unsigned int,3> label(geometry);


    unsigned int total_track_number = tract_data.size();
    unsigned int no_fiber_label = total_track_number;
    unsigned int have_multiple_fiber_label = total_track_number+1;

    unsigned int width = label.width();
    unsigned int height = label.height();
    unsigned int depth = label.depth();
    unsigned int wh = width*height;
    std::fill(label.begin(),label.end(),no_fiber_label);
    for (unsigned int index = 0;index < total_track_number;++index)
    {
        const float* ptr = &*tract_data[index].begin();
        const float* end = ptr + tract_data[index].size();
        for (;ptr < end;ptr += 3)
        {
            int x = std::floor(*ptr+0.5);
            if (x < 0 || x >= width)
                continue;
            int y = std::floor(*(ptr+1)+0.5);
            if (y < 0 || y >= height)
                continue;
            int z = std::floor(*(ptr+2)+0.5);
            if (z < 0 || z >= depth)
                continue;
            unsigned int pixel_index = z*wh+y*width+x;
            if (pixel_index >= label.size())
                continue;
            unsigned int cur_label = label[pixel_index];
            if (cur_label == have_multiple_fiber_label || cur_label == index)
                continue;
            if (cur_label == no_fiber_label)
                label[pixel_index] = index;
            else
                label[pixel_index] = have_multiple_fiber_label;
        }
    }

    std::set<unsigned int> tracts_to_delete;
    for (unsigned int index = 0;index < label.size();++index)
        if (label[index] < total_track_number)
            tracts_to_delete.insert(label[index]);
    delete_tracts((std::vector<unsigned int>(tracts_to_delete.begin(),tracts_to_delete.end())));
}
//---------------------------------------------------------------------------
void TractModel::undo(void)
{
    if (deleted_count.empty())
        return;
    redo_size.push_back(std::make_pair((unsigned int)tract_data.size(),deleted_count.back()));
    for (unsigned int index = 0;index < deleted_count.back();++index)
    {
        tract_data.push_back(std::vector<float>());
        tract_data.back().swap(deleted_tract_data.back());
        tract_color.push_back(deleted_tract_color.back());
        deleted_tract_data.pop_back();
        deleted_tract_color.pop_back();
    }
    deleted_count.pop_back();
}
//---------------------------------------------------------------------------
void TractModel::redo(void)
{
    if(redo_size.empty())
        return;
    std::vector<unsigned int> redo_tracts(redo_size.back().second);
    for (unsigned int index = 0;index < redo_tracts.size();++index)
        redo_tracts[index] = redo_size.back().first + index;
    redo_size.pop_back();
    delete_tracts(redo_tracts);
}
//---------------------------------------------------------------------------
void TractModel::add_tracts(std::vector<std::vector<float> >& new_tract)
{
    tract_data.reserve(tract_data.size()+new_tract.size());
    image::rgb_color def_color(200,100,30);
    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        if (new_tract[index].empty())
            continue;
        tract_data.push_back(std::vector<float>());
        tract_data.back().swap(new_tract[index]);
        tract_color.push_back(def_color);
    }
}
//---------------------------------------------------------------------------
void TractModel::get_density_map(image::basic_image<unsigned int,3>& mapping,
                                 const std::vector<float>& transformation,bool endpoint)
{
    image::geometry<3> geometry = mapping.geometry();
    begin_prog("calculating");
    for (unsigned int i = 0;check_prog(i,tract_data.size());++i)
    {
        std::set<unsigned int> point_set;
        for (unsigned int j = 0;j < tract_data[i].size();j+=3)
        {
            if(j && endpoint)
                j = tract_data[i].size()-3;
            image::vector<3,float> tmp;
            image::vector_transformation(tract_data[i].begin()+j, tmp.begin(),
                transformation, image::vdim<3>());

            int x = std::floor(tmp[0]+0.5);
            int y = std::floor(tmp[1]+0.5);
            int z = std::floor(tmp[2]+0.5);
            if (!geometry.is_valid(x,y,z))
                continue;
            point_set.insert((z*mapping.height()+y)*mapping.width()+x);
        }

        std::vector<unsigned int> point_list(point_set.begin(),point_set.end());
        for(unsigned int j = 0;j < point_list.size();++j)
            ++mapping[point_list[j]];
    }
}
//---------------------------------------------------------------------------
void TractModel::get_density_map(
        image::basic_image<image::rgb_color,3>& mapping,
        const std::vector<float>& transformation,bool endpoint)
{
    image::geometry<3> geometry = mapping.geometry();
    image::basic_image<float,3> map_r(geometry),
                            map_g(geometry),map_b(geometry);
    for (unsigned int i = 0;i < tract_data.size();++i)
    {
        const float* buf = &*tract_data[i].begin();
        for (unsigned int j = 3;j < tract_data[i].size();j+=3)
        {
            if(j > 3 && endpoint)
                j = tract_data[i].size()-3;
            image::vector<3,float>  tmp,dir;
            image::vector_transformation(buf+j-3, dir.begin(),
                transformation, image::vdim<3>());
            image::vector_transformation(buf+j, tmp.begin(),
                transformation, image::vdim<3>());
            dir -= tmp;
            dir.normalize();
            int x = std::floor(tmp[0]+0.5);
            int y = std::floor(tmp[1]+0.5);
            int z = std::floor(tmp[2]+0.5);
            if (!geometry.is_valid(x,y,z))
                continue;
            unsigned int ptr = (z*mapping.height()+y)*mapping.width()+x;
            map_r[ptr] += std::fabs(dir[0]);
            map_g[ptr] += std::fabs(dir[1]);
            map_b[ptr] += std::fabs(dir[2]);
        }
    }
    float max_value = 0;
    for(unsigned int index = 0;index < mapping.size();++index)
    {
        float sum = map_r[index]+map_g[index]+map_b[index];
        if(sum > max_value)
            max_value = sum;
    }
    for(unsigned int index = 0;index < mapping.size();++index)
    {
        float sum = map_r[index]+map_g[index]+map_b[index];
        image::vector<3,float> cmap(map_r[index],map_g[index],map_b[index]);
        cmap.normalize();
        cmap *= 255.0*sum/max_value;
        mapping[index] = image::rgb_color(cmap[0],cmap[1],cmap[2]);
    }
}

void TractModel::save_tdi(const char* file_name,bool sub_voxel,bool endpoint)
{
    std::vector<float> tr(16);
    tr[0] = tr[5] = tr[10] = tr[15] = (sub_voxel ? 4.0:1.0);
    image::vector<3,float> new_vs(vs);
    if(sub_voxel)
        new_vs /= 4.0;
    image::basic_image<unsigned int,3> tdi;

    if(sub_voxel)
        tdi.resize(image::geometry<3>(geometry[0]*4,geometry[1]*4,geometry[2]*4));
    else
        tdi.resize(geometry);

    get_density_map(tdi,tr,endpoint);
    gz_nifti nii_header;
    image::flip_xy(tdi);
    nii_header << tdi;
    nii_header.set_voxel_size(new_vs.begin());
    nii_header.save_to_file(file_name);

}


void TractModel::get_quantitative_data(std::vector<float>& data)
{
    if(tract_data.empty())
        return;
    float voxel_volume = vs[0]*vs[1]*vs[2];

    data.push_back(tract_data.size());

    // mean length
    {
        float sum_length = 0.0;
        float sum_length2 = 0.0;
        for (unsigned int i = 0;i < tract_data.size();++i)
        {
            float length = 0.0;
            for (unsigned int j = 3;j < tract_data[i].size();j += 3)
            {
                length += image::vector<3,float>(
                    vs[0]*(tract_data[i][j]-tract_data[i][j-3]),
                    vs[1]*(tract_data[i][j+1]-tract_data[i][j-2]),
                    vs[2]*(tract_data[i][j+2]-tract_data[i][j-1])).length();

            }
            sum_length += length;
            sum_length2 += length*length;
        }
        data.push_back(sum_length/((float)tract_data.size()));
        data.push_back(std::sqrt(sum_length2/(double)tract_data.size()-
                                 sum_length*sum_length/(double)tract_data.size()/(double)tract_data.size()));
    }


    // tract volume
    {

        std::set<image::vector<3,int> > pass_map;
        for (unsigned int i = 0;i < tract_data.size();++i)
            for (unsigned int j = 0;j < tract_data[i].size();j += 3)
                pass_map.insert(image::vector<3,int>(std::floor(tract_data[i][j]+0.5),
                                              std::floor(tract_data[i][j+1]+0.5),
                                              std::floor(tract_data[i][j+2]+0.5)));

        data.push_back(pass_map.size()*voxel_volume);
    }

    // output mean and std of each index
    for(int data_index = 0;
        data_index < handle->fib_data.view_item.size();++data_index)
    {
        if(data_index > 0 && data_index < handle->fib_data.other_mapping_index)
            continue;

        float sum_data = 0.0;
        float sum_data2 = 0.0;
        unsigned int total = 0;
        for (unsigned int i = 0;i < tract_data.size();++i)
        {
            std::vector<float> data;
            if(data_index == 0)
                get_tract_fa(i,data);
            else
                get_tract_data(i,data_index,data);
            for(int j = 0;j < data.size();++j)
            {
                float value = data[j];
                sum_data += value;
                sum_data2 += value*value;
                ++total;
            }
        }

        data.push_back(sum_data/((double)total));
        data.push_back(std::sqrt(sum_data2/(double)total-sum_data*sum_data/(double)total/(double)total));
    }
}

void TractModel::get_quantitative_info(std::string& result)
{
    if(tract_data.empty())
        return;
    std::ostringstream out;
    std::vector<std::string> titles;
    std::vector<float> data;
    titles.push_back("number of tracts");
    titles.push_back("tract length mean(mm)");
    titles.push_back("tract length sd(mm)");
    titles.push_back("tracts volume (mm^3)");
    handle->get_index_titles(titles);
    get_quantitative_data(data);
    for(unsigned int index = 0;index < data.size() && index < titles.size();++index)
        out << titles[index] << "\t" << data[index] << std::endl;
    result = out.str();
}


void TractModel::get_report(unsigned int profile_dir,float band_width,const std::string& index_name,
                            std::vector<float>& values,
                            std::vector<float>& data_profile)
{
    if(tract_data.empty())
        return;
    int profile_on_length = 0;// 1 :along tract 2: mean value
    if(profile_dir > 2)
    {
        profile_on_length = profile_dir-2;
        profile_dir = 0;
    }
    double detail = profile_on_length ? 1.0 : 2.0;
    unsigned int profile_width = (geometry[profile_dir]+1)*detail;


    std::vector<float> weighting((int)(1.0+band_width*3.0));
    for(int index = 0;index < weighting.size();++index)
    {
        float x = index;
        weighting[index] = std::exp(-x*x/2.0/band_width/band_width);
    }
    // along tract profile
    if(profile_on_length == 1)
        profile_width = tract_data[0].size()/2.0;
    // mean value of each tract
    if(profile_on_length == 2)
        profile_width = tract_data.size();

    values.resize(profile_width);
    data_profile.resize(profile_width);
    std::vector<float> data_profile_w(profile_width);


    {
        std::vector<std::vector<float> > data;
        if(index_name == "qa" || index_name == "fa")
            get_tracts_fa(data);
        else
            get_tracts_data(index_name,data);

        if(profile_on_length == 2)// list the mean fa value of each tract
        {
            data_profile.resize(data.size());
            data_profile_w.resize(data.size());
            for(unsigned int index = 0;index < data_profile.size();++index)
            {
                data_profile[index] = image::mean(data[index].begin(),data[index].end());
                data_profile_w[index] = 1.0;
            }
        }
        else
            for(int i = 0;i < data.size();++i)
                for(int j = 0;j < data[i].size();++j)
                {
                    int pos = profile_on_length ?
                              j*(int)profile_width/data[i].size() :
                              std::floor(tract_data[i][j + j + j + profile_dir]*detail+0.5);
                    if(pos < 0)
                        pos = 0;
                    if(pos >= profile_width)
                        pos = profile_width-1;

                    data_profile[pos] += data[i][j]*weighting[0];
                    data_profile_w[pos] += weighting[0];
                    for(int k = 1;k < weighting.size();++k)
                    {
                        if(pos > k)
                        {
                            data_profile[pos-k] += data[i][j]*weighting[k];
                            data_profile_w[pos-k] += weighting[k];
                        }
                        if(pos+k < data_profile.size())
                        {
                            data_profile[pos+k] += data[i][j]*weighting[k];
                            data_profile_w[pos+k] += weighting[k];
                        }
                    }
                }
    }

    for(unsigned int j = 0;j < data_profile.size();++j)
    {
        values[j] = (double)j/detail;
        if(data_profile_w[j] + 1.0 != 1.0)
            data_profile[j] /= data_profile_w[j];
        else
            data_profile[j] = 0.0;
    }
}


void TractModel::get_tract_data(unsigned int fiber_index,
                    unsigned int index_num,
                    std::vector<float>& data)
{
    data.clear();
    if(index_num >= handle->fib_data.view_item.size())
        return;
    data.resize(tract_data[fiber_index].size()/3);
    for (unsigned int data_index = 0,index = 0;index < tract_data[fiber_index].size();index += 3,++data_index)
        image::linear_estimate(handle->fib_data.view_item[index_num].image_data,&(tract_data[fiber_index][index]),data[data_index]);
}

void TractModel::get_tracts_data(
        const std::string& index_name,
        std::vector<std::vector<float> >& data)
{
    data.clear();
    unsigned int index_num = handle->get_name_index(index_name);
    if(index_num == handle->fib_data.view_item.size())
        return;
    data.resize(tract_data.size());
    for (unsigned int i = 0;i < tract_data.size();++i)
        get_tract_data(i,index_num,data[i]);
}

void TractModel::get_tract_fa(unsigned int fiber_index,std::vector<float>& data)
{
    unsigned int count = tract_data[fiber_index].size()/3;
    data.resize(count);
    if(tract_data[fiber_index].empty())
        return;
    std::vector<image::vector<3,float> > gradient(count);
    const float (*tract_ptr)[3] = (const float (*)[3])&(tract_data[fiber_index][0]);
    ::gradient(tract_ptr,tract_ptr+count,gradient.begin());
    for (unsigned int point_index = 0,tract_index = 0;
         point_index < count;++point_index,tract_index += 3)
    {
        image::interpolation<image::linear_weighting,3> tri_interpo;
        gradient[point_index].normalize();
        if (tri_interpo.get_location(fib->dim,&(tract_data[fiber_index][tract_index])))
        {
            float value,average_value = 0.0;
            float sum_value = 0.0;
            for (unsigned int index = 0;index < 8;++index)
            {
                if ((value = fib->get_fa(tri_interpo.dindex[index],gradient[point_index])) == 0.0)
                    continue;
                average_value += value*tri_interpo.ratio[index];
                sum_value += tri_interpo.ratio[index];
            }
            if (sum_value > 0.5)
                data[point_index] = average_value/sum_value;
            else
            data[point_index] = fib->threshold;
        }
        else
            data[point_index] = fib->threshold;
    }
}
void TractModel::get_tracts_fa(std::vector<std::vector<float> >& data)
{
    data.resize(tract_data.size());
    for(unsigned int index = 0;index < tract_data.size();++index)
        get_tract_fa(index,data[index]);
}

double TractModel::get_spin_volume(void)
{
    std::map<image::vector<3,short>,image::vector<3,float> > passing_regions;
    for (unsigned int i = 0;i < tract_data.size();++i)
    {
        std::vector<image::vector<3,float> > point(tract_data[i].size() / 3);
        std::vector<image::vector<3,float> > gradient(tract_data[i].size() / 3);
        for (unsigned int j = 0,index = 0;j < tract_data[i].size();j += 3,++index)
            point[index] = &(tract_data[i][j]);

        ::gradient(point.begin(),point.end(),gradient.begin());

        for (unsigned int j = 0;j < point.size();++j)
        {
            gradient[j].normalize();
            point[j] += 0.5;
            point[j].floor();
            passing_regions[image::vector<3,short>(point[j])] += gradient[j];
        }
    }
    double result = 0.0;
    std::map<image::vector<3,short>,image::vector<3,float> >::iterator iter = passing_regions.begin();
    std::map<image::vector<3,short>,image::vector<3,float> >::iterator end = passing_regions.end();
    for (;iter != end;++iter)
    {
        iter->second.normalize();
        result += fib->get_fa(
                      image::pixel_index<3>(iter->first[0],iter->first[1],iter->first[2],fib->dim).index(),iter->second);
    }
    return result;
}

void TractModel::get_connectivity_matrix(const std::vector<std::vector<image::vector<3,short> > >& regions,
                                         std::vector<std::vector<connectivity_info> >& matrix,
                                         bool use_end_only) const
{
    matrix.clear();
    matrix.resize(regions.size());
    for(unsigned int index = 0;index < regions.size();++index)
        matrix[index].resize(regions.size());

    // create regions maps
    std::vector<std::vector<short> > region_map(geometry.size());
    {
        std::vector<std::set<short> > regions_set(geometry.size());
        for(unsigned int roi = 0;roi < regions.size();++roi)
        {
            for(unsigned int index = 0;index < regions[roi].size();++index)
            {
                image::vector<3,short> pos = regions[roi][index];
                regions_set[image::pixel_index<3>(pos[0],pos[1],pos[2],geometry).index()].insert(roi);
            }
        }

        for(unsigned int index = 0;index < geometry.size();++index)
            if(!regions_set[index].empty())
                region_map[index] = std::vector<short>(regions_set[index].begin(),regions_set[index].end());
    }

    for(unsigned int index = 0;index < tract_data.size();++index)
    {
        if(tract_data[index].size() < 6)
            continue;
        std::vector<unsigned char> has_region(regions.size());
        for(unsigned int ptr = 0;ptr < tract_data[index].size();ptr += 3)
        {
            image::pixel_index<3> pos(std::floor(tract_data[index][ptr]+0.5),
                                        std::floor(tract_data[index][ptr+1]+0.5),
                                        std::floor(tract_data[index][ptr+2]+0.5),geometry);
            if(!geometry.is_valid(pos))
                continue;
            unsigned int pos_index = pos.index();
            for(unsigned int j = 0;j < region_map[pos_index].size();++j)
                has_region[region_map[pos_index][j]] = 1;
            if(!ptr && use_end_only)
                ptr = tract_data[index].size()-6;
        }
        std::vector<unsigned int> region_list;
        for(unsigned int i = 0;i < has_region.size();++i)
            if(has_region[i])
                region_list.push_back(i);
        for(unsigned int i = 0;i < region_list.size();++i)
            for(unsigned int j = i+1;j < region_list.size();++j)
            {
                matrix[region_list[i]][region_list[j]].add(tract_data[index]);
                matrix[region_list[j]][region_list[i]].add(tract_data[index]);
            }
    }
}




void ConnectivityMatrix::save_to_image(image::color_image& cm,bool log,bool norm)
{
    if(matrix.empty())
        return;
    cm.resize(image::geometry<2>(matrix.size(),matrix.size()));
    std::vector<float> values(cm.size());
    std::copy(connectivity_count.begin(),connectivity_count.end(),values.begin());
    for(unsigned int index = 0;index < values.size();++index)
    {
        if(log)
            values[index] = std::log(values[index] + 1.0);
        if(norm && tract_median_length[index] > 0)
            values[index] /= tract_median_length[index];
    }
    image::normalize(values,255.99);
    for(unsigned int index = 0;index < values.size();++index)
    {
        cm[index] = image::rgb_color((unsigned char)values[index],(unsigned char)values[index],(unsigned char)values[index]);
    }
}

void ConnectivityMatrix::save_to_file(const char* file_name)
{
    image::io::mat_write mat_header(file_name);
    mat_header.write("connectivity",&*connectivity_count.begin(),matrix.size(),matrix.size());
    mat_header.write("tract_median_length",&*tract_median_length.begin(),matrix.size(),matrix.size());
    mat_header.write("tract_mean_length",&*tract_mean_length.begin(),matrix.size(),matrix.size());
    std::ostringstream out;
    std::copy(region_name.begin(),region_name.end(),std::ostream_iterator<std::string>(out,"\n"));
    std::string result(out.str());
    mat_header.write("name",result.c_str(),1,result.length());
}

void ConnectivityMatrix::set_atlas(const atlas& data,const image::basic_image<image::vector<3,float>,3 >& mni_position)
{
    image::geometry<3> geo(mni_position.geometry());
    region_table_type region_table;
    image::vector<3> null;
    for (unsigned int label = 0; label < data.get_list().size(); ++label)
    {
        std::vector<image::vector<3,short> > cur_region;
        image::vector<3,float> mni_avg_pos;
        float min_x = 200,max_x = -200;
        for (image::pixel_index<3>index; index.is_valid(geo);index.next(geo))
            if (mni_position[index.index()] != null &&
                data.label_matched(data.get_label_at(mni_position[index.index()]),label))
            {
                cur_region.push_back(image::vector<3,short>(index.begin()));
                mni_avg_pos += mni_position[index.index()];
                float x = mni_position[index.index()][0];
                if(x > max_x)
                   max_x = x;
                if(x < min_x)
                   min_x = x;
            }
        if(cur_region.empty())
            continue;
        mni_avg_pos /= cur_region.size();
        const std::vector<std::string>& region_names = data.get_list();
        float order;
        if(mni_avg_pos[0] > 0)
            order = 500.0-mni_avg_pos[1];
        else
            order = mni_avg_pos[1]-500.0;
        // is at middle?
        if((max_x-min_x)/8.0 > std::fabs(mni_avg_pos[0]))
            order = mni_avg_pos[1];
        region_table[order] = std::make_pair(cur_region,region_names[label]);
    }
    set_regions(region_table);
}

void ConnectivityMatrix::set_regions(const region_table_type& region_table)
{
    regions.resize(region_table.size());
    region_name.resize(region_table.size());
    region_table_type::const_iterator iter = region_table.begin();
    region_table_type::const_iterator end = region_table.end();
    for(unsigned int index = 0;iter != end;++iter,++index)
    {
        regions[index] = iter->second.first;
        region_name[index] = iter->second.second;
        // replace space by _
        std::replace(region_name[index].begin(),region_name[index].end(),' ','_');
    }
}

void ConnectivityMatrix::calculate(const TractModel& tract_model,bool use_end_only)
{
    if(regions.size() == 0)
        return;

    tract_model.get_connectivity_matrix(regions,matrix,use_end_only);
    connectivity_count.resize(matrix.size()*matrix.size());
    tract_median_length.resize(matrix.size()*matrix.size());
    tract_mean_length.resize(matrix.size()*matrix.size());

    for(unsigned int i = 0,pos = 0;i < matrix.size();++i)
        for(unsigned int j = 0;j < matrix[i].size();++j,++pos)
        {
            connectivity_count[pos] = matrix[i][j].count;
            if(!connectivity_count[pos])
            {
                tract_median_length[pos] = 0;
                tract_mean_length[pos] = 0;
                continue;
            }
            std::nth_element(matrix[i][j].length.begin(),
                             matrix[i][j].length.begin()+(matrix[i][j].length.size() >> 1),
                             matrix[i][j].length.end());
            tract_mean_length[pos] = image::mean(matrix[i][j].length.begin(),matrix[i][j].length.end());
            tract_median_length[pos] = matrix[i][j].length[matrix[i][j].length.size() >> 1];
        }
}
