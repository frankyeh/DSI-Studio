//---------------------------------------------------------------------------
#include <QString>
#include <fstream>
#include <sstream>
#include <iterator>
#include <set>
#include <map>
#include "roi.hpp"
#include "tract_model.hpp"
#include "prog_interface_static_link.h"
#include "fib_data.hpp"
#include "gzip_interface.hpp"
#include "mapping/atlas.hpp"
#include "gzip_interface.hpp"
#include "tract_cluster.hpp"
#include "../../tracking/region/Regions.h"

void smoothed_tracks(const std::vector<float>& track,std::vector<float>& smoothed)
{
    smoothed.clear();
    smoothed.resize(track.size());
    float w[5] = {1.0,2.0,4.0,2.0,1.0};
    int shift[5] = {-6, -3, 0, 3, 6};
    for(int index = 0;index < track.size();++index)
    {
        float sum_w = 0.0;
        float sum = 0.0;
        for(char i = 0;i < 5;++i)
        {
            int cur_index = index + shift[i];
            if(cur_index < 0 || cur_index >= track.size())
                continue;
            sum += w[i]*track[cur_index];
            sum_w += w[i];
        }
        if(sum_w != 0.0)
            smoothed[index] = sum/sum_w;
    }
}

void resample_tracks(const std::vector<float>& track,std::vector<float>& new_track,float interval)
{
    if(track.size() < 6)
        return;
    float step_size = image::vector<3>(track[0]-track[3],track[1]-track[4],track[2]-track[5]).length();
    if(std::fabs(step_size - interval)/interval < 0.1)
        return;
    float dis = interval;
    image::vector<3> v1(&track[0]),v2,v3;
    new_track.push_back(track[0]);
    new_track.push_back(track[1]);
    new_track.push_back(track[2]);
    bool new_step = true;
    float now_dis = step_size;
    for(int i = 3;i < track.size();)
    {
        if(new_step)
        {
            v2 = image::vector<3>(&(track[i]));
            v3[0] = v2[0]-v1[0];
            v3[1] = v2[1]-v1[1];
            v3[2] = v2[2]-v1[2];
            v3 *= 1.0f/step_size;
            new_step = false;
        }
        if(dis > now_dis)
        {
            dis -= now_dis;
            v1 = v2;
            i += 3;
            now_dis = step_size;
            new_step = true;
            continue;
        }
        now_dis -= dis;
        v1[0] += v3[0]*dis;
        v1[1] += v3[1]*dis;
        v1[2] += v3[2]*dis;
        dis = interval;
        new_track.push_back(v1[0]);
        new_track.push_back(v1[1]);
        new_track.push_back(v1[2]);
    }
    if(dis < interval*0.1f)
    {
        new_track.push_back(track[track.size()-3]);
        new_track.push_back(track[track.size()-2]);
        new_track.push_back(track[track.size()-1]);
    }
}


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
        vox_to_ras[0][0] = -voxel_size[0]; // L to R
        vox_to_ras[1][1] = -voxel_size[1]; // P to A
        vox_to_ras[2][2] = voxel_size[2];
        vox_to_ras[3][3] = 1;
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
TractModel::TractModel(std::shared_ptr<fib_data> handle_):handle(handle_),
        report(handle_->report),geometry(handle_->dim),vs(handle_->vs),fib(new tracking_data)
{
    fib->read(*handle_);
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
    deleted_cut_count.insert(deleted_cut_count.end(),
                              rhs.deleted_cut_count.begin(),
                              rhs.deleted_cut_count.end());
}
//---------------------------------------------------------------------------
bool TractModel::load_from_file(const char* file_name_,bool append)
{
    std::string file_name(file_name_);
    std::vector<std::vector<float> > loaded_tract_data;
    std::vector<unsigned int> loaded_tract_cluster;

    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    if(ext == std::string(".trk") || ext == std::string("k.gz"))
        {
            TrackVis trk;
            gz_istream in;
            if (!in.open(file_name_))
                return false;
            in.read((char*)&trk,1000);
            unsigned int track_number = trk.n_count;
            if(!track_number) // number is not stored
                track_number = 100000000;
            begin_prog("loading");
            for (unsigned int index = 0;!(!in) && check_prog(index,track_number);++index)
            {
                unsigned int n_point;
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
                    if(trk.voxel_order[1] == 'R')
                        to[0] = trk.dim[0]-x-1;
                    else
                        to[0] = x;
                    if(trk.voxel_order[1] == 'A')
                        to[1] = trk.dim[1]-y-1;
                    else
                        to[1] = y;
                    to[2] = z;
                }
                if(trk.n_properties == 1)
                    loaded_tract_cluster.push_back(from[0]);
            }
        }
        else
        if (ext == std::string(".txt"))
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

                if(loaded_tract_data.back().size() == 1)// cluster info
                    loaded_tract_cluster.push_back(loaded_tract_data.back()[0]);
            }

        }
        else
            if (ext == std::string(".mat"))
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
                if (ext == std::string(".tck"))
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
                        image::divide_constant(loaded_tract_data.back().begin(),loaded_tract_data.back().end(),handle->vs[0]);
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
    deleted_count.clear();
    deleted_cut_count.clear();
    redo_size.clear();
    return true;
}

//---------------------------------------------------------------------------
bool TractModel::save_data_to_file(const char* file_name,const std::string& index_name)
{
    std::vector<std::vector<float> > data;
    if(!get_tracts_data(index_name,data) || data.empty())
        return false;

    std::string file_name_s(file_name);
    std::string ext;
    if(file_name_s.length() > 4)
        ext = std::string(file_name_s.end()-4,file_name_s.end());

    if (ext == std::string(".txt"))
    {
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
    if (ext == std::string(".mat"))
    {
        image::io::mat_write out(file_name);
        if(!out)
            return false;
        std::vector<float> buf;
        std::vector<unsigned int> length;
        for(unsigned int index = 0;index < data.size();++index)
        {
            length.push_back((unsigned int)data[index].size());
            std::copy(data[index].begin(),data[index].end(),std::back_inserter(buf));
        }
        out.write("data",&*buf.begin(),1,(unsigned int)buf.size());
        out.write("length",&*length.begin(),1,(unsigned int)length.size());
        return true;
    }

    return false;
}
//---------------------------------------------------------------------------
bool TractModel::save_tracts_in_native_space(const char* file_name,image::basic_image<image::vector<3,float>,3 > native_position)
{
    std::vector<std::vector<float> > keep_tract_data(tract_data);
    image::par_for(tract_data.size(),[&](int i)
    {
        for(int j = 0;j < tract_data[i].size();j += 3)
        {
            image::vector<3> pos(&tract_data[i][0]+j),new_pos;
            image::estimate(native_position,pos,new_pos);
            tract_data[i][j] = new_pos[0];
            tract_data[i][j+1] = new_pos[1];
            tract_data[i][j+2] = new_pos[2];
        }
    });
    bool result = save_tracts_to_file(file_name);
    keep_tract_data.swap(tract_data);
    return result;
}
//---------------------------------------------------------------------------
bool TractModel::save_tracts_to_file(const char* file_name_)
{
    std::string file_name(file_name_);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());
    if (ext == std::string(".trk") || ext == std::string("k.gz"))
    {
        if(ext == std::string(".trk"))
            file_name += ".gz";
        gz_ostream out;
        if (!out.open(file_name.c_str()))
            return false;
        {
            TrackVis trk;
            trk.init(geometry,vs);
            trk.n_count = tract_data.size();
            out.write((const char*)&trk,1000);
        }
        begin_prog("saving");
        for (unsigned int i = 0;check_prog(i,tract_data.size());++i)
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
    if (ext == std::string(".txt"))
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
    if (ext == std::string(".mat"))
    {
        image::io::mat_write out(file_name.c_str());
        if(!out)
            return false;
        std::vector<float> buf;
        std::vector<unsigned int> length;
        for(unsigned int index = 0;index < tract_data.size();++index)
        {
            length.push_back((unsigned int)tract_data[index].size()/3);
            std::copy(tract_data[index].begin(),tract_data[index].end(),std::back_inserter(buf));
        }
        out.write("tracts",&*buf.begin(),3,(unsigned int)buf.size()/3);
        out.write("length",&*length.begin(),1,(unsigned int)length.size());
        return true;
    }
    if (ext == std::string(".nii") || ext == std::string("i.gz"))
    {
        std::vector<image::vector<3,float> >points;
        get_tract_points(points);
        ROIRegion region(geometry,vs);
        region.add_points(points,false);
        std::vector<float> no_trans;
        region.SaveToFile(file_name_,handle->is_qsdr ? handle->trans_to_mni: no_trans);
        return true;
    }
    return false;
}
void TractModel::save_vrml(const char* file_name,
                           unsigned char tract_style,
                           unsigned char tract_color_style,
                           float tube_diameter,
                           unsigned char tract_tube_detail,
                           const std::string& surface_text)
{
    std::ofstream out(file_name);
    out << "#VRML V2.0 utf8" << std::endl;
    out << "Viewpoint { description \"Initial view\" position 0 0 9 }" << std::endl;
    out << "NavigationInfo { type \"EXAMINE\" }" << std::endl;




    std::vector<image::vector<3,float> > points(8),previous_points(8);
    image::rgb_color paint_color;
    image::vector<3,float> paint_color_f;

    const float detail_option[] = {1.0,0.5,0.25,0.0,0.0};
    float tube_detail = tube_diameter*detail_option[tract_tube_detail]*4.0;


    QString Coordinate,CoordinateIndex,Color;
    unsigned int vrml_coordinate_count = 0,vrml_color_count = 0;
    for (unsigned int data_index = 0; data_index < tract_data.size(); ++data_index)
    {
        unsigned int vertex_count = get_tract_length(data_index)/3;
        if (vertex_count <= 1)
            continue;

        const float* data_iter = &tract_data[data_index][0];

        switch(tract_color_style)
        {
        case 1:
            paint_color = get_tract_color(data_index);
            paint_color_f = image::vector<3,float>(paint_color.r,paint_color.g,paint_color.b);
            paint_color_f /= 255.0;
            break;
        break;
        }
        unsigned int prev_coordinate_count = vrml_coordinate_count;
        image::vector<3,float> last_pos(data_iter),
                vec_a(1,0,0),vec_b(0,1,0),
                vec_n,prev_vec_n,vec_ab,vec_ba,cur_color;

        for (unsigned int j = 0, index = 0; index < vertex_count;j += 3, data_iter += 3, ++index)
        {
            image::vector<3,float> pos(data_iter);
            if (index + 1 < vertex_count)
            {
                vec_n[0] = data_iter[3] - data_iter[0];
                vec_n[1] = data_iter[4] - data_iter[1];
                vec_n[2] = data_iter[5] - data_iter[2];
                vec_n.normalize();
            }

            switch(tract_color_style)
            {
            case 0://directional
                cur_color[0] = std::fabs(vec_n[0]);
                cur_color[1] = std::fabs(vec_n[1]);
                cur_color[2] = std::fabs(vec_n[2]);
                break;
            case 1://manual assigned
                cur_color = paint_color_f;
                break;
            }

            if (index != 0 && index+1 != vertex_count)
            {
                image::vector<3,float> displacement(data_iter+3);
                displacement -= last_pos;
                displacement -= prev_vec_n*(prev_vec_n*displacement);
                if (displacement.length() < tube_detail)
                    continue;
            }
            // add end
            if(tract_style == 0)// line
            {
                Coordinate += QString("%1 %2 %3 ").arg(pos[0]).arg(pos[1]).arg(pos[2]);
                Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
                prev_vec_n = vec_n;
                last_pos = pos;
                ++vrml_coordinate_count;
                ++vrml_color_count;
                continue;
            }

            if (index == 0 && std::fabs(vec_a*vec_n) > 0.5)
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

            static const unsigned char end_sequence[8] = {4,3,5,2,6,1,7,0};
            if (index == 0)
            {
                for (unsigned int k = 0;k < 8;++k)
                {
                    Coordinate += QString("%1 %2 %3 ").arg(points[end_sequence[k]][0]).arg(points[end_sequence[k]][1]).arg(points[end_sequence[k]][2]);
                    Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
                }
                vrml_coordinate_count+=8;
            }
            else
            // add tube
            {

                Coordinate += QString("%1 %2 %3 ").arg(points[0][0]).arg(points[0][1]).arg(points[0][2]);
                Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
                for (unsigned int k = 1;k < 8;++k)
                {
                    Coordinate += QString("%1 %2 %3 ").arg(previous_points[k][0]).arg(previous_points[k][1]).arg(previous_points[k][2]);
                    Coordinate += QString("%1 %2 %3 ").arg(points[k][0]).arg(points[k][1]).arg(points[k][2]);
                    Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
                    Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
                }
                Coordinate += QString("%1 %2 %3 ").arg(points[0][0]).arg(points[0][1]).arg(points[0][2]);
                Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
                vrml_coordinate_count+=16;

                if(index +1 == vertex_count)
                {
                    for (int k = 7;k >= 0;--k)
                    {
                        Coordinate += QString("%1 %2 %3 ").arg(points[end_sequence[k]][0]).arg(points[end_sequence[k]][1]).arg(points[end_sequence[k]][2]);
                        Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
                    }
                    vrml_coordinate_count+=8;
                }
            }
            previous_points.swap(points);
            prev_vec_n = vec_n;
            last_pos = pos;
            ++vrml_color_count;

        }

        if(tract_style == 0)// line
        {
            for (unsigned int j = prev_coordinate_count;j < vrml_coordinate_count;++j)
                CoordinateIndex += QString("%1 ").arg(j);
            CoordinateIndex += QString("-1 ");
        }
        else
        {
            for (unsigned int j = prev_coordinate_count;j+2 < vrml_coordinate_count;++j)
                CoordinateIndex += QString("%1 %2 %3 -1 ").arg(j).arg(j+1).arg(j+2);
        }
    }


    if(tract_style == 0)// line
    {

        out << "Shape {" << std::endl;
        out << "geometry IndexedLineSet {" << std::endl;
        out << "coord Coordinate { point [" << Coordinate.toStdString() << " ] }" << std::endl;
        out << "color Color { color ["<< Color.toStdString() <<"] } " << std::endl;
        out << "coordIndex ["<< CoordinateIndex.toStdString() <<"] }" << std::endl;
    }
    else
    {

        out << "Shape {" << std::endl;
        out << "appearance Appearance { " << std::endl;
        out << "material Material { " << std::endl;
        out << "ambientIntensity 0.0" << std::endl;
        out << "diffuseColor 0.6 0.6 0.6" << std::endl;
        out << "specularColor 0.1 0.1 0.1" << std::endl;
        out << "emissiveColor 0.0 0.0 0.0" << std::endl;
        out << "shininess 0.1" << std::endl;
        out << "transparency 0" << std::endl;
        out << "} }" << std::endl;
        out << "geometry IndexedFaceSet {" << std::endl;
        out << "creaseAngle 3.14" << std::endl;
        out << "solid TRUE" << std::endl;
        out << "coord Coordinate { point [" << Coordinate.toStdString() << " ] }" << std::endl;
        out << "color Color { color ["<< Color.toStdString() <<"] }" << std::endl;
        out << "coordIndex ["<< CoordinateIndex.toStdString() <<"] } }" << std::endl;
    }

    out << surface_text;
}

//---------------------------------------------------------------------------
bool TractModel::save_all(const char* file_name_,const std::vector<TractModel*>& all)
{
    if(all.empty())
        return false;
    std::string file_name(file_name_);
    std::string ext;
    if(file_name.length() > 4)
        ext = std::string(file_name.end()-4,file_name.end());

    if (ext == std::string(".txt"))
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

    if (ext == std::string(".trk") || ext == std::string("k.gz"))
    {
        gz_ostream out;
        if (!out.open(file_name_))
            return false;
        begin_prog("saving");
        {
            TrackVis trk;
            trk.init(all[0]->geometry,all[0]->vs);
            trk.n_count = 0;
            trk.n_properties = 1;
            std::copy(all[0]->report.begin(),all[0]->report.begin()+std::min<int>(444,all[0]->report.length()),trk.reserved);
            for(unsigned int index = 0;index < all.size();++index)
                trk.n_count += all[index]->tract_data.size();
            out.write((const char*)&trk,1000);

        }
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

    if (ext == std::string(".mat"))
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
    std::vector<float> colors;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),
              std::back_inserter(colors));
    if(colors.size() <= tract_color.size())
        std::copy(colors.begin(),colors.begin()+colors.size(),tract_color.begin());
    if(colors.size()/3 <= tract_color.size())
        for(unsigned int index = 0,pos = 0;pos+2 < colors.size();++index,pos += 3)
            tract_color[index] = image::rgb_color(std::min<int>(colors[pos],255),
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
        image::rgb_color color;
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
void TractModel::get_end_points(std::vector<image::vector<3,float> >& points)
{
    for (unsigned int index = 0;index < tract_data.size();++index)
    {
        if (tract_data[index].size() < 3)
            return;
        points.push_back(image::vector<3,float>(&tract_data[index][0]));
        points.push_back(image::vector<3,float>(&tract_data[index][tract_data[index].size()-3]));
    }
}
//---------------------------------------------------------------------------
void TractModel::get_tract_points(std::vector<image::vector<3,float> >& points)
{
    for (unsigned int index = 0;index < tract_data.size();++index)
        for (unsigned int j = 0;j < tract_data[index].size();j += 3)
        {
            image::vector<3,float> point(&tract_data[index][j]);
            points.push_back(point);
        }
}
//---------------------------------------------------------------------------
void TractModel::select(float select_angle,
                        const std::vector<image::vector<3,float> >& dirs,
                        const image::vector<3,float>& from_pos,std::vector<unsigned int>& selected)
{
    selected.resize(tract_data.size());
    std::fill(selected.begin(),selected.end(),0);
    for(int i = 1;i < dirs.size();++i)
    {
        image::vector<3,float> from_dir = dirs[i-1];
        image::vector<3,float> to_dir = (i+1 < dirs.size() ? dirs[i+1] : dirs[i]);
        image::vector<3,float> z_axis = from_dir.cross_product(to_dir);
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
    deleted_count.push_back(tracts_to_delete.size());
    deleted_cut_count.push_back(std::make_pair(tract_data.size(),0));
    // no redo once track deleted
    redo_size.clear();
}
//---------------------------------------------------------------------------
void TractModel::select_tracts(const std::vector<unsigned int>& tracts_to_select)
{
    std::vector<unsigned int> selected(tract_data.size());
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
void TractModel::delete_repeated(double d)
{
    auto norm1 = [](const float* v1,const float* v2){return std::fabs(v1[0]-v2[0])+std::fabs(v1[1]-v2[1])+std::fabs(v1[2]-v2[2]);};
    std::vector<bool> repeated(tract_data.size());
    image::par_for(tract_data.size(),[&](int i)
    {
        if(!repeated[i])
        {
        for(int j = i+1;j < tract_data.size();++j)
            if(!repeated[j])
            {
                // check endpoints
                if(norm1(&tract_data[i][0],&tract_data[j][0]) > d ||
                   norm1(&tract_data[i][tract_data[i].size()-3],&tract_data[j][tract_data[j].size()-3]) > d)
                    continue;

                bool not_repeated = false;
                for(int m = 0;m < tract_data[i].size();m += 3)
                {
                    float min_dis = norm1(&tract_data[i][m],&tract_data[j][0]);
                    for(int n = 3;n < tract_data[j].size();n += 3)
                        min_dis = std::min<float>(min_dis,norm1(&tract_data[i][m],&tract_data[j][n]));
                    if(min_dis > d)
                    {
                        not_repeated = true;
                        break;
                    }
                }
                if(!not_repeated)
                for(int m = 0;m < tract_data[j].size();m += 3)
                {
                    float min_dis = norm1(&tract_data[j][m],&tract_data[i][0]);
                    for(int n = 3;n < tract_data[i].size();n += 3)
                        min_dis = std::min<float>(min_dis,norm1(&tract_data[j][m],&tract_data[i][n]));
                    if(min_dis > d)
                    {
                        not_repeated = true;
                        break;
                    }
                }
                if(!not_repeated)
                    repeated[j] = true;
            }
        }
    });
    std::vector<unsigned int> track_to_delete;
    for(unsigned int i = 0;i < tract_data.size();++i)
        if(repeated[i])
            track_to_delete.push_back(i);
    delete_tracts(track_to_delete);
}
//---------------------------------------------------------------------------
void TractModel::delete_by_length(float length)
{
    std::vector<unsigned int> track_to_delete;
    for(unsigned int i = 0;i < tract_data.size();++i)
    {
        if(tract_data[i].size() <= 6)
        {
            track_to_delete.push_back(i);
            continue;
        }
        image::vector<3> v1(&tract_data[i][0]),v2(&tract_data[i][3]);
        v1 -= v2;
        if((((tract_data[i].size()/3)-1)*v1.length()) < length)
            track_to_delete.push_back(i);
    }
    delete_tracts(track_to_delete);
}
//---------------------------------------------------------------------------
void TractModel::cut(float select_angle,const std::vector<image::vector<3,float> >& dirs,
                     const image::vector<3,float>& from_pos)
{
    std::vector<unsigned int> selected;
    select(select_angle,dirs,from_pos,selected);
    std::vector<std::vector<float> > new_tract;
    std::vector<unsigned int> new_tract_color;

    std::vector<unsigned int> tract_to_delete;
    for (unsigned int index = 0;index < selected.size();++index)
        if (selected[index] && tract_data[index].size() > 6)
        {
            new_tract.push_back(std::vector<float>(tract_data[index].begin(),tract_data[index].begin()+selected[index]));
            new_tract_color.push_back(tract_color[index]);
            new_tract.push_back(std::vector<float>(tract_data[index].begin() + selected[index],tract_data[index].end()));
            new_tract_color.push_back(tract_color[index]);
            tract_to_delete.push_back(index);
        }
    if(tract_to_delete.empty())
        return;
    delete_tracts(tract_to_delete);
    deleted_cut_count.back().second = new_tract.size();
    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        tract_data.push_back(std::vector<float>());
        tract_data.back().swap(new_tract[index]);
        tract_color.push_back(new_tract_color[index]);
    }
    redo_size.clear();

}
void TractModel::cut_by_slice(unsigned int dim, unsigned int pos,bool greater)
{
    std::vector<std::vector<float> > new_tract;
    std::vector<unsigned int> new_tract_color;
    std::vector<unsigned int> tract_to_delete;
    for(unsigned int i = 0;i < tract_data.size();++i)
    {
        bool adding = false;
        for(unsigned int j = 0;j < tract_data[i].size();j += 3)
        {
            if((tract_data[i][j+dim] < pos) ^ greater)
            {
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
    delete_tracts(tract_to_delete);
    for (unsigned int index = 0;index < new_tract.size();++index)
    if(new_tract[index].size() >= 6)
        {
            tract_data.push_back(std::vector<float>());
            tract_data.back().swap(new_tract[index]);
            tract_color.push_back(new_tract_color[index]);
            ++deleted_cut_count.back().second;
        }
    redo_size.clear();
}
//---------------------------------------------------------------------------
void TractModel::filter_by_roi(RoiMgr& roi_mgr)
{
    std::vector<unsigned int> tracts_to_delete;
    for (unsigned int index = 0;index < tract_data.size();++index)
    if(tract_data[index].size() >= 6)
    {
        if(!roi_mgr.have_include(&(tract_data[index][0]),tract_data[index].size()) ||
           !roi_mgr.fulfill_end_point(image::vector<3,float>(tract_data[index][0],
                                                             tract_data[index][1],
                                                             tract_data[index][2]),
                                      image::vector<3,float>(tract_data[index][tract_data[index].size()-3],
                                                             tract_data[index][tract_data[index].size()-2],
                                                             tract_data[index][tract_data[index].size()-1])))
        {
            tracts_to_delete.push_back(index);
            continue;
        }
        if(!roi_mgr.exclusive.empty())
        {
            for(unsigned int i = 0;i < tract_data[index].size();i+=3)
                if(roi_mgr.is_excluded_point(image::vector<3,float>(tract_data[index][i],
                                                                    tract_data[index][i+1],
                                                                    tract_data[index][i+2])))
                {
                    tracts_to_delete.push_back(index);
                    break;
                }
        }
    }
    delete_tracts(tracts_to_delete);
}
//---------------------------------------------------------------------------
void TractModel::cull(float select_angle,
                      const std::vector<image::vector<3,float> >& dirs,
                      const image::vector<3,float>& from_pos,
                      bool delete_track)
{
    std::vector<unsigned int> selected;
    select(select_angle,dirs,from_pos,selected);
    std::vector<unsigned int> tracts_to_delete;
    tracts_to_delete.reserve(100 + (selected.size() >> 4));
    for (unsigned int index = 0;index < selected.size();++index)
        if (!((selected[index] > 0) ^ delete_track))
            tracts_to_delete.push_back(index);
    delete_tracts(tracts_to_delete);
}
//---------------------------------------------------------------------------
void TractModel::paint(float select_angle,
                       const std::vector<image::vector<3,float> >& dirs,
                       const image::vector<3,float>& from_pos,
                       unsigned int color)
{
    std::vector<unsigned int> selected;
    select(select_angle,dirs,from_pos,selected);
    for (unsigned int index = 0;index < selected.size();++index)
        if (selected[index] > 0)
            tract_color[index] = color;
}

//---------------------------------------------------------------------------
void TractModel::cut_by_mask(const char*)
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
            image::vector<3,short> p(std::round(iter[0]),
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


bool TractModel::trim(void)
{
    /*
    std::vector<char> continuous(tract_data.size());
    float epsilon = 2.0f;
    image::par_for(tract_data.size(),[&](int i)
    {
        if(tract_data[i].empty() || continuous[i])
            return;
        const float* t1 = &tract_data[i][0];
        const float* t1_end = &tract_data[i][tract_data[i].size()-3];
        for(int j = i+1;j < tract_data.size();++j)
        {
            if(tract_data[j].empty())
                continue;
            const float* t2 = &tract_data[j][0];
            const float* t2_end = &tract_data[j][tract_data[j].size()-3];
            if(std::min<float>(
                        image::vector<3>(t1[0]-t2[0],t1[1]-t2[1],t1[2]-t2[2]).length(),
                        image::vector<3>(t1[0]-t2_end[0],t1[1]-t2_end[1],t1[2]-t2_end[2]).length()) > epsilon)
                continue;
            if(std::min<float>(
                        image::vector<3>(t1_end[0]-t2[0],t1_end[1]-t2[1],t1_end[2]-t2[2]).length(),
                        image::vector<3>(t1_end[0]-t2_end[0],t1_end[1]-t2_end[1],t1_end[2]-t2_end[2]).length()) > epsilon)
                continue;
            unsigned int length1 = tract_data[i].size()-3;
            unsigned int length2 = tract_data[j].size()-3;

            bool con = true;
            for(int m = 3;m < length1;m += 3)
            {
                bool has_c = false;
                for(int n = 3;n < length2;n += 3)
                    if(image::vector<3>(t1[m]-t2[n],t1[m+1]-t2[n+1],t1[m+2]-t2[n+2]).length() < epsilon)
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
        if (!continuous[index])
            tracts_to_delete.push_back(index);
    if(tracts_to_delete.empty())
        return false;
    delete_tracts(tracts_to_delete);
    */

    image::basic_image<unsigned int,3> label(geometry);


    int total_track_number = tract_data.size();
    int no_fiber_label = total_track_number;
    int have_multiple_fiber_label = total_track_number+1;

    int width = label.width();
    int height = label.height();
    int depth = label.depth();
    int wh = width*height;
    std::fill(label.begin(),label.end(),no_fiber_label);
    int shift[8] = {0,1,width,wh,1+width,1+wh,width+wh,1+width+wh};
    image::par_for(total_track_number,[&](int index)
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
                    label[pixel_index] = index;
                else
                    label[pixel_index] = have_multiple_fiber_label;
            }
        }
    });

    std::set<unsigned int> tracts_to_delete;
    for (unsigned int index = 0;index < label.size();++index)
        if (label[index] < total_track_number)
            tracts_to_delete.insert(label[index]);
    if(tracts_to_delete.empty())
        return false;
    delete_tracts(std::vector<unsigned int>(tracts_to_delete.begin(),tracts_to_delete.end()));

    return true;
}
//---------------------------------------------------------------------------
void TractModel::clear_deleted(void)
{
    deleted_count.clear();
    deleted_cut_count.clear();
    deleted_tract_data.clear();
    deleted_tract_color.clear();
    redo_size.clear();
}

void TractModel::undo(void)
{
    if (deleted_count.empty())
        return;
    redo_size.push_back(std::make_pair((unsigned int)tract_data.size(),deleted_count.back()));
    if(deleted_cut_count.back().second)
    {
        std::vector<unsigned int> cut_tracts(deleted_cut_count.back().second);
        for (unsigned int index = 0;index < cut_tracts.size();++index)
            cut_tracts[index] = deleted_cut_count.back().first + index;
        delete_tracts(cut_tracts);
        for(unsigned int index = 0;index < cut_tracts.size();++index)
            deleted_tract_data.pop_back();
        deleted_count.pop_back();
        deleted_cut_count.pop_back();
    }
    for (unsigned int index = 0;index < deleted_count.back();++index)
    {
        tract_data.push_back(std::vector<float>());
        tract_data.back().swap(deleted_tract_data.back());
        tract_color.push_back(deleted_tract_color.back());
        deleted_tract_data.pop_back();
        deleted_tract_color.pop_back();
    }
    deleted_count.pop_back();
    deleted_cut_count.pop_back();

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
    // keep redo because delete tracts will clear redo
    std::vector<std::pair<unsigned int,unsigned int> > keep_redo;
    keep_redo.swap(redo_size);
    delete_tracts(redo_tracts);
    keep_redo.swap(redo_size);
}
//---------------------------------------------------------------------------
void TractModel::add_tracts(std::vector<std::vector<float> >& new_tracks)
{
    add_tracts(new_tracks,tract_color.empty() ? image::rgb_color(255,160,60) : image::rgb_color(tract_color.back()));
}
//---------------------------------------------------------------------------
void TractModel::add_tracts(std::vector<std::vector<float> >& new_tract,image::rgb_color color)
{
    tract_data.reserve(tract_data.size()+new_tract.size());

    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        if (new_tract[index].empty())
            continue;
        tract_data.push_back(std::vector<float>());
        tract_data.back().swap(new_tract[index]);
        tract_color.push_back(color);
    }
}

void TractModel::add_tracts(std::vector<std::vector<float> >& new_tract, unsigned int length_threshold)
{
    tract_data.reserve(tract_data.size()+new_tract.size()/2.0);
    image::rgb_color def_color(200,100,30);
    for (unsigned int index = 0;index < new_tract.size();++index)
    {
        if (new_tract[index].size()/3-1 < length_threshold)
            continue;
        tract_data.push_back(std::vector<float>());
        tract_data.back().swap(new_tract[index]);
        tract_color.push_back(def_color);
    }
}
//---------------------------------------------------------------------------
void TractModel::get_density_map(image::basic_image<unsigned int,3>& mapping,
                                 const image::matrix<4,4,float>& transformation,bool endpoint)
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
                transformation.begin(), image::vdim<3>());

            int x = std::round(tmp[0]);
            int y = std::round(tmp[1]);
            int z = std::round(tmp[2]);
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
        const image::matrix<4,4,float>& transformation,bool endpoint)
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
                transformation.begin(), image::vdim<3>());
            image::vector_transformation(buf+j, tmp.begin(),
                transformation.begin(), image::vdim<3>());
            dir -= tmp;
            dir.normalize();
            int x = std::round(tmp[0]);
            int y = std::round(tmp[1]);
            int z = std::round(tmp[2]);
            if (!geometry.is_valid(x,y,z))
                continue;
            unsigned int ptr = (z*mapping.height()+y)*mapping.width()+x;
            map_r[ptr] += std::fabs(dir[0]);
            map_g[ptr] += std::fabs(dir[1]);
            map_b[ptr] += std::fabs(dir[2]);
        }
    }
    float max_value = 0.0f;
    for(unsigned int index = 0;index < mapping.size();++index)
        max_value = std::max<float>(max_value,map_r[index]+map_g[index]+map_b[index]);

    for(unsigned int index = 0;index < mapping.size();++index)
    {
        float sum = map_r[index]+map_g[index]+map_b[index];
        if(sum == 0.0f)
            continue;
        image::vector<3> v(map_r[index],map_g[index],map_b[index]);
        sum = v.normalize();
        v*=255.0*std::log(200.0f*sum/max_value+1)/2.303f;
        mapping[index] = image::rgb_color(
                (unsigned char)std::min<float>(255,v[0]),
                (unsigned char)std::min<float>(255,v[1]),
                (unsigned char)std::min<float>(255,v[2]));
    }
}

void TractModel::save_tdi(const char* file_name,bool sub_voxel,bool endpoint,const std::vector<float>& trans)
{
    image::matrix<4,4,float> tr;
    tr.zero();
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
    nii_header.set_voxel_size(new_vs.begin());
    if(!trans.empty())
    {
        if(sub_voxel)
        {
            std::vector<float> new_trans(trans);
            new_trans[0] /= 4.0;
            new_trans[4] /= 4.0;
            new_trans[8] /= 4.0;
            nii_header.set_LPS_transformation(new_trans.begin(),tdi.geometry());
        }
        else
            nii_header.set_LPS_transformation(trans.begin(),tdi.geometry());
    }
    image::flip_xy(tdi);
    nii_header << tdi;
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
                pass_map.insert(image::vector<3,int>(std::round(tract_data[i][j]),
                                              std::round(tract_data[i][j+1]),
                                              std::round(tract_data[i][j+2])));

        data.push_back(pass_map.size()*voxel_volume);
    }

    // output mean and std of each index
    for(int data_index = 0;data_index < handle->view_item.size();++data_index)
    {
        if(handle->view_item[data_index].name == "color")
            continue;
        float mean,sd;
        get_tracts_data(data_index,mean,sd);
        data.push_back(mean);
        data.push_back(sd);
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


    if(handle->db.has_db()) // connectometry database
    {
        std::vector<const float*> old_index_data(fib->other_index[0]);
        for(int i = 0;i < handle->db.num_subjects;++i)
        {
            std::vector<std::vector<float> > fa_data;
            handle->db.get_subject_fa(i,fa_data);
            for(int j = 0;j < fa_data.size();++j)
                fib->other_index[0][j] = &fa_data[j][0];
            float mean,sd;
            get_tracts_data(0,mean,sd);
            out << handle->db.subject_names[i] << " " << handle->db.index_name << " mean\t" << mean << std::endl;
            out << handle->db.subject_names[i] << " " << handle->db.index_name << " sd\t" << sd << std::endl;
        }
        fib->other_index[0] = old_index_data;
    }
    result = out.str();
}

extern track_recognition track_network;


bool TractModel::recognize(std::map<float,std::string,std::greater<float> >& result)
{
    if(!track_network.can_recognize())
        return false;
    std::vector<float> accu_input(track_network.cnn.get_output_size());
    image::par_for(tract_data.size(),[&](int i)
    {
        std::vector<float> input;
        if(!handle->get_profile(tract_data[i],input))
            return;
        track_network.cnn.predict(input);
        image::minus_constant(input,*std::min_element(input.begin(),input.end()));
        image::multiply_constant(input,1.0f/std::accumulate(input.begin(),input.end(),0.0f));
        image::add(accu_input,input);
    });
    image::multiply_constant(accu_input,1.0f/std::accumulate(accu_input.begin(),accu_input.end(),0.0f));
    for(int i = 0;i < accu_input.size();++i)
        result[accu_input[i]] = track_network.track_name[i];
    return true;
}
extern atlas* track_atlas;
void TractModel::recognize_report(std::string& report)
{
    /*
    if(track_atlas)
    {
        image::vector<3> dummy;
        track_atlas->is_labeled_as(dummy,0);// invoke loading file
        std::vector<int> recog_count(track_atlas->get_list().size());
        image::par_for(tract_data.size(),[&](int i)
        {
            std::vector<image::vector<3> > points;
            for(int j = 0;j < tract_data[i].size();j += 3)
            {
                points.push_back(image::vector<3>(&(tract_data[i][j])));
                handle->subject2mni(points.back());
            }
            int result = track_atlas->get_track_label(points);
            if(result >= 0 && result < recog_count.size())
                ++recog_count[result];
        });
        int sum = std::accumulate(recog_count.begin(),recog_count.end(),(int)0);
        std::multimap<float,std::string,std::greater<float> > sorted_result;
        for(int i = 0;i < recog_count.size();++i)
        {
            float p = recog_count[i];
            p /= sum;
            if(p > 0.05)
                sorted_result.insert(std::make_pair(p,track_atlas->get_list()[i]));
        }
        std::ostringstream out;
        int n = 0;
        for(auto& r : sorted_result)
        {
            if(n)
                out << ((n == sorted_result.size()-1 ? (sorted_result.size() == 2 ? " and ":", and ") : ", "));
            out << r.second << " (" << (float)(int(r.first*10000.0))/100.0 << "%)";
            ++n;
        }
        report = out.str();
    }
    else
        report = "tracks";

    */
    if(!handle->is_human_data || !track_network.can_recognize())
        return;
    std::vector<int> recog_count(track_network.cnn.get_output_size());
    image::par_for(tract_data.size(),[&](int i)
    {
        std::vector<float> input;
        if(!handle->get_profile(tract_data[i],input))
            return;
        track_network.cnn.predict(input);
        input[80] = -100;// suppress false tracks ID:20
        ++recog_count[std::max_element(input.begin(),input.end())-input.begin()];
    });
    {
        std::map<int,std::string,std::greater<int> > sorted_result;
        unsigned int report_threshold = tract_data.size()/20; //5%
        for(unsigned int i = 0;i < recog_count.size();++i)
            if(recog_count[i] > report_threshold)
                sorted_result[recog_count[i]] = track_network.track_name[i];
        int n = 0;
        for(auto& r : sorted_result)
        {
            if(!report.empty())
                report += (n == sorted_result.size()-1 ? (sorted_result.size() == 2 ? " and ":", and ") : ", ");
            report += r.second;
            ++n;
        }
    }
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
                              std::round(tract_data[i][j + j + j + profile_dir]*detail);
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

void TractModel::get_tract_data(unsigned int fiber_index,unsigned int index_num,std::vector<float>& data) const
{
    data.clear();
    if(tract_data[fiber_index].empty())
        return;
    unsigned int count = tract_data[fiber_index].size()/3;
    data.resize(count);
    // track specific index
    if(index_num < fib->other_index.size())
    {
        auto base_image = image::make_image(fib->other_index[index_num][0],fib->dim);
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
                    if ((value = fib->get_track_specific_index(tri_interpo.dindex[index],index_num,gradient[point_index])) == 0.0)
                        continue;
                    average_value += value*tri_interpo.ratio[index];
                    sum_value += tri_interpo.ratio[index];
                }
                if (sum_value > 0.5)
                    data[point_index] = average_value/sum_value;
                else
                    image::estimate(base_image,&(tract_data[fiber_index][tract_index]),data[point_index],image::linear);
            }
            else
                image::estimate(base_image,&(tract_data[fiber_index][tract_index]),data[point_index],image::linear);
        }
    }
    else
    // voxel-based index
    {
        for (unsigned int data_index = 0,index = 0;index < tract_data[fiber_index].size();index += 3,++data_index)
            image::estimate(handle->view_item[index_num].image_data,&(tract_data[fiber_index][index]),data[data_index],image::linear);
    }
}

bool TractModel::get_tracts_data(
        const std::string& index_name,
        std::vector<std::vector<float> >& data) const
{
    unsigned int index_num = handle->get_name_index(index_name);
    if(index_num == handle->view_item.size())
        return false;
    data.clear();
    data.resize(tract_data.size());
    for (unsigned int i = 0;i < tract_data.size();++i)
        get_tract_data(i,index_num,data[i]);
    return true;
}
void TractModel::get_tracts_data(unsigned int data_index,float& mean, float& sd) const
{
    float sum_data = 0.0;
    float sum_data2 = 0.0;
    unsigned int total = 0;
    for (unsigned int i = 0;i < tract_data.size();++i)
    {
        std::vector<float> data;
        get_tract_data(i,data_index,data);
        for(int j = 0;j < data.size();++j)
        {
            float value = data[j];
            sum_data += value;
            sum_data2 += value*value;
            ++total;
        }
    }

    mean = sum_data/((double)total);
    sd = std::sqrt(sum_data2/(double)total-sum_data*sum_data/(double)total/(double)total);

}
// return region overlapped ratio
float create_region_map(const image::geometry<3>& geometry,
                      const std::vector<std::vector<image::vector<3,short> > >& regions,
                      std::vector<std::vector<short> >& region_map)
{
    std::vector<std::set<short> > regions_set(geometry.size());
    region_map.resize(geometry.size());
    for(unsigned int roi = 0;roi < regions.size();++roi)
    {
        for(unsigned int index = 0;index < regions[roi].size();++index)
        {
            image::vector<3,short> pos = regions[roi][index];
            if(geometry.is_valid(pos))
                regions_set[image::pixel_index<3>(pos[0],pos[1],pos[2],geometry).index()].insert(roi);

        }
    }
    unsigned int overlap_count = 0,total_count = 0;
    for(unsigned int index = 0;index < geometry.size();++index)
        if(!regions_set[index].empty())
        {
            for(auto i : regions_set[index])
                region_map[index].push_back(i);
            ++total_count;
            if(region_map[index].size() > 1)
                ++overlap_count;
        }
    return (float)overlap_count/(float)total_count;
}

void TractModel::get_passing_list(const std::vector<std::vector<image::vector<3,short> > >& regions,
                                  std::vector<std::vector<short> >& passing_list1,
                                  std::vector<std::vector<short> >& passing_list2,
                                  float& overlap_ratio) const
{
    passing_list1.clear();
    passing_list1.resize(tract_data.size());
    passing_list2.clear();
    passing_list2.resize(tract_data.size());
    // create regions maps
    std::vector<std::vector<short> > region_map;
    overlap_ratio = create_region_map(geometry,regions,region_map);

    for(unsigned int index = 0;index < tract_data.size();++index)
    {
        if(tract_data[index].size() < 6)
            continue;
        std::vector<unsigned char> has_region(regions.size());
        unsigned int half_length = tract_data[index].size()/2;
        for(unsigned int ptr = 0;ptr < tract_data[index].size();ptr += 3)
        {
            image::pixel_index<3> pos(std::round(tract_data[index][ptr]),
                                        std::round(tract_data[index][ptr+1]),
                                        std::round(tract_data[index][ptr+2]),geometry);
            if(!geometry.is_valid(pos))
                continue;
            unsigned int pos_index = pos.index();
            for(unsigned int j = 0;j < region_map[pos_index].size();++j)
                has_region[region_map[pos_index][j]] = (ptr > half_length ? 1: 2);
        }
        for(unsigned int i = 0;i < has_region.size();++i)
        {
            if(has_region[i] == 1)
                passing_list1[index].push_back(i);
            if(has_region[i] == 2)
                passing_list2[index].push_back(i);
        }
    }
}

void TractModel::get_end_list(const std::vector<std::vector<image::vector<3,short> > >& regions,
                                  std::vector<std::vector<short> >& end_pair1,
                                  std::vector<std::vector<short> >& end_pair2,
                                    float& overlap_ratio) const
{
    end_pair1.clear();
    end_pair1.resize(tract_data.size());
    end_pair2.clear();
    end_pair2.resize(tract_data.size());
    // create regions maps
    std::vector<std::vector<short> > region_map;
    overlap_ratio = create_region_map(geometry,regions,region_map);

    for(unsigned int index = 0;index < tract_data.size();++index)
    {
        if(tract_data[index].size() < 6)
            continue;
        image::pixel_index<3> end1(std::round(tract_data[index][0]),
                                    std::round(tract_data[index][1]),
                                    std::round(tract_data[index][2]),geometry);
        image::pixel_index<3> end2(std::round(tract_data[index][tract_data[index].size()-3]),
                                    std::round(tract_data[index][tract_data[index].size()-2]),
                                    std::round(tract_data[index][tract_data[index].size()-1]),geometry);
        if(!geometry.is_valid(end1) || !geometry.is_valid(end2))
            continue;
        end_pair1[index] = region_map[end1.index()];
        end_pair2[index] = region_map[end2.index()];
    }
}


void TractModel::run_clustering(unsigned char method_id,unsigned int cluster_count,float detail)
{
    float param[4] = {0};
    if(method_id)// k-means or EM
        param[0] = cluster_count;
    else
    {
        std::copy(handle->dim.begin(),
                  handle->dim.end(),param);
        param[3] = detail;
    }
    std::unique_ptr<BasicCluster> c;
    switch (method_id)
    {
    case 0:
        c.reset(new TractCluster(param));
        break;
    case 1:
        c.reset(new FeatureBasedClutering<image::ml::k_means<double,unsigned char> >(param));
        break;
    case 2:
        c.reset(new FeatureBasedClutering<image::ml::expectation_maximization<double,unsigned char> >(param));
        break;
    case 3:
        {
            tract_cluster.resize(tract_data.size());
            std::fill(tract_cluster.begin(),tract_cluster.end(),80);
            if(!track_network.can_recognize())
                return;
            image::par_for(tract_data.size(),[&](int i)
            {
                std::vector<float> input;
                if(!handle->get_profile(tract_data[i],input))
                    return;
                tract_cluster[i] = track_network.cnn.predict_label(input);
            });
        }
        return;
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

void ConnectivityMatrix::save_to_image(image::color_image& cm)
{
    if(matrix_value.empty())
        return;
    cm.resize(matrix_value.geometry());
    std::vector<float> values(matrix_value.size());
    std::copy(matrix_value.begin(),matrix_value.end(),values.begin());
    image::normalize(values,255.99f);
    for(unsigned int index = 0;index < values.size();++index)
    {
        cm[index] = image::rgb_color((unsigned char)values[index],(unsigned char)values[index],(unsigned char)values[index]);
    }
}

void ConnectivityMatrix::save_to_file(const char* file_name)
{
    image::io::mat_write mat_header(file_name);
    mat_header.write("connectivity",&*matrix_value.begin(),matrix_value.width(),matrix_value.height());
    std::ostringstream out;
    std::copy(region_name.begin(),region_name.end(),std::ostream_iterator<std::string>(out,"\n"));
    std::string result(out.str());
    mat_header.write("name",result.c_str(),1,result.length());
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
    int w = matrix_value.width();
    std::vector<float> sum(w);
    out << "data\tdata\t";
    for(int i = 0;i < w;++i)
    {
        sum[i] = std::max<int>(1,std::accumulate(matrix_value.begin()+i*w,matrix_value.begin()+i*w+w,0.0)*2);
        out << sum[i] << "\t";
    }
    out << std::endl;
    out << "data\tdata\t";
    for(int i = 0;i < w;++i)
        out << region_name[i] << "\t";
    out << std::endl;

    for(int i = 0;i < w;++i)
    {
        out << sum[i] << "\t" << region_name[i] << "\t";
        for(int j = 0;j < w;++j)
            out << matrix_value[i*w+j] << "\t";
        out << std::endl;
    }
}

void ConnectivityMatrix::set_atlas(atlas& data,const image::basic_image<image::vector<3,float>,3 >& mni_position)
{
    if(mni_position.empty())
        return;
    image::geometry<3> geo(mni_position.geometry());
    image::vector<3> null;
    regions.clear();
    region_name.clear();
    for (unsigned int label_index = 0; label_index < data.get_list().size(); ++label_index)
    {
        std::vector<image::vector<3,short> > cur_region;
        for (image::pixel_index<3> index(geo); index < geo.size();++index)
            if(mni_position[index.index()] != null && data.is_labeled_as(mni_position[index.index()],label_index))
                cur_region.push_back(image::vector<3,short>(index.begin()));
        regions.push_back(cur_region);
        region_name.push_back(data.get_list()[label_index]);
    }
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
        for(unsigned int i = 0;i < r1.size();++i)
            for(unsigned int j = 0;j < r2.size();++j)
                if(r1[i] != r2[j])
                {
                    lambda_fun(index,r1[i],r2[j]);
                    lambda_fun(index,r2[j],r1[i]);
                }
    }
}

bool ConnectivityMatrix::calculate(TractModel& tract_model,std::string matrix_value_type,bool use_end_only,float threshold)
{
    if(regions.size() == 0)
    {
        error_msg = "No region information. Please assign regions";
        return false;
    }

    std::vector<std::vector<short> > end_list1,end_list2;
    if(use_end_only)
        tract_model.get_end_list(regions,end_list1,end_list2,overlap_ratio);
    else
        tract_model.get_passing_list(regions,end_list1,end_list2,overlap_ratio);
    if(matrix_value_type == "trk")
    {
        std::vector<std::vector<std::vector<unsigned int> > > region_passing_list;
        init_matrix(region_passing_list,regions.size());

        for_each_connectivity(end_list1,end_list2,
                              [&](unsigned int index,short i,short j){
            region_passing_list[i][j].push_back(index);
        });

        for(unsigned int i = 0;i < region_passing_list.size();++i)
            for(unsigned int j = i+1;j < region_passing_list.size();++j)
            {
                if(region_passing_list[i][j].empty())
                    continue;
                std::string file_name = region_name[i]+"_"+region_name[j]+".trk";
                TractModel tm(tract_model.get_handle());
                std::vector<std::vector<float> > new_tracts;
                for (unsigned int k = 0;k < region_passing_list[i][j].size();++k)
                    new_tracts.push_back(tract_model.get_tract(region_passing_list[i][j][k]));
                tm.add_tracts(new_tracts);
                if(!tm.save_tracts_to_file(file_name.c_str()))
                    return false;
            }
        return true;
    }
    matrix_value.clear();
    matrix_value.resize(image::geometry<2>(regions.size(),regions.size()));
    std::vector<std::vector<unsigned int> > count;
    init_matrix(count,regions.size());

    for_each_connectivity(end_list1,end_list2,
                          [&](unsigned int,short i,short j){
        ++count[i][j];
    });

    // determine the threshold for counting the connectivity
    unsigned int threshold_count = 0;
    for(unsigned int i = 0,index = 0;i < count.size();++i)
        for(unsigned int j = 0;j < count[i].size();++j,++index)
            threshold_count = std::max<unsigned int>(threshold_count,count[i][j]);
    threshold_count *= threshold;

    if(matrix_value_type == "count")
    {
        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                matrix_value[index] = (count[i][j] > threshold_count ? count[i][j] : 0);
        return true;
    }
    if(matrix_value_type == "ncount")
    {
        std::vector<std::vector<std::vector<unsigned int> > > length_matrix;
        init_matrix(length_matrix,regions.size());

        for_each_connectivity(end_list1,end_list2,
                              [&](unsigned int index,short i,short j){
            length_matrix[i][j].push_back(tract_model.get_tract_length(index));
        });

        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                if(!length_matrix[i][j].empty())
                {
                    std::nth_element(length_matrix[i][j].begin(),
                                     length_matrix[i][j].begin()+(length_matrix[i][j].size() >> 1),
                                     length_matrix[i][j].end());
                    float length = (float)length_matrix[i][j][length_matrix[i][j].size() >> 1];
                    matrix_value[index] = ((length == 0 || count[i][j] < threshold_count )? 0:count[i][j]/length);
                }
            else
                    matrix_value[index] = 0;

        return true;
    }

    if(matrix_value_type == "mean_length")
    {
        std::vector<std::vector<unsigned int> > sum_length,sum_n;
        init_matrix(sum_length,regions.size());
        init_matrix(sum_n,regions.size());

        for_each_connectivity(end_list1,end_list2,
                              [&](unsigned int index,short i,short j){
            sum_length[i][j] += tract_model.get_tract_length(index);
            ++sum_n[i][j];
        });

        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                if(sum_n[i][j] && count[i][j] > threshold_count)
                    matrix_value[index] = (float)sum_length[i][j]/(float)sum_n[i][j]/3.0;
        return true;
    }
    std::vector<std::vector<float> > data;
    if(!tract_model.get_tracts_data(matrix_value_type,data))
    {
        error_msg = "Cannot quantify matrix value using ";
        error_msg += matrix_value_type;
        return false;
    }
    std::vector<std::vector<float> > sum;
    init_matrix(sum,regions.size());

    std::vector<float> m(data.size());
    for(unsigned int index = 0;index < data.size();++index)
        m[index] = image::mean(data[index].begin(),data[index].end());

    for_each_connectivity(end_list1,end_list2,
                          [&](unsigned int index,short i,short j){
        sum[i][j] += m[index];
    });


    for(unsigned int i = 0,index = 0;i < count.size();++i)
        for(unsigned int j = 0;j < count[i].size();++j,++index)
            matrix_value[index] = (count[i][j] > threshold_count ? sum[i][j]/(float)count[i][j] : 0);
    return true;

}
template<class matrix_type>
void distance_bin(const matrix_type& bin,image::basic_image<float,2>& D)
{
    unsigned int n = bin.width();
    image::basic_image<unsigned int,2> A,Lpath;
    A = bin;
    Lpath = bin;
    D = bin;
    for(unsigned int l = 2;1;++l)
    {
        image::basic_image<unsigned int,2> t(A.geometry());
        image::mat::product(Lpath.begin(),A.begin(),t.begin(),image::dyndim(n,n),image::dyndim(n,n));
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
void distance_wei(const matrix_type& W_,image::basic_image<float,2>& D)
{
    image::basic_image<float,2> W(W_);
    for(unsigned int i = 0;i < W.size();++i)
        W[i] = (W[i] != 0) ? 1.0/W[i]:0;
    unsigned int n = W.width();
    D.clear();
    D.resize(W.geometry());
    std::fill(D.begin(),D.end(),std::numeric_limits<float>::max());
    for(unsigned int i = 0,dg = 0;i < n;++i,dg += n + 1)
        D[dg] = 0;
    for(unsigned int i = 0,in = 0;i < n;++i,in += n)
    {
        std::vector<unsigned char> S(n);
        image::basic_image<float,2> W1(W);

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
    if(e.begin() != D.begin())
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
    image::basic_image<unsigned char,2> binary_matrix(matrix_value.geometry());
    image::basic_image<float,2> norm_matrix(matrix_value.geometry());

    float max_value = *std::max_element(matrix_value.begin(),matrix_value.end());
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
    float cc_bin = image::mean(cluster_co.begin(),cluster_co.end());
    out << "clustering_coeff_average(binary)\t" << cc_bin << std::endl;

    // calculate weighted clustering coefficient
    image::basic_image<float,2> cyc3(norm_matrix.geometry());
    std::vector<float> wcluster_co(n);
    {
        image::basic_image<float,2> root(norm_matrix);
        // root = W.^ 1/3
        for(unsigned int j = 0;j < root.size();++j)
            root[j] = std::pow(root[j],(float)(1.0/3.0));
        // cyc3 = (W.^1/3)^3
        image::basic_image<float,2> t(root.geometry());
        image::mat::product(root.begin(),root.begin(),t.begin(),image::dyndim(n,n),image::dyndim(n,n));
        image::mat::product(t.begin(),root.begin(),cyc3.begin(),image::dyndim(n,n),image::dyndim(n,n));
        // wcc = diag(cyc3)/(K.*(K-1));
        for(unsigned int i = 0;i < n;++i)
        if(degree[i] >= 2)
        {
            float d = degree[i];
            wcluster_co[i] = cyc3[i*(n+1)]/(d*d-d);
        }
    }
    float cc_wei = image::mean(wcluster_co.begin(),wcluster_co.end());
    out << "clustering_coeff_average(weighted)\t" << cc_wei << std::endl;


    // transitivity
    {
        image::basic_image<float,2> norm_matrix2(norm_matrix.geometry());
        image::basic_image<float,2> norm_matrix3(norm_matrix.geometry());
        image::mat::product(norm_matrix.begin(),norm_matrix.begin(),norm_matrix2.begin(),image::dyndim(n,n),image::dyndim(n,n));
        image::mat::product(norm_matrix2.begin(),norm_matrix.begin(),norm_matrix3.begin(),image::dyndim(n,n),image::dyndim(n,n));
        out << "transitivity(binary)\t" << image::mat::trace(norm_matrix3.begin(),image::dyndim(n,n)) /
                (std::accumulate(norm_matrix2.begin(),norm_matrix2.end(),0.0) - image::mat::trace(norm_matrix2.begin(),image::dyndim(n,n))) << std::endl;
        float k = 0;
        for(unsigned int i = 0;i < n;++i)
            k += degree[i]*(degree[i]-1);
        out << "transitivity(weighted)\t" << (k == 0 ? 0 : image::mat::trace(cyc3.begin(),image::dyndim(n,n))/k) << std::endl;
    }

    std::vector<float> eccentricity_bin(n),eccentricity_wei(n);

    {
        image::basic_image<float,2> dis_bin,dis_wei;
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
        image::basic_image<float,2> invD;
        inv_dis(dis_bin,invD);
        out << "global_efficiency(binary)\t" << std::accumulate(invD.begin(),invD.end(),0.0)/(n*n-inf_count_bin) << std::endl;
        inv_dis(dis_wei,invD);
        out << "global_efficiency(weighted)\t" << std::accumulate(invD.begin(),invD.end(),0.0)/(n*n-inf_count_wei) << std::endl;

        for(unsigned int i = 0,ipos = 0;i < n;++i,ipos += n)
        {
            eccentricity_bin[i] = *std::max_element(dis_bin.begin()+ipos,
                                                 dis_bin.begin()+ipos+n);
            eccentricity_wei[i] = *std::max_element(dis_wei.begin()+ipos,
                                                 dis_wei.begin()+ipos+n);

        }
        out << "diameter_of_graph(binary)\t" << *std::max_element(eccentricity_bin.begin(),eccentricity_bin.end()) <<std::endl;
        out << "diameter_of_graph(weighted)\t" << *std::max_element(eccentricity_wei.begin(),eccentricity_wei.end()) <<std::endl;


        std::replace(eccentricity_bin.begin(),eccentricity_bin.end(),(float)0,std::numeric_limits<float>::max());
        std::replace(eccentricity_wei.begin(),eccentricity_wei.end(),(float)0,std::numeric_limits<float>::max());
        out << "radius_of_graph(binary)\t" << *std::min_element(eccentricity_bin.begin(),eccentricity_bin.end()) <<std::endl;
        out << "radius_of_graph(weighted)\t" << *std::min_element(eccentricity_wei.begin(),eccentricity_wei.end()) <<std::endl;
        std::replace(eccentricity_bin.begin(),eccentricity_bin.end(),std::numeric_limits<float>::max(),(float)0);
        std::replace(eccentricity_wei.begin(),eccentricity_wei.end(),std::numeric_limits<float>::max(),(float)0);
    }

    std::vector<float> local_efficiency_bin(n);
    //claculate local efficiency
    {
        for(unsigned int i = 0,ipos = 0;i < n;++i,ipos += n)
        {
            unsigned int new_n = std::accumulate(binary_matrix.begin()+ipos,
                                                 binary_matrix.begin()+ipos+n,0);
            if(new_n < 2)
                continue;
            image::basic_image<float,2> newA(image::geometry<2>(new_n,new_n));
            unsigned int pos = 0;
            for(unsigned int j = 0,index = 0;j < n;++j)
                for(unsigned int k = 0;k < n;++k,++index)
                    if(binary_matrix[ipos+j] && binary_matrix[ipos+k])
                    {
                        if(pos < newA.size())
                            newA[pos] = binary_matrix[index];
                        ++pos;
                    }
            image::basic_image<float,2> invD;
            distance_bin(newA,invD);
            inv_dis(invD,invD);
            local_efficiency_bin[i] = std::accumulate(invD.begin(),invD.end(),0.0)/(new_n*new_n-new_n);
        }
        out << "local_efficiency(binary)\t" << std::accumulate(local_efficiency_bin.begin(),local_efficiency_bin.end(),0.0) << std::endl;

    }

    std::vector<float> local_efficiency_wei(n);
    {

        for(unsigned int i = 0,ipos = 0;i < n;++i,ipos += n)
        {
            unsigned int new_n = std::accumulate(binary_matrix.begin()+ipos,
                                                 binary_matrix.begin()+ipos+n,0);
            if(new_n < 2)
                continue;
            image::basic_image<float,2> newA(image::geometry<2>(new_n,new_n));
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
            image::basic_image<float,2> invD;
            distance_wei(newA,invD);
            inv_dis(invD,invD);
            float numer = 0.0;
            for(unsigned int j = 0,index = 0;j < new_n;++j)
                for(unsigned int k = 0;k < new_n;++k,++index)
                    numer += std::pow(invD[index],(float)(1.0/3.0))*sw[j]*sw[k];
            local_efficiency_wei[i] = numer/(new_n*new_n-new_n);
        }
        out << "local_efficiency(weighted)\t" << std::accumulate(local_efficiency_wei.begin(),local_efficiency_wei.end(),0.0) << std::endl;

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
        float sum = image::vec::dot(degi.begin(),degi.end(),degj.begin())/degi.size();
        image::square(degi);
        image::square(degj);
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
        float sum = image::vec::dot(degi.begin(),degi.end(),degj.begin())/degi.size();
        image::square(degi);
        image::square(degj);
        float b = (std::accumulate(degi.begin(),degi.end(),0.0)+
                   std::accumulate(degj.begin(),degj.end(),0.0))/2.0/degi.size();
        out << "assortativity_coefficient(weighted)\t" << ( sum  - a*a)/ ( b - a*a ) << std::endl;
    }
    // betweenness
    std::vector<float> betweenness_bin(n);
    {

        image::basic_image<unsigned int,2> NPd(binary_matrix),NSPd(binary_matrix),NSP(binary_matrix);
        for(unsigned int i = 0,dg = 0;i < n;++i,dg += n+1)
            NSP[dg] = 1;
        image::basic_image<unsigned int,2> L(NSP);
        unsigned int d = 2;
        for(;std::find(NSPd.begin(),NSPd.end(),1) != NSPd.end();++d)
        {
            image::basic_image<unsigned int,2> t(binary_matrix.geometry());
            image::mat::product(NPd.begin(),binary_matrix.begin(),t.begin(),image::dyndim(n,n),image::dyndim(n,n));
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
        image::basic_image<float,2> DP(binary_matrix.geometry());
        for(--d;d >= 2;--d)
        {
            image::basic_image<float,2> t(DP),DPd1(binary_matrix.geometry());
            t += 1.0;
            for(unsigned int i = 0;i < t.size();++i)
                if(L[i] != d)
                    t[i] = 0;
                else
                    t[i] /= NSP[i];
            image::mat::product(t.begin(),binary_matrix.begin(),DPd1.begin(),image::dyndim(n,n),image::dyndim(n,n));
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
            image::basic_image<unsigned char,2> P(binary_matrix.geometry());
            image::basic_image<float,2> G1(norm_matrix);
            std::vector<unsigned int> V;
            V.push_back(i);
            while(1)
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
        image::basic_image<float,2> bin;
        bin = binary_matrix;
        std::vector<float> V(binary_matrix.size()),d(n);
        image::mat::eigen_decomposition_sym(bin.begin(),V.begin(),d.begin(),image::dyndim(n,n));
        std::copy(V.begin(),V.begin()+n,eigenvector_centrality_bin.begin());
        image::mat::eigen_decomposition_sym(norm_matrix.begin(),V.begin(),d.begin(),image::dyndim(n,n));
        std::copy(V.begin(),V.begin()+n,eigenvector_centrality_wei.begin());
    }

    std::vector<float> pagerank_centrality_bin(n),pagerank_centrality_wei(n);
    {
        float d = 0.85f;
        std::vector<float> deg_bin(degree.begin(),degree.end()),deg_wei(strength.begin(),strength.end());
        std::replace(deg_bin.begin(),deg_bin.end(),0.0f,1.0f);
        std::replace(deg_wei.begin(),deg_wei.end(),0.0f,1.0f);

        image::basic_image<float,2> B_bin(binary_matrix.geometry()),B_wei(binary_matrix.geometry());
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
        image::mat::lu_decomposition(B_bin.begin(),pivot.begin(),image::dyndim(n,n));
        image::mat::lu_solve(B_bin.begin(),pivot.begin(),b.begin(),pagerank_centrality_bin.begin(),image::dyndim(n,n));
        image::mat::lu_decomposition(B_wei.begin(),pivot.begin(),image::dyndim(n,n));
        image::mat::lu_solve(B_wei.begin(),pivot.begin(),b.begin(),pagerank_centrality_wei.begin(),image::dyndim(n,n));

        float sum_bin = std::accumulate(pagerank_centrality_bin.begin(),pagerank_centrality_bin.end(),0.0);
        float sum_wei = std::accumulate(pagerank_centrality_wei.begin(),pagerank_centrality_wei.end(),0.0);

        if(sum_bin != 0)
            image::divide_constant(pagerank_centrality_bin,sum_bin);
        if(sum_wei != 0)
            image::divide_constant(pagerank_centrality_wei,sum_wei);
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


    if(overlap_ratio > 0.5)
    {
        out << " The brain parcellations have a large overlapping area (ratio="
            << overlap_ratio << "). The network measure calculated may not be reliable.";
    }

    report = out.str();
}
