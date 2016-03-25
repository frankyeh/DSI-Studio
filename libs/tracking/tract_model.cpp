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
TractModel::TractModel(std::shared_ptr<fib_data> handle_):handle(handle_),report(handle_->report),geometry(handle_->dim),vs(handle_->vs),fib(new tracking)
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
            begin_prog("loading");
            for (unsigned int index = 0;check_prog(index,track_number);++index)
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
                    to[0] = x;
                    to[1] = y;
                    to[2] = z;
                }
                if(trk.n_properties == 1)
                    loaded_tract_cluster.push_back(from[0]);
            }
            unsigned int report_size = 0;
            in.read(&report_size,sizeof(unsigned int));
            if(in && report_size)
            {
                report.resize(report_size);
                in.read(&*report.begin(),report_size);
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
        unsigned int report_size = report.size();
        out.write((const char*)&report_size,sizeof(unsigned int));
        out.write((const char*)&*report.begin(),report_size);
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
    return false;
}
void TractModel::save_vrml(const char* file_name,
                           unsigned char tract_style,
                           unsigned char tract_color_style,
                           float tube_diameter,
                           unsigned char tract_tube_detail)
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


    QString CoordinateIndex;
    std::vector<int> CoordinateIndexPos(2);
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
        unsigned int vrml_coordinate_count = 0,vrml_color_count = 0;
        QString Coordinate,Color,ColorIndex;
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
            Color += QString("%1 %2 %3 ").arg(cur_color[0]).arg(cur_color[1]).arg(cur_color[2]);
            // add end
            if(tract_style == 0)// line
            {
                Coordinate += QString("%1 %2 %3 ").arg(pos[0]).arg(pos[1]).arg(pos[2]);
                ColorIndex += QString("%1 ").arg(vrml_color_count);
                prev_vec_n = vec_n;
                last_pos = pos;
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
                    ColorIndex += QString("%1 ").arg(vrml_color_count);
                }
                vrml_coordinate_count+=8;
            }
            else
            // add tube
            {

                Coordinate += QString("%1 %2 %3 ").arg(points[0][0]).arg(points[0][1]).arg(points[0][2]);
                ColorIndex += QString("%1, ").arg(vrml_color_count);
                for (unsigned int k = 1;k < 8;++k)
                {
                    Coordinate += QString("%1 %2 %3 ").arg(previous_points[k][0]).arg(previous_points[k][1]).arg(previous_points[k][2]);
                    Coordinate += QString("%1 %2 %3 ").arg(points[k][0]).arg(points[k][1]).arg(points[k][2]);
                    ColorIndex += QString("%1, ").arg(vrml_color_count);
                    ColorIndex += QString("%1, ").arg(vrml_color_count);
                }
                Coordinate += QString("%1 %2 %3 ").arg(points[0][0]).arg(points[0][1]).arg(points[0][2]);
                ColorIndex += QString("%1 ").arg(vrml_color_count);
                vrml_coordinate_count+=16;

                if(index +1 == vertex_count)
                {
                    for (int k = 7;k >= 0;--k)
                    {
                        Coordinate += QString("%1 %2 %3 ").arg(points[end_sequence[k]][0]).arg(points[end_sequence[k]][1]).arg(points[end_sequence[k]][2]);
                        ColorIndex += QString("%1 ").arg(vrml_color_count);
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
            for (unsigned int j = CoordinateIndexPos.size();j < vrml_coordinate_count;++j)
            {
                CoordinateIndex += QString("%1 ").arg(j-2).arg(j-1).arg(j);
                CoordinateIndexPos.push_back(CoordinateIndex.size());
            }
            out << "Shape {" << std::endl;
            out << "geometry DEF IFS IndexedLineSet {" << std::endl;
            out << "point [" << Coordinate.toStdString() << " ]" << std::endl;
            out << "coordIndex ["<< ColorIndex.toStdString() <<"]" << std::endl;
            out << "color Color { color ["<< Color.toStdString() <<"] } } }" << std::endl;
            continue;
        }
        for (unsigned int j = CoordinateIndexPos.size();j < vrml_coordinate_count;++j)
        {
            CoordinateIndex += QString("%1 %2 %3 -1 ").arg(j-2).arg(j-1).arg(j);
            CoordinateIndexPos.push_back(CoordinateIndex.size());
        }
        out << "Shape {" << std::endl;
        out << "geometry DEF IFS IndexedFaceSet {" << std::endl;
        out << "coord Coordinate { point [" << Coordinate.toStdString() << " ] }" << std::endl;
        out << "coordIndex ["<< CoordinateIndex.left(CoordinateIndexPos[vrml_coordinate_count-1]-1).toStdString() <<"]" << std::endl;
        out << "color Color { color ["<< Color.toStdString() <<"] }" << std::endl;
        out << "colorPerVertex FALSE" << std::endl;
        out << "colorIndex ["<< ColorIndex.toStdString() << "] } }" << std::endl;
    }
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
            if(tract_data[i][j+dim] < pos ^ greater)
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
        if(roi_mgr.exclusive.get())
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


    int total_track_number = tract_data.size();
    int no_fiber_label = total_track_number;
    int have_multiple_fiber_label = total_track_number+1;

    int width = label.width();
    int height = label.height();
    int depth = label.depth();
    int wh = width*height;
    std::fill(label.begin(),label.end(),no_fiber_label);
    int shift[8] = {0,1,width,wh,1+width,1+wh,width+wh,1+width+wh};
    for (unsigned int index = 0;index < total_track_number;++index)
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
    }

    std::set<unsigned int> tracts_to_delete;
    for (unsigned int index = 0;index < label.size();++index)
        if (label[index] < total_track_number)
            tracts_to_delete.insert(label[index]);
    delete_tracts(std::vector<unsigned int>(tracts_to_delete.begin(),tracts_to_delete.end()));
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
    add_tracts(new_tracks,image::rgb_color(200,100,30));
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
    if(trans.empty())
        image::flip_xy(tdi);
    else
    {
        if(sub_voxel)
        {
            std::vector<float> new_trans(trans);
            new_trans[0] /= 4.0;
            new_trans[4] /= 4.0;
            new_trans[8] /= 4.0;
            nii_header.set_image_transformation(new_trans.begin());
        }
        else
            nii_header.set_image_transformation(trans.begin());
    }
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
                pass_map.insert(image::vector<3,int>(std::floor(tract_data[i][j]+0.5),
                                              std::floor(tract_data[i][j+1]+0.5),
                                              std::floor(tract_data[i][j+2]+0.5)));

        data.push_back(pass_map.size()*voxel_volume);
    }

    // output mean and std of each index
    for(int data_index = 0;data_index < handle->view_item.size();++data_index)
    {
        if(handle->view_item[data_index].name == "color")
            continue;
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

    if(handle->db.has_db()) // connectometry database
    {
        std::vector<const float*> old_fa(fib->fa);
        std::vector<std::vector<float> > fa_data;
        for(unsigned int subject_index = 0;subject_index < handle->db.num_subjects;++subject_index)
        {
            handle->db.get_subject_fa(subject_index,fa_data);
            for(unsigned int i = 0;i < fib->fib_num;++i)
                fib->fa[i] = &(fa_data[i][0]);

            data.clear();
            get_quantitative_data(data);
            for(unsigned int index = 4;index < data.size() && index < titles.size();++index)
                out << handle->db.subject_names[subject_index] << " " <<
                       titles[index] << "\t" << data[index] << std::endl;
        }
        fib->fa = old_fa;
    }
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




template<typename input_iterator,typename output_iterator>
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
    if(index_num < handle->get_name_index("color"))
    {
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
                    image::estimate(handle->view_item[index_num].image_data,&(tract_data[fiber_index][tract_index]),data[point_index],image::linear);
            }
            else
                image::estimate(handle->view_item[index_num].image_data,&(tract_data[fiber_index][tract_index]),data[point_index],image::linear);
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


void TractModel::get_passing_list(const std::vector<std::vector<image::vector<3,short> > >& regions,
                                         std::vector<std::vector<unsigned int> >& passing_list,
                                         bool use_end_only) const
{
    passing_list.clear();
    passing_list.resize(tract_data.size());
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
        for(unsigned int i = 0;i < has_region.size();++i)
            if(has_region[i])
                passing_list[index].push_back(i);
    }
}




void ConnectivityMatrix::save_to_image(image::color_image& cm)
{
    if(matrix_value.empty())
        return;
    cm.resize(matrix_value.geometry());
    std::vector<float> values(matrix_value.size());
    std::copy(matrix_value.begin(),matrix_value.end(),values.begin());
    image::normalize(values,255.99);
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

void ConnectivityMatrix::set_atlas(atlas& data,const image::basic_image<image::vector<3,float>,3 >& mni_position)
{
    image::geometry<3> geo(mni_position.geometry());
    image::vector<3> null;
    regions.clear();
    region_name.clear();
    for (unsigned int label_index = 0; label_index < data.get_list().size(); ++label_index)
    {
        std::vector<image::vector<3,short> > cur_region;
        for (image::pixel_index<3> index(geo); index < geo.size();++index)
            if(mni_position[index.index()] != null &&
               data.label_matched(data.get_label_at(mni_position[index.index()]),label_index))
                cur_region.push_back(image::vector<3,short>(index.begin()));
        regions.push_back(cur_region);
        region_name.push_back(data.get_list()[label_index]);
    }
}

bool ConnectivityMatrix::calculate(TractModel& tract_model,std::string matrix_value_type,bool use_end_only)
{
    if(regions.size() == 0)
    {
        error_msg = "No region information. Please assign regions";
        return false;
    }
    tract_model.get_passing_list(regions,passing_list,use_end_only);
    if(matrix_value_type == "trk")
    {
        std::vector<std::vector<std::vector<unsigned int> > > region_passing_list(regions.size());
        for(unsigned int i = 0;i < regions.size();++i)
            region_passing_list[i].resize(regions.size());
        for(unsigned int index = 0;index < passing_list.size();++index)
        {
            std::vector<unsigned int>& region_passed = passing_list[index];
            for(unsigned int i = 0;i < region_passed.size();++i)
                    for(unsigned int j = i+1;j < region_passed.size();++j)
                    {
                        region_passing_list[region_passed[i]][region_passed[j]].push_back(index);
                        region_passing_list[region_passed[j]][region_passed[i]].push_back(index);
                    }
        }

        for(unsigned int i = 0;i < region_passing_list.size();++i)
            for(unsigned int j = i+1;j < region_passing_list.size();++j)
            {
                std::string file_name = region_name[i]+"_"+region_name[j]+".trk";
                tract_model.select_tracts(region_passing_list[i][j]);
                tract_model.save_tracts_to_file(file_name.c_str());
                tract_model.undo();
            }
        return true;
    }
    matrix_value.clear();
    matrix_value.resize(image::geometry<2>(regions.size(),regions.size()));
    std::vector<std::vector<unsigned int> > count(regions.size());
    for(unsigned int i = 0;i < count.size();++i)
        count[i].resize(regions.size());
    for(unsigned int index = 0;index < passing_list.size();++index)
    {
        std::vector<unsigned int>& region_passed = passing_list[index];
        for(unsigned int i = 0;i < region_passed.size();++i)
                for(unsigned int j = i+1;j < region_passed.size();++j)
                {
                    ++count[region_passed[i]][region_passed[j]];
                    ++count[region_passed[j]][region_passed[i]];
                }
    }
    if(matrix_value_type == "count")
    {
        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                matrix_value[index] = count[i][j];
        return true;
    }
    if(matrix_value_type == "ncount")
    {
        std::vector<std::vector<std::vector<unsigned int> > > length_matrix(regions.size());
        for(unsigned int i = 0;i < regions.size();++i)
            length_matrix[i].resize(regions.size());
        for(unsigned int index = 0;index < passing_list.size();++index)
        {
            std::vector<unsigned int>& region_passed = passing_list[index];
            for(unsigned int i = 0;i < region_passed.size();++i)
                    for(unsigned int j = i+1;j < region_passed.size();++j)
                    {
                        length_matrix[region_passed[i]][region_passed[j]].push_back(tract_model.get_tract_length(index));
                        length_matrix[region_passed[j]][region_passed[i]].push_back(tract_model.get_tract_length(index));
                    }
        }
        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                if(!length_matrix[i][j].empty())
                {
                    std::nth_element(length_matrix[i][j].begin(),
                                     length_matrix[i][j].begin()+(length_matrix[i][j].size() >> 1),
                                     length_matrix[i][j].end());
                    matrix_value[index] = count[i][j]/(float)length_matrix[i][j][length_matrix[i][j].size() >> 1];
                }
        return true;
    }
    if(matrix_value_type == "mean_length")
    {
        std::vector<std::vector<unsigned int> > sum_length(regions.size());
        std::vector<std::vector<unsigned int> > sum_n(regions.size());

        for(unsigned int i = 0;i < regions.size();++i)
        {
            sum_length[i].resize(regions.size());
            sum_n[i].resize(regions.size());
        }
        for(unsigned int index = 0;index < passing_list.size();++index)
        {
            std::vector<unsigned int>& region_passed = passing_list[index];
            for(unsigned int i = 0;i < region_passed.size();++i)
                    for(unsigned int j = i+1;j < region_passed.size();++j)
                    {
                        sum_length[region_passed[i]][region_passed[j]] += tract_model.get_tract_length(index);
                        sum_length[region_passed[j]][region_passed[i]] += tract_model.get_tract_length(index);
                        ++sum_n[region_passed[i]][region_passed[j]];
                        ++sum_n[region_passed[j]][region_passed[i]];
                    }
        }
        for(unsigned int i = 0,index = 0;i < count.size();++i)
            for(unsigned int j = 0;j < count[i].size();++j,++index)
                if(sum_n[i][j])
                    matrix_value[index] = (float)sum_length[i][j]/(float)sum_n[i][j]/3.0;
        return true;
    }
    std::vector<std::vector<float> > data;
    if(!tract_model.get_tracts_data(matrix_value_type,data) || data.empty())
    {
        error_msg = "Cannot quantify matrix value using ";
        error_msg += matrix_value_type;
        return false;
    }
    std::vector<std::vector<float> > sum(regions.size());
    for(unsigned int i = 0;i < sum.size();++i)
        sum[i].resize(regions.size());

    for(unsigned int index = 0;index < data.size();++index)
    {
        float m = image::mean(data[index].begin(),data[index].end());
        std::vector<unsigned int>& region_passed = passing_list[index];
        for(unsigned int i = 0;i < region_passed.size();++i)
            for(unsigned int j = i+1;j < region_passed.size();++j)
                {
                    sum[region_passed[i]][region_passed[j]] += m;
                    sum[region_passed[j]][region_passed[i]] += m;
                }
    }
    for(unsigned int i = 0,index = 0;i < count.size();++i)
        for(unsigned int j = 0;j < count[i].size();++j,++index)
            matrix_value[index] = (count[i][j] ? sum[i][j]/(float)count[i][j] : 0);
    return true;

}
template<typename matrix_type>
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
template<typename matrix_type>
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
template<typename matrix_type>
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

template<typename vec_type>
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
    float threshold = max_value*0.001;
    for(unsigned int i = 0;i < binary_matrix.size();++i)
    {
        binary_matrix[i] = matrix_value[i] > threshold ? 1 : 0;
        norm_matrix[i] = matrix_value[i]/max_value;
    }
    // density
    size_t edge = std::accumulate(binary_matrix.begin(),binary_matrix.end(),size_t(0))/2;
    out << "density=" << (float)edge*2.0/(float)(n*n-n) << std::endl;

    // calculate degree
    std::vector<float> degree(n);
    for(unsigned int i = 0;i < n;++i)
        degree[i] = std::accumulate(binary_matrix.begin()+i*n,binary_matrix.begin()+(i+1)*n,0.0);
    // calculate strength
    std::vector<float> nstrength(n),strength(n);
    for(unsigned int i = 0;i < n;++i)
    {
        strength[i] = std::accumulate(matrix_value.begin()+i*n,matrix_value.begin()+(i+1)*n,0.0);
        nstrength[i] = std::accumulate(norm_matrix.begin()+i*n,norm_matrix.begin()+(i+1)*n,0.0);
    }
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

    // calculate weighted clustering coefficient
    image::basic_image<float,2> cyc3(matrix_value.geometry());
    std::vector<float> wcluster_co(n);
    {
        image::basic_image<float,2> root(norm_matrix);
        for(unsigned int j = 0;j < root.size();++j)
            root[j] = std::pow(root[j],(float)(1.0/3.0));
        image::basic_image<float,2> t(root.geometry());
        image::mat::product(root.begin(),root.begin(),t.begin(),image::dyndim(n,n),image::dyndim(n,n));
        image::mat::product(t.begin(),root.begin(),cyc3.begin(),image::dyndim(n,n),image::dyndim(n,n));
        for(unsigned int i = 0;i < strength.size();++i)
        if(degree[i] >= 2)
        {
            float d = degree[i];
            wcluster_co[i] = cyc3[i*(n+1)]/(d*d-d);
        }
    }

    // transitivity
    {
        image::basic_image<float,2> norm_matrix2(norm_matrix.geometry());
        image::basic_image<float,2> norm_matrix3(norm_matrix.geometry());
        image::mat::product(norm_matrix.begin(),norm_matrix.begin(),norm_matrix2.begin(),image::dyndim(n,n),image::dyndim(n,n));
        image::mat::product(norm_matrix2.begin(),norm_matrix.begin(),norm_matrix3.begin(),image::dyndim(n,n),image::dyndim(n,n));
        out << "transitivity(binary)=" << image::mat::trace(norm_matrix3.begin(),image::dyndim(n,n)) /
                (std::accumulate(norm_matrix2.begin(),norm_matrix2.end(),0.0) - image::mat::trace(norm_matrix2.begin(),image::dyndim(n,n))) << std::endl;
        float k = 0;
        for(unsigned int i = 0;i < n;++i)
            k += degree[i]*(degree[i]-1);
        out << "transitivity(weighted)=" << (k == 0 ? 0 : image::mat::trace(cyc3.begin(),image::dyndim(n,n))/k) << std::endl;
    }

    std::vector<float> eccentricity_bin(n),eccentricity_wei(n);

    {
        image::basic_image<float,2> dis_bin,dis_wei;
        distance_bin(binary_matrix,dis_bin);
        distance_wei(matrix_value,dis_wei);
        unsigned int inf_count_bin = std::count(dis_bin.begin(),dis_bin.end(),std::numeric_limits<float>::max());
        unsigned int inf_count_wei = std::count(dis_wei.begin(),dis_wei.end(),std::numeric_limits<float>::max());
        std::replace(dis_bin.begin(),dis_bin.end(),std::numeric_limits<float>::max(),(float)0);
        std::replace(dis_wei.begin(),dis_wei.end(),std::numeric_limits<float>::max(),(float)0);
        out << "network_characteristic_path_length(binary)=" << std::accumulate(dis_bin.begin(),dis_bin.end(),0.0)/(n*n-inf_count_bin) << std::endl;
        out << "network_characteristic_path_length(weighted)=" << std::accumulate(dis_wei.begin(),dis_wei.end(),0.0)/(n*n-inf_count_wei) << std::endl;
        image::basic_image<float,2> invD;
        inv_dis(dis_bin,invD);
        out << "global_efficiency(binary)=" << std::accumulate(invD.begin(),invD.end(),0.0)/(n*n-inf_count_bin) << std::endl;
        inv_dis(dis_wei,invD);
        out << "global_efficiency(weighted)=" << std::accumulate(invD.begin(),invD.end(),0.0)/(n*n-inf_count_wei) << std::endl;

        for(unsigned int i = 0,ipos = 0;i < n;++i,ipos += n)
        {
            eccentricity_bin[i] = *std::max_element(dis_bin.begin()+ipos,
                                                 dis_bin.begin()+ipos+n);
            eccentricity_wei[i] = *std::max_element(dis_wei.begin()+ipos,
                                                 dis_wei.begin()+ipos+n);

        }
        out << "diameter_of_graph(binary)=" << *std::max_element(eccentricity_bin.begin(),eccentricity_bin.end()) <<std::endl;
        out << "diameter_of_graph(weighted)=" << *std::max_element(eccentricity_wei.begin(),eccentricity_wei.end()) <<std::endl;


        std::replace(eccentricity_bin.begin(),eccentricity_bin.end(),(float)0,std::numeric_limits<float>::max());
        std::replace(eccentricity_wei.begin(),eccentricity_wei.end(),(float)0,std::numeric_limits<float>::max());
        out << "radius_of_graph(binary)=" << *std::min_element(eccentricity_bin.begin(),eccentricity_bin.end()) <<std::endl;
        out << "radius_of_graph(weighted)=" << *std::min_element(eccentricity_wei.begin(),eccentricity_wei.end()) <<std::endl;
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
                            newA[pos] = matrix_value[index];
                        ++pos;
                    }
            std::vector<float> sw;
            for(unsigned int j = 0;j < n;++j)
                if(binary_matrix[ipos+j])
                    sw.push_back(std::pow(matrix_value[ipos+j],(float)(1.0/3.0)));
            image::basic_image<float,2> invD;
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
        float sum = image::vec::dot(degi.begin(),degi.end(),degj.begin())/degi.size();
        image::square(degi);
        image::square(degj);
        float b = (std::accumulate(degi.begin(),degi.end(),0.0)+
                   std::accumulate(degj.begin(),degj.end(),0.0))/2.0/degi.size();
        a = a*a;
        out << "assortativity_coefficient(binary) = " << (b == a ? 0 : ( sum  - a)/ ( b - a )) << std::endl;
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
        out << "assortativity_coefficient(weighted) = " << ( sum  - a*a)/ ( b - a*a ) << std::endl;
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
            image::basic_image<float,2> G1(matrix_value);
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
        image::mat::eigen_decomposition_sym(matrix_value.begin(),V.begin(),d.begin(),image::dyndim(n,n));
        std::copy(V.begin(),V.begin()+n,eigenvector_centrality_wei.begin());
    }

    std::vector<float> pagerank_centrality_bin(n),pagerank_centrality_wei(n);
    {
        float d = 0.85;
        std::vector<float> deg_bin(degree.begin(),degree.end()),deg_wei(strength.begin(),strength.end());
        std::replace(deg_bin.begin(),deg_bin.end(),0.0,1.0);
        std::replace(deg_wei.begin(),deg_wei.end(),0.0,1.0);

        image::basic_image<float,2> B_bin(binary_matrix.geometry()),B_wei(binary_matrix.geometry());
        for(unsigned int i = 0,index = 0;i < n;++i)
            for(unsigned int j = 0;j < n;++j,++index)
            {
                B_bin[index] = -d*((float)binary_matrix[index])*1.0/deg_bin[j];
                B_wei[index] = -d*matrix_value[index]*1.0/deg_wei[j];
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




    report = out.str();




}
