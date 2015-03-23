#include "atlas.hpp"
#include <fstream>
#include <sstream>
#include "libs/gzip_interface.hpp"




void atlas::load_from_file(const char* file_name)
{
    {
        gz_nifti nii;
        if(!nii.load_from_file(file_name))
            throw std::runtime_error("Cannot load atlas file");
        nii >> I;
        transform.clear();
        transform.resize(16);
        transform[15] = 1.0;
        nii.get_image_transformation(transform.begin());
        image::matrix::inverse(transform.begin(),image::dim<4,4>());
    }
    std::string file_name_str(file_name);
    std::string text_file_name;

    if (file_name_str.length() > 3 &&
            file_name_str[file_name_str.length()-3] == '.' &&
            file_name_str[file_name_str.length()-2] == 'g' &&
            file_name_str[file_name_str.length()-1] == 'z')
        text_file_name = std::string(file_name_str.begin(),file_name_str.end()-6);
    else
        text_file_name = std::string(file_name_str.begin(),file_name_str.end()-3);
    text_file_name += "txt";
    if(image::geometry<3>(141,172,110) == I.geometry())
    {
        std::ifstream in(text_file_name.c_str());
        if(!in)
            throw std::runtime_error("Cannot load atlas label file");
        std::map<std::string,std::set<unsigned int> > regions;
        std::string line;
        for(int i = 0;std::getline(in,line);++i)
        {
            std::istringstream read_line(line);
            int num;
            read_line >> num;
            std::string region;
            while (read_line >> region)
            {
                if(region == "*")
                    continue;
                regions[region].insert(i);
            }
            index2label.resize(i+1);
        }

        std::map<std::string,std::set<unsigned int> >::iterator iter = regions.begin();
        std::map<std::string,std::set<unsigned int> >::iterator end = regions.end();
        for (int i = 0;iter != end;++iter,++i)
        {
            labels.push_back(iter->first);
            label_num.push_back(label_num.size());// dummy
            label2index.push_back(std::vector<unsigned int>(iter->second.begin(),iter->second.end()));
            for(int j = 0;j < label2index.back().size();++j)
                index2label[label2index[i][j]].push_back(i);
        }
    }
    else
    {
        std::vector<unsigned short> hist(1+*std::max_element(I.begin(),I.end()));
        for(int index = 0;index < I.size();++index)
            hist[I[index]] = 1;

        std::ifstream in(text_file_name.c_str());
        if(in)
        {
            std::string line,txt;
            while(std::getline(in,line))
            {
                if(line.empty() || line[0] == '#')
                    continue;
                std::istringstream read_line(line);
                int num = 0;
                read_line >> num >> txt;
                if(num < 0 || num >= hist.size() || !hist[num])
                    continue;
                label_num.push_back(num);
                labels.push_back(txt);
            }
        }
        else
        {
            for(int index = 1;index < hist.size();++index)
                if(hist[index])
                {
                    std::ostringstream out_name;
                    label_num.push_back(index);
                    out_name << "region " << index;
                    labels.push_back(out_name.str());
                }
        }
    }
}


void mni_to_tal(float& x,float &y, float &z)
{
    x *= 0.9900;
    float ty = 0.9688*y + ((z >= 0) ? 0.0460*z : 0.0420*z) ;
    float tz = -0.0485*y + ((z >= 0) ? 0.9189*z : 0.8390*z) ;
    y = ty;
    z = tz;
}


short atlas::get_label_at(const image::vector<3,float>& mni_space)
{
    if(I.empty())
        load_from_file(filename.c_str());
    image::vector<3,float> atlas_space(mni_space);
    image::vector_transformation(mni_space.begin(),atlas_space.begin(),transform.begin(),image::vdim<3>());
    atlas_space += 0.5;
    atlas_space.floor();
    if(!I.geometry().is_valid(atlas_space))
        return 0;
    return I.at(atlas_space[0],atlas_space[1],atlas_space[2]);
}

std::string atlas::get_label_name_at(const image::vector<3,float>& mni_space)
{
    if(I.empty())
        load_from_file(filename.c_str());
    short l = get_label_at(mni_space);
    if(!l)
        return std::string();
    if(index2label.empty())
    {
        unsigned int pos = std::find(label_num.begin(),label_num.end(),l)-label_num.begin();
        return pos >= labels.size() ? std::string() : labels[pos];
    }
    if(l >= index2label.size())
        return std::string();
    std::string result;
    for(int i = 0;i < index2label[l].size();++i)
    {
        result += labels[index2label[l][i]];
        result += " ";
    }
    return result;
}

bool atlas::is_labeled_as(const image::vector<3,float>& mni_space,short label_name_index)
{
    if(I.empty())
        load_from_file(filename.c_str());
    return label_matched(get_label_at(mni_space),label_name_index);
}
bool atlas::label_matched(short l,short label_name_index)
{
    if(I.empty())
        load_from_file(filename.c_str());
    if(index2label.empty())
        return label_name_index >= label_num.size() ? false:l == label_num[label_name_index];
    if(l >= index2label.size())
        return false;
    return std::find(index2label[l].begin(),index2label[l].end(),label_name_index) != index2label[l].end();
}
void atlas::calculate_order(std::vector<float>& order)
{
    if(I.empty())
        load_from_file(filename.c_str());

    if(!index2label.empty())
    {
        order.resize(labels.size());
        for(unsigned int index = 0;index < order.size();++index)
            order[index] = index;
        return;
    }
    //setBoundingBox(-78,-112,-50,78,76,85,1.0);
    std::vector<image::vector<3> > mean(label_num.size());
    std::vector<unsigned int> count(label_num.size());
    std::vector<float> max_x(label_num.size()),min_x(label_num.size());
    std::vector<unsigned short> index_map(*std::max_element(label_num.begin(),label_num.end())+1);
    for(unsigned short index = 0;index < label_num.size();++index)
        index_map[label_num[index]] = index;
    for(float z = -50;z < 85;z += 1.0)
        for(float y = -112;y < 76;y += 1.0)
            for(float x = -78;x < 78;x += 1.0)
                {
                    image::vector<3> pos(x,y,z);
                    short label = get_label_at(pos);
                    if(!label || label >= index_map.size())
                        continue;
                    unsigned int label_index = index_map[label];
                    count[label_index]++;
                    mean[label_index] += pos;
                    if(x > max_x[label_index])
                        max_x[label_index] = x;
                    if(x < min_x[label_index])
                        min_x[label_index] = x;
                }
    order.resize(label_num.size());
    for(int index = 0;index < label_num.size();++index)
    {
        if(count[index])
            mean[index] /= count[index];
        // separate left right first
        order[index] = (mean[index][0] > 0) ? 500.0-mean[index][1]:mean[index][1]-500.0;
        // is at middle?
        if((max_x[index]-min_x[index])/8.0 > std::fabs(mean[index][0]))
            order[index] = mean[index][1];
    }
}
