#include "device.h"
extern std::vector<std::vector<float> > device_seg_length;
extern std::vector<std::vector<char> > device_seg_type;
extern std::vector<float> device_radius;
extern std::string device_content_file;

std::vector<std::string> device_types;
std::vector<std::vector<float> > device_seg_length;
std::vector<std::vector<char> > device_seg_type;
std::vector<float> device_radius;

bool load_device_content(void)
{
    std::ifstream in(device_content_file.c_str());
    if(!in)
        return false;
    std::string line;
    while(std::getline(in,line))
    {
        device_types.push_back(line);
        // read device segmentation length
        {
            std::getline(in,line);
            std::istringstream in2(line);
            std::vector<float> seg_length;
            std::copy(std::istream_iterator<float>(in2),std::istream_iterator<float>(),std::back_inserter(seg_length));
            device_seg_length.push_back(std::move(seg_length));
        }
        // read device segmentation length
        {
            std::getline(in,line);
            std::istringstream in2(line);
            std::vector<char> seg_type;
            std::copy(std::istream_iterator<int>(in2),std::istream_iterator<int>(),std::back_inserter(seg_type));
            device_seg_type.push_back(std::move(seg_type));
        }
        // read radius
        {
            std::getline(in,line);
            std::istringstream in2(line);
            float value;
            in2 >> value;
            device_radius.push_back(value);
        }
    }
    return true;
}

void Device::get_rendering(std::vector<float>& seg_length,
                   std::vector<char>& seg_type,
                   float& radius)
{
    // DBS Lead
    for(size_t i = 0;i < device_types.size();++i)
        if(type == device_types[i])
        {
            seg_length = device_seg_length[i];
            seg_type = device_seg_type[i];
            rendering_radius = radius = device_radius[i]*0.5f;
            break;
        }
    auto pos = std::find(seg_type.begin(),seg_type.end(),3);
    if(pos != seg_type.end())
         seg_length[size_t(pos-seg_type.begin())] = length;
}

Device::Device()
{
}


bool Device::selected(const tipl::vector<3>& p,float vs,float& device_selected_length,float& distance_in_voxel)
{
    auto dis = p-pos;
    float dis_length = float(dis.length());
    if(dis_length > length/vs)
        return false;
    float radius = 0.5f+rendering_radius/vs;
    if(dis_length < radius) // selecting the tip
    {
        device_selected_length = 0.0f;
        distance_in_voxel = dis_length;
        return true;
    }
    // now consider selecting the shaft
    auto proj = dir;
    device_selected_length = dis*proj;
    if(device_selected_length/dis_length < 0.8f)
        return false;
    proj *= device_selected_length;
    dis -= proj;
    distance_in_voxel = float(dis.length());
    return float(dis.length()) < radius;
}

void Device::move(float device_selected_length,const tipl::vector<3>& dis)
{
    if(device_selected_length < 5.0f)
        pos += dis;
    else
    {
        dir = dir*device_selected_length+dis;
        dir.normalize();
    }
}

std::string Device::to_str(void)
{
    std::ostringstream out;
    out << name << ',' << type << ',' << pos << ',' << dir << ',' << length << ',' << uint32_t(color) << std::endl;
    return out.str();
}
bool Device::from_str(const std::string& str)
{
    std::istringstream in(str); //create string stream from the string
    std::string pos_str,dir_str,length_str,color_str;
    if(!std::getline(in, name, ',') ||
       !std::getline(in, type, ',') ||
       !std::getline(in, pos_str, ',') ||
       !std::getline(in, dir_str, ',') ||
       !std::getline(in, length_str, ',') ||
       !std::getline(in, color_str, ','))
        return false;
    std::istringstream(pos_str) >> pos;
    std::istringstream(dir_str) >> dir;
    std::istringstream(length_str) >> length;
    uint32_t color_t = 0;
    std::istringstream(color_str) >> color_t;
    color = color_t;
    dir.normalize();
    return true;
}

