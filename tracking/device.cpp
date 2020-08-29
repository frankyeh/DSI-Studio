#include "libs/gzip_interface.hpp"
#include "device.h"


char device_types[7][40]=
{"DBS Lead:Medtronic 3387",
 "DBS Lead:Medtronic 3389",
 "DBS Lead:Abbott Infinity",
 "DBS Lead:Boston Scientific",
 "Probe",
 "Obturator:11 mm",
 "Obturator:13.5 mm"};

void Device::get_rendering(std::vector<float>& seg_length,
                   std::vector<char>& seg_type,
                   float& radius)
{
    if(type == device_types[0])
    {
        seg_length = {0.0f,1.0f,1.5f,1.5f,1.5f,1.5f,1.5f,1.5f,1.5f,length};
        seg_type = {-1,0,1,0,1,0,1,0,1,0};
        radius = 1.27f*0.5f;
    }
    if(type == device_types[1])
    {
        seg_length = {0.0f,1.0f,1.5f,0.5f,1.5f,0.5f,1.5f,0.5f,1.5f,length};
        seg_type = {-1,0,1,0,1,0,1,0,1,0};
        radius = 1.27f*0.5f;
    }
    if(type == device_types[2])
    {
        seg_length = {0.0f,1.0f,1.5f,0.5f,1.5f,0.5f,1.5f,0.5f,1.5f,length};
        seg_type = {-1,0,1,0,2,0,2,0,1,0};
        radius = 1.27f*0.5f;
    }
    if(type == device_types[3])
    {
        seg_length = {0.0f,1.0f,0.5f,1.5f,0.5f,1.5f,0.5f,1.5f,length};
        seg_type = {-1,1,0,2,0,2,0,1,0};
        radius = 1.3f*0.5f;
    }
    if(type == device_types[4])
    {
        seg_length = {0.0f,length};
        seg_type = {-1,0};
        radius = 1.0f*0.5f;
    }
    if(type == device_types[5])
    {
        seg_length = {length,2.0f};
        seg_type = {0,-2};
        radius = 11.0f*0.5f;
    }
    if(type == device_types[6])
    {
        seg_length = {length,2.0f};
        seg_type = {0,-2};
        radius = 13.5f*0.5f;
    }

}

Device::Device()
{
}


bool Device::has_point(const tipl::vector<3>& p,float& device_selected_length)
{
    auto dis = p-pos;
    if(float(dis.length()) < length) // check if the body is selected
    {
        auto proj = dir;
        device_selected_length = dis*proj;
        proj *= device_selected_length;
        dis -= proj;
        if(dis.length() < 1.0)
            return true;
    }
    return false;
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

