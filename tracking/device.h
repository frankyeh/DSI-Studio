#ifndef DEVICE_H
#define DEVICE_H
#include <string>
#include "zlib.h"
#include "TIPL/tipl.hpp"
class Device
{
public:
    std::string name;
    std::string type;
public:
    tipl::vector<3> pos;
    tipl::vector<3> dir;
    tipl::rgb color = 0xFFFFFFFF;
public: // for electrode
    float length = 30.0f;
private:
    float rendering_radius = 0.0; // for internnal use only to select device
public:
    Device();
    bool selected(const tipl::vector<3>& p,float vs,float& device_selected_length,float& distance_in_voxel);
    void move(float select_length,const tipl::vector<3>& dis);
    std::string to_str(void);
    bool from_str(const std::string& str);
    bool load_from_file(const char* file_name);
    void get_rendering(std::vector<float>& seg_length,
                       std::vector<char>& seg_type,
                       float& radius);
};

extern std::vector<std::string> device_types;
bool load_device_content(void);
#endif // DEVICE_H
