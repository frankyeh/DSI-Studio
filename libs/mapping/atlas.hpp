#ifndef ATLAS_HPP
#define ATLAS_HPP
#include "image/image.hpp"
#include <vector>
#include <string>
class atlas{
private:
    image::basic_image<short,3> I;
    std::vector<short> label_num;
    std::vector<std::string> labels;
    image::matrix<4,4,float> transform;
    void load_from_file(const char* file_name);

private:// for talairach only
    std::vector<std::vector<unsigned int> > index2label;
    std::vector<std::vector<unsigned int> > label2index;
public:
    std::string name,filename;
public:
    const std::vector<std::string>& get_list(void) {if(I.empty())load_from_file(filename.c_str());return labels;}
    const std::vector<short>& get_num(void) {if(I.empty())load_from_file(filename.c_str());return label_num;}
    short get_label_at(const image::vector<3,float>& mni_space);
    std::string get_label_name_at(const image::vector<3,float>& mni_space);
    bool is_labeled_as(const image::vector<3,float>& mni_space,short label);
    bool label_matched(short image_label,short region_label);
};

#endif // ATLAS_HPP
