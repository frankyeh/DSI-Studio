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
    void load_from_file(void);
    void load_label(void);
private:// for talairach only
    std::vector<std::vector<unsigned int> > index2label;
    std::vector<std::vector<unsigned int> > label2index;
public:
    std::string name,filename;
public:
    const std::vector<std::string>& get_list(void) {if(labels.empty())load_label();return labels;}
    const std::vector<short>& get_num(void) {if(labels.empty())load_label();return label_num;}
    short get_label_at(const image::vector<3,float>& mni_space);
    std::string get_label_name_at(const image::vector<3,float>& mni_space);
    bool is_labeled_as(const image::vector<3,float>& mni_space,unsigned int label);
    bool label_matched(short image_label,unsigned int region_label);
};

#endif // ATLAS_HPP
