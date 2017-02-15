#ifndef ATLAS_HPP
#define ATLAS_HPP
#include "image/image.hpp"
#include <vector>
#include <string>
class atlas{
private:
    image::basic_image<int,3> I;
    std::vector<int> label_num;
    std::vector<std::string> labels;
    image::matrix<4,4,float> transform;
    void load_from_file(void);
    void load_label(void);
private:// for talairach only
    std::vector<std::vector<unsigned int> > index2label;
    std::vector<std::vector<unsigned int> > label2index;
private:// for track atlas only
    image::basic_image<char,4> track;
    std::vector<unsigned int> track_base_pos;
    bool is_track;
public:
    std::string name,filename;
public:
    const std::vector<std::string>& get_list(void)
    {
        if(labels.empty())
        {
            load_label();
            if(labels.empty())
                load_from_file();
        }
        return labels;
    }
    const std::vector<int>& get_num(void)
    {
        if(labels.empty())
        {
            load_label();
            if(labels.empty())
                load_from_file();
        }
        return label_num;
    }
    //std::string get_label_name_at(const image::vector<3,float>& mni_space);
    bool is_labeled_as(const image::vector<3,float>& mni_space,unsigned int label);
};

#endif // ATLAS_HPP
