#ifndef ATLAS_HPP
#define ATLAS_HPP
#include "tipl/tipl.hpp"
#include <vector>
#include <string>
class atlas{
private:
    tipl::image<uint32_t,3> I;
    std::vector<uint32_t> label_num;
    std::vector<std::string> labels;
    tipl::matrix<4,4,float> T;
    void load_label(void);
    size_t get_index(tipl::vector<3,float> atlas_space);
private:// for talairach only
    std::vector<std::vector<size_t> > index2label;
    std::vector<std::vector<size_t> > label2index;
private:// for track atlas only
    tipl::image<char,4> track;
    std::vector<uint32_t> track_base_pos;
    bool is_track;
public:
    std::string name,filename,error_msg;
public:
    bool load_from_file(void);
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
    const std::vector<uint32_t>& get_num(void)
    {
        if(labels.empty())
        {
            load_label();
            if(labels.empty())
                load_from_file();
        }
        return label_num;
    }
    //std::string get_label_name_at(const tipl::vector<3,float>& mni_space);
    bool is_labeled_as(const tipl::vector<3,float>& mni_space,unsigned int label);
    int get_track_label(const std::vector<tipl::vector<3> >& points);
};

#endif // ATLAS_HPP
