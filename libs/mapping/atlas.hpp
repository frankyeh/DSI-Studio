#ifndef ATLAS_HPP
#define ATLAS_HPP
#include "tipl/tipl.hpp"
#include <vector>
#include <string>
class atlas{
private:
    tipl::image<int,3> I;
    std::vector<int> label_num;
    std::vector<std::string> labels;
    float T[12];
    void load_label(void);
    int get_index(tipl::vector<3,float> atlas_space);
private:
    tipl::image<tipl::vector<3,float>,3 > mapping; // between template mapping
    float T1[12],T2[12]; // template trans matrix
private:// for talairach only
    std::vector<std::vector<unsigned int> > index2label;
    std::vector<std::vector<unsigned int> > label2index;
private:// for track atlas only
    tipl::image<char,4> track;
    std::vector<unsigned int> track_base_pos;
    bool is_track;
public:
    std::string name,filename,error_msg;
    int template_from = -1;
    int template_to = -1;
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
    //std::string get_label_name_at(const tipl::vector<3,float>& mni_space);
    bool is_labeled_as(const tipl::vector<3,float>& mni_space,unsigned int label);
    int get_track_label(const std::vector<tipl::vector<3> >& points);
};

#endif // ATLAS_HPP
