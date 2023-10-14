#include <fstream>
#include <sstream>
#include <QCoreApplication>
#include <QDir>
#include "atlas.hpp"


void apply_trans(tipl::vector<3>& pos,const tipl::matrix<4,4>& trans);

std::string get_label_file_name(const std::string& file_name)
{
    if(tipl::ends_with(file_name,".nii.gz"))
        return file_name.substr(0,file_name.size()-6) +"txt";
    if(tipl::ends_with(file_name,".nii"))
        return file_name.substr(0,file_name.size()-3) +"txt";
    return file_name + ".txt";
}
void atlas::load_label(void)
{
    std::string text_file_name = get_label_file_name(filename);
    std::ifstream in(text_file_name.c_str());
    if(!in)
    {
        error_msg = "cannot find label file at ";
        error_msg += text_file_name;
        return;
    }
    std::string line;
    while(std::getline(in,line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        std::string txt;
        uint32_t num = 0;
        std::istringstream(line) >> num >> txt;
        if(txt.empty() || num > std::numeric_limits<uint16_t>::max())
            continue;
        if(num >= value2index.size())
            value2index.resize(num+1);

        region_value.push_back(num);
        value2index[num] = uint16_t(region_value.size());
        labels.push_back(txt);
    }
}
extern std::vector<std::string> fa_template_list;
bool atlas::load_from_file(void)
{
    if(!I.empty())
        return true;
    tipl::io::gz_nifti nii;
    if(!nii.load_from_file(filename.c_str()))
    {
        error_msg = nii.error_msg;
        return false;
    }
    if(name.empty())
        name = QFileInfo(filename.c_str()).baseName().toStdString();
    is_multiple_roi = (nii.dim(4) > 1); // 4d nifti as multiple roi
    if(is_multiple_roi)
    {
        nii.toLPS(multiple_I);
        I.resize(tipl::shape<3>(multiple_I.width(),multiple_I.height(),multiple_I.depth()));
        for(unsigned int i = 0;i < multiple_I.size();i += I.size())
            multiple_I_pos.push_back(i);
    }
    else
        nii.toLPS(I);

    nii.get_image_transformation(T);
    if(T == template_to_mni)
    {
        in_template_space = true;
        T.identity();
    }
    else
        T = tipl::from_space(template_to_mni).to(T);

    if(labels.empty())
        load_label();

    if(label2index.empty() && !is_multiple_roi) // not talairach not tracks
    {
        std::vector<unsigned char> hist(1+tipl::max_value(I));
        for(size_t index = 0;index < I.size();++index)
            hist[size_t(I[index])] = 1;
        if(labels.empty())
        {
            for(uint32_t index = 1;index < hist.size();++index)
                if(hist[index])
                {
                    std::ostringstream out_name;
                    region_value.push_back(index);
                    out_name << "region " << index;
                    labels.push_back(out_name.str());
                }
        }
        else
        {
            bool modified_atlas = false;
            for(size_t i = 0;i < labels.size();)
                if(region_value[i] >= hist.size() || !hist[region_value[i]])
                {
                    labels.erase(labels.begin()+long(i));
                    region_value.erase(region_value.begin()+long(i));
                    modified_atlas = true;
                }
            else
                ++i;
            if(modified_atlas)
                for(size_t i = 0;i < region_value.size();++i)
                    value2index[region_value[i]] = uint16_t(i+1);
        }
    }
    return true;
}

size_t atlas::get_index(tipl::vector<3,float> p)
{
    // template to atlas space
    if(!in_template_space)
        apply_trans(p,T);
    p.round();
    if(!I.shape().is_valid(p))
        return 0;
    return size_t((int64_t(p[2])*int64_t(I.height())+int64_t(p[1]))*int64_t(I.width())+int64_t(p[0]));
}

bool atlas::is_labeled_as(const tipl::vector<3,float>& template_space,unsigned int region_index)
{
    if(I.empty())
        load_from_file();
    if(region_index >= region_value.size())
        return false;
    size_t offset = get_index(template_space);
    if(!offset || offset >= I.size())
        return false;
    if(is_multiple_roi)
    {
        if(region_index >= multiple_I_pos.size())
            return false;
        size_t pos = multiple_I_pos[region_index] + offset;
        if(pos >= multiple_I.size())
            return false;
        return multiple_I[pos];
    }
    return I[offset] == region_value[region_index];
}
int atlas::region_index_at(const tipl::vector<3,float>& template_space)
{
    if(is_multiple_roi)
        return -1;
    if(I.empty())
        load_from_file();
    size_t offset = get_index(template_space);
    if(!offset || offset >= I.size())
        return -1;
    auto value = I[offset];
    if(value >= value2index.size())
        return -1;
    return int(value2index[value])-1;
}
void atlas::region_indices_at(const tipl::vector<3,float>& template_space,std::vector<uint16_t>& indices)
{
    if(!is_multiple_roi)
        return;
    if(I.empty())
        load_from_file();
    size_t offset = get_index(template_space);
    if(!offset || offset >= I.size())
        return;
    for(uint32_t region_index = 0;region_index < multiple_I_pos.size();++region_index)
    {
        size_t pos = multiple_I_pos[region_index] + offset;
        if(pos < multiple_I.size() && multiple_I[pos])
            indices.push_back(region_index);
    }
}


