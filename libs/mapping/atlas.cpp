#include <fstream>
#include <sstream>
#include <QCoreApplication>
#include <QDir>
#include "atlas.hpp"


void apply_trans(tipl::vector<3>& pos,const tipl::matrix<4,4>& trans);

std::string get_label_file_name(const std::string& file_name)
{
    std::string label_name(file_name);
    tipl::remove_suffix(label_name,".nii.gz");
    tipl::remove_suffix(label_name,".nii");
    return label_name + ".txt";
}
void atlas::load_label(void)
{
    std::string text_file_name = get_label_file_name(filename);
    if(!std::filesystem::exists(text_file_name))
    {
        error_msg = "cannot find label file at ";
        error_msg += text_file_name;
        return;
    }
    for(const auto& line: tipl::read_text_file(text_file_name))
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
    tipl::io::gz_nifti nii(filename,std::ios::in);
    if(!nii)
    {
        error_msg = nii.error_msg;
        return false;
    }
    if(name.empty())
        name = QFileInfo(filename.c_str()).baseName().toStdString();
    is_multiple_roi = (nii.dim(4) > 1); // 4d nifti as multiple roi
    if(is_multiple_roi)
    {
        nii >> multiple_I;
        I.resize(tipl::shape<3>(multiple_I.width(),multiple_I.height(),multiple_I.depth()));
        multiple_I_3d.clear();
        for(size_t pos = 0;pos < multiple_I.size();pos += I.size())
            multiple_I_3d.push_back(tipl::make_image(&*multiple_I.data() + pos,I.shape()));
    }
    else
        nii >> I;

    nii >> T;
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
            // remove empty regions
            /*
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
                    */
        }
    }
    return true;
}


bool atlas::is_labeled_as(tipl::vector<3,float> template_space,unsigned int region_index)
{
    if(I.empty())
        load_from_file();
    if(region_index >= region_value.size())
        return false;
    if(!in_template_space)
        apply_trans(template_space,T);
    unsigned int voxel_index = 0;
    if(is_multiple_roi)
    {
        if(region_index >= multiple_I_3d.size())
            return false;
        return tipl::estimate<tipl::interpolation::majority>(multiple_I_3d[region_index],template_space);
    }
    if(!tipl::estimate<tipl::interpolation::majority>(I,template_space,voxel_index))
        return false;
    return voxel_index == region_value[region_index];
}
int atlas::region_index_at(tipl::vector<3,float> template_space)
{
    if(is_multiple_roi)
        return -1;
    if(I.empty())
        load_from_file();
    if(!in_template_space)
        apply_trans(template_space,T);
    unsigned int value = 0;
    if(!tipl::estimate<tipl::interpolation::majority>(I,template_space,value))
        return -1;
    if(value >= value2index.size())
        return -1;
    return int(value2index[value])-1;
}
void atlas::region_indices_at(tipl::vector<3,float> template_space,std::vector<uint16_t>& indices)
{
    if(!is_multiple_roi)
        return;
    if(I.empty())
        load_from_file();
    if(!in_template_space)
        apply_trans(template_space,T);

    for(uint32_t region_index = 0;region_index < multiple_I_3d.size();++region_index)
        if(tipl::estimate<tipl::interpolation::majority>(multiple_I_3d[region_index],template_space))
            indices.push_back(region_index);
}


