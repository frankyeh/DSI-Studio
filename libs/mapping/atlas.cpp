#include "atlas.hpp"
#include <fstream>
#include <sstream>
#include "libs/gzip_interface.hpp"
#include <QCoreApplication>
#include <QDir>

void sub2mni(tipl::vector<3>& pos,const tipl::matrix<4,4,float>& trans);
void mni2sub(tipl::vector<3>& pos,const tipl::matrix<4,4,float>& trans);

void atlas::load_label(void)
{
    std::string file_name_str(filename);
    std::string text_file_name;
    if (file_name_str.length() > 3 &&
            file_name_str[file_name_str.length()-3] == '.' &&
            file_name_str[file_name_str.length()-2] == 'g' &&
            file_name_str[file_name_str.length()-1] == 'z')
        text_file_name = std::string(file_name_str.begin(),file_name_str.end()-6);
    else
        text_file_name = std::string(file_name_str.begin(),file_name_str.end()-3);
    text_file_name += "txt";
    std::ifstream in(text_file_name.c_str());
    if(!in)
        return;
    std::vector<std::string> text;
    std::string str;
    while(std::getline(in,str))
        text.push_back(str);

    if(text[0] == "0\t* * * * *")//talairach
    {
        std::map<std::string,std::set<size_t> > regions;
        for(size_t i = 0;i < text.size();++i)
        {
            std::istringstream read_line(text[i]);
            int num;
            read_line >> num;
            std::string region;
            while (read_line >> region)
            {
                if(region == "*")
                    continue;
                regions[region].insert(i);
            }
            index2label.resize(i+1);
        }

        auto iter = regions.begin();
        auto end = regions.end();
        for (size_t i = 0;iter != end;++iter,++i)
        {
            labels.push_back(iter->first);
            label_num.push_back(uint32_t(label_num.size()));// dummy
            label2index.push_back(std::vector<size_t>(iter->second.begin(),iter->second.end()));
            for(size_t j = 0;j < label2index.back().size();++j)
                index2label[label2index[i][j]].push_back(i);
        }
    }
    else
    {
        for(auto& line : text)
        {
            if(line.empty() || line[0] == '#')
                continue;
            std::string txt;
            uint32_t num = 0;
            std::istringstream(line) >> num >> txt;
            if(txt.empty())
                continue;
            label_num.push_back(num);
            labels.push_back(txt);
        }

    }
}
extern std::vector<std::string> fa_template_list;
bool atlas::load_from_file(void)
{
    if(!I.empty())
        return true;
    gz_nifti nii;
    if(!nii.load_from_file(filename.c_str()))
    {
        error_msg = "Cannot load atlas file";
        return false;
    }
    if(name.empty())
        name = QFileInfo(filename.c_str()).baseName().toStdString();
    is_track = (nii.dim(4) > 1); // 4d nifti as track files
    if(is_track)
    {
        nii.toLPS(track);
        I.resize(tipl::geometry<3>(track.width(),track.height(),track.depth()));
        for(unsigned int i = 0;i < track.size();i += I.size())
            track_base_pos.push_back(i);
    }
    else
        nii.toLPS(I);
    nii.get_image_transformation(T);

    if(labels.empty())
        load_label();

    if(label2index.empty() && !is_track) // not talairach not tracks
    {
        std::vector<unsigned char> hist(1+*std::max_element(I.begin(),I.end()));
        for(size_t index = 0;index < I.size();++index)
            hist[size_t(I[index])] = 1;
        if(labels.empty())
        {
            for(uint32_t index = 1;index < hist.size();++index)
                if(hist[index])
                {
                    std::ostringstream out_name;
                    label_num.push_back(index);
                    out_name << "region " << index;
                    labels.push_back(out_name.str());
                }
        }
        else
        {
            //bool modified_atlas = false;
            for(size_t i = 0;i < labels.size();)
                if(label_num[i] >= hist.size() || !hist[label_num[i]])
                {
                    labels.erase(labels.begin()+long(i));
                    label_num.erase(label_num.begin()+long(i));
                    //modified_atlas = true;
                }
            else
                ++i;
            // used to removed empty label
            /*
            if(modified_atlas)
            {
                std::string file_name_str(filename);
                std::string text_file_name;
                if (file_name_str.length() > 3 &&
                        file_name_str[file_name_str.length()-3] == '.' &&
                        file_name_str[file_name_str.length()-2] == 'g' &&
                        file_name_str[file_name_str.length()-1] == 'z')
                    text_file_name = std::string(file_name_str.begin(),file_name_str.end()-6);
                else
                    text_file_name = std::string(file_name_str.begin(),file_name_str.end()-3);
                text_file_name += "txt";
                std::ofstream out(text_file_name.c_str());
                for(int i = 0;i < labels.size();++i)
                    out << label_num[i] << " " << labels[i] << std::endl;
            }*/
        }
    }
    return true;
}

size_t atlas::get_index(tipl::vector<3,float> p)
{
    mni2sub(p,T);
    p.round();
    if(!I.geometry().is_valid(p))
        return 0;
    return size_t((int(p[2])*I.height()+int(p[1]))*I.width()+int(p[0]));
}

bool atlas::is_labeled_as(const tipl::vector<3,float>& mni_space,unsigned int label_name_index)
{
    if(I.empty())
        load_from_file();
    if(label_name_index >= label_num.size())
        return false;
    size_t offset = get_index(mni_space);
    if(!offset || offset >= I.size())
        return false;
    if(is_track)
    {
        if(label_name_index >= track_base_pos.size())
            return false;
        size_t pos = track_base_pos[label_name_index] + offset;
        if(pos >= track.size())
            return false;
        return track[pos];
    }
    size_t l = I[offset];
    if(index2label.empty()) // not talairach
        return l == label_num[label_name_index];

    // The following is for talairach
    if(l >= index2label.size())
        return false;
    return std::find(index2label[l].begin(),index2label[l].end(),label_name_index) != index2label[l].end();
}
int atlas::get_track_label(const std::vector<tipl::vector<3> >& points)
{
    if(I.empty())
        load_from_file();
    if(!is_track)
        return -1;
    std::vector<int> vote(track_base_pos.size());
    for(size_t i = 0;i < points.size();++i)
    {
        size_t offset = get_index(points[i]);
        if(!offset)
            continue;
        for(size_t j = 0;j < track_base_pos.size();++j)
            if(track[track_base_pos[j] + offset])
                ++vote[j];
    }
    return int(std::max_element(vote.begin(),vote.end())-vote.begin());
}

