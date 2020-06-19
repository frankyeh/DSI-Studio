#include <sstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <cctype>
#include "fib_data.hpp"
#include "program_option.hpp"

bool find_string_case_insensitive(const std::string & str1, const std::string & str2)
{
  auto it = std::search(
    str1.begin(), str1.end(),
    str2.begin(),   str2.end(),
    [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
  );
  return (it != str1.end() );
}

std::string run_auto_track(
                    const std::vector<std::string>& file_list,
                    const std::vector<unsigned int>& track_id,
                    float length_ratio,
                    float tolerance,
                    unsigned int track_count,
                    int interpolation,int tip,
                    bool export_stat,
                    bool export_trk,
                    bool overwrite,
                    bool default_mask,
                    int& progress);

int atk(void)
{
    std::vector<std::string> file_list;
    file_list.push_back(po.get("source"));
    std::vector<unsigned int> track_id;
    {
        fib_data fib;
        fib.set_template_id(0);
        std::string track_id_str = po.get("track_id","Arcuate,Cingulum,Fornix,Aslant,Superior_L,Inferior_L,Inferior_F,Uncinate");
        std::replace(track_id_str.begin(),track_id_str.end(),',',' ');
        std::istringstream in(track_id_str);
        std::string str;
        while(in >> str)
        {
            try {
                track_id.push_back(uint32_t(std::stoi(str)));
            }
            catch (...)
            {
                bool find = false;
                for(size_t index = 0;index < fib.tractography_name_list.size();++index)
                    if(find_string_case_insensitive(fib.tractography_name_list[index],str))
                    {
                        track_id.push_back(index);
                        find = true;
                    }
                if(!find)
                {
                    std::cout << "invalid track_id: cannot find track name containing " << str << std::endl;
                    return false;
                }
            }

        }
        std::sort(track_id.begin(),track_id.end());
        track_id.erase(std::unique(track_id.begin(), track_id.end()), track_id.end());

        std::cout << "target tracks:";
        for(unsigned int index = 0;index < track_id.size();++index)
            std::cout << " " << fib.tractography_name_list[track_id[index]];
        std::cout << std::endl;
    }
    int progress;
    std::cout << run_auto_track(file_list,track_id,
                                po.get("length_ratio",1.25f),
                                po.get("tolerance",16.0f),
                                po.get("track_count",uint32_t(8000)),
                                po.get("interpolation",2),
                                po.get("tip",16),
                                po.get("export_stat",1),
                                po.get("export_trk",1),
                                po.get("overwrite",0),
                                po.get("default_mask",0),progress) << std::endl;
    return 1;
}
