#include <sstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <cctype>
#include <QDir>
#include <QFileInfo>
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
                    program_option& po,
                    const std::vector<std::string>& file_list,
                    const std::vector<unsigned int>& track_id,
                    float length_ratio,
                    std::string tolerance_string,
                    float track_voxel_ratio,
                    int tip,
                    bool export_stat,
                    bool export_trk,
                    bool overwrite,
                    bool default_mask,
                    bool export_template_trk,
                    size_t thread_count,
                    int& progress);

extern std::string auto_track_report;
void get_filenames_from(const std::string param,std::vector<std::string>& filenames);
int atk(program_option& po)
{
    std::vector<std::string> file_list;
    get_filenames_from(po.get("source"),file_list);
    if(file_list.empty())
    {
        std::cout << "no file listed in --source" << std::endl;
        return 1;
    }
    std::vector<unsigned int> track_id;
    {
        fib_data fib;
        fib.set_template_id(0);
        std::istringstream in(po.get("track_id","Fasciculus,Cingulum,Aslant,Corticos,Thalamic_R,Reticular,Optic,Fornix,Corpus"));
        std::string str;
        while(std::getline(in,str,','))
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
                    std::cout << "ERROR: track_id: cannot find track name containing " << str << std::endl;
                    return 1;
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
    std::string error = run_auto_track(po,file_list,track_id,
                                po.get("length_ratio",1.25f),
                                po.get("tolerance","16,18,20"),
                                po.get("track_voxel_ratio",2),
                                po.get("tip",32),
                                po.get("export_stat",1),
                                po.get("export_trk",1),
                                po.get("overwrite",0),
                                po.get("default_mask",0),
                                po.get("export_template_trk",0),
                                po.get("thread_count",std::thread::hardware_concurrency()),progress);
    if(error.empty())
    {
        if(po.has("report"))
        {
            std::ofstream out(po.get("report").c_str());
            out << auto_track_report;
        }
        return 0;
    }
    std::cout << "ERROR:" << error << std::endl;
    return 1;
}
