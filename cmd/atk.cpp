#include <sstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <cctype>
#include <QDir>
#include <QFileInfo>
#include "fib_data.hpp"
#include "TIPL/tipl.hpp"

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
                    tipl::io::program_option<show_progress>& po,
                    const std::vector<std::string>& file_list,
                    const std::vector<unsigned int>& track_id,int& progress);

extern std::string auto_track_report;
void get_filenames_from(const std::string param,std::vector<std::string>& filenames);
bool get_track_id(std::string track_id_text,std::vector<unsigned int>& track_id)
{
    if(track_id_text.empty())
        return true;
    fib_data fib;
    fib.set_template_id(0);
    std::istringstream in(track_id_text);
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
                show_progress() << "ERROR: track_id: cannot find track name containing " << str << std::endl;
                return false;
            }
        }

    }
    std::sort(track_id.begin(),track_id.end());
    track_id.erase(std::unique(track_id.begin(), track_id.end()), track_id.end());

    show_progress pout;
    pout << "target tracks:";
    for(unsigned int index = 0;index < track_id.size();++index)
        pout << " " << fib.tractography_name_list[track_id[index]];
    pout << std::endl;
    return true;
}
int atk(tipl::io::program_option<show_progress>& po)
{
    std::vector<std::string> file_list;
    get_filenames_from(po.get("source"),file_list);
    if(file_list.empty())
    {
        show_progress() << "no file listed in --source" << std::endl;
        return 1;
    }
    std::vector<unsigned int> track_id;
    if(!get_track_id(po.get("track_id","Fasciculus,Cingulum,Aslant,Corticos,Thalamic_R,Optic,Fornix,Corpus"),
                     track_id))
        return 1;
    int progress;
    std::string error = run_auto_track(po,file_list,track_id,progress);
    if(error.empty())
    {
        if(po.has("report"))
        {
            std::ofstream out(po.get("report").c_str());
            out << auto_track_report;
        }
        return 0;
    }
    show_progress() << "ERROR:" << error << std::endl;
    return 1;
}
