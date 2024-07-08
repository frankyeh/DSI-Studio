#include <sstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <cctype>
#include <QDir>
#include <QFileInfo>
#include "fib_data.hpp"



std::string run_auto_track(
                    tipl::program_option<tipl::out>& po,
                    const std::vector<std::string>& file_list,int& progress);

extern std::string auto_track_report;
int atk(tipl::program_option<tipl::out>& po)
{
    std::vector<std::string> file_list;
    if(!po.get_files("source",file_list))
    {
        tipl::error() << po.error_msg << std::endl;
        return 1;
    }


    if(file_list.empty())
    {
        tipl::error() << "no file listed in --source" << std::endl;
        return 1;
    }

    int progress;
    std::string error = run_auto_track(po,file_list,progress);
    if(error.empty())
    {
        if(po.has("report"))
        {
            std::ofstream out(po.get("report").c_str());
            out << auto_track_report;
        }
        return 0;
    }
    tipl::error() << error << std::endl;
    return 1;
}
