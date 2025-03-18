#include <QApplication>
#include <QFileInfo>
#include "tracking/tracking_window.h"
extern std::vector<tracking_window*> tracking_windows;
int vis(tipl::program_option<tipl::out>& po)
{
    if(tracking_windows.empty())
    {
        tipl::error() << "please load a fib file to run --action=vis";
        return 1;
    }
    if(tracking_windows.size() != 1)
    {
        tipl::error() << "multiple fib files are currently opened. please close others.";
        return 1;
    }
    if(!po.has("cmd"))
    {
        tipl::error() << "please specify command using --cmd";
        return 1;
    }
    po.mute("cmd");
    for(auto each : tipl::split(po.get("cmd"),'+'))
    {
        auto param = tipl::split(each,',');
        param.resize(3);
        if(!tracking_windows.back()->command(param[0],param[1],param[2]))
        {
            tipl::error() << tracking_windows.back()->error_msg << std::endl;
            return 1;
        }
    }
    return 0;
}

