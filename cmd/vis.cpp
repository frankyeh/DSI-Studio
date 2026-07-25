#include "mainwindow.h"
#include "tracking/tracking_window.h"
extern std::vector<tracking_window*> tracking_windows;
extern MainWindow* main_window;
int vis(tipl::program_option<tipl::out>& po)
{
    if(!po.check("cmd"))
        return 1;
    if(tracking_windows.empty())
    {
        if(!po.check("source"))
            return 1;
        main_window->loadFib(po.get("source").c_str());
    }
    if(tracking_windows.empty())
        return 1;
    po.mute("cmd");
    for(const auto& each : tipl::split(po.get("cmd"),'+'))
        if(auto* window = tracking_windows.back();
            !window->command(tipl::split(each,',')))
            return tipl::error() << window->error_msg,1;
    return 0;
}

