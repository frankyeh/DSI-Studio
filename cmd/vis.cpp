#include <QApplication>
#include <QFileInfo>
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
    for(auto each : tipl::split(po.get("cmd"),'+'))
    {
        if(!tracking_windows.back()->command(tipl::split(each,',')))
        {
            tipl::error() << tracking_windows.back()->error_msg << std::endl;
            return 1;
        }
    }
    return 0;
}

