#include <QApplication>
#include <QFileInfo>
#include "libs/tracking/tract_model.hpp"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"

std::shared_ptr<fib_data> cmd_load_fib(tipl::program_option<tipl::out>& po);
int vis(tipl::program_option<tipl::out>& po)
{
    std::shared_ptr<fib_data> new_handle = cmd_load_fib(po);
    if(!new_handle.get())
        return 1;
    auto prior_show_prog = tipl::show_prog;
    tipl::show_prog = false;
    tipl::out() << "starting gui" << std::endl;
    tracking_window* new_mdi = new tracking_window(nullptr,new_handle);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(po.get("source").c_str());
    new_mdi->show();
    new_mdi->resize(1980,1000);

    if(po.has("tract"))
    {
        std::vector<std::string> filenames;
        if(!po.get_files("tract",filenames))
        {
            tipl::out() << "ERROR: " << po.error_msg << std::endl;
            return 1;
        }
        QStringList tracts;
        for(auto file : filenames)
            tracts << file.c_str();
        new_mdi->tractWidget->load_tracts(tracts);
        new_mdi->tractWidget->check_all();
    }
    QStringList cmd = QString(po.get("cmd").c_str()).split('+');
    for(unsigned int index = 0;index < cmd.size();++index)
    {
        QStringList param = cmd[index].split(',');
        if(!new_mdi->command(param[0],param.size() > 1 ? param[1]:QString(),param.size() > 2 ? param[2]:QString()))
        {
            tipl::out() << "ERROR: " << new_mdi->error_msg << std::endl;
            break;
        }
    }
    if(!po.has("stay_open"))
    {
        new_mdi->close();
        delete new_mdi;
    }
    tipl::show_prog = prior_show_prog;
    return 0;
}

