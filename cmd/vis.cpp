#include <QApplication>
#include <QFileInfo>
#include "libs/tracking/tract_model.hpp"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"
#include "program_option.hpp"

std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);

int vis(program_option& po)
{
    std::shared_ptr<fib_data> new_handle = cmd_load_fib(po.get("source"));
    if(!new_handle.get())
        return 1;
    show_progress() << "starting gui" << std::endl;
    tracking_window* new_mdi = new tracking_window(nullptr,new_handle);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(po.get("source").c_str());
    new_mdi->show();
    new_mdi->resize(1980,1000);

    if(po.has("track"))
    {
        if(!std::filesystem::exists(po.get("track")))
        {
            show_progress() << "ERROR:" << po.get("track") << " does not exist" << std::endl;
            return 1;
        }
        new_mdi->tractWidget->load_tracts(QString(po.get("track").c_str()).split(","));
    }
    QStringList cmd = QString(po.get("cmd").c_str()).split('+');
    for(unsigned int index = 0;index < cmd.size();++index)
    {
        QStringList param = cmd[index].split(',');
        if(!new_mdi->command(param[0],param.size() > 1 ? param[1]:QString(),param.size() > 2 ? param[2]:QString()))
        {
            show_progress() << "unknown command:" << param[0].toStdString() << std::endl;
            break;
        }

    }

    if(!po.has("stay_open"))
    {
        new_mdi->close();
        delete new_mdi;
    }
    return 0;
}
