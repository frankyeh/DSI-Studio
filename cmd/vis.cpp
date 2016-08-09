
#include <QApplication>
#include <QFileInfo>
#include "libs/tracking/tract_model.hpp"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"
#include "program_option.hpp"

int vis(void)
{
    if(!po.has("cmd"))
    {
        std::cout << "No command specified" << std::endl;
    }
    std::string file_name = po.get("source");
    std::cout << "loading " << file_name << std::endl;
    std::shared_ptr<fib_data> new_handle(new fib_data);
    if (!new_handle->load_from_file(&*file_name.begin()))
    {
        std::cout << "load fib file failed: " << new_handle->error_msg << std::endl;
        return 0;
    }
    std::cout << "starting gui" << std::endl;
    tracking_window* new_mdi = new tracking_window(0,new_handle);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(file_name.c_str());
    new_mdi->show();
    if(!po.has("stay_open"))
        new_mdi->hide();

    if(po.has("track"))
    {
        std::cout << "loading tracks" << po.get("track") << std::endl;
        new_mdi->tractWidget->load_tracts(QString(po.get("track").c_str()).split(","));
    }
    QStringList cmd = QString(po.get("cmd").c_str()).split(';');
    for(unsigned int index = 0;index < cmd.size();++index)
    {
        QStringList param = cmd[index].split(',');
        std::cout << "run ";
        for(unsigned int j = 0;j < param.size();++j)
            std::cout << param[j].toStdString() << " ";
        std::cout << std::endl;
        if(!new_mdi->glWidget->command(param[0],param.size() > 1 ? param[1]:QString(),param.size() > 2 ? param[2]:QString()) &&
           !new_mdi->scene.command(param[0],param.size() > 1 ? param[1]:QString(),param.size() > 2 ? param[2]:QString()) &&
           !new_mdi->tractWidget->command(param[0],param.size() > 1 ? param[1]:QString(),param.size() > 2 ? param[2]:QString()))
        {
            std::cout << "unknown command:" << param[0].toStdString() << std::endl;
        }
    }

    if(!po.has("stay_open"))
    {
        new_mdi->close();
        delete new_mdi;
    }
    return 0;
}
