#include <QApplication>
#include <QFileInfo>
#include "libs/tracking/tract_model.hpp"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"

std::shared_ptr<fib_data> cmd_load_fib(std::string file_name);
void get_filenames_from(const std::string param,std::vector<std::string>& filenames);
extern bool has_gui;
int vis(tipl::io::program_option<show_progress>& po)
{
    std::shared_ptr<fib_data> new_handle = cmd_load_fib(po.get("source"));
    if(!new_handle.get())
        return 1;
    bool has_gui_ = has_gui;
    has_gui = false;
    show_progress() << "starting gui" << std::endl;
    tracking_window* new_mdi = new tracking_window(nullptr,new_handle);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(po.get("source").c_str());
    new_mdi->show();
    new_mdi->resize(1980,1000);

    if(po.has("tract"))
    {
        std::vector<std::string> filenames;
        get_filenames_from(po.get("tract").c_str(),filenames);
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
            show_progress() << "ERROR: " << new_mdi->error_msg << std::endl;
            break;
        }
    }
    if(!po.has("stay_open"))
    {
        new_mdi->close();
        delete new_mdi;
    }
    has_gui = has_gui_;
    return 0;
}
