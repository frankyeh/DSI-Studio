
#include <QApplication>
#include <QFileInfo>
#include "boost/program_options.hpp"
#include "libs/tracking/tract_model.hpp"
#include "tracking/tracking_window.h"
#include "opengl/glwidget.h"

namespace po = boost::program_options;

// test example
// --action=ana --source=20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000

int vis(int ac, char *av[])
{
    // options for fiber tracking
    po::options_description ana_desc("analysis options");
    ana_desc.add_options()
    ("help", "help message")
    ("action", po::value<std::string>(), "cnt: connectometry analysis")
    ("source", po::value<std::string>(), "assign the .fib file name")
    ("track", po::value<std::string>(), "assign the .trk file name")
    ("cmd", po::value<std::string>()->default_value("save_image"), "specify the command")

    ;

    if(!ac)
    {
        std::cout << ana_desc << std::endl;
        return 1;
    }


    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(ana_desc).run(), vm);
    po::notify(vm);

    std::string file_name = vm["source"].as<std::string>();
    std::cout << "loading " << file_name << std::endl;
    std::auto_ptr<FibData> new_handle(new FibData);
    if (!new_handle->load_from_file(&*file_name.begin()))
    {
        std::cout << "load fib file failed: " << new_handle->error_msg << std::endl;
        return 0;
    }
    std::cout << "starting gui" << std::endl;
    std::auto_ptr<tracking_window> new_mdi(new tracking_window(0,new_handle.release()));
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(file_name.c_str());
    new_mdi->show();
    new_mdi->hide();

    if(vm.count("track"))
    {
        std::cout << "loading tracks" << vm["track"].as<std::string>() << std::endl;
        new_mdi->tractWidget->load_tracts(QString(vm["track"].as<std::string>().c_str()).split(","));
    }
    QStringList cmd = QString(vm["cmd"].as<std::string>().c_str()).split(';');
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
    new_mdi->close();
    return 0;
}
