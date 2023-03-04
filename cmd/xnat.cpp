
#include <xnat_dialog.h>


int xnat(tipl::io::program_option<tipl::out>& po)
{
    std::string output = po.get("output");
    if(QFileInfo(output.c_str()).isDir() && output.back() != '\\' && output.back() != '/')
        output += '/';

    if(!po.has("id"))
    {
        if(output.empty() || QFileInfo(output.c_str()).isDir())
            output = "data.txt";
        if(!QString(output.c_str()).endsWith(".txt"))
            output += ".txt";
        tipl::out() << "writing output to " << output << std::endl;
        xnat_connection.get_experiments_info(po.get("source","https://central.xnat.org/"),po.get("auth"));
    }
    else
    {
        if(output.empty())
            output = QDir::current().path().toStdString();
        if(!QFileInfo(output.c_str()).isDir())
        {
            tipl::out() << "ERROR: please specify output directory using --output" << std::endl;
            return 1;
        }
        tipl::out() << "writing output to " << output << std::endl;
        xnat_connection.get_scans_data(po.get("source","https://central.xnat.org/"),po.get("auth"),po.get("id"),output);
    }

    while(xnat_connection.is_running())
        QApplication::processEvents();

    if (xnat_connection.has_error())
    {
        tipl::out() << "ERROR: " << xnat_connection.error_msg << std::endl;
        return 1;
    }

    if(!po.has("id"))
    {
        tipl::out() << "write experiment info to " << output << std::endl;
        std::ofstream(output) << xnat_connection.result;
    }
    else
    {
        tipl::out() << "data saved to " << output << std::endl;
    }
    return 0;
}
