
#include <xnat_dialog.h>


int xnat(tipl::program_option<tipl::out>& po)
{
    xnat_facade xnat_connection;
    std::string output = po.get("output");
    if(output.empty())
    {
        tipl::error() << "please specify --output" << std::endl;
        return 1;
    }
    if(QFileInfo(output.c_str()).isDir() && output.back() != '\\' && output.back() != '/')
        output += '/';

    if(!po.has("id"))
    {
        if(QFileInfo(output.c_str()).isDir())
            output += "data.txt";
        if(!tipl::ends_with(output,".txt"))
            output += ".txt";
        tipl::out() << "writing output to " << output << std::endl;
        xnat_connection.get_experiments_info(po.get("source","https://central.xnat.org/"),po.get("auth"));
    }
    else
    {
        if(!QFileInfo(output.c_str()).isDir())
        {
            tipl::error() << "please specify output directory using --output" << std::endl;
            return 1;
        }
        tipl::out() << "saving output to " << output << std::endl;
        xnat_connection.get_scans_data(po.get("source","https://central.xnat.org/"),po.get("auth"),po.get("id"),output);
    }

    while(xnat_connection.is_running())
        QCoreApplication::processEvents();

    if (xnat_connection.has_error())
    {
        tipl::error() << xnat_connection.error_msg << std::endl;
        return 1;
    }

    if(!po.has("id"))
    {
        tipl::out() << "write experiment info to " << output;
        if(!(std::ofstream(output) << xnat_connection.result))
            return tipl::error() << "cannot write to " << output,1;
    }
    return 0;
}
