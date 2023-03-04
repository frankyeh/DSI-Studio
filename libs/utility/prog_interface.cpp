#include <QProgressDialog>
#include <QApplication>
#include <QObject>
#include <memory>
#include <ctime>
#include <iostream>
#include <chrono>
#include <thread>
#include <QTime>
#include "prog_interface_static_link.h"
bool has_gui = false;
std::shared_ptr<QProgressDialog> progressDialog;
static std::vector<std::chrono::high_resolution_clock::time_point> process_time,t_last;
std::vector<std::string> progress::status_list;
std::vector<std::string> progress::at_list;

inline bool processing_time_less_than(int time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::high_resolution_clock::now() - process_time.back()).count() < time;
}
void progress::update_prog(bool show_now)
{
    if(!has_gui || !tipl::is_main_thread<0>() ||
        (!show_now && processing_time_less_than(250)))
        return;
    if(!progressDialog.get())
    {
        progressDialog.reset(new QProgressDialog(get_status().c_str(),"Cancel",0,100));
        progressDialog->setAttribute(Qt::WA_ShowWithoutActivating);
        progressDialog->activateWindow();
    }
    else
        progressDialog->setLabelText(get_status().c_str());

    progressDialog->show();
    progressDialog->raise();
    QApplication::processEvents();
}
std::string progress::get_status(void)
{
    std::string result;
    for(size_t i = 0;i < status_list.size();++i)
    {
        if(status_list[i].empty())
            continue;
        if(i && (result.empty() || result.back() != '\n'))
            result += "\n";
        result += QStringList(status_list[i].c_str())[0].toStdString();
        if(i < at_list.size())
        {
            result += " ";
            result += at_list[i];
        }
    }
    if(!result.empty() && result.back() == '\n')
        result.pop_back();
    return result;
}
void progress::begin_prog(const char* status,bool show_now)
{
    if(!tipl::is_main_thread<0>())
        return;
    status_list.push_back(status);
    process_time.resize(status_list.size());
    process_time.back() = std::chrono::high_resolution_clock::now();
    t_last.resize(status_list.size());
    t_last.back() = std::chrono::high_resolution_clock::now()+std::chrono::milliseconds(200);
    update_prog(show_now);
}


progress::~progress(void)
{
    if(!tipl::is_main_thread<0>())
        return;

    std::ostringstream out;

    {
        std::string unit("ms");
        float count = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - process_time.back()).count();
        if(count > 1000.0f)
        {
            count /= 1000.0f;
            unit = "s";
            if(count > 60.0f)
            {
                count /= 60.0;
                unit = "m";
                if(count > 60.0f)
                {
                    count /= 60.0f;
                    unit = "h";
                }
            }
        }
        out << count << " " << unit;
    }
    status_list.pop_back();
    print(out.str().c_str(),false,true);

    process_time.pop_back();
    t_last.pop_back();
    if(!has_gui)
        return;
    if(!status_list.empty())
    {
        update_prog();
        return;
    }
    at_list.clear();
    if(progressDialog.get())
    {
        prog_aborted_ = progressDialog->wasCanceled();
        progressDialog->close();
        progressDialog.reset();
    }
    QApplication::processEvents();
}
bool progress::check_prog(unsigned int now,unsigned int total)
{
    if(!has_gui || !tipl::is_main_thread<0>() || status_list.empty())
        return now < total;
    if(now >= total)
    {
        if(at_list.size() == status_list.size())
            at_list.back().clear();
        return false;
    }
    if(std::chrono::high_resolution_clock::now() > t_last.back())
    {
        t_last.back() = std::chrono::high_resolution_clock::now()+std::chrono::milliseconds(200);
        int expected_sec = (
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                            process_time.back()).count()*int(total-now)/int(now+1)/1000/60);
        at_list.resize(status_list.size());
        at_list.back() = QString("(%1/%2)").arg(now).arg(total).toStdString();
        if(expected_sec)
            at_list.back() += QString(" %1 min").arg(expected_sec).toStdString();
        update_prog();
        if(progressDialog.get())
        {
            progressDialog->setRange(0, int(total));
            progressDialog->setValue(int(now));
            if(progressDialog->wasCanceled())
                return false;
            QApplication::processEvents();
        }
    }
    return now < total;
}

bool progress::aborted(void)
{
    if(!has_gui || !tipl::is_main_thread<0>())
        return prog_aborted_;
    if(progressDialog.get())
        return progressDialog->wasCanceled();
    return prog_aborted_;
}



